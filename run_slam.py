"""
run_slam.py — the orchestrator.

Replaces: mono_video.cc + System.cc's thread launching.
Real ORB-SLAM3 runs Tracking / LocalMapping / LoopClosing on three parallel
threads. We run them sequentially in one loop — simpler to follow, and the
result is identical for offline processing.

Two input modes:
  --realsense       live D435 (RGB-D, uses depth for instant init)
  --frames <dir>    replay pre-extracted PNG frames (monocular)

Usage:
  python3 run_slam.py --realsense --seconds 30
  python3 run_slam.py --frames ~/orb_scratch/IMG_1112 --calib calibration/x.json
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from camera import Camera
from extractor import Extractor
from matcher import Matcher
from frame import Frame
from atlas import Atlas
from tracking import Tracking
from local_mapping import LocalMapping
from vocabulary import Vocabulary
from loop_closing import LoopClosing
import initializer
from bundle_adjust import local_bundle_adjust, pose_only_optimize


class SLAMSystem:
    def __init__(self, camera, use_depth=False, verbose=True):
        self.camera = camera
        self.use_depth = use_depth
        self.verbose = verbose

        self.extractor = Extractor(nfeatures=1200, scale_factor=1.2, nlevels=8,
                                   ini_th_fast=20, min_th_fast=7)
        self.matcher = Matcher()
        self.atlas = Atlas()
        self.tracking = Tracking(camera, self.extractor, self.matcher,
                                 self.atlas.active_map)
        self.local_mapping = LocalMapping(camera, self.matcher,
                                          self.atlas.active_map)
        self.vocab = Vocabulary(n_words=64)
        self.loop_closer = LoopClosing(camera, self.matcher, self.vocab)

        self.frames = []
        self.init_candidate = None
        self.consecutive_lost = 0
        self.stats = {'tracked': 0, 'lost': 0, 'keyframes': 0,
                      'points_created': 0, 'points_culled': 0, 'loops': 0}

    def process(self, image, timestamp, depth_image=None):
        frame = Frame(image, timestamp, self.camera, self.extractor,
                      depth_image=depth_image)
        self.frames.append(frame)
        world_map = self.atlas.active_map

        # ── initialisation ───────────────────────────────────────────────
        if self.tracking.state == "NOT_INITIALIZED":
            if self.use_depth:
                ok, new_pts = initializer.init_rgbd(frame, world_map)
                if ok:
                    self.tracking.state = "OK"
                    self.tracking.last_frame = frame
                    self.tracking.mark_keyframe(frame)
                    self.local_mapping.recent_points.extend(new_pts)
                    self.stats['keyframes'] += 1
                    self.stats['points_created'] += len(new_pts)
                    self._log(f"[init] RGB-D success at frame {frame.id} "
                              f"({len(new_pts)} points)")
            else:
                if self.init_candidate is None:
                    self.init_candidate = frame
                    return frame
                ok, new_pts = initializer.init_monocular(
                    self.init_candidate, frame, self.matcher,
                    self.camera, world_map)
                if ok:
                    self.tracking.state = "OK"
                    self.tracking.last_frame = frame
                    self.tracking.mark_keyframe(frame)
                    self.local_mapping.recent_points.extend(new_pts)
                    self.stats['keyframes'] += 2
                    self.stats['points_created'] += len(new_pts)
                    self._log(f"[init] mono success at frame {frame.id} "
                              f"({len(new_pts)} points)")
                else:
                    self.init_candidate = frame   # slide the window forward
                    if frame.id % 30 == 0:
                        self._log(f"[init] waiting for parallax... frame {frame.id}")
            return frame

        # ── tracking ─────────────────────────────────────────────────────
        ok = self.tracking.track(frame)
        if not ok:
            self.stats['lost'] += 1
            self.consecutive_lost += 1
            self._log(f"[track] LOST at frame {frame.id} "
                      f"(consecutive: {self.consecutive_lost})")

            # Atlas behaviour: after sustained loss, abandon and start fresh
            if self.consecutive_lost >= 10:
                self._log(f"[atlas] starting NEW MAP after "
                          f"{self.consecutive_lost} lost frames")
                new_map = self.atlas.start_new_map()
                self.tracking.set_map(new_map)
                self.local_mapping.set_map(new_map)
                self.tracking.state = "NOT_INITIALIZED"
                self.tracking.last_frame = None
                self.tracking.last_keyframe = None
                self.tracking.velocity = None
                self.init_candidate = None
                self.consecutive_lost = 0
            return frame

        self.consecutive_lost = 0
        self.stats['tracked'] += 1
        pose_only_optimize(frame, world_map, self.camera)

        # ── keyframe -> local mapping ────────────────────────────────────
        if self.tracking.needs_new_keyframe(frame):
            new_pts, n_culled = self.local_mapping.process_new_keyframe(
                frame, use_depth=self.use_depth)
            self.tracking.mark_keyframe(frame)
            self.stats['keyframes'] += 1
            self.stats['points_created'] += len(new_pts)
            self.stats['points_culled'] += n_culled

            n_kf = world_map.n_keyframes()
            #if n_kf >= 3:
               # local_bundle_adjust(world_map, self.camera, window=5,
                        #            verbose=self.verbose)

            # ── loop closing ─────────────────────────────────────────────
            if n_kf % 10 == 0 and n_kf >= 20:
                descs = [kf.descriptors for kf in world_map.keyframes
                         if kf.descriptors is not None and len(kf.descriptors) > 0]
                if descs:
                    self.vocab.build(np.vstack(descs))
                result = self.loop_closer.detect_loop(frame, world_map)
                if result is not None:
                    matched_kf, sim, inliers = result
                    drift = self.loop_closer.measure_drift(frame, matched_kf)
                    self.stats['loops'] += 1
                    self._log(f"[LOOP] kf {frame.id} <-> kf {matched_kf.id} | "
                              f"sim={sim:.3f} inliers={inliers} | "
                              f"drift={drift:.3f}m (NOT corrected)")
        return frame

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ── outputs ──────────────────────────────────────────────────────────

    def save_trajectory_plot(self, path="trajectory_python.png"):
        poses = [f.pose for f in self.frames if f.pose is not None]
        if len(poses) < 2:
            print("Not enough poses to plot.")
            return None

        c = np.array([p[:3, 3] for p in poses])
        kf_c = np.array([kf.camera_center() for m in self.atlas.maps
                            for kf in m.keyframes if kf.pose is not None])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (i, j), (xl, yl), title in (
                (axes[0], (0, 2), ("X (m)", "Z (m)"), "Bird's-eye (X-Z)"),
                (axes[1], (0, 1), ("X (m)", "Y (m)"), "Front (X-Y)")):
            ax.plot(c[:, i], c[:, j], '-', lw=1.2, color='tab:orange', label='trajectory')
            if len(kf_c):
                ax.scatter(kf_c[:, i], kf_c[:, j], s=14, color='tab:blue',
                           zorder=4, label='keyframes')
            ax.scatter([c[0, i]], [c[0, j]], s=70, c='green', marker='o',
                       edgecolors='k', zorder=5, label='start')
            ax.scatter([c[-1, i]], [c[-1, j]], s=70, c='red', marker='s',
                       edgecolors='k', zorder=5, label='end')
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(alpha=0.3); ax.legend(fontsize=8)

        fig.suptitle("Python ORB-SLAM3 reimplementation")
        fig.tight_layout()
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {path}")
        return path

    def save_map_ply(self, path="map_points.ply"):
        pts = [mp.position for m in self.atlas.maps
               for mp in m.good_map_points()]
        if not pts:
            return None
        pts = np.asarray(pts)
        with open(path, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for p in pts:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        print(f"Saved {path} ({len(pts)} points)")
        return path

    def print_summary(self):
        n_posed = sum(1 for f in self.frames if f.pose is not None)
        total = len(self.frames)
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        print(f"Frames processed:    {total}")
        print(f"Frames with pose:    {n_posed} ({100.0 * n_posed / max(1, total):.1f}%)")
        print(f"Tracking successes:  {self.stats['tracked']}")
        print(f"Tracking losses:     {self.stats['lost']}")
        print(f"Keyframes:           {self.stats['keyframes']}")
        print(f"Map points created:  {self.stats['points_created']}")
        print(f"Map points culled:   {self.stats['points_culled']}")
        print(f"Loop closures found: {self.stats['loops']}")
        print()
        print(self.atlas.summary())

        poses = [f.pose for f in self.frames if f.pose is not None]
        if len(poses) >= 2:
            c = np.array([p[:3, 3] for p in poses])
            path_len = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
            net = float(np.linalg.norm(c[-1] - c[0]))
            print(f"\nPath length:      {path_len:.3f}"
                  f"{' m' if self.use_depth else ' (arbitrary units)'}")
            print(f"Net displacement: {net:.3f}"
                  f"{' m' if self.use_depth else ' (arbitrary units)'}")
            if not self.use_depth:
                print("  NOTE: monocular = no absolute scale. Distances are "
                      "relative only.")
        print("=" * 60)


# ── runners ──────────────────────────────────────────────────────────────

def run_realsense(args):
    import pyrealsense2 as rs

    camera = Camera.from_realsense(args.width, args.height, args.fps)
    print(camera)
    os.makedirs("calibration", exist_ok=True)
    camera.to_json("calibration/realsense_d435.json")

    slam = SLAMSystem(camera, use_depth=not args.mono, verbose=not args.quiet)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)

    profile = pipeline.start(config)
    print(f"\nStreaming at {args.fps} FPS. Capturing exactly 30 target frames (1 per second).\n")

    t0 = time.time()
    processed_count = 0
    frame_counter = 0
    skip_interval = 1  # process 6 frames per second for smoother tracking

    try:
        while processed_count < 30:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color:
                continue

            frame_counter += 1
            if frame_counter % skip_interval != 0:
                continue

            color_img = np.asanyarray(color.get_data())
            depth_img = np.asanyarray(depth.get_data()) if (depth and not args.mono) else None
            
            processed_count += 1
            print(f"Processing target frame {processed_count}/30")
            slam.process(color_img, time.time() - t0, depth_image=depth_img)
            
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        pipeline.stop()

    return slam


def run_frames(args):
    if args.calib:
        camera = Camera.from_json(args.calib)
    else:
        print("ERROR: --frames needs --calib <calibration.json>")
        sys.exit(1)
    print(camera)

    paths = sorted(glob.glob(os.path.join(args.frames, "*.png")) +
                   glob.glob(os.path.join(args.frames, "*.jpg")))
    if args.max_frames:
        paths = paths[:args.max_frames]
    if not paths:
        print(f"No images found in {args.frames}")
        sys.exit(1)
    print(f"Found {len(paths)} frames\n")

    slam = SLAMSystem(camera, use_depth=False, verbose=not args.quiet)
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        slam.process(img, i / float(args.fps))
    return slam


def main():
    ap = argparse.ArgumentParser(description="Python ORB-SLAM3 reimplementation")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--realsense", action="store_true", help="live D435 input")
    src.add_argument("--frames", type=str, help="directory of image frames")

    ap.add_argument("--calib", type=str, help="calibration JSON (for --frames)")
    ap.add_argument("--seconds", type=int, default=30, help="capture duration")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mono", action="store_true",
                    help="ignore depth even on RealSense (monocular mode)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    slam = run_realsense(args) if args.realsense else run_frames(args)

    slam.print_summary()
    slam.save_trajectory_plot()
    slam.save_map_ply()


if __name__ == "__main__":
    main()

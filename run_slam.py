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
from bundle_adjust import local_bundle_adjust, local_inertial_bundle_adjust, pose_only_optimize
# Phase 2: GTSAM backend is optional -- only imported if config actually
# requests it, so `gtsam` is not a hard dependency for anyone still on the
# scipy backend. See bundle_adjust_gtsam.py's docstring for why this
# exists and PROGRESS.md for the three bugs found verifying it.
try:
    from bundle_adjust_gtsam import local_bundle_adjust_gtsam
except ImportError:
    local_bundle_adjust_gtsam = None
import imu
import imu_init
from config import load_config


class SLAMSystem:
    def __init__(self, camera, use_depth=False, verbose=True, use_imu=False, config_path=None):
        self.camera = camera
        self.use_depth = use_depth
        self.verbose = verbose
        self.use_imu = use_imu
        self.cfg = load_config(config_path)

        # BUGFIX: config.py was loaded (self.cfg) but every module below
        # used to be constructed with hardcoded defaults instead of cfg's
        # values -- e.g. cfg["loop_closing"]["vocab_words"]=10000 existed on
        # disk but Vocabulary(n_words=64) below ignored it completely, and
        # nothing in cfg["extractor"]/["tracking"]/["local_mapping"] had any
        # effect at all. Wired through properly now so config.yaml edits
        # actually change behavior. See PROGRESS.md.
        self.extractor = Extractor(**self.cfg["extractor"])
        self.matcher = Matcher()
        self.atlas = Atlas()
        self.tracking = Tracking(camera, self.extractor, self.matcher,
                                 self.atlas.active_map, **self.cfg["tracking"])
        self.local_mapping = LocalMapping(camera, self.matcher,
                                          self.atlas.active_map,
                                          extractor=self.extractor,
                                          **self.cfg["local_mapping"])
        # Shared covisibility graph: LocalMapping rebuilds this dict IN
        # PLACE (see covisibility.py) after every keyframe, so Tracking
        # always sees the latest version through this one reference.
        self.tracking.covis_graph = self.local_mapping.covis_graph
        self.vocab = Vocabulary(n_words=self.cfg["loop_closing"]["vocab_words"])
        self.loop_closer = LoopClosing(camera, self.matcher, self.vocab,
                                       min_keyframe_gap=self.cfg["loop_closing"]["min_keyframe_gap"],
                                       consistency_checks=self.cfg["loop_closing"]["consistency_checks"])

        # ── Visual-inertial state (Phase 3/4) ───────────────────────────
        # imu_preint: the RUNNING preintegration accumulator since the last
        # keyframe. Created once the first keyframe exists, reset every
        # time a new keyframe is marked (see process() below). bias_* are
        # carried forward as constants once imu_init.py succeeds (see that
        # module's docstring for why bias isn't re-estimated online here).
        self.imu_preint = None
        self.imu_raw_buffer = []     # accumulated synchronized [t,gx,gy,gz,ax,ay,az] rows
        self._last_imu_t = None
        self.bias_gyro = np.zeros(3)
        self.bias_accel = np.zeros(3)
        self.last_ok_timestamp = None

        self.frames = []
        self.init_candidate = None
        self.consecutive_lost = 0
        self.stats = {'tracked': 0, 'lost': 0, 'keyframes': 0,
                      'points_created': 0, 'points_culled': 0, 'loops': 0,
                      'points_fused': 0, 'keyframes_culled': 0,
                      'imu_init_attempts': 0}

    def process(self, image, timestamp, depth_image=None, imu_samples=None):
        """
        imu_samples: optional (K,7) array of synchronized [t,gx,gy,gz,ax,ay,az]
        rows (see imu.py) covering the interval since the previous process()
        call. Already rotated into the camera frame. Ignored entirely
        unless use_imu=True.
        """
        frame = Frame(image, timestamp, self.camera, self.extractor,
                      depth_image=depth_image)
        
        # ── Diagnostic output immediately after creating each Frame ──
        # BUGFIX: Frame has no `valid_depth_count` attribute -- this used to
        # silently print 0 forever via getattr's fallback. n_valid_depths()
        # is the real accessor (see frame.py).
        if self.verbose:
            print(f"[RGBD] Frame {frame.id}: keypoints={len(frame.keypoints)}, "
                  f"valid_depth={frame.n_valid_depths()}")

        self.frames.append(frame)
        world_map = self.atlas.active_map

        if self.use_imu and imu_samples is not None and len(imu_samples) > 0:
            self._integrate_imu(imu_samples)

        # ── initialisation ───────────────────────────────────────────────
        if self.tracking.state == "NOT_INITIALIZED":
            if self.use_depth:
                ok, new_pts, fail_reason = initializer.init_rgbd(
                    frame, world_map, min_points=self.cfg["initializer"]["rgbd_min_points"])
                if ok:
                    self.tracking.state = "OK"
                    self.tracking.last_frame = frame
                    self.tracking.mark_keyframe(frame)
                    self.local_mapping.recent_points.extend(new_pts)
                    self.stats['keyframes'] += 1
                    self.stats['points_created'] += len(new_pts)
                    self.last_ok_timestamp = timestamp
                    if self.use_imu:
                        self._start_imu_segment()
                    self._log(f"[init] RGB-D success at frame {frame.id} "
                              f"({len(new_pts)} points)")
                else:
                    self._log(f"[init] RGB-D failed at frame {frame.id}: {fail_reason}")
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
                    self.last_ok_timestamp = timestamp
                    if self.use_imu:
                        self._start_imu_segment()
                    self._log(f"[init] mono success at frame {frame.id} "
                              f"({len(new_pts)} points)")
                else:
                    self.init_candidate = frame   # slide the window forward
                    if frame.id % 30 == 0:
                        self._log(f"[init] waiting for parallax... frame {frame.id}")
            return frame

        # ── tracking ─────────────────────────────────────────────────────
        ok = self.tracking.track(frame, imu_preint=self.imu_preint if self.use_imu else None)
        if ok:
            self.last_ok_timestamp = timestamp
        if not ok:
            self.stats['lost'] += 1
            self.consecutive_lost += 1
            self._log(f"[track] LOST at frame {frame.id} "
                      f"(consecutive: {self.consecutive_lost})")

            # Atlas behaviour: after sustained loss, abandon and start fresh.
            # With IMU initialized, give the pipeline a TIME-based grace
            # window instead of a fixed frame count -- tracking.py already
            # keeps propagating frame.pose via IMU alone during this
            # window (state RECENTLY_LOST), so a few seconds of bad visual
            # conditions (motion blur, a blank wall) can recover instead of
            # immediately abandoning the map.
            elapsed_lost = timestamp - self.last_ok_timestamp if self.last_ok_timestamp is not None else 0.0
            if self.use_imu and self.tracking.imu_initialized:
                give_up = (elapsed_lost >= self.cfg["imu"]["recently_lost_max_seconds"]
                           and self.consecutive_lost >= 3)
            else:
                give_up = self.consecutive_lost >= 10

            if give_up:
                self._log(f"[atlas] starting NEW MAP after "
                          f"{self.consecutive_lost} lost frames ({elapsed_lost:.2f}s)")
                new_map = self.atlas.start_new_map()
                self.tracking.set_map(new_map)
                self.local_mapping.set_map(new_map)
                self.tracking.covis_graph = self.local_mapping.covis_graph
                self.tracking.state = "NOT_INITIALIZED"
                self.tracking.last_frame = None
                self.tracking.last_keyframe = None
                self.tracking.velocity = None
                self.tracking.imu_initialized = False
                self.init_candidate = None
                self.consecutive_lost = 0
                self.imu_preint = None
            return frame

        self.consecutive_lost = 0
        self.stats['tracked'] += 1
        pose_only_optimize(frame, world_map, self.camera)

        # ── keyframe -> local mapping ────────────────────────────────────
        if self.tracking.needs_new_keyframe(frame):
            new_pts, n_culled, n_fused = self.local_mapping.process_new_keyframe(
                frame, use_depth=self.use_depth, depth_image=depth_image)
            self.tracking.mark_keyframe(frame)
            self.stats['keyframes'] += 1
            self.stats['points_created'] += len(new_pts)
            self.stats['points_culled'] += n_culled
            self.stats['points_fused'] += n_fused

            if self.use_imu:
                self._finalize_imu_segment(frame)

            # KeyFrameCulling: drop redundant keyframes so covisibility
            # rebuilds / BA windows / local-map matching don't grow forever
            # over a facility-length walk. Cheap enough to run every
            # keyframe; internally protects the most recent few.
            #
            # SKIPPED when IMU is active: each keyframe's imu_preint spans
            # from its IMMEDIATE predecessor at recording time. If that
            # predecessor gets culled, the chain imu_init.py and
            # local_inertial_bundle_adjust walk (consecutive keyframes in
            # world_map.keyframes) would silently desync from what each
            # segment actually covers. Fixing that properly means re-
            # concatenating preintegration segments across a cull, which
            # is real work deferred for now -- see PROGRESS notes.
            if not self.use_imu:
                n_kf_culled = self.local_mapping.cull_keyframes()
                self.stats['keyframes_culled'] += n_kf_culled

            # Re-enabled: the old dense/unbounded BA is what was hanging.
            # bundle_adjust.py now uses a bounded window, a sparse
            # graph-colored Jacobian, and only optimizes points with >=2
            # observations -- see that module's docstring for why this no
            # longer hangs. Real-time isn't a priority here, so this is
            # allowed to take however long it needs; it just won't grow
            # unbounded with map size anymore.
            n_kf = world_map.n_keyframes()
            if self.use_imu and not self.tracking.imu_initialized:
                self._try_imu_init(world_map)

            if self.use_imu and self.tracking.imu_initialized:
                if n_kf >= 3:
                    local_inertial_bundle_adjust(
                        world_map, self.camera, gravity=self.tracking.gravity,
                        bias_gyro=self.bias_gyro, bias_accel=self.bias_accel,
                        window=self.cfg["imu"]["ba_window"], verbose=self.verbose)
            elif n_kf >= 3:
                # Phase 2: pick the BA backend from config. GTSAM path
                # falls back to scipy with a warning if gtsam isn't
                # installed or config asks for it incorrectly -- never
                # silently no-ops bundle adjustment entirely.
                backend = self.cfg["bundle_adjust"].get("backend", "scipy")
                if backend == "gtsam" and local_bundle_adjust_gtsam is not None:
                    local_bundle_adjust_gtsam(world_map, self.camera,
                                              window=self.cfg["bundle_adjust"]["window"],
                                              max_iter=self.cfg["bundle_adjust"]["max_iter"],
                                              min_obs_to_optimize=self.cfg["bundle_adjust"]["min_obs_to_optimize"],
                                              huber_f_scale=self.cfg["bundle_adjust"]["huber_f_scale"],
                                              verbose=self.verbose)
                else:
                    if backend == "gtsam" and self.verbose:
                        print("[BA] config requested gtsam backend but gtsam "
                             "is not installed -- falling back to scipy")
                    local_bundle_adjust(world_map, self.camera,
                                        window=self.cfg["bundle_adjust"]["window"],
                                        max_iter=self.cfg["bundle_adjust"]["max_iter"],
                                        min_obs_to_optimize=self.cfg["bundle_adjust"]["min_obs_to_optimize"],
                                        huber_f_scale=self.cfg["bundle_adjust"]["huber_f_scale"],
                                        verbose=self.verbose)

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

    # ── IMU orchestration (Phase 3/4) ───────────────────────────────────

    def _integrate_imu(self, imu_samples):
        """Append raw synchronized samples to the running buffer (used by
        imu_init's observability gate) and integrate them into the RUNNING
        preintegration segment since the last keyframe, if one exists yet."""
        self.imu_raw_buffer.extend(imu_samples.tolist())
        if self.imu_preint is None:
            return
        t_prev = self._last_imu_t
        for row in imu_samples:
            t, gx, gy, gz, ax, ay, az = row
            if t_prev is not None:
                self.imu_preint.integrate_sample([gx, gy, gz], [ax, ay, az], t - t_prev)
            t_prev = t
        self._last_imu_t = t_prev

    def _start_imu_segment(self):
        """Begin accumulating a fresh preintegration segment from the
        just-created (first) keyframe forward."""
        self.imu_preint = imu.Preintegration(
            self.bias_gyro, self.bias_accel,
            noise_gyro=self.cfg["imu"]["noise_gyro"],
            noise_accel=self.cfg["imu"]["noise_accel"])
        self._last_imu_t = None

    def _finalize_imu_segment(self, keyframe):
        """Attach the just-completed segment to the new keyframe and start
        the next one."""
        if self.imu_preint is None:
            self._start_imu_segment()
            return
        keyframe.imu_preint = self.imu_preint
        keyframe.bias_gyro = self.bias_gyro.copy()
        keyframe.bias_accel = self.bias_accel.copy()
        self._start_imu_segment()

    def _try_imu_init(self, world_map):
        """
        Attempt staged inertial initialization (imu_init.py) once enough
        keyframes with attached preintegration segments exist. On success,
        assigns velocities to those keyframes, stores gravity/bias, and
        flips tracking.imu_initialized so future frames get IMU prediction
        and future keyframes get inertial BA.
        """
        kfs = [kf for kf in world_map.keyframes if kf.imu_preint is not None or kf.kf_seq == 0]
        kfs = sorted(kfs, key=lambda kf: kf.kf_seq)
        min_kf = self.cfg["imu"]["init_min_keyframes"]
        if len(kfs) < min_kf:
            return
        if kfs[-1].timestamp - kfs[0].timestamp < self.cfg["imu"]["init_window_seconds"]:
            return

        self.stats['imu_init_attempts'] += 1
        sync_samples = np.asarray(self.imu_raw_buffer) if self.imu_raw_buffer else np.zeros((0, 7))
        result = imu_init.initialize(
            kfs, sync_samples,
            gravity_mag=self.cfg["imu"]["gravity_magnitude"],
            min_gyro_std=self.cfg["imu"]["observability_min_gyro_std"],
            min_accel_std=self.cfg["imu"]["observability_min_accel_std"])

        if not result["success"]:
            self._log(f"[imu-init] not ready yet: {result['reason']}")
            return

        for kf, v in zip(kfs, result["velocities"]):
            kf.velocity = v
        self.bias_gyro = result["bias_gyro"]
        self.bias_accel = result["bias_accel"]
        self.tracking.gravity = result["gravity"]
        self.tracking.imu_initialized = True
        self._log(f"[imu-init] SUCCESS: gravity={np.round(result['gravity'], 3)} "
                  f"|g|={np.linalg.norm(result['gravity']):.3f} "
                  f"bias_gyro={np.round(result['bias_gyro'], 4)} "
                  f"bias_accel={np.round(result['bias_accel'], 4)}")

        # Fold the initialization straight into the map once (stands in for
        # ORB-SLAM3's VIBA1/VIBA2 re-refinement passes -- see
        # imu_init.py's docstring for why those aren't implemented here).
        local_inertial_bundle_adjust(world_map, self.camera, gravity=self.tracking.gravity,
                                     bias_gyro=self.bias_gyro, bias_accel=self.bias_accel,
                                     window=len(kfs), verbose=self.verbose)

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
        print(f"Keyframes culled:    {self.stats['keyframes_culled']}")
        print(f"Map points created:  {self.stats['points_created']}")
        print(f"Map points culled:   {self.stats['points_culled']}")
        print(f"Map points fused:    {self.stats['points_fused']}")
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
    """
    Live capture. Default path tracks on the COLOR image (fixed: intrinsics
    now genuinely come from the color stream that gets aligned-to and fed
    to the extractor, and baseline comes from real extrinsics -- see
    camera.py's docstring for what was wrong before).

    --ir switches to the recommended pipeline: track on the left infrared
    image instead. Global-shutter IR pairs better with future IMU
    preintegration than the rolling-shutter RGB sensor, and depth is
    natively registered to it (no rs.align needed at all). Only worth it if
    the facility has enough natural texture for emitter-off IR frames to be
    usable -- see the architecture notes. Uses emitter_on_off alternating
    mode: even frames keep the projector pattern (for depth), odd frames
    are clean for ORB (real-time isn't a priority, so halving the rate is fine).
    """
    import pyrealsense2 as rs

    if args.ir:
        camera = Camera.from_realsense_ir(args.width, args.height, args.fps)
    else:
        camera = Camera.from_realsense(args.width, args.height, args.fps)
    print(camera)
    os.makedirs("calibration", exist_ok=True)
    camera.to_json("calibration/realsense_d435.json")

    slam = SLAMSystem(camera, use_depth=not args.mono, verbose=not args.quiet)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.ir:
        config.enable_stream(rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps)
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    else:
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        align = rs.align(rs.stream.color)

    profile = pipeline.start(config)

    if args.ir:
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.global_time_enabled):
            depth_sensor.set_option(rs.option.global_time_enabled, 1)
        for s in profile.get_device().sensors:
            if s.supports(rs.option.global_time_enabled):
                s.set_option(rs.option.global_time_enabled, 1)
        if depth_sensor.supports(rs.option.emitter_on_off):
            depth_sensor.set_option(rs.option.emitter_on_off, 1)
            depth_sensor.set_option(rs.option.emitter_enabled, 1)

    total_target_frames = args.fps * args.seconds
    print(f"\nStreaming at {args.fps} FPS. Capturing approximately {total_target_frames} target frames ({args.fps} per second over {args.seconds} seconds).\n")

    t0 = time.time()
    processed_count = 0
    frame_counter = 0
    pending_ir_clean = None   # holds the last emitter-off IR frame while we wait for the paired depth

    try:
        while processed_count < total_target_frames:
            frames = pipeline.wait_for_frames()

            if args.ir:
                ir = frames.get_infrared_frame(1)
                depth = frames.get_depth_frame()
                if not ir:
                    continue
                try:
                    emitter_on = bool(ir.get_frame_metadata(
                        rs.frame_metadata_value.frame_laser_power_mode))
                except Exception:
                    emitter_on = (frame_counter % 2 == 0)

                if emitter_on:
                    if pending_ir_clean is not None and depth:
                        color_img = pending_ir_clean
                        depth_img = np.asanyarray(depth.get_data()) if not args.mono else None
                        pending_ir_clean = None
                    else:
                        frame_counter += 1
                        continue
                else:
                    pending_ir_clean = np.asanyarray(ir.get_data())
                    frame_counter += 1
                    continue
            else:
                frames = align.process(frames)
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if not color:
                    continue
                frame_counter += 1
                color_img = np.asanyarray(color.get_data())
                depth_img = np.asanyarray(depth.get_data()) if (depth and not args.mono) else None

            processed_count += 1
            print(f"Processing target frame {processed_count}/{total_target_frames}")
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
    ap.add_argument("--ir", action="store_true",
                    help="track on left IR (global shutter, natively depth-"
                         "registered) instead of RGB. See camera.py docstring. "
                         "Only worth it if the facility has enough natural "
                         "texture for emitter-off IR frames to be usable.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    slam = run_realsense(args) if args.realsense else run_frames(args)

    slam.print_summary()
    slam.save_trajectory_plot()
    slam.save_map_ply()


if __name__ == "__main__":
    main()

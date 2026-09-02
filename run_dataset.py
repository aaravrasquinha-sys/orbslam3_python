"""
run_dataset.py — the offline, deterministic entry point.

Reads a dataset written by record.py and replays it through the SAME
SLAMSystem pipeline run_slam.py's live mode uses (tracking, local mapping,
BA, loop detection) -- nothing about the per-frame logic changes here, only
where the frames come from.

WHY THIS SOLVES "keyframes were queuing and it was hanging":
run_slam.py never had threads or a queue to begin with -- see bundle_adjust.py's
docstring for the actual root cause (a dense, unbounded, rank-deficient
least-squares problem that would never converge). But a LIVE loop still
couples the camera's frame rate to however long local mapping + BA take for
that keyframe: if BA needs 30 seconds, the RealSense pipeline buffer fills
and frames get dropped or the process appears to hang regardless of how
fast the underlying math is fixed to be.

This script removes that coupling entirely: it's a for-loop over a fixed
list of files on disk. There is no clock, nothing to fall behind, and
nothing to buffer. Each iteration's local_mapping + bundle_adjust are free
to take as long as they need -- the "queue" conceptually described in the
architecture (track cheap frames immediately, drain local mapping to empty
before moving on) is satisfied trivially because we only ever have one
keyframe in flight at a time by construction.

Bonus over live capture: fixed RANSAC/numpy seeds make runs bit-identical,
which you need when comparing BA experiments against each other.

Usage:
    python3 run_dataset.py --dataset dataset/hallway_01
    python3 run_dataset.py --dataset dataset/hallway_01 --mono --max_frames 500
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np

from camera import Camera
from run_slam import SLAMSystem
import imu
from config import load_config


def _read_manifest(dataset_dir, manifest_filename="manifest.csv"):
    path = os.path.join(dataset_dir, manifest_filename)
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def run_dataset(args):
    # Fixed seeds -> bit-identical reruns, needed for debugging BA changes
    # against each other rather than against sensor/RANSAC noise.
    np.random.seed(args.seed)
    random.seed(args.seed)
    cv2.setRNGSeed(args.seed)

    calib_path = args.calib or os.path.join(args.dataset, "calibration", "camera.json")
    if not os.path.exists(calib_path):
        raise SystemExit(f"No calibration found at {calib_path}. "
                         f"Pass --calib explicitly if it lives elsewhere.")
    camera = Camera.from_json(calib_path)
    print(camera)

    rows = _read_manifest(args.dataset)
    if args.max_frames:
        rows = rows[:args.max_frames]
    if not rows:
        raise SystemExit(f"No frames listed in {args.dataset}/manifest.csv")
    print(f"Found {len(rows)} recorded frames in {args.dataset}\n")

    images_dir = os.path.join(args.dataset, "images")
    depth_dir = os.path.join(args.dataset, "depth")

    # ── IMU (Phase 3/4) ──────────────────────────────────────────────────
    sync_samples = None
    if args.imu:
        cfg = load_config(args.config)
        imu_csv = os.path.join(args.dataset, cfg["dataset"]["imu_filename"])
        if not os.path.exists(imu_csv):
            raise SystemExit(f"--imu given but no {imu_csv} found (record with `record.py --imu`)")
        T_cam_imu_path = os.path.join(args.dataset, "calibration", "T_cam_imu.txt")
        R_cam_imu = imu.load_T_cam_imu(T_cam_imu_path)
        raw = imu.load_imu_csv(imu_csv)
        sync_samples = imu.synchronize(raw, R_cam_imu=R_cam_imu)
        print(f"Loaded {len(sync_samples)} synchronized IMU samples "
             f"(R_cam_imu from {T_cam_imu_path if os.path.exists(T_cam_imu_path) else 'identity fallback'})")

    slam = SLAMSystem(camera, use_depth=not args.mono, verbose=not args.quiet,
                      use_imu=args.imu, config_path=args.config)

    t0 = float(rows[0]["timestamp_s"])
    prev_ts = None
    for i, row in enumerate(rows):
        img_path = os.path.join(images_dir, row["image_file"])
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"  [warn] could not read {img_path}, skipping")
            continue

        depth_image = None
        if not args.mono and row.get("depth_file"):
            depth_path = os.path.join(depth_dir, row["depth_file"])
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        raw_ts = float(row["timestamp_s"])
        timestamp = raw_ts - t0

        imu_slice = None
        if args.imu and sync_samples is not None and len(sync_samples) > 0:
            lo = prev_ts if prev_ts is not None else raw_ts
            rows_slice = imu.slice_between(sync_samples, lo, raw_ts)
            if len(rows_slice) > 0:
                imu_slice = rows_slice.copy()
                imu_slice[:, 0] -= t0   # same time origin as `timestamp`
        prev_ts = raw_ts

        # This single call does: track -> pose-only optimize -> (if this
        # frame becomes a keyframe) local mapping -> bundle adjust (visual
        # or visual-inertial) -> (every 10 keyframes) loop detection. All
        # synchronous, all sequential, no backlog possible -- see module
        # docstring.
        slam.process(image, timestamp, depth_image=depth_image, imu_samples=imu_slice)

        if i % 30 == 0 and not args.quiet:
            imu_flag = " | IMU init" if (args.imu and slam.tracking.imu_initialized) else ""
            print(f"  [{i}/{len(rows)}] t={timestamp:7.2f}s | "
                  f"kfs={slam.atlas.active_map.n_keyframes()} | "
                  f"pts={slam.atlas.active_map.n_map_points()}{imu_flag}")

    return slam


def main():
    ap = argparse.ArgumentParser(description="Replay a recorded dataset through the SLAM pipeline")
    ap.add_argument("--dataset", type=str, required=True, help="directory written by record.py")
    ap.add_argument("--calib", type=str, default=None,
                    help="override calibration JSON (default: <dataset>/calibration/camera.json)")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--mono", action="store_true", help="ignore depth even if recorded")
    ap.add_argument("--imu", action="store_true",
                    help="use recorded IMU for visual-inertial tracking + BA "
                         "(dataset must have been recorded with record.py --imu)")
    ap.add_argument("--config", type=str, default=None, help="optional config.yaml")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible runs")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    slam = run_dataset(args)
    slam.print_summary()
    slam.save_trajectory_plot()
    slam.save_map_ply()


if __name__ == "__main__":
    main()

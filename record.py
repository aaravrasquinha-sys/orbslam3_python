"""
record.py — dataset capture. Does ZERO SLAM.

This is the architectural fix for "keyframes were queuing and hanging":
that wasn't a threading problem (run_slam.py has no threads -- see
run_dataset.py's docstring for the full diagnosis), it was that a live
RealSense loop couples capture to an optimizer that can take 30+ seconds
per keyframe. Splitting into record.py (capture, full rate, disk) +
run_dataset.py (process, single-threaded, deterministic, reads from disk)
means the camera never waits on anything, and processing is a for-loop over
a finite file list -- nothing can fall behind because there's no clock.

Bonus, not possible with a live pipeline:
  - re-run BA experiments without re-recording
  - bit-identical reproducibility with fixed RANSAC/numpy seeds
  - a bootstrap pass over the whole dataset to train the loop-closure vocabulary

Two capture modes:
  RGB  (default)  color + depth, aligned. Simple, but rolling-shutter RGB
                   is a poor pairing with IMU preintegration later.
  --ir             left infrared + depth, natively registered (no align).
                   Global shutter. Uses emitter_on_off alternating mode:
                   patterned frames give good depth, clean frames give good
                   ORB features -- see camera.py's docstring for the
                   reasoning. Only worth it with enough natural texture.

IMU (--imu): enables accel (250 Hz) + gyro (200 Hz) as separate async
streams and writes them RAW to imu.csv, timestamped on the SAME clock as
the images (global_time_enabled is set on every sensor -- without this,
preintegration intervals are silently wrong). Synchronizing accel onto
gyro timestamps by linear interpolation happens at LOAD time (in imu.py,
Phase 3), not here -- record.py's only job is to get raw data onto disk
faithfully.

Usage:
    python3 record.py --out dataset/hallway_01 --seconds 120
    python3 record.py --out dataset/hallway_01 --ir --imu --seconds 120
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np


def _make_dirs(root, cfg):
    images_dir = os.path.join(root, cfg["dataset"]["images_dirname"])
    depth_dir = os.path.join(root, cfg["dataset"]["depth_dirname"])
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    return images_dir, depth_dir


def record(args):
    import pyrealsense2 as rs
    from config import load_config
    from camera import Camera

    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    images_dir, depth_dir = _make_dirs(args.out, cfg)

    pipeline = rs.pipeline()
    rs_config = rs.config()

    if args.ir:
        rs_config.enable_stream(rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps)
        rs_config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    else:
        rs_config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        rs_config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    if args.imu:
        rs_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 0)
        rs_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 0)

    profile = pipeline.start(rs_config)

    # Critical for IMU: put every sensor's timestamps on ONE clock.
    for sensor in profile.get_device().sensors:
        if sensor.supports(rs.option.global_time_enabled):
            sensor.set_option(rs.option.global_time_enabled, 1)

    align = None if args.ir else rs.align(rs.stream.color)

    if args.ir:
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_on_off):
            depth_sensor.set_option(rs.option.emitter_on_off, 1)
            depth_sensor.set_option(rs.option.emitter_enabled, 1)

    # Save calibration once, from the SAME profile images will be read from.
    calib_dir = os.path.join(args.out, "calibration")
    os.makedirs(calib_dir, exist_ok=True)
    if args.ir:
        ir_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        ir2_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile() \
            if profile.get_stream(rs.stream.infrared, 2) else None
        intr = ir_profile.get_intrinsics()
        baseline = 0.05
        if ir2_profile:
            extr = ir_profile.get_extrinsics_to(ir2_profile)
            baseline = float(np.linalg.norm(np.array(extr.translation)))
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        camera = Camera(intr.fx, intr.fy, intr.ppx, intr.ppy, intr.width, intr.height,
                        baseline=baseline, dist_coeffs=intr.coeffs, depth_scale=depth_scale)
    else:
        camera = Camera.from_realsense(args.width, args.height, args.fps)
    camera.to_json(os.path.join(calib_dir, "camera.json"))
    print(camera)

    if args.imu:
        imu_extrinsics = _get_imu_extrinsics(profile, rs, args.ir)
        if imu_extrinsics is not None:
            np.savetxt(os.path.join(calib_dir, "T_cam_imu.txt"), imu_extrinsics)
            print("Saved nominal T_cam_imu (librealsense factory extrinsic -- "
                  "treat as an initial guess, not a per-device calibration; "
                  "see imu_init.py notes once IMU lands).")

    manifest_path = os.path.join(args.out, cfg["dataset"]["manifest_filename"])
    imu_path = os.path.join(args.out, cfg["dataset"]["imu_filename"])

    manifest_f = open(manifest_path, "w", newline="")
    manifest_w = csv.writer(manifest_f)
    manifest_w.writerow(["frame_idx", "timestamp_s", "image_file", "depth_file", "emitter_on"])

    imu_f = open(imu_path, "w", newline="") if args.imu else None
    imu_w = csv.writer(imu_f) if imu_f else None
    if imu_w:
        imu_w.writerow(["timestamp_s", "stream", "x", "y", "z"])

    print(f"Recording to {args.out} ({'IR' if args.ir else 'RGB'} + depth"
          f"{' + IMU' if args.imu else ''}). Ctrl-C to stop early.\n")

    t0 = time.time()
    frame_idx = 0
    n_imu = 0
    pending_ir_clean = None
    pending_ir_ts = None

    try:
        while (time.time() - t0) < args.seconds:
            frames = pipeline.wait_for_frames()

            if imu_w:
                for f in frames:
                    if f.is_motion_frame():
                        mf = f.as_motion_frame()
                        md = mf.get_motion_data()
                        ts = mf.get_timestamp() / 1000.0   # ms -> s, global clock
                        stream = "accel" if mf.get_profile().stream_type() == rs.stream.accel else "gyro"
                        imu_w.writerow([f"{ts:.6f}", stream, md.x, md.y, md.z])
                        n_imu += 1

            if args.ir:
                ir = frames.get_infrared_frame(1)
                depth = frames.get_depth_frame()
                if not ir:
                    continue
                ts = ir.get_timestamp() / 1000.0
                try:
                    emitter_on = bool(ir.get_frame_metadata(
                        rs.frame_metadata_value.frame_laser_power_mode))
                except Exception:
                    emitter_on = (frame_idx % 2 == 0)

                if not emitter_on:
                    pending_ir_clean = np.asanyarray(ir.get_data())
                    pending_ir_ts = ts
                    continue
                if pending_ir_clean is None or not depth:
                    continue
                img = pending_ir_clean
                img_ts = pending_ir_ts
                depth_img = np.asanyarray(depth.get_data())
                pending_ir_clean = None
            else:
                frames = align.process(frames)
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if not color:
                    continue
                img = np.asanyarray(color.get_data())
                img_ts = color.get_timestamp() / 1000.0
                depth_img = np.asanyarray(depth.get_data()) if depth else None

            img_name = f"{frame_idx:06d}.png"
            depth_name = f"{frame_idx:06d}.png"
            cv2.imwrite(os.path.join(images_dir, img_name), img)
            if depth_img is not None:
                cv2.imwrite(os.path.join(depth_dir, depth_name), depth_img)
            manifest_w.writerow([frame_idx, f"{img_ts:.6f}", img_name,
                                 depth_name if depth_img is not None else "",
                                 int(args.ir)])

            if frame_idx % 30 == 0:
                elapsed = time.time() - t0
                print(f"  frame {frame_idx:5d} | t={elapsed:6.1f}s | "
                      f"imu samples={n_imu}")
            frame_idx += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        pipeline.stop()
        manifest_f.close()
        if imu_f:
            imu_f.close()

    print(f"\nWrote {frame_idx} frames"
          f"{f' and {n_imu} IMU samples' if args.imu else ''} to {args.out}")
    print(f"Replay with: python3 run_dataset.py --dataset {args.out}")


def _get_imu_extrinsics(profile, rs, use_ir):
    """
    T_cam_imu from gyro_profile.get_extrinsics_to(cam_profile), as a 4x4.
    This is librealsense's NOMINAL D435I extrinsic, not a per-device
    calibration -- treat it as an initial guess (see imu_init.py, Phase 4).
    """
    try:
        gyro_profile = profile.get_stream(rs.stream.gyro)
        cam_stream = rs.stream.infrared if use_ir else rs.stream.color
        cam_idx = 1 if use_ir else 0
        cam_profile = (profile.get_stream(cam_stream, cam_idx) if use_ir
                       else profile.get_stream(cam_stream))
        extr = gyro_profile.get_extrinsics_to(cam_profile)
        T = np.eye(4)
        T[:3, :3] = np.array(extr.rotation).reshape(3, 3)
        T[:3, 3] = np.array(extr.translation)
        return T
    except Exception as e:
        print(f"Warning: could not read IMU extrinsics ({e})")
        return None


def main():
    ap = argparse.ArgumentParser(description="Record a RealSense dataset for offline SLAM processing")
    ap.add_argument("--out", type=str, required=True, help="output dataset directory")
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--ir", action="store_true",
                    help="record left IR (alternating emitter) instead of RGB")
    ap.add_argument("--imu", action="store_true", help="also record accel+gyro")
    ap.add_argument("--config", type=str, default=None, help="optional config.yaml")
    args = ap.parse_args()
    record(args)


if __name__ == "__main__":
    main()

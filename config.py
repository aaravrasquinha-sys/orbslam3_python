"""
config.py — one YAML file for every tunable that used to be a hardcoded
default scattered across camera.py, tracking.py, local_mapping.py, and
bundle_adjust.py. Doesn't change any behavior by itself (defaults below
match what the code already used) -- it's here so that per-facility tuning
(texture, room scale, how aggressive to gate depth) doesn't require editing
source files, and so record.py / run_dataset.py can share one source of
truth for capture and replay settings.

Usage:
    from config import load_config
    cfg = load_config("config.yaml")   # or load_config() for defaults
    extractor = Extractor(**cfg["extractor"])
    tracking = Tracking(camera, extractor, matcher, world_map, **cfg["tracking"])
    local_mapping = LocalMapping(camera, matcher, world_map, **cfg["local_mapping"])
"""

import copy
import os
import yaml

DEFAULTS = {
    "capture": {
        "width": 640,
        "height": 480,
        "fps": 30,
        "use_ir": False,          # track on left IR instead of RGB -- see camera.py
        "use_depth": True,
        "enable_imu": False,      # set True once imu.py lands (Phase 3)
        "accel_rate_hz": 250,
        "gyro_rate_hz": 200,
    },
    "extractor": {
        "nfeatures": 1200,
        "scale_factor": 1.2,
        "nlevels": 8,
        "ini_th_fast": 20,
        "min_th_fast": 7,
    },
    "tracking": {
        "min_matches_for_pose": 15,
        "keyframe_min_matches": 50,
        "keyframe_min_displacement": 0.10,
        "keyframe_max_frames": 20,
    },
    "local_mapping": {
        "min_parallax_px": 2.0,
        "max_neighbors": 5,
        "culling_found_ratio": 0.25,
        "culling_min_obs": 3,
        # ORB-SLAM3's RGB-D point-creation rule (see local_mapping.py):
        "max_new_points_per_kf": 100,
        "depth_min": 0.3,
        "depth_max": 3.5,
        "depth_patch_radius": 2,
        "depth_rel_std_max": 0.02,
    },
    "bundle_adjust": {
        "window": 8,
        "max_iter": 15,
        "min_obs_to_optimize": 2,
        "huber_f_scale": 2.0,
    },
    "loop_closing": {
        "min_keyframe_gap": 30,
        "vocab_words": 10000,
        "consistency_checks": 3,   # require same candidate over N consecutive KFs
    },
    "imu": {
        # BMI085 datasheet fallbacks -- measured (record.py's static-camera
        # calibration recording) is better, but these work to get running.
        "noise_gyro": 1.7e-2,      # rad/s / sqrt(Hz)
        "noise_accel": 2.0e-2,     # m/s^2 / sqrt(Hz)
        "gravity_magnitude": 9.81,
        "init_min_keyframes": 6,
        "init_window_seconds": 2.0,
        "observability_min_gyro_std": 0.05,
        "observability_min_accel_std": 0.3,
        "recently_lost_max_seconds": 2.0,   # IMU-only survival window (Phase 4)
        "ba_window": 8,
    },
    "dataset": {
        "images_dirname": "images",
        "depth_dirname": "depth",
        "imu_filename": "imu.csv",
        "manifest_filename": "manifest.csv",
    },
}


def load_config(path=None):
    """
    Returns a deep-merged dict: DEFAULTS overridden by whatever's in the
    YAML file at `path` (only the keys present in the file are overridden;
    everything else keeps its default). Returns DEFAULTS untouched if path
    is None or doesn't exist.
    """
    cfg = copy.deepcopy(DEFAULTS)
    if path and os.path.exists(path):
        with open(path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        for section, values in user_cfg.items():
            if section in cfg and isinstance(values, dict):
                cfg[section].update(values)
            else:
                cfg[section] = values
    return cfg


def save_default_config(path="config.yaml"):
    """Write DEFAULTS to disk as a starting point for editing."""
    with open(path, "w") as f:
        yaml.safe_dump(DEFAULTS, f, sort_keys=False, default_flow_style=False)
    return path


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    save_default_config(out)
    print(f"Wrote default config to {out}")

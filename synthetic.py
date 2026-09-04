"""
synthetic.py — deterministic synthetic RGB-D sequences for regression testing.

WHY THIS EXISTS: every bug found during the Phase-0/1 forensic audit (the
initializer crash, the erase_observation threshold, the cull_keyframes
over-culling) was found by running a synthetic sequence and measuring map
statistics, NOT by running on real RealSense hardware. A synthetic sequence
is deterministic (same seed -> bit-identical run), needs no camera, runs in
seconds, and lets you isolate ONE variable (motion, texture, rotation rate)
at a time -- which is impossible with a physical camera and a 30cm cable.

This is meant to be a permanent fixture of the test suite, not a one-off
script: every future change to tracking.py / local_mapping.py /
bundle_adjust.py should be checked against scroll_sequence() (or a purpose-
built variant) before being trusted on real hardware.

Two sequence types:
  scroll_sequence()   — camera translates in front of a fixed random-texture
                        plane at constant depth. No rotation, no noise, no
                        depth holes. This is the "does the map lifecycle
                        even work in the easiest possible case" test -- if a
                        change makes scroll_sequence() worse, it is not the
                        camera's fault.
  yaw_sequence()      — adds a homography-warped rotation component, so the
                        frame-to-frame appearance genuinely changes (not
                        just translates), exercising the same failure mode
                        that collapsed real tracking at frame ~145 in the
                        original D435i logs.

Both return (images, depths, gt_poses) with gt_poses as 4x4 camera-to-world
matrices, so metrics.py can compute ATE/RPE directly against a real ground
truth instead of only inspecting internal map statistics.
"""

import numpy as np
import cv2


def _make_texture(rng, height, width_total):
    """A blurred-noise 'wall' -- rich, stationary, uncorrelated texture that
    ORB can always find plenty of features on. NOT meant to be realistic;
    meant to isolate map-lifecycle bugs from feature-detection bugs (use
    low_texture_patch() for the latter)."""
    tex = rng.randint(0, 255, (height, width_total), np.uint8)
    return cv2.GaussianBlur(tex, (3, 3), 0)


def low_texture_patch(height=480, width=640, rng=None):
    """A painted-wall-like low-texture image: smooth gradient + faint noise
    + one or two edges. Used to test the dense-RGB-D fallback path (Phase
    4) since NO sparse detector (ORB/SIFT/AKAZE) reliably finds features
    here -- see PROGRESS.md's feature-detector benchmark."""
    rng = rng or np.random.RandomState(0)
    img = np.tile(np.linspace(70, 95, width), (height, 1))
    img = np.clip(img + rng.normal(0, 2.0, (height, width)), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.line(img, (width // 5, 0), (width // 5 + 20, height), (150,), 2)
    return img


def scroll_sequence(n_frames=120, height=480, width=640, depth_m=2.0,
                    px_per_frame=8, seed=1, fx=385.0, fy=385.0):
    """
    Pure lateral translation in front of a fronto-parallel textured plane at
    constant depth. Ground truth is exact: camera moves +x at a constant
    rate, everything else fixed.

    Returns (images, depths, gt_poses, camera_dict).
    """
    rng = np.random.RandomState(seed)
    tex = _make_texture(rng, height, width + px_per_frame * n_frames)

    images, depths, gt_poses = [], [], []
    # meters per pixel at this depth, given fx -- keeps the translation
    # numerically consistent with the camera model instead of an arbitrary
    # unit, so ATE against gt_poses means something.
    m_per_px = depth_m / fx

    for i in range(n_frames):
        images.append(tex[:, i * px_per_frame: i * px_per_frame + width].copy())
        depths.append(np.full((height, width), int(depth_m * 1000), np.uint16))
        pose = np.eye(4)
        pose[0, 3] = i * px_per_frame * m_per_px
        pose[2, 3] = 0.0
        gt_poses.append(pose)

    cam = dict(fx=fx, fy=fy, cx=width / 2, cy=height / 2,
              width=width, height=height, depth_scale=0.001)
    return images, depths, gt_poses, cam


def yaw_sequence(n_frames=120, height=480, width=640, depth_m=2.0,
                 max_yaw_deg=25.0, seed=1, fx=385.0, fy=385.0):
    """
    Camera yaws back and forth (sinusoidal) in front of a large textured
    plane, simulating the "turn to look down a corridor" motion that
    collapsed tracking at frame ~145 in the original D435i logs. Depth
    stays constant (fronto-parallel plane) so the RGB-D initializer and
    unprojection stay exact -- this isolates ROTATION-under-matching from
    depth-quality issues, which is a separate failure mode (test with
    low_texture_patch() instead).
    """
    rng = np.random.RandomState(seed)
    base = _make_texture(rng, height, width * 3)
    base = base[:, width:2 * width]   # center crop, reused via homography

    K = np.array([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])
    images, depths, gt_poses = [], [], []
    for i in range(n_frames):
        yaw = np.deg2rad(max_yaw_deg * np.sin(2 * np.pi * i / n_frames))
        R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                     [0, 1, 0],
                     [-np.sin(yaw), 0, np.cos(yaw)]])
        H = K @ R @ np.linalg.inv(K)
        H /= H[2, 2]
        img = cv2.warpPerspective(base, H, (width, height))
        images.append(img)
        depths.append(np.full((height, width), int(depth_m * 1000), np.uint16))
        pose = np.eye(4)
        pose[:3, :3] = R
        gt_poses.append(pose)

    cam = dict(fx=fx, fy=fy, cx=width / 2, cy=height / 2,
              width=width, height=height, depth_scale=0.001)
    return images, depths, gt_poses, cam

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
    """
    A synthetic 'wall' with enough real structure for ORB and a ratio-test
    matcher to work the way they would on an actual facility environment.

    BUGFIX (found during Phase 3 verification): this used to be pure
    Gaussian-blurred random noise. That's fine for exercising the map
    LIFECYCLE (Phase 1's original bug-finding ablations never needed
    anything more), but it is a genuinely adversarial input for Lowe's
    ratio test specifically: blurred noise is highly self-similar, so a
    given patch frequently has multiple near-identical-looking
    neighboring patches elsewhere in the image, and the ratio test
    correctly flags these as ambiguous and rejects them -- that's the
    ratio test doing exactly its job, not a bug in it. Measured directly:
    on this texture, crossCheck matching found 456 matches between two
    consecutive scroll frames, but the ratio test (rightly suspicious of
    the texture's repetitiveness) accepted only 253 (45% rejected). On a
    more structured synthetic scene (below), crossCheck found 444 and the
    ratio test accepted 434 -- a 2% difference, consistent with what
    real, non-repetitive facility imagery should look like to a matcher.
    Real ORB-SLAM3's own test sequences are real photographs for exactly
    this reason: synthetic noise is not a neutral stand-in once a test
    exercises anything beyond basic map bookkeeping.

    Kept the SAME function name/signature so scroll_sequence() and
    yaw_sequence() need no changes -- only what's inside the wall changed.
    """
    canvas = np.zeros((height, width_total), np.uint8)
    n_shapes = int(width_total * height / 4000)  # scale with canvas area
    # BUGFIX (found immediately after switching from noise to circles):
    # filled CIRCLES fixed the ratio-test self-similarity problem (see
    # above) but introduced a NEW one -- smooth curved boundaries give
    # FAST/Harris corner detection much less precise sub-pixel
    # localization than sharp corners do (a circle's boundary has few
    # true corner points at all). Measured directly: switching this test
    # texture to circles alone took ATE on a 120-frame scroll sequence
    # from 0.0012m (blurred-noise baseline) to 0.24m with a 1.95m max
    # error and a 6.4% scale error -- a real, new problem, not
    # progress. RECTANGLES give both properties at once: sharp, precisely
    # localizable corners (measured: matched-point shift std dev 0.83px,
    # consistent with the original noise-texture baseline) AND enough
    # real distinctiveness for a ratio-test matcher to behave sanely
    # (measured: crossCheck 486 vs ratio 475 matches, ~2% difference,
    # matching what real non-repetitive imagery should look like).
    for _ in range(n_shapes):
        x, y = rng.randint(0, width_total), rng.randint(0, height)
        w, h = rng.randint(10, min(50, width_total // 8)), rng.randint(10, min(50, height // 6))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), int(rng.randint(50, 220)), -1)
    n_lines = n_shapes // 3
    for _ in range(n_lines):
        p1 = (rng.randint(0, width_total), rng.randint(0, height))
        p2 = (rng.randint(0, width_total), rng.randint(0, height))
        cv2.line(canvas, p1, p2, int(rng.randint(40, 230)), rng.randint(1, 4))
    return cv2.GaussianBlur(canvas, (3, 3), 0)


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


def two_depth_scroll_sequence(n_frames=60, height=480, width=640,
                              near_depth_m=1.0, far_depth_m=3.0,
                              px_per_frame=8, seed=2, fx=385.0, fy=385.0):
    """
    PHASE 7: two fronto-parallel textured planes at different depths --
    top half of the image at near_depth_m, bottom half at far_depth_m --
    camera translates laterally in front of both, same convention as
    scroll_sequence. Ground truth translation is calibrated to the NEAR
    plane; the far plane shows correspondingly less apparent parallax
    per frame, which is physically correct (a farther plane appears to
    move less for the same camera translation), not a modeling error.

    WHY THIS EXISTS: scroll_sequence() and yaw_sequence() both use a
    SINGLE fixed depth for the whole scene, so every point is either
    all-close or all-far -- neither can exercise local_mapping.py's
    close/far split (th_depth) or the Phase 7 fix that makes RGB-D mode
    call _triangulate_new_points for far/no-depth points (see
    PHASE7_ARCHITECTURE.md finding 3.3 and test_phase7_gate.py's
    far-point coverage test). With near_depth_m=1.0 and far_depth_m=3.0
    against the default th_depth=2.0, the top half is unambiguously
    close and the bottom half is unambiguously far.
    """
    rng = np.random.RandomState(seed)
    half_h = height // 2
    tex_near = _make_texture(rng, half_h, width + px_per_frame * n_frames)
    tex_far = _make_texture(rng, height - half_h, width + px_per_frame * n_frames)

    images, depths, gt_poses = [], [], []
    for i in range(n_frames):
        img = np.zeros((height, width), np.uint8)
        img[:half_h] = tex_near[:, i * px_per_frame: i * px_per_frame + width]
        img[half_h:] = tex_far[:, i * px_per_frame: i * px_per_frame + width]
        images.append(img)

        depth = np.zeros((height, width), np.uint16)
        depth[:half_h] = int(near_depth_m * 1000)
        depth[half_h:] = int(far_depth_m * 1000)
        depths.append(depth)

        pose = np.eye(4)
        pose[0, 3] = i * px_per_frame * (near_depth_m / fx)
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

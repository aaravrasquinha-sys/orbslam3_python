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
                    px_per_frame=8, seed=1, noise_std=15.0,
                    fx=379.81365966796875, fy=379.81365966796875,
                    cx=322.3706970214844, cy=238.14862060546875):
    """
    Pure lateral translation in front of a fronto-parallel textured plane at
    constant depth, with empirical depth jitter applied.

    Returns (images, depths, gt_poses, camera_dict).
    """
    rng = np.random.RandomState(seed)
    tex = _make_texture(rng, height, width + px_per_frame * n_frames)

    images, depths, gt_poses = [], [], []
    # meters per pixel at this depth, given fx -- keeps the translation
    # numerically consistent with the camera model instead of an arbitrary
    # unit, so ATE against gt_poses means something.
    m_per_px = depth_m / fx
    base_depth_mm = depth_m * 1000.0

    for i in range(n_frames):
        images.append(tex[:, i * px_per_frame: i * px_per_frame + width].copy())
        
        # Inject empirical depth jitter matching the physical D435i
        if noise_std > 0:
            noisy_depth = rng.normal(base_depth_mm, noise_std, (height, width))
            noisy_depth = np.clip(noisy_depth, 0, 65535).astype(np.uint16)
        else:
            noisy_depth = np.full((height, width), int(base_depth_mm), np.uint16)
            
        depths.append(noisy_depth)
        
        pose = np.eye(4)
        pose[0, 3] = i * px_per_frame * m_per_px
        pose[2, 3] = 0.0
        gt_poses.append(pose)

    cam = dict(fx=fx, fy=fy, cx=cx, cy=cy,
               width=width, height=height, depth_scale=0.001)
    return images, depths, gt_poses, cam


def yaw_sequence(n_frames=120, height=480, width=640, depth_m=2.0,
                 max_yaw_deg=25.0, seed=1, noise_std=15.0,
                 fx=379.81365966796875, fy=379.81365966796875,
                 cx=322.3706970214844, cy=238.14862060546875):
    """
    Camera yaws back and forth (sinusoidal) in front of a large textured
    plane, simulating the "turn to look down a corridor" motion that
    collapsed tracking at frame ~145 in the original D435i logs. Depth
    stays constant (fronto-parallel plane) with empirical depth jitter applied.
    """
    rng = np.random.RandomState(seed)
    base = _make_texture(rng, height, width * 3)
    base = base[:, width:2 * width]   # center crop, reused via homography

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    images, depths, gt_poses = [], [], []
    base_depth_mm = depth_m * 1000.0

    for i in range(n_frames):
        yaw = np.deg2rad(max_yaw_deg * np.sin(2 * np.pi * i / n_frames))
        R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                      [0, 1, 0],
                      [-np.sin(yaw), 0, np.cos(yaw)]])
        H = K @ R @ np.linalg.inv(K)
        H /= H[2, 2]
        img = cv2.warpPerspective(base, H, (width, height))
        images.append(img)
        
        # Inject empirical depth jitter matching the physical D435i
        if noise_std > 0:
            noisy_depth = rng.normal(base_depth_mm, noise_std, (height, width))
            noisy_depth = np.clip(noisy_depth, 0, 65535).astype(np.uint16)
        else:
            noisy_depth = np.full((height, width), int(base_depth_mm), np.uint16)
            
        depths.append(noisy_depth)
        
        pose = np.eye(4)
        pose[:3, :3] = R
        gt_poses.append(pose)

    cam = dict(fx=fx, fy=fy, cx=cx, cy=cy,
               width=width, height=height, depth_scale=0.001)
    return images, depths, gt_poses, cam

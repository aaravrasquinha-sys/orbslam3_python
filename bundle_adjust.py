"""
bundle_adjust.py — local bundle adjustment.

Maps to: ORB_SLAM3/src/Optimizer.cc
  LocalBundleAdjustment -> local_bundle_adjust()
  PoseOptimization      -> pose_only_optimize()

SIMPLIFICATION: real ORB-SLAM3 uses g2o (sparse Levenberg-Marquardt on SE3/Sim3
manifolds, robust Huber kernels, Schur-complement marginalisation). We use
scipy.optimize.least_squares on a dense parameterisation. Same objective
(minimise reprojection error jointly over poses and points), simpler solver,
no outlier down-weighting.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares


def _pose_to_vec(pose):
    """4x4 camera-to-world -> 6-vector (rotvec, translation)."""
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    return np.concatenate([rvec.flatten(), pose[:3, 3]])


def _vec_to_pose(v):
    R, _ = cv2.Rodrigues(v[:3].reshape(3, 1))
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = v[3:6]
    return pose


def _project(K, pose_cw, p_world):
    """pose_cw = world-to-camera 4x4. Returns pixel or None if behind camera."""
    p_cam = pose_cw[:3, :3] @ p_world + pose_cw[:3, 3]
    if p_cam[2] <= 1e-6:
        return None
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    return np.array([u, v])


def local_bundle_adjust(world_map, camera, window=5, max_iter=30, verbose=False):
    """
    Jointly refine the last `window` keyframes' poses and the 3D points they
    observe. The OLDEST keyframe in the window is held fixed as the anchor —
    without it the whole solution could drift/rotate freely.
    """
    kfs = [kf for kf in world_map.keyframes[-window:] if kf.pose is not None]
    if len(kfs) < 2:
        return 0

    kf_index = {kf.id: i for i, kf in enumerate(kfs)}

    # gather observations
    observations, point_ids = [], set()
    for kf in kfs:
        for kp_i, mp_id in enumerate(kf.map_point_ids):
            if mp_id is None:
                continue
            mp = world_map.map_points.get(mp_id)
            if mp is None or mp.is_bad:
                continue
            observations.append((kf_index[kf.id], mp_id,
                                 kf.points_undistorted[kp_i].astype(np.float64)))
            point_ids.add(mp_id)

    point_ids = sorted(point_ids)
    if len(observations) < 10 or not point_ids:
        return 0

    pt_index = {pid: i for i, pid in enumerate(point_ids)}
    n_free_poses = len(kfs) - 1     # kfs[0] is the fixed anchor
    n_points = len(point_ids)

    # pack parameters
    x0 = []
    for kf in kfs[1:]:
        x0.append(_pose_to_vec(kf.pose))
    for pid in point_ids:
        x0.append(world_map.map_points[pid].position)
    x0 = np.concatenate(x0) if x0 else np.zeros(0)
    if x0.size == 0:
        return 0

    anchor_pose = kfs[0].pose.copy()

    def unpack(x):
        poses = [anchor_pose]
        for i in range(n_free_poses):
            poses.append(_vec_to_pose(x[i * 6:(i + 1) * 6]))
        base = n_free_poses * 6
        pts = {pid: x[base + pt_index[pid] * 3: base + pt_index[pid] * 3 + 3]
               for pid in point_ids}
        return poses, pts

    def residuals(x):
        poses, pts = unpack(x)
        poses_cw = [np.linalg.inv(p) for p in poses]
        res = np.zeros(len(observations) * 2)
        for k, (ki, pid, obs) in enumerate(observations):
            proj = _project(camera.K, poses_cw[ki], pts[pid])
            if proj is None:
                res[2 * k:2 * k + 2] = 100.0     # heavy penalty, behind camera
            else:
                res[2 * k:2 * k + 2] = proj - obs
        return res

    before = float(np.sum(residuals(x0) ** 2))
    result = least_squares(residuals, x0, method='lm', max_nfev=max_iter * len(x0))
    after = float(np.sum(result.fun ** 2))

    # write back
    poses, pts = unpack(result.x)
    for i, kf in enumerate(kfs[1:], start=1):
        kf.set_pose(poses[i])
    for pid in point_ids:
        world_map.map_points[pid].position = np.asarray(pts[pid], dtype=np.float64)

    if verbose:
        print(f"    [BA] {len(kfs)} kfs, {n_points} pts, {len(observations)} obs | "
              f"cost {before:.1f} -> {after:.1f}")

    return len(observations)


def pose_only_optimize(frame, world_map, camera, max_iter=20):
    """
    Optimizer::PoseOptimization() — refine ONE frame's pose with all 3D points
    held fixed. Much cheaper than full BA; runs per-frame in the real system.
    """
    if frame.pose is None:
        return False

    obs = []
    for kp_i, mp_id in enumerate(frame.map_point_ids):
        if mp_id is None:
            continue
        mp = world_map.map_points.get(mp_id)
        if mp is None or mp.is_bad:
            continue
        obs.append((mp.position.copy(), frame.points_undistorted[kp_i].astype(np.float64)))

    if len(obs) < 6:
        return False

    x0 = _pose_to_vec(np.linalg.inv(frame.pose))   # optimise world-to-camera

    def residuals(x):
        pose_cw = _vec_to_pose(x)
        res = np.zeros(len(obs) * 2)
        for k, (p3d, px) in enumerate(obs):
            proj = _project(camera.K, pose_cw, p3d)
            res[2 * k:2 * k + 2] = (proj - px) if proj is not None else 100.0
        return res

    result = least_squares(residuals, x0, method='lm', max_nfev=max_iter * 6)
    frame.set_pose(np.linalg.inv(_vec_to_pose(result.x)))
    return True
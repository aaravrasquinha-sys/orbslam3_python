"""
bundle_adjust.py — local bundle adjustment.

Maps to: ORB_SLAM3/src/Optimizer.cc
   LocalBundleAdjustment -> local_bundle_adjust()
   PoseOptimization      -> pose_only_optimize()

WHY THE OLD VERSION HUNG (three compounding causes, worst first):
  1. Rank deficiency. Every unmatched depth pixel became a MapPoint with
     exactly ONE observation (2 residuals, 3 free unknowns). Feeding
     thousands of these into the optimizer made it wander a degenerate
     null space -- LM effectively never converges. FIX: only points with
     >=2 observations are optimized at all; everything else is excluded
     from the parameter vector and instead re-anchored rigidly to its host
     keyframe's pose after the solve (see _reattach_single_obs_points).
  2. Dense, finite-differenced Jacobian. With window=5 and RGB-D point
     creation, the parameter vector was ~15,000-dimensional; finite-
     differencing a DENSE Jacobian costs ~15,000 residual evaluations per
     LM iteration. FIX: pass a sparse `jac_sparsity` pattern -- scipy then
     uses graph coloring and needs ~O(20) evaluations regardless of problem
     size, because unrelated parameter blocks (a distant keyframe's pose vs
     a point it never observed) share color groups.
  3. Per-observation Python loop building the residual vector. FIX: fully
     vectorized with numpy (batched rotation application via
     scipy.spatial.transform.Rotation, einsum for the batched projection).

Also new: the window is no longer anchored by arbitrarily fixing its oldest
keyframe. Instead, any keyframe OUTSIDE the window that observes a point
INSIDE the window is included as a FIXED camera -- it contributes residuals
(anchoring the solution against real covisibility) but no free parameters.
This is closer to what real windowed BA does and is more robust than a
single fixed anchor, especially early in a sequence. If no such external
keyframes exist (e.g. very early on), the oldest keyframe in the window is
fixed as a fallback so the solve isn't gauge-free.

SIMPLIFICATION: real ORB-SLAM3 uses g2o (sparse LM on SE3 manifolds, robust
kernels, Schur-complement marginalisation). We use scipy.optimize.least_squares
with a Huber loss and a supplied sparsity pattern -- same objective, much
cheaper solver than the old dense version, still not the hand-rolled
Schur-complement LM described as the eventual endgame for facility-scale maps.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from imu import log_so3


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


PENALTY = 1e3   # residual value assigned to observations that fall behind the camera


def local_bundle_adjust(world_map, camera, window=8, max_iter=15, verbose=False,
                         min_obs_to_optimize=2, huber_f_scale=2.0):
    """
    Jointly refine the last `window` keyframes' poses and the points they
    observe with >=2 total observations. Cost is roughly constant per call
    regardless of total map size (see module docstring for why).
    """
    all_kfs = [kf for kf in world_map.keyframes if kf.pose is not None]
    if len(all_kfs) < 2:
        return 0

    free_kfs = all_kfs[-window:]
    free_ids = {kf.id for kf in free_kfs}

    # ── choose which points are optimized ───────────────────────────────
    # A point is a candidate if a free keyframe observes it, has a valid
    # (non-bad) MapPoint, and that MapPoint has >=2 total observations.
    free_point_ids = []
    seen = set()
    for kf in free_kfs:
        for kp_i, mp_id in enumerate(kf.map_point_ids):
            if mp_id is None or mp_id in seen:
                continue
            mp = world_map.map_points.get(mp_id)
            if mp is None or mp.is_bad:
                continue
            seen.add(mp_id)
            if mp.n_observations() >= min_obs_to_optimize:
                free_point_ids.append(mp_id)

    if not free_point_ids:
        return 0
    pt_index = {pid: i for i, pid in enumerate(free_point_ids)}
    free_point_set = set(free_point_ids)
    n_points = len(free_point_ids)

    # ── fixed anchor keyframes: outside the window, observe a free point ──
    fixed_kfs = []
    for kf in all_kfs:
        if kf.id in free_ids or kf.pose is None:
            continue
        if any(mp_id in free_point_set for mp_id in kf.map_point_ids):
            fixed_kfs.append(kf)

    if not fixed_kfs:
        # No external anchor available (e.g. very early in the sequence) --
        # fall back to fixing the oldest keyframe in the window so the
        # solution can't drift/rotate for free.
        fixed_kfs = [free_kfs[0]]
        free_kfs = free_kfs[1:]
        if not free_kfs:
            return 0

    n_free_poses = len(free_kfs)
    n_fixed_poses = len(fixed_kfs)
    cam_index = {kf.id: i for i, kf in enumerate(free_kfs)}
    cam_index.update({kf.id: n_free_poses + i for i, kf in enumerate(fixed_kfs)})

    # ── gather observations from BOTH free and fixed cameras ───────────
    obs_cam, obs_pt, obs_uv = [], [], []
    for kf in free_kfs + fixed_kfs:
        for kp_i, mp_id in enumerate(kf.map_point_ids):
            if mp_id not in free_point_set:
                continue
            obs_cam.append(cam_index[kf.id])
            obs_pt.append(pt_index[mp_id])
            obs_uv.append(kf.points_undistorted[kp_i])

    n_obs = len(obs_cam)
    if n_obs < 10:
        return 0
    obs_cam = np.asarray(obs_cam, dtype=np.int64)
    obs_pt = np.asarray(obs_pt, dtype=np.int64)
    obs_uv = np.asarray(obs_uv, dtype=np.float64)

    # ── fixed cameras' world-to-camera R,t are constants for this solve ──
    fixed_R = np.stack([np.linalg.inv(kf.pose)[:3, :3] for kf in fixed_kfs]) \
        if fixed_kfs else np.zeros((0, 3, 3))
    fixed_t = np.stack([np.linalg.inv(kf.pose)[:3, 3] for kf in fixed_kfs]) \
        if fixed_kfs else np.zeros((0, 3))

    # ── pack initial parameter vector: [free pose vecs..., point xyz...] ──
    x0 = np.concatenate(
        [_pose_to_vec(kf.pose) for kf in free_kfs] +
        [world_map.map_points[pid].position for pid in free_point_ids]
    )
    n_pose_params = n_free_poses * 6

    fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy

    def unpack_cams(x):
        """x -> world-to-camera (R, t) for every camera, free then fixed."""
        pose_vecs = x[:n_pose_params].reshape(n_free_poses, 6)
        rotvecs = pose_vecs[:, :3]
        t_c2w = pose_vecs[:, 3:6]
        R_c2w = Rotation.from_rotvec(rotvecs).as_matrix()          # (n_free,3,3)
        # invert camera-to-world -> world-to-camera: R' = R^T, t' = -R^T t
        R_free = np.transpose(R_c2w, (0, 2, 1))
        t_free = -np.einsum('nij,nj->ni', R_free, t_c2w)
        R_all = np.concatenate([R_free, fixed_R], axis=0) if n_fixed_poses else R_free
        t_all = np.concatenate([t_free, fixed_t], axis=0) if n_fixed_poses else t_free
        return R_all, t_all

    def residuals(x):
        R_all, t_all = unpack_cams(x)
        pts = x[n_pose_params:].reshape(n_points, 3)

        R_obs = R_all[obs_cam]                       # (n_obs,3,3)
        t_obs = t_all[obs_cam]                        # (n_obs,3)
        p_world = pts[obs_pt]                          # (n_obs,3)
        p_cam = np.einsum('nij,nj->ni', R_obs, p_world) + t_obs   # (n_obs,3)

        behind = p_cam[:, 2] <= 1e-6
        z_safe = np.where(behind, 1.0, p_cam[:, 2])
        u = fx * p_cam[:, 0] / z_safe + cx
        v = fy * p_cam[:, 1] / z_safe + cy
        res = np.empty((n_obs, 2), dtype=np.float64)
        res[:, 0] = u - obs_uv[:, 0]
        res[:, 1] = v - obs_uv[:, 1]
        res[behind] = PENALTY
        return res.reshape(-1)

    # ── sparse Jacobian pattern: which params affect which residual rows ──
    n_params = n_pose_params + n_points * 3
    sparsity = lil_matrix((n_obs * 2, n_params), dtype=bool)
    row_pairs = np.arange(n_obs) * 2
    for k in range(n_obs):
        r0 = row_pairs[k]
        cam = obs_cam[k]
        if cam < n_free_poses:      # only free cameras have parameters
            c0 = cam * 6
            sparsity[r0:r0 + 2, c0:c0 + 6] = True
        pc0 = n_pose_params + obs_pt[k] * 3
        sparsity[r0:r0 + 2, pc0:pc0 + 3] = True

    before = float(np.sum(residuals(x0) ** 2))
    try:
        result = least_squares(
            residuals, x0,
            jac_sparsity=sparsity.tocsr(),
            method='trf', tr_solver='lsmr',
            x_scale='jac', loss='huber', f_scale=huber_f_scale,
            max_nfev=max_iter * 25,   # ~25 evals/iter with graph-colored sparse jac
        )
        x_opt = result.x
    except ValueError as e:
        if verbose:
            print(f"    [BA] sparse solve failed ({e}); skipping this call")
        return 0
    after = float(np.sum(residuals(x_opt) ** 2))

    # ── write back ───────────────────────────────────────────────────────
    old_poses = {kf.id: kf.pose.copy() for kf in free_kfs}
    pose_vecs = x_opt[:n_pose_params].reshape(n_free_poses, 6)
    for kf, vec in zip(free_kfs, pose_vecs):
        kf.set_pose(_vec_to_pose(vec))

    new_points = x_opt[n_pose_params:].reshape(n_points, 3)
    for pid, p in zip(free_point_ids, new_points):
        world_map.map_points[pid].position = np.asarray(p, dtype=np.float64)

    n_reattached = _reattach_single_obs_points(world_map, free_kfs, old_poses)

    if verbose:
        print(f"    [BA] {n_free_poses} free + {n_fixed_poses} fixed kfs, "
              f"{n_points} pts, {n_obs} obs | cost {before:.1f} -> {after:.1f} "
              f"| {n_reattached} single-obs pts reattached")

    return n_obs


def _reattach_single_obs_points(world_map, moved_kfs, old_poses):
    """
    Points with exactly one observation were excluded from the optimization
    (see module docstring). If their host keyframe's pose moved, keep them
    rigidly attached: recompute their world position from the SAME camera-
    frame coordinates under the keyframe's NEW pose, rather than leaving
    them stale in the old frame or dragging them along naively.
    """
    n = 0
    for kf in moved_kfs:
        old_pose = old_poses.get(kf.id)
        new_pose = kf.pose
        if old_pose is None or new_pose is None:
            continue
        old_R, old_t = old_pose[:3, :3], old_pose[:3, 3]
        new_R, new_t = new_pose[:3, :3], new_pose[:3, 3]
        for mp_id in kf.map_point_ids:
            if mp_id is None:
                continue
            mp = world_map.map_points.get(mp_id)
            if mp is None or mp.is_bad or mp.n_observations() != 1:
                continue
            p_cam = old_R.T @ (mp.position - old_t)
            mp.position = new_R @ p_cam + new_t
            n += 1
    return n


def local_inertial_bundle_adjust(world_map, camera, gravity, bias_gyro=None, bias_accel=None,
                                 window=8, max_iter=15,
                                 verbose=False, min_obs_to_optimize=2,
                                 huber_f_scale=2.0, imu_weight_scale=1.0):
    """
    Visual-inertial windowed BA (Phase 3/4). Adds preintegration + gravity
    residuals between consecutive keyframes to the same windowed structure
    local_bundle_adjust() uses, so poses AND velocities are refined jointly
    with the points, using gravity as a shared constant.

    SCOPE SIMPLIFICATIONS (moving fast on purpose -- see imu.py/imu_init.py
    docstrings for the rest of this pipeline's simplifications):
      - Per-keyframe gyro/accel BIAS is NOT optimized here -- it's carried
        forward from initialization (imu_init.py) as a constant. Real
        ORB-SLAM3 adds bias as 6 more free parameters per keyframe with a
        random-walk residual between consecutive keyframes; skipping that
        shrinks the parameter count a lot for a first working version at
        the cost of not re-estimating bias drift during normal operation.
      - Gravity is a fixed constant passed in (from imu_init.py), not a
        jointly-refined parameter in every call.
      - IMU residuals are weighted by imu.Preintegration.weight() (a fixed
        diagonal approximation, not the properly propagated 9x9 covariance
        -- see imu.py's docstring).
      - The oldest keyframe in the window is always the (fixed) anchor,
        rather than local_bundle_adjust()'s "external covisible keyframe"
        anchoring -- IMU residuals between consecutive keyframes already
        constrain the window tightly, so a single anchor is enough here.

    bias_gyro/bias_accel: the CURRENT global bias estimate (from
    imu_init.py). Each keyframe's imu_preint was integrated with whatever
    bias was current AT THAT TIME (see run_slam.py) -- passing the latest
    estimate here lets Preintegration.corrected() first-order-correct each
    segment for any drift between its own recorded bias and the current
    estimate, rather than assuming every segment used identical bias.

    Only keyframes with valid pose, velocity, AND imu_preint (i.e. created
    after IMU initialization succeeded) participate.
    """
    kfs = [kf for kf in world_map.keyframes
          if kf.pose is not None and kf.velocity is not None and kf.imu_preint is not None]
    if len(kfs) < 3:
        return 0

    kfs = kfs[-window:]
    anchor = kfs[0]
    free_kfs = kfs[1:]
    if len(free_kfs) < 2:
        return 0
    free_ids = {kf.id for kf in free_kfs}

    # ── points: >=2 observations, seen by anchor or a free keyframe ─────
    free_point_ids, seen = [], set()
    for kf in [anchor] + free_kfs:
        for mp_id in kf.map_point_ids:
            if mp_id is None or mp_id in seen:
                continue
            mp = world_map.map_points.get(mp_id)
            if mp is None or mp.is_bad:
                continue
            seen.add(mp_id)
            if mp.n_observations() >= min_obs_to_optimize:
                free_point_ids.append(mp_id)
    pt_index = {pid: i for i, pid in enumerate(free_point_ids)}
    n_points = len(free_point_ids)
    free_point_set = set(free_point_ids)
    if n_points == 0:
        return 0

    n_free = len(free_kfs)
    fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy

    # ── visual observations, anchor + free cameras ──────────────────────
    obs_cam, obs_pt, obs_uv = [], [], []
    cam_index = {anchor.id: -1}   # -1 marks the fixed anchor camera
    cam_index.update({kf.id: i for i, kf in enumerate(free_kfs)})
    for kf in [anchor] + free_kfs:
        for kp_i, mp_id in enumerate(kf.map_point_ids):
            if mp_id not in free_point_set:
                continue
            obs_cam.append(cam_index[kf.id])
            obs_pt.append(pt_index[mp_id])
            obs_uv.append(kf.points_undistorted[kp_i])
    obs_cam = np.asarray(obs_cam, dtype=np.int64)
    obs_pt = np.asarray(obs_pt, dtype=np.int64)
    obs_uv = np.asarray(obs_uv, dtype=np.float64)
    n_obs = len(obs_cam)

    anchor_R_cw = np.linalg.inv(anchor.pose)[:3, :3]
    anchor_t_cw = np.linalg.inv(anchor.pose)[:3, 3]

    # ── pack x0: [ (pose6, vel3) per free kf ..., point xyz ... ] ───────
    n_state_per_kf = 9
    n_pose_vel_params = n_free * n_state_per_kf
    x0 = np.concatenate(
        [np.concatenate([_pose_to_vec(kf.pose), kf.velocity]) for kf in free_kfs] +
        [world_map.map_points[pid].position for pid in free_point_ids]
    )

    imu_pairs = [anchor] + free_kfs   # consecutive-pair residuals walk this list

    def unpack(x):
        c2w_R, c2w_t, vels = [], [], []
        for i in range(n_free):
            base = i * n_state_per_kf
            vec = x[base:base + 6]
            R, _ = cv2.Rodrigues(vec[:3].reshape(3, 1))
            c2w_R.append(R)
            c2w_t.append(vec[3:6])
            vels.append(x[base + 6:base + 9])
        pts = x[n_pose_vel_params:].reshape(n_points, 3)
        return c2w_R, c2w_t, vels, pts

    def residuals(x):
        c2w_R, c2w_t, vels, pts = unpack(x)
        # world-to-camera R,t per camera index (-1 = anchor, 0..n_free-1 = free)
        R_wc = [anchor_R_cw] + [R.T for R in c2w_R]
        t_wc = [anchor_t_cw] + [-R.T @ t for R, t in zip(c2w_R, c2w_t)]
        R_wc = np.stack(R_wc)
        t_wc = np.stack(t_wc)

        cam_idx_arr = obs_cam + 1   # shift so anchor (-1) -> 0
        p_world = pts[obs_pt]
        p_cam = np.einsum('nij,nj->ni', R_wc[cam_idx_arr], p_world) + t_wc[cam_idx_arr]
        behind = p_cam[:, 2] <= 1e-6
        z_safe = np.where(behind, 1.0, p_cam[:, 2])
        u = fx * p_cam[:, 0] / z_safe + cx
        v = fy * p_cam[:, 1] / z_safe + cy
        vis_res = np.empty((n_obs, 2))
        vis_res[:, 0] = u - obs_uv[:, 0]
        vis_res[:, 1] = v - obs_uv[:, 1]
        vis_res[behind] = PENALTY

        # inertial residuals: one (R,v,p) triple per consecutive pair
        inertial_res = []
        c2w_R_all = [anchor.pose[:3, :3]] + c2w_R
        c2w_t_all = [anchor.pose[:3, 3]] + c2w_t
        v_all = [anchor.velocity] + vels
        for i in range(1, len(imu_pairs)):
            kf_cur = imu_pairs[i]
            preint = kf_cur.imu_preint
            dt = preint.dt
            R_i, R_j = c2w_R_all[i - 1], c2w_R_all[i]
            p_i, p_j = c2w_t_all[i - 1], c2w_t_all[i]
            v_i, v_j = v_all[i - 1], v_all[i]

            if bias_gyro is not None and bias_accel is not None:
                dR_meas, dv_meas, dp_meas = preint.corrected(
                    bias_gyro - preint.bias_gyro, bias_accel - preint.bias_accel)
            else:
                dR_meas, dv_meas, dp_meas = preint.dR, preint.dv, preint.dp

            r_R = log_so3(dR_meas.T @ R_i.T @ R_j)
            r_v = R_i.T @ (v_j - v_i - gravity * dt) - dv_meas
            r_p = R_i.T @ (p_j - p_i - v_i * dt - 0.5 * gravity * dt ** 2) - dp_meas

            w_r, w_v, w_p = preint.weight()
            inertial_res.append(r_R * w_r * imu_weight_scale)
            inertial_res.append(r_v * w_v * imu_weight_scale)
            inertial_res.append(r_p * w_p * imu_weight_scale)

        return np.concatenate([vis_res.reshape(-1)] + inertial_res) if inertial_res \
            else vis_res.reshape(-1)

    before = float(np.sum(residuals(x0) ** 2))
    try:
        result = least_squares(residuals, x0, method='trf', loss='huber',
                               f_scale=huber_f_scale, max_nfev=max_iter * 30)
        x_opt = result.x
    except ValueError as e:
        if verbose:
            print(f"    [VI-BA] solve failed ({e}); skipping this call")
        return 0
    after = float(np.sum(residuals(x_opt) ** 2))

    old_poses = {kf.id: kf.pose.copy() for kf in free_kfs}
    c2w_R, c2w_t, vels, pts = unpack(x_opt)
    for kf, R, t, vel in zip(free_kfs, c2w_R, c2w_t, vels):
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        kf.set_pose(pose)
        kf.velocity = vel
    for pid, p in zip(free_point_ids, pts):
        world_map.map_points[pid].position = np.asarray(p, dtype=np.float64)

    n_reattached = _reattach_single_obs_points(world_map, free_kfs, old_poses)

    if verbose:
        print(f"    [VI-BA] {n_free} free + 1 anchor kfs, {n_points} pts, "
              f"{n_obs} visual obs, {len(imu_pairs)-1} imu pairs | "
              f"cost {before:.1f} -> {after:.1f} | {n_reattached} single-obs pts reattached")

    return n_obs


def pose_only_optimize(frame, world_map, camera, max_iter=20):
    """
    Optimizer::PoseOptimization() — refine ONE frame's pose with all 3D
    points held fixed. Cheap (6 unknowns): left as a small dense solve,
    vectorized so it stays fast even with a few thousand observations.
    """
    if frame.pose is None:
        return False

    p3ds, pxs = [], []
    for kp_i, mp_id in enumerate(frame.map_point_ids):
        if mp_id is None:
            continue
        mp = world_map.map_points.get(mp_id)
        if mp is None or mp.is_bad:
            continue
        p3ds.append(mp.position)
        pxs.append(frame.points_undistorted[kp_i])

    if len(p3ds) < 6:
        return False

    p3ds = np.asarray(p3ds, dtype=np.float64)
    pxs = np.asarray(pxs, dtype=np.float64)
    fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
    x0 = _pose_to_vec(np.linalg.inv(frame.pose))   # optimise world-to-camera

    def residuals(x):
        R, _ = cv2.Rodrigues(x[:3].reshape(3, 1))
        t = x[3:6]
        p_cam = p3ds @ R.T + t
        behind = p_cam[:, 2] <= 1e-6
        z_safe = np.where(behind, 1.0, p_cam[:, 2])
        u = fx * p_cam[:, 0] / z_safe + cx
        v = fy * p_cam[:, 1] / z_safe + cy
        res = np.stack([u - pxs[:, 0], v - pxs[:, 1]], axis=1)
        res[behind] = PENALTY
        return res.reshape(-1)

    try:
        result = least_squares(residuals, x0, method='trf', loss='huber',
                               f_scale=2.0, max_nfev=max_iter * 6)
    except ValueError:
        result = least_squares(residuals, x0, method='lm', max_nfev=max_iter * 6)
    frame.set_pose(np.linalg.inv(_vec_to_pose(result.x)))
    return True

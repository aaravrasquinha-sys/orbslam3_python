"""
imu_init.py — staged inertial initialization.

Maps to: ORB_SLAM3/src/LocalMapping.cc InitializeIMU() (the "actual
contribution" of ORB-SLAM3, per the architecture notes).

Because this project is RGB-D, scale is already metric -- fix s=1 and drop
it entirely. That removes the worst-conditioned unknown in the monocular
version of this problem and makes initialization far more reliable.

SCOPE NOTE (moving fast on purpose): this implements the two staged linear
solves (gyro bias, then gravity+accel-bias+velocities) described in the
architecture, run ONCE when enough keyframes + IMU excitation exist. It
does NOT implement:
  - the VIBA1/VIBA2 re-refinement passes at ~5s/~15s (real ORB-SLAM3 refines
    twice more after the initial solve; here initialization happens once,
    followed by one inertial BA call in run_slam.py to fold it into the map)
  - joint nonlinear refinement of all unknowns together (stage 3 in the
    architecture notes) -- the two linear stages are used as the final
    estimate, not just an initial guess for a joint solve
Both are real accuracy left on the table, but the pipeline is genuinely
using preintegrated IMU + solving for gravity/bias/velocity from it, which
is the actual "does this component work at all" bar for a first pass.
"""

import numpy as np

from imu import exp_so3, log_so3, slice_between


def observability_gate(sync_samples, t_start, t_end,
                       min_gyro_std=0.05, min_accel_std=0.3):
    """
    Gravity direction and accel bias are only observable under sufficient
    rotational and accelerational excitation (see architecture notes: this
    is where most VI-SLAM init attempts fail, and it's not a code problem
    -- walking a straight corridor at constant speed will silently produce
    garbage). Checks variance of gyro and accel norm over the init window;
    refuses to initialize until it passes.
    """
    rows = slice_between(sync_samples, t_start, t_end)
    if len(rows) < 20:
        return False, "not enough IMU samples in window"

    gyro_norm = np.linalg.norm(rows[:, 1:4], axis=1)
    accel_norm = np.linalg.norm(rows[:, 4:7], axis=1)
    gyro_std = float(np.std(gyro_norm))
    accel_std = float(np.std(accel_norm))

    if gyro_std < min_gyro_std:
        return False, (f"insufficient rotational excitation (gyro std "
                       f"{gyro_std:.4f} < {min_gyro_std}) -- operator should "
                       f"rotate the camera deliberately during the first few seconds")
    if accel_std < min_accel_std:
        return False, (f"insufficient accelerational excitation (accel std "
                       f"{accel_std:.4f} < {min_accel_std}) -- try some "
                       f"stop-start motion during the first few seconds")
    return True, "ok"


def _estimate_gyro_bias(keyframes):
    """
    Stage 1: Solve for gyro bias delta aligning preintegrated dR with visual relative rotation.
    """
    A_rows, b_rows = [], []
    for i in range(1, len(keyframes)):
        kf_prev, kf_cur = keyframes[i - 1], keyframes[i]
        preint = kf_cur.imu_preint
        if preint is None or preint.dt <= 0:
            continue
        R_prev, R_cur = kf_prev.pose[:3, :3], kf_cur.pose[:3, :3]
        R_rel_visual = R_prev.T @ R_cur
        residual0 = log_so3(preint.dR.T @ R_rel_visual)
        A_rows.append(preint.dR_dbg)
        b_rows.append(residual0)

    if len(A_rows) < 2:
        return np.zeros(3), False

    A = np.concatenate(A_rows, axis=0)
    b = np.concatenate(b_rows, axis=0)
    delta_bg, *_ = np.linalg.lstsq(A, b, rcond=None)
    return delta_bg, True


def _estimate_gravity_bias_velocities(keyframes, bias_gyro, gravity_mag=9.81):
    """
    Stage 2: Solve for velocities, gravity, and accel bias using direct kinematic equations.
    """
    n = len(keyframes)
    n_unknowns = n * 3 + 3 + 3
    rows_A, rows_b = [], []

    def vel_cols(i):
        return slice(i * 3, i * 3 + 3)

    g_cols = slice(n * 3, n * 3 + 3)
    ba_cols = slice(n * 3 + 3, n * 3 + 6)

    for i in range(n - 1):
        kf_i, kf_j = keyframes[i], keyframes[i + 1]
        preint = kf_j.imu_preint
        if preint is None or preint.dt <= 0:
            continue
        dt = preint.dt
        R_i = kf_i.pose[:3, :3]
        p_i, p_j = kf_i.pose[:3, 3], kf_j.pose[:3, 3]

        _, dv_raw, dp_raw = preint.corrected(bias_gyro - preint.bias_gyro, np.zeros(3))

        # Position equation: p_j = p_i + v_i*dt + 0.5*g*dt^2 + R_i @ dp
        A_p = np.zeros((3, n_unknowns))
        A_p[:, vel_cols(i)] = dt * np.eye(3)
        A_p[:, g_cols] = 0.5 * (dt ** 2) * np.eye(3)
        A_p[:, ba_cols] = R_i @ preint.dp_dba
        b_p = p_j - p_i - R_i @ dp_raw
        rows_A.append(A_p)
        rows_b.append(b_p)

        # Velocity equation: v_{i+1} = v_i + g*dt + R_i @ dv
        A_v = np.zeros((3, n_unknowns))
        A_v[:, vel_cols(i)] = -np.eye(3)
        A_v[:, vel_cols(i + 1)] = np.eye(3)
        A_v[:, g_cols] = -dt * np.eye(3)
        A_v[:, ba_cols] = R_i @ preint.dv_dba
        b_v = R_i @ dv_raw
        rows_A.append(A_v)
        rows_b.append(b_v)

    if len(rows_A) < 4:
        return None

    A = np.concatenate(rows_A, axis=0)
    b = np.concatenate(rows_b, axis=0)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)

    velocities = [x[vel_cols(i)] for i in range(n)]
    gravity_raw = x[g_cols]
    accel_bias = x[ba_cols]

    g_norm = np.linalg.norm(gravity_raw)
    gravity = (gravity_raw / g_norm * gravity_mag) if g_norm > 1e-6 else np.array([0, -gravity_mag, 0])

    return {"velocities": velocities, "gravity": gravity, "bias_accel": accel_bias}


def initialize(keyframes, sync_samples, gravity_mag=9.81,
              min_gyro_std=0.05, min_accel_std=0.3):
    """
    Run both stages. `keyframes` must be in chronological order, each with
    a valid `.pose` and `.imu_preint` linking it to the previous one (the
    first keyframe's imu_preint is ignored/may be None).

    Returns dict{success, reason, gravity, bias_gyro, bias_accel,
    velocities (list aligned with keyframes)} -- velocities/gravity/biases
    are None on failure.
    """
    if len(keyframes) < 4:
        return {"success": False, "reason": "need at least 4 keyframes to initialize"}

    t_start = keyframes[0].timestamp
    t_end = keyframes[-1].timestamp
    ok, reason = observability_gate(sync_samples, t_start, t_end,
                                    min_gyro_std=min_gyro_std, min_accel_std=min_accel_std)
    if not ok:
        return {"success": False, "reason": reason}

    delta_bg, ok1 = _estimate_gyro_bias(keyframes)
    if not ok1:
        return {"success": False, "reason": "gyro bias stage failed (too few preintegrated pairs)"}
    bias_gyro = keyframes[1].imu_preint.bias_gyro + delta_bg   # base bias + correction

    result = _estimate_gravity_bias_velocities(keyframes, bias_gyro, gravity_mag=gravity_mag)
    if result is None:
        return {"success": False, "reason": "gravity/velocity stage failed (too few preintegrated pairs)"}

    return {
        "success": True, "reason": "ok",
        "gravity": result["gravity"],
        "bias_gyro": bias_gyro,
        "bias_accel": result["bias_accel"],
        "velocities": result["velocities"],
    }

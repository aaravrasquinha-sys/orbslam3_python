"""
imu.py — loading/synchronizing raw IMU, and on-manifold preintegration.

Maps to: ORB_SLAM3/src/ImuTypes.cc (Preintegrated, IntegrateNewMeasurement)

SCOPE NOTE (moving fast on purpose): this implements the real Forster et al.
preintegration recursion for (ΔR, Δv, Δp) and the bias Jacobians used for
first-order correction, but SKIPS full 9x9 covariance propagation --
residual weighting in bundle_adjust.py uses a fixed diagonal weight scaled
by elapsed time instead of the properly propagated measurement covariance.
That's a real simplification (real ORB-SLAM3 uses the propagated
covariance as the residual's information matrix), but it doesn't change
what's being optimized, only how precisely different residuals are
relatively weighted -- reasonable to defer for a first working pipeline.

SIMPLIFICATION: raw gyro/accel samples are rotated into the CAMERA frame
ONCE, using only the ROTATIONAL part of T_cam_imu (see rotate_imu_to_camera
below). The translational lever arm between the IMU and camera optical
center is ignored. This means all preintegrated quantities and all
keyframe states (pose, velocity) live directly in the camera frame, which
keeps tracking.py and bundle_adjust.py free of per-residual frame
conversions -- at the cost of a small, constant modeling error proportional
to the lever arm length (a few cm on a D435i) times angular velocity.
"""

import csv
import os

import numpy as np


# ── SO(3) helpers ───────────────────────────────────────────────────────

def hat(w):
    """3-vector -> 3x3 skew-symmetric matrix."""
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]], dtype=np.float64)


def exp_so3(w):
    """so(3) vector -> SO(3) rotation matrix (Rodrigues)."""
    theta = np.linalg.norm(w)
    if theta < 1e-8:
        return np.eye(3) + hat(w)
    axis = w / theta
    K = hat(axis)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def log_so3(R):
    """SO(3) rotation matrix -> so(3) vector."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    w_hat = (R - R.T) / (2.0 * np.sin(theta))
    w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])
    return w * theta


def right_jacobian_so3(w):
    """SO(3) right Jacobian Jr(w), used to relate angular-velocity-space
    perturbations to the bias Jacobian of ΔR."""
    theta = np.linalg.norm(w)
    if theta < 1e-8:
        return np.eye(3) - 0.5 * hat(w)
    axis = w / theta
    K = hat(axis)
    A = (1 - np.cos(theta)) / theta
    B = (theta - np.sin(theta)) / theta
    return np.eye(3) - A * K + B * (K @ K)


def rotate_imu_to_camera(gyro, accel, R_cam_imu):
    """Apply the (rotation-only) extrinsic to raw IMU samples -- see module docstring."""
    return R_cam_imu @ gyro, R_cam_imu @ accel


# ── loading + synchronization ───────────────────────────────────────────

def load_imu_csv(path):
    """record.py's imu.csv -> dict{'accel': (N,4), 'gyro': (M,4)}, columns [t,x,y,z]."""
    accel, gyro = [], []
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["timestamp_s"])
            xyz = [float(row["x"]), float(row["y"]), float(row["z"])]
            (accel if row["stream"] == "accel" else gyro).append([t] + xyz)
    return {"accel": np.asarray(accel, dtype=np.float64).reshape(-1, 4),
           "gyro": np.asarray(gyro, dtype=np.float64).reshape(-1, 4)}


def synchronize(imu_data, R_cam_imu=None):
    """
    Linearly interpolate accel onto gyro timestamps (per the architecture
    notes: BMI085 delivers these as separate async streams). Returns
    (N,7) array: [t, gx,gy,gz, ax,ay,az], already rotated into the camera
    frame if R_cam_imu is given.
    """
    accel, gyro = imu_data["accel"], imu_data["gyro"]
    if len(gyro) == 0 or len(accel) < 2:
        return np.zeros((0, 7))

    t_g = gyro[:, 0]
    t_a, xyz_a = accel[:, 0], accel[:, 1:]
    # keep only gyro samples within the accel time range (no extrapolation)
    mask = (t_g >= t_a[0]) & (t_g <= t_a[-1])
    t_g = t_g[mask]
    gyro_xyz = gyro[mask, 1:]
    accel_interp = np.stack([np.interp(t_g, t_a, xyz_a[:, k]) for k in range(3)], axis=1)

    if R_cam_imu is not None:
        gyro_xyz = gyro_xyz @ R_cam_imu.T
        accel_interp = accel_interp @ R_cam_imu.T

    return np.concatenate([t_g[:, None], gyro_xyz, accel_interp], axis=1)


def slice_between(sync_samples, t0, t1):
    """sync_samples rows with t0 < t <= t1 -- the interval feeding one process() call."""
    if len(sync_samples) == 0:
        return sync_samples
    mask = (sync_samples[:, 0] > t0) & (sync_samples[:, 0] <= t1)
    return sync_samples[mask]


def load_T_cam_imu(path):
    """record.py's T_cam_imu.txt (4x4) -> rotation-only 3x3 (see module docstring)."""
    if not path or not os.path.exists(path):
        return np.eye(3)
    T = np.loadtxt(path)
    return T[:3, :3]


# ── preintegration ───────────────────────────────────────────────────────

class Preintegration:
    """
    Forster et al. (2015) on-manifold IMU preintegration between two
    keyframes. Accumulates one sample at a time via integrate_sample();
    query the running (dR, dv, dp, dt) at any point (it's a valid partial
    preintegration, which is what lets tracking.py use this same object for
    per-frame pose prediction before the segment is finalized at the next
    keyframe -- see run_slam.py).
    """

    def __init__(self, bias_gyro, bias_accel, noise_gyro=1.7e-2, noise_accel=2.0e-2):
        self.bias_gyro = np.asarray(bias_gyro, dtype=np.float64).copy()
        self.bias_accel = np.asarray(bias_accel, dtype=np.float64).copy()
        self.noise_gyro = noise_gyro     # rad/s / sqrt(Hz) -- BMI085 datasheet fallback
        self.noise_accel = noise_accel   # m/s^2 / sqrt(Hz)

        self.dR = np.eye(3)
        self.dv = np.zeros(3)
        self.dp = np.zeros(3)
        self.dt = 0.0

        # Bias Jacobians for first-order correction (Forster eq. 21-27,
        # accumulated with a simplified -- not exactly-manifold-consistent
        # in the dR term's use of a linear correction rather than a second
        # right-Jacobian factor -- but standard-practice recursion).
        self.dR_dbg = np.zeros((3, 3))
        self.dv_dbg = np.zeros((3, 3))
        self.dv_dba = np.zeros((3, 3))
        self.dp_dbg = np.zeros((3, 3))
        self.dp_dba = np.zeros((3, 3))

        self.n_samples = 0

    def integrate_sample(self, gyro, accel, dt):
        if dt <= 0:
            return
        w = np.asarray(gyro, dtype=np.float64) - self.bias_gyro
        a = np.asarray(accel, dtype=np.float64) - self.bias_accel

        dR_k = exp_so3(w * dt)
        Jr = right_jacobian_so3(w * dt)
        a_hat = hat(a)

        # Jacobian recursion (uses PRE-update dR)
        self.dp_dba += self.dv_dba * dt - 0.5 * self.dR * dt ** 2
        self.dp_dbg += self.dv_dbg * dt - 0.5 * self.dR @ a_hat @ self.dR_dbg * dt ** 2
        self.dv_dba += -self.dR * dt
        self.dv_dbg += -self.dR @ a_hat @ self.dR_dbg * dt
        self.dR_dbg = dR_k.T @ self.dR_dbg - Jr * dt

        # state recursion
        self.dp += self.dv * dt + 0.5 * (self.dR @ a) * dt ** 2
        self.dv += (self.dR @ a) * dt
        self.dR = self.dR @ dR_k

        self.dt += dt
        self.n_samples += 1

    def integrate_span(self, sync_samples, t_start, t_end):
        """Integrate every synchronized sample with t_start < t <= t_end."""
        rows = slice_between(sync_samples, t_start, t_end)
        t_prev = t_start
        for row in rows:
            t, gx, gy, gz, ax, ay, az = row
            self.integrate_sample([gx, gy, gz], [ax, ay, az], t - t_prev)
            t_prev = t

    def corrected(self, delta_bg, delta_ba):
        """
        First-order bias correction (Forster eq. 44): if the bias estimate
        changes by (delta_bg, delta_ba) since this was integrated, correct
        (dR, dv, dp) via the stored Jacobians instead of re-integrating from
        raw samples.
        """
        dR_corr = self.dR @ exp_so3(self.dR_dbg @ delta_bg)
        dv_corr = self.dv + self.dv_dbg @ delta_bg + self.dv_dba @ delta_ba
        dp_corr = self.dp + self.dp_dbg @ delta_bg + self.dp_dba @ delta_ba
        return dR_corr, dv_corr, dp_corr

    def weight(self, max_weight=50.0):
        """
        Rough fixed-diagonal residual weight (1/sigma), scaled by elapsed
        time, standing in for the properly propagated covariance -- see
        module docstring. Larger dt -> more accumulated noise -> lower
        confidence -> smaller weight.

        Clamped to max_weight: without full covariance propagation (or the
        VIBA re-refinement passes that would improve a weak initial accel-
        bias estimate -- see imu_init.py), a badly-observed bias can
        translate into real position drift between keyframes. An
        unclamped 1/sigma weight then makes that residual dominate the
        cost function by orders of magnitude and drags out convergence,
        without actually being "more correct" data -- it's compounding a
        known-weak estimate, not a well-characterized measurement. Capping
        it keeps VI-BA calls fast and stable while that limitation exists.
        """
        sigma_r = self.noise_gyro * np.sqrt(max(self.dt, 1e-3))
        sigma_v = self.noise_accel * np.sqrt(max(self.dt, 1e-3))
        sigma_p = 0.5 * self.noise_accel * max(self.dt, 1e-3) ** 1.5
        return (min(1.0 / max(sigma_r, 1e-6), max_weight),
               min(1.0 / max(sigma_v, 1e-6), max_weight),
               min(1.0 / max(sigma_p, 1e-6), max_weight))

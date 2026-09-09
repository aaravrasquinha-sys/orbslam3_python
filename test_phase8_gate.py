"""
test_phase8_gate.py — Phase 8: IMU-based motion prediction for tracking
robustness.

Run: python3 test_phase8_gate.py

CONTEXT: unlike every other phase, this one did NOT start from "build X".
The tracking-side algorithm (imu.py's preintegration, imu_init.py's
staged initialization, tracking.py's _predict_pose_imu prediction
strategy) already existed in the codebase, apparently unmodified since
before Phase 6. What this phase actually found: NOTHING had ever
exercised any of it end-to-end -- no gate test, no field log entry, not
even a mention in PROGRESS.md beyond "deferred to Phase 7" (a phase
number that, by the time it arrived, ended up covering something else
entirely). Four real bugs were sitting there as a direct result:

  1. run_slam.py's `_try_imu_init` had no `def` line at all -- the
     docstring and body were dead code hanging off the end of
     `_handle_loop_result`. `hasattr(SLAMSystem, '_try_imu_init')` was
     False. The call site would have raised AttributeError on the very
     first keyframe processed with use_imu=True.
  2. run_realsense() (the live hardware capture path) never enabled the
     D435i's accel/gyro motion streams, never passed use_imu=True to
     SLAMSystem, and main() had no --imu flag at all. Even with bug 1
     fixed, live hardware could never have fed IMU data into the
     pipeline.
  3. tracking.py only ever set frame.velocity inside Strategy 0's own
     success branch. The first time ANY other strategy is what actually
     succeeds -- entirely possible, including during the exact fast-
     motion conditions IMU prediction exists to help with -- that
     keyframe's velocity silently stays None, which permanently disables
     Strategy 0 for every subsequent frame with no recovery path.
  4. Once bug 1 was fixed and _try_imu_init could actually run,
     local_inertial_bundle_adjust (bundle_adjust.py) hung indefinitely --
     traced via faulthandler to scipy's dense finite-difference Jacobian
     path. This is the exact "BA hang" failure mode local_bundle_adjust
     (the non-inertial sibling) was already fixed for early in this
     project; the fix (a `jac_sparsity` pattern) was just never applied
     to this function, because nothing had ever called it with a real
     multi-keyframe inertial window before.

What WASN'T broken, verified here with actual ground truth rather than
taken on faith: imu.Preintegration's core recursion, and imu_init.py's
staged linear-least-squares initialization. Both check out to within
the accuracy their own documented scope (first-order, non-iterated)
implies.

See PROGRESS.md's Phase 8 section for the full account and
RUNBOOK_PHASE8.md for symptom-to-cause mapping if something in this file
starts failing later.
"""

import sys
import numpy as np

from camera import Camera
from frame import Frame
import imu
import imu_init
import synthetic
from run_slam import SLAMSystem
from tracking import Tracking

FAILURES = []


def gate(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def _dummy_camera():
    return Camera(fx=385.0, fy=385.0, cx=320.0, cy=240.0,
                 width=640, height=480, baseline=0.05, depth_scale=0.001)


# ── ground-truth correctness: the algorithm itself ──────────────────────

def test_preintegration_matches_analytic_ground_truth():
    print("\n--- Test 1: imu.Preintegration reconstructs exact analytic "
         "ground truth ---")
    gt = synthetic.imu_ground_truth_sequence(duration=6.0)
    samples = gt["imu_samples"]
    pose_fn, velocity_fn, gravity = gt["pose_fn"], gt["velocity_fn"], gt["gravity"]

    t0, t1 = 1.0, 3.0
    preint = imu.Preintegration(np.zeros(3), np.zeros(3))
    mask = (samples[:, 0] > t0) & (samples[:, 0] <= t1)
    t_prev = t0
    for row in samples[mask]:
        t, gx, gy, gz, ax, ay, az = row
        preint.integrate_sample([gx, gy, gz], [ax, ay, az], t - t_prev)
        t_prev = t

    R0, p0 = pose_fn(t0)[:3, :3], pose_fn(t0)[:3, 3]
    R1, p1 = pose_fn(t1)[:3, :3], pose_fn(t1)[:3, 3]
    v0, v1 = velocity_fn(t0), velocity_fn(t1)
    dt = t1 - t0

    dR_true = R0.T @ R1
    dv_true = R0.T @ (v1 - v0 - gravity * dt)
    dp_true = R0.T @ (p1 - p0 - v0 * dt - 0.5 * gravity * dt ** 2)

    dR_err = float(np.linalg.norm(preint.dR - dR_true))
    dv_err = float(np.linalg.norm(preint.dv - dv_true))
    dp_err = float(np.linalg.norm(preint.dp - dp_true))

    gate("dR matches ground truth", dR_err < 1e-8, f"err={dR_err:.2e}")
    gate("dv matches ground truth (< 0.1% of true norm)",
        dv_err < 1e-3 * np.linalg.norm(dv_true), f"err={dv_err:.2e}")
    gate("dp matches ground truth (< 1% of true norm)",
        dp_err < 1e-2 * np.linalg.norm(dp_true), f"err={dp_err:.2e}")


def test_imu_init_recovers_ground_truth():
    print("\n--- Test 2: imu_init.initialize() recovers gravity/bias/"
         "velocity from synthetic ground truth ---")
    gt = synthetic.imu_ground_truth_sequence(duration=6.0)
    samples = gt["imu_samples"]
    pose_fn, velocity_fn, gravity_true = gt["pose_fn"], gt["velocity_fn"], gt["gravity"]

    class FakeKF:
        pass

    kf_times = np.arange(0, 6.01, 0.5)
    kfs = []
    for k, t in enumerate(kf_times):
        kf = FakeKF()
        kf.timestamp, kf.pose, kf.kf_seq, kf.imu_preint = t, pose_fn(t), k, None
        kfs.append(kf)
    for k in range(1, len(kfs)):
        t0, t1 = kfs[k - 1].timestamp, kfs[k].timestamp
        preint = imu.Preintegration(np.zeros(3), np.zeros(3))
        mask = (samples[:, 0] > t0) & (samples[:, 0] <= t1)
        t_prev = t0
        for row in samples[mask]:
            t, gx, gy, gz, ax, ay, az = row
            preint.integrate_sample([gx, gy, gz], [ax, ay, az], t - t_prev)
            t_prev = t
        kfs[k].imu_preint = preint

    result = imu_init.initialize(kfs, samples, gravity_mag=9.81)
    gate("initialize() succeeds", result["success"], result.get("reason"))
    if not result["success"]:
        return

    g_err = float(np.linalg.norm(result["gravity"] - gravity_true))
    gate("gravity within 0.2 m/s^2 of true [0,-9.81,0]", g_err < 0.2, f"err={g_err:.4f}")
    gate("bias_gyro within 0.01 rad/s of zero",
        float(np.linalg.norm(result["bias_gyro"])) < 0.01)
    gate("bias_accel within 0.15 m/s^2 of zero",
        float(np.linalg.norm(result["bias_accel"])) < 0.15)

    v_errs = [float(np.linalg.norm(v - velocity_fn(kf.timestamp)))
             for kf, v in zip(kfs, result["velocities"])]
    gate("all recovered velocities within 0.1 m/s of ground truth",
        max(v_errs) < 0.1, f"max err={max(v_errs):.4f}")


# ── bug 1 regression: _try_imu_init must exist and be callable ──────────

def test_try_imu_init_exists_and_runs():
    print("\n--- Test 3: SLAMSystem._try_imu_init exists and runs without "
         "crashing (regression for the missing `def` line) ---")
    gate("_try_imu_init is a real method", hasattr(SLAMSystem, "_try_imu_init"))

    cam = _dummy_camera()
    slam = SLAMSystem(cam, use_depth=True, verbose=False, use_imu=True)
    # A world_map with too few keyframes should just return quietly, not
    # crash -- this alone would have raised AttributeError before the fix.
    try:
        slam._try_imu_init(slam.atlas.active_map)
        ran_ok = True
    except AttributeError as e:
        ran_ok = False
        print(f"    AttributeError: {e}")
    gate("calling it on an empty map does not raise", ran_ok)


# ── bug 4 regression: VI-BA sparsity pattern must be both fast AND correct ──

def test_vi_ba_sparsity_pattern_correct_and_fast():
    print("\n--- Test 4: local_inertial_bundle_adjust's sparse Jacobian "
         "pattern is correct (matches dense) and fast (regression for the "
         "BA-hang bug) ---")
    import time
    from bundle_adjust import local_inertial_bundle_adjust
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    # Small synthetic VI window built directly (bypassing the full visual
    # pipeline) so this test is fast and isolates the optimizer itself.
    gt = synthetic.imu_ground_truth_sequence(duration=3.0)
    samples = gt["imu_samples"]
    pose_fn, velocity_fn, gravity = gt["pose_fn"], gt["velocity_fn"], gt["gravity"]

    cam = _dummy_camera()
    from map import Map
    from map_point import MapPoint
    world_map = Map()

    kf_times = [0.0, 0.5, 1.0, 1.5]
    kfs = []
    rng = np.random.RandomState(0)
    for k, t in enumerate(kf_times):
        kf = Frame.__new__(Frame)
        kf.id, kf.kf_seq, kf.timestamp = 1000 + k, k, t
        kf.pose = pose_fn(t)
        kf.velocity = velocity_fn(t) if k == 0 else velocity_fn(t) + rng.normal(0, 0.05, 3)
        kf.map_point_ids = [None] * 50
        kf.points_undistorted = np.zeros((50, 2))
        kf.imu_preint = None
        world_map.add_keyframe(kf)
        kfs.append(kf)

    # a handful of shared points, each observed by every keyframe (with a
    # small perturbation on the free keyframes so there's real work for
    # the optimizer to do)
    for j in range(30):
        p_world = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), 2.0 + rng.uniform(-0.5, 0.5)])
        mp = MapPoint(p_world, np.zeros(32, np.uint8), ref_keyframe_id=0)
        world_map.add_map_point(mp)
        for kf in kfs:
            Rcw = kf.pose[:3, :3].T
            tcw = -Rcw @ kf.pose[:3, 3]
            p_cam = Rcw @ p_world + tcw
            u = cam.fx * p_cam[0] / p_cam[2] + cam.cx
            v = cam.fy * p_cam[1] / p_cam[2] + cam.cy
            kf.map_point_ids[j] = mp.id
            kf.points_undistorted[j] = [u, v]
            mp.add_observation(kf.id, j)

    for k in range(1, len(kfs)):
        t0, t1 = kfs[k - 1].timestamp, kfs[k].timestamp
        preint = imu.Preintegration(np.zeros(3), np.zeros(3))
        mask = (samples[:, 0] > t0) & (samples[:, 0] <= t1)
        t_prev = t0
        for row in samples[mask]:
            t, gx, gy, gz, ax, ay, az = row
            preint.integrate_sample([gx, gy, gz], [ax, ay, az], t - t_prev)
            t_prev = t
        kfs[k].imu_preint = preint

    t_start = time.time()
    n_optimized = local_inertial_bundle_adjust(
        world_map, cam, gravity=gravity, bias_gyro=np.zeros(3), bias_accel=np.zeros(3),
        window=4, max_iter=15, verbose=False)
    elapsed = time.time() - t_start

    gate("completes in under 15 seconds (was: hangs indefinitely)",
        elapsed < 15.0, f"{elapsed:.2f}s")

    # correctness: after optimizing, free keyframes should have moved
    # CLOSER to ground truth than their perturbed starting point, not
    # just converged to something arbitrary -- a wrong sparsity pattern
    # (a missed true dependency) would leave those parameters essentially
    # un-optimized instead.
    final_pos_errs = [float(np.linalg.norm(kf.pose[:3, 3] - pose_fn(kf.timestamp)[:3, 3]))
                      for kf in kfs[1:]]
    gate("free keyframes end up close to ground truth pose",
        max(final_pos_errs) < 0.1, f"max pos err={max(final_pos_errs):.4f}")


# ── bug 3 regression: velocity propagation must not have a single point of failure ──

def test_velocity_propagates_even_when_strategy_zero_fails():
    print("\n--- Test 5: frame.velocity gets set even when a NON-IMU "
         "strategy is what succeeds (regression for the silent permanent-"
         "disable bug) ---")
    cam = _dummy_camera()
    tracking = Tracking(cam, extractor=None, matcher=None, world_map=None)

    kf0 = Frame.__new__(Frame)
    kf0.pose = np.eye(4)
    kf0.velocity = np.array([1.0, 0.0, 0.0])
    tracking.last_keyframe = kf0
    tracking.imu_initialized = True
    tracking.gravity = np.array([0.0, -9.81, 0.0])
    tracking.last_frame = None
    tracking.velocity = None
    tracking.covis_graph = {}
    tracking.state = "OK"

    preint = imu.Preintegration(np.zeros(3), np.zeros(3))
    preint.integrate_sample([0.0, 0.0, 0.0], [0.0, 9.81, 0.0], 0.05)

    frame = Frame.__new__(Frame)
    frame.id = 1
    frame.map_point_ids = [None] * 10
    frame.velocity = None

    call_count = {"n": 0}
    def fake_motion_model(f, initial_pose=None):
        call_count["n"] += 1
        if initial_pose is not None:
            return False   # Strategy 0 (IMU-predicted) rejected
        return False       # Strategy 1 (constant velocity) also not applicable here
    def fake_reference_keyframe(f):
        f.set_pose(np.eye(4))   # Strategy 2 succeeds
        return True
    def fake_local_map(f):
        return 0
    def fake_needs_new_kf(f):
        return False

    tracking._track_with_motion_model = fake_motion_model
    tracking._track_reference_keyframe = fake_reference_keyframe
    tracking._track_local_map = fake_local_map

    ok = tracking.track(frame, imu_preint=preint)

    gate("tracking succeeds via the non-IMU strategy", ok is True)
    gate("frame.velocity is still populated (not None)", frame.velocity is not None,
        f"velocity={frame.velocity}")
    if frame.velocity is not None:
        expected = kf0.velocity + tracking.gravity * preint.dt + kf0.pose[:3, :3] @ preint.dv
        err = float(np.linalg.norm(frame.velocity - expected))
        gate("propagated velocity matches the IMU-derived expectation",
            err < 1e-9, f"err={err:.2e}")


# ── full-pipeline integration ────────────────────────────────────────────

def test_full_pipeline_with_imu():
    print("\n--- Test 6: full SLAMSystem run with use_imu=True initializes "
         "IMU, tracks with zero loss, and every keyframe gets a velocity ---")
    gt = synthetic.imu_ground_truth_sequence(duration=5.0)
    images, depths, gt_poses, frame_times, cam_dict = synthetic.imu_visual_sequence(
        gt["pose_fn"], duration=5.0, camera_hz=20.0)

    cam = Camera(fx=cam_dict["fx"], fy=cam_dict["fy"], cx=cam_dict["cx"], cy=cam_dict["cy"],
                width=cam_dict["width"], height=cam_dict["height"],
                baseline=0.05, depth_scale=0.001)
    slam = SLAMSystem(cam, use_depth=True, verbose=False, use_imu=True)

    samples = gt["imu_samples"]
    last_t = 0.0
    for img, d, t in zip(images, depths, frame_times):
        mask = (samples[:, 0] > last_t) & (samples[:, 0] <= t)
        imu_slice = samples[mask]
        last_t = t
        slam.process(img, t, depth_image=d, imu_samples=imu_slice)

    gate("imu_initialized flips True within the run", slam.tracking.imu_initialized)
    gate("zero tracking losses", slam.stats["lost"] == 0, f"lost={slam.stats['lost']}")

    kfs = slam.atlas.active_map.keyframes
    n_no_vel = sum(1 for kf in kfs if kf.velocity is None)
    gate("every keyframe has a non-None velocity", n_no_vel == 0,
        f"{n_no_vel}/{len(kfs)} missing")

    g_err = float(np.linalg.norm(slam.tracking.gravity - gt["gravity"]))
    # NOTE: this tolerance is deliberately much looser than Test 2's. Test
    # 2 feeds imu_init.py exact ground-truth keyframe poses -- it verifies
    # the ALGORITHM. This test runs the full noisy visual pipeline (real
    # feature tracking, real PnP, real keyframe selection) -- it verifies
    # the algorithm doesn't fall over when given realistically-imperfect
    # keyframe poses, not that it matches Test 2's precision. imu_init.py's
    # own docstring documents why this is expected to be a first-order,
    # non-iterated estimate (no VIBA1/VIBA2 re-refinement passes).
    gate("recovered gravity within 1.5 m/s^2 of true (full noisy visual pipeline)",
        g_err < 1.5, f"err={g_err:.3f}")

    import validate
    report = validate.validate_atlas(slam.atlas)
    gate("validate.py reports zero errors on this run", report.ok(),
        f"{len(report.errors)} error(s)")


# ── full regression ──────────────────────────────────────────────────────

def test_full_pipeline_regression():
    print("\n--- Test 7: Phase 1-7 gates still pass with Phase 8 changes ---")
    import subprocess
    for script in ["test_phase1_gate.py", "test_phase2_gate.py",
                   "test_phase3_gate.py", "test_phase5_gate.py",
                   "test_phase6_gate.py", "test_phase7_gate.py"]:
        result = subprocess.run([sys.executable, script], capture_output=True,
                               text=True, timeout=900)
        passed = "ALL GATES PASSED" in result.stdout
        print(f"  {script}: {'PASSED' if passed else 'FAILED'}")
        if not passed:
            print("\n".join(result.stdout.splitlines()[-20:]))
        gate(f"{script} still passes with Phase 8 changes", passed)


if __name__ == "__main__":
    print("=" * 64)
    print("  PHASE 8 GATE — IMU-based motion prediction for tracking robustness")
    print("=" * 64)

    test_preintegration_matches_analytic_ground_truth()
    test_imu_init_recovers_ground_truth()
    test_try_imu_init_exists_and_runs()
    test_vi_ba_sparsity_pattern_correct_and_fast()
    test_velocity_propagates_even_when_strategy_zero_fails()
    test_full_pipeline_with_imu()
    test_full_pipeline_regression()

    print("\n" + "=" * 64)
    if FAILURES:
        print(f"  RESULT: {len(FAILURES)} GATE(S) FAILED")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    else:
        print("  RESULT: ALL GATES PASSED")
    print("=" * 64)

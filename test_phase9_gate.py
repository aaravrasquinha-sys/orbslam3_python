"""
test_phase9_gate.py — Phase 9: real-time incremental iSAM2 back end.

Run: python3 test_phase9_gate.py

WHY THIS FILE EXISTS: isam2_backend.py replaces two previously-verified
optimizers (bundle_adjust_gtsam.py's windowed BA, pose_graph.py's scipy
loop correction) with one new, genuinely different architecture
(continuous incremental factor graph). Three real bugs were found and
fixed WHILE BUILDING this gate, not anticipated in the original design
-- see PROGRESS.md's Phase 9 section and isam2_backend.py's own module
docstring for the full account of each:
  1. Missing odometry factor: a keyframe sharing no map points with
     anything already in the graph was a genuinely disconnected
     component with no path back to the gauge anchor --
     IndeterminantLinearSystemException, not a numerical fluke.
  2. The init keyframe (from initializer.init_rgbd/init_monocular) never
     reached the backend at all -- run_slam.py's per-keyframe block,
     where add_keyframe() is normally called, is never reached on the
     init frame's own process() call (that branch returns early).
     Whichever keyframe reached add_keyframe() FIRST became the gauge
     anchor instead, anchored at the WRONG pose, with the true first
     keyframe's already-created points inserted as a large single-
     keyframe batch.
  3. A brand-new landmark's first-ever factor could be mono-only (2
     constraints for 3 unknowns) if its depth was invalid in the
     keyframe that happened to see it first -- genuinely underdetermined
     regardless of how well-connected the observing keyframe otherwise
     is.

Gates:
  - Ground-truth convergence: incremental updates recover an exact known
    trajectory + point cloud (no ambiguity, dense observations).
  - The odometry-factor regression: a keyframe sharing zero points with
    the graph so far must not crash (bug 1).
  - Loop-closure correction propagates through the WHOLE drifted
    trajectory, not just the two endpoint keyframes.
  - The defensive-filtering regression: a landmark with a wildly bad
    position, or whose first-ever observation is mono-only, must not
    reach GTSAM and must not crash the update.
  - Map-merge integration: absorb_map() correctly brings a second map's
    keyframes into the survivor's graph and bridges them.
  - A timing benchmark (this phase's own "measure before adding fixed-
    lag marginalization" plan from PHASE7_ARCHITECTURE_V2_REALTIME.md
    sec 2.4/7).
  - Full Phase 1-8 regression (includes test_phase6_gate.py's own
    247-frame fragmentation+merge scenario, which now runs against
    isam2 as a side effect of it being the new default backend --
    exercises the full realistic pipeline, not just this file's own
    smaller focused scenarios).
"""

import sys
import time
import numpy as np

from camera import Camera
from frame import Frame
from map import Map
from map_point import MapPoint
from isam2_backend import Isam2Backend

FAILURES = []


def gate(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def _cam():
    return Camera(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480,
                 baseline=0.05, depth_scale=0.001)


def _make_kf(kf_id, kf_seq, pose, n_pts, world_map):
    kf = Frame.__new__(Frame)
    kf.id = kf_id
    kf.kf_seq = kf_seq
    kf.pose = pose
    kf.velocity = None
    kf.map_point_ids = [None] * n_pts
    kf.points_undistorted = np.zeros((n_pts, 2))
    kf.depths = np.full(n_pts, -1.0)
    world_map.add_keyframe(kf)
    return kf


def _project(cam, T_cw_pose, pts_world):
    """T_cw_pose: camera-to-world (Twc). Returns (u, v, depth) or None per point."""
    Rcw = T_cw_pose[:3, :3].T
    tcw = -Rcw @ T_cw_pose[:3, 3]
    out = []
    for p in pts_world:
        p_cam = Rcw @ p + tcw
        if p_cam[2] <= 0.1:
            out.append(None)
            continue
        u = cam.fx * p_cam[0] / p_cam[2] + cam.cx
        v = cam.fy * p_cam[1] / p_cam[2] + cam.cy
        if not (0 <= u < cam.width and 0 <= v < cam.height):
            out.append(None)
            continue
        out.append((u, v, p_cam[2]))
    return out


# ── ground-truth convergence ─────────────────────────────────────────────

def test_ground_truth_convergence():
    print("\n--- Test 1: incremental updates converge to exact ground truth "
         "(dense, unambiguous observations) ---")
    cam = _cam()
    world_map = Map()
    rng = np.random.RandomState(0)

    n_kf = 15
    gt_poses = [np.eye(4) for _ in range(n_kf)]
    for k in range(n_kf):
        gt_poses[k][0, 3] = 0.2 * k

    n_pts = 60
    pts_world = np.stack([rng.uniform(-1.5, 1.5, n_pts), rng.uniform(-1, 1, n_pts),
                          np.full(n_pts, 2.0) + rng.uniform(-0.3, 0.3, n_pts)], axis=1)

    kfs = []
    for k, T in enumerate(gt_poses):
        # PHASE 9 test design note: NOT perturbing starting poses here --
        # this test verifies exact recovery from a good initial guess
        # with fully-consistent measurements. The odometry factor (see
        # isam2_backend.py) uses the CURRENT kf.pose at insertion time as
        # its own "measurement" -- perturbing that would make the
        # odometry factor's measurement wrong too, which is a different,
        # legitimate scenario (recovering from real drift) already
        # covered by Test 3 below, not a bug in this test's own
        # convergence check.
        kf = _make_kf(k, k, T.copy(), n_pts, world_map)
        kfs.append(kf)

    mps = []
    for j in range(n_pts):
        mp = MapPoint(pts_world[j].copy(), np.zeros(32, np.uint8), ref_keyframe_id=0)
        world_map.add_map_point(mp)
        mps.append(mp)
        for kf, T in zip(kfs, gt_poses):
            proj = _project(cam, T, [pts_world[j]])[0]
            if proj is None:
                continue
            u, v, z = proj
            kf.map_point_ids[j] = mp.id
            kf.points_undistorted[j] = [u, v]
            kf.depths[j] = z
            mp.add_observation(kf.id, j)

    backend = Isam2Backend(cam)
    update_times = []
    for kf in kfs:
        _, _, dt = backend.add_keyframe(kf, world_map)
        gate(f"kf {kf.id} update did not fail (dt is not None)", dt is not None)
        if dt is not None:
            update_times.append(dt)

    pos_errs = [np.linalg.norm(kf.pose[:3, 3] - T[:3, 3]) for kf, T in zip(kfs, gt_poses)]
    pt_errs = [np.linalg.norm(mp.position - p) for mp, p in zip(mps, pts_world)]
    gate("max keyframe position error < 1mm", max(pos_errs) < 1e-3, f"max={max(pos_errs):.2e}")
    gate("max point position error < 1mm", max(pt_errs) < 1e-3, f"max={max(pt_errs):.2e}")
    gate("per-keyframe update time stays well under 100ms (real-time budget)",
        max(update_times) < 0.1, f"max={max(update_times)*1000:.1f}ms")


# ── the odometry-factor regression (bug 1) ──────────────────────────────

def test_odometry_factor_prevents_disconnected_component():
    print("\n--- Test 2: a keyframe sharing ZERO points with the graph so far "
         "does not crash (regression for the missing-odometry-factor bug) ---")
    cam = _cam()
    world_map = Map()

    kf0 = _make_kf(0, 0, np.eye(4), 1, world_map)
    mp = MapPoint(np.array([0.0, 0.0, 2.0]), np.zeros(32, np.uint8), ref_keyframe_id=0)
    world_map.add_map_point(mp)
    kf0.map_point_ids[0] = mp.id
    kf0.points_undistorted[0] = [cam.cx, cam.cy]
    kf0.depths[0] = 2.0
    mp.add_observation(kf0.id, 0)
    mp.add_observation(999, 0)   # fake extra observation so n_observations() >= 2

    T1 = np.eye(4)
    T1[0, 3] = 0.3
    kf1 = _make_kf(1, 1, T1, 1, world_map)   # deliberately observes NOTHING kf0 does

    backend = Isam2Backend(cam)
    ok = True
    try:
        backend.add_keyframe(kf0, world_map)
        _, _, dt = backend.add_keyframe(kf1, world_map)
        ok = dt is not None
    except RuntimeError as e:
        ok = False
        print(f"    EXCEPTION: {e}")
    gate("adding a keyframe with zero shared points does not crash", ok)
    gate("both keyframes end up in the graph", {0, 1} <= backend.kf_ids_in_graph)


# ── loop-closure correction propagation ──────────────────────────────────

def test_loop_closure_propagates_through_trajectory():
    print("\n--- Test 3: a loop-closure factor corrects the WHOLE drifted "
         "trajectory, not just the two endpoint keyframes ---")
    cam = _cam()
    world_map = Map()
    rng = np.random.RandomState(1)

    n_kf = 20
    angles = np.linspace(0, 2 * np.pi, n_kf, endpoint=False)
    radius = 2.0
    gt_poses = []
    for k in range(n_kf):
        T = np.eye(4)
        T[0, 3] = radius * np.cos(angles[k]) - radius
        T[2, 3] = radius * np.sin(angles[k])
        gt_poses.append(T)

    n_pts = 80
    pts_world = np.stack([rng.uniform(-radius * 1.5, radius * 0.5, n_pts),
                          rng.uniform(-1, 1, n_pts),
                          rng.uniform(-radius * 1.5, radius * 1.5, n_pts)], axis=1)

    kfs = []
    drift_accum = np.zeros(3)
    for k, T in enumerate(gt_poses):
        drift_accum = drift_accum + np.array([0.01, 0.0, 0.008])
        drifted = T.copy()
        if k > 0:
            drifted[:3, 3] += drift_accum
        kf = _make_kf(k, k, drifted, n_pts, world_map)
        kfs.append(kf)

    mps = []
    for j in range(n_pts):
        mp = MapPoint(pts_world[j].copy(), np.zeros(32, np.uint8), ref_keyframe_id=0)
        world_map.add_map_point(mp)
        mps.append(mp)
        for kf, T in zip(kfs, gt_poses):
            proj = _project(cam, T, [pts_world[j]])[0]
            if proj is None:
                continue
            u, v, z = proj
            kf.map_point_ids[j] = mp.id
            kf.points_undistorted[j] = [u, v]
            kf.depths[j] = z
            mp.add_observation(kf.id, j)

    backend = Isam2Backend(cam)
    for kf in kfs:
        backend.add_keyframe(kf, world_map)

    errs_before = [np.linalg.norm(kf.pose[:3, 3] - T[:3, 3]) for kf, T in zip(kfs, gt_poses)]

    T_rel_true = np.linalg.inv(gt_poses[0]) @ gt_poses[-1]
    backend.add_loop_factor(kfs[0], kfs[-1], T_rel_true, loop_weight=100.0, world_map=world_map)

    errs_after = [np.linalg.norm(kf.pose[:3, 3] - T[:3, 3]) for kf, T in zip(kfs, gt_poses)]

    gate("last keyframe's error drops substantially after the loop factor",
        errs_after[-1] < 0.3 * errs_before[-1],
        f"before={errs_before[-1]:.4f} after={errs_after[-1]:.4f}")
    # the correction should be SPREAD across intermediate keyframes, not
    # just snap the two endpoints -- check a keyframe in the middle of
    # the path also moved (didn't stay bit-for-bit at its pre-loop pose).
    mid = n_kf // 2
    gate("a keyframe in the MIDDLE of the loop also moved (correction "
        "is distributed, not just applied at the endpoints)",
        abs(errs_after[mid] - errs_before[mid]) > 1e-4)


# ── defensive filtering (bugs found while building this) ────────────────

def test_bad_point_position_is_skipped_not_crashed():
    print("\n--- Test 4: a landmark with a wildly out-of-range position is "
         "skipped, not sent to GTSAM ---")
    cam = _cam()
    world_map = Map()
    kf0 = _make_kf(0, 0, np.eye(4), 2, world_map)

    good = MapPoint(np.array([0.1, 0.0, 2.0]), np.zeros(32, np.uint8), ref_keyframe_id=0)
    bad = MapPoint(np.array([1e8, -1e8, 5e7]), np.zeros(32, np.uint8), ref_keyframe_id=0)
    world_map.add_map_point(good)
    world_map.add_map_point(bad)
    for i, (mp, u_offset) in enumerate([(good, 0), (bad, 50)]):
        kf0.map_point_ids[i] = mp.id
        kf0.points_undistorted[i] = [cam.cx + u_offset, cam.cy]
        kf0.depths[i] = 2.0
        mp.add_observation(kf0.id, i)
        mp.add_observation(999 + i, i)   # n_observations() >= 2

    backend = Isam2Backend(cam)
    n_pts_added, n_obs_added, dt = backend.add_keyframe(kf0, world_map)
    gate("update succeeds despite the bad point", dt is not None)
    gate("only the GOOD point was actually inserted into the graph",
        good.id in backend.point_ids_in_graph and bad.id not in backend.point_ids_in_graph)


def test_mono_only_first_observation_is_deferred():
    print("\n--- Test 5: a brand-new landmark whose first observation is "
         "mono-only (no valid depth) is deferred, not inserted underdetermined ---")
    cam = _cam()
    world_map = Map()
    kf0 = _make_kf(0, 0, np.eye(4), 1, world_map)

    mp = MapPoint(np.array([0.1, 0.0, 2.0]), np.zeros(32, np.uint8), ref_keyframe_id=0)
    world_map.add_map_point(mp)
    kf0.map_point_ids[0] = mp.id
    kf0.points_undistorted[0] = [cam.cx, cam.cy]
    kf0.depths[0] = -1.0   # invalid depth -- forces mono fallback
    mp.add_observation(kf0.id, 0)
    mp.add_observation(999, 0)

    backend = Isam2Backend(cam)
    n_pts_added, n_obs_added, dt = backend.add_keyframe(kf0, world_map)
    gate("update succeeds", dt is not None)
    gate("the mono-only point was NOT inserted as a brand-new landmark "
        "(would be underdetermined -- 2 constraints for 3 unknowns)",
        mp.id not in backend.point_ids_in_graph)
    gate("no points were reported added", n_pts_added == 0)


# ── map-merge integration ────────────────────────────────────────────────

def test_absorb_map_bridges_two_backends():
    print("\n--- Test 6: absorb_map() brings a second map's keyframes into "
         "the survivor's graph and a bridging loop factor connects them ---")
    cam = _cam()
    rng = np.random.RandomState(2)

    map_a = Map()
    map_b = Map()
    n_pts = 40
    pts_a = np.stack([rng.uniform(-1, 1, n_pts), rng.uniform(-1, 1, n_pts),
                      np.full(n_pts, 2.0)], axis=1)
    pts_b = np.stack([rng.uniform(-1, 1, n_pts), rng.uniform(-1, 1, n_pts),
                      np.full(n_pts, 2.0)], axis=1)

    kfs_a = [_make_kf(k, k, np.eye(4), n_pts, map_a) for k in range(3)]
    for k in kfs_a:
        k.pose = np.eye(4)
        k.pose[0, 3] = 0.1 * k.id
    kfs_b = [_make_kf(100 + k, k, np.eye(4), n_pts, map_b) for k in range(3)]
    for k in kfs_b:
        k.pose = np.eye(4)
        k.pose[0, 3] = 5.0 + 0.1 * (k.id - 100)   # far away, disjoint coordinate area pre-merge

    for pts, kfs, wm in [(pts_a, kfs_a, map_a), (pts_b, kfs_b, map_b)]:
        for j in range(n_pts):
            mp = MapPoint(pts[j].copy(), np.zeros(32, np.uint8), ref_keyframe_id=0)
            wm.add_map_point(mp)
            for kf in kfs:
                proj = _project(cam, kf.pose, [pts[j]])[0]
                if proj is None:
                    continue
                u, v, z = proj
                kf.map_point_ids[j] = mp.id
                kf.points_undistorted[j] = [u, v]
                kf.depths[j] = z
                mp.add_observation(kf.id, j)

    backend_a = Isam2Backend(cam)
    for kf in kfs_a:
        backend_a.add_keyframe(kf, map_a)
    n_kfs_before = len(backend_a.kf_ids_in_graph)

    # simulate "post-merge": map_b's keyframes get moved into map_a's
    # coordinate frame (merge_maps.py's job, not re-tested here) --
    # apply a trivial identity-ish shift for this test's purposes.
    for kf in kfs_b:
        kf.pose[0, 3] -= 4.7   # pretend the merge transform brought it close to map_a's frame

    ok = True
    try:
        backend_a.absorb_map(kfs_b, map_a, bridge_kf=kfs_a[-1])
    except RuntimeError as e:
        ok = False
        print(f"    EXCEPTION during absorb_map: {e}")
    gate("absorb_map completes without crashing", ok)
    gate("all of map_b's keyframes are now in backend_a's graph",
        {kf.id for kf in kfs_b} <= backend_a.kf_ids_in_graph)
    gate("backend_a's graph grew by exactly len(kfs_b) keyframes",
        len(backend_a.kf_ids_in_graph) == n_kfs_before + len(kfs_b))

    # bridge with an explicit loop factor, same pattern run_slam.py uses
    T_bridge = np.linalg.inv(kfs_a[-1].pose) @ kfs_b[0].pose
    dt = backend_a.add_loop_factor(kfs_a[-1], kfs_b[0], T_bridge, loop_weight=50.0, world_map=map_a)
    gate("bridging loop factor between the two formerly-separate maps succeeds",
        dt is not None)


def test_reconciliation_after_partial_commit_failure():
    print("\n--- Test 7: after isam.update() fails, bookkeeping is reconciled "
         "against GTSAM's ACTUAL state, not assumed -- regression for the "
         "cascading 'key already exists' bug ---")
    cam = _cam()
    world_map = Map()
    backend = Isam2Backend(cam)

    kf0 = _make_kf(0, 0, np.eye(4), 3, world_map)
    pts = [np.array([0.1, 0.0, 2.0]), np.array([-0.1, 0.1, 2.0]), np.array([0.0, -0.1, 2.0])]
    mps = []
    for i, p in enumerate(pts):
        mp = MapPoint(p.copy(), np.zeros(32, np.uint8), ref_keyframe_id=0)
        world_map.add_map_point(mp)
        mps.append(mp)
        kf0.map_point_ids[i] = mp.id
        kf0.points_undistorted[i] = [cam.cx + i * 5, cam.cy]
        kf0.depths[i] = 2.0
        mp.add_observation(kf0.id, i)
        mp.add_observation(999 + i, i)

    # Force isam.update() to raise on THIS call, simulating exactly what
    # was observed on real data (GTSAM partially committing values before
    # throwing during elimination). gtsam.ISAM2's methods are read-only
    # (pybind11-wrapped C++), so wrap the instance instead of patching
    # the method directly -- forwards everything except update(), which
    # calls through to the REAL update() first (so the values really do
    # get committed, exactly like real GTSAM) and then raises, mirroring
    # the actual failure mode precisely.
    class _FailOnceWrapper:
        def __init__(self, real_isam):
            self._real = real_isam
            self.calls = 0
        def update(self, graph, initial):
            self.calls += 1
            result = self._real.update(graph, initial)
            if self.calls == 1:
                raise RuntimeError("simulated IndeterminantLinearSystemException")
            return result
        def __getattr__(self, name):
            return getattr(self._real, name)

    backend.isam = _FailOnceWrapper(backend.isam)
    n_pts_added, n_obs_added, dt = backend.add_keyframe(kf0, world_map)
    backend.isam = backend.isam._real

    gate("caller is told this attempt failed (dt is None)", dt is None)
    gate("kf0 IS marked in kf_ids_in_graph (GTSAM actually committed it, "
        "confirmed via valueExists -- must not be silently forgotten)",
        kf0.id in backend.kf_ids_in_graph)
    gate("all 3 points ARE marked in point_ids_in_graph (also genuinely "
        "committed, confirmed via valueExists)",
        len(backend.point_ids_in_graph) == 3)

    # THE regression check: a SECOND keyframe re-observing the SAME
    # points must NOT hit "key already exists" -- this is exactly the
    # cascade that happened on real data when bookkeeping and GTSAM's
    # actual state disagreed.
    kf1 = _make_kf(1, 1, np.eye(4), 3, world_map)
    for i, mp in enumerate(mps):
        kf1.map_point_ids[i] = mp.id
        kf1.points_undistorted[i] = [cam.cx + i * 5, cam.cy]
        kf1.depths[i] = 2.0
        mp.add_observation(kf1.id, i)

    ok = True
    try:
        _, _, dt2 = backend.add_keyframe(kf1, world_map)
        ok = dt2 is not None
    except (RuntimeError, IndexError) as e:
        ok = False
        print(f"    EXCEPTION (this is exactly the cascade bug if it fires): {e}")
    gate("a later keyframe re-observing the same points does NOT crash "
        "with 'key already exists' (the actual cascade bug)", ok)


# ── full regression ──────────────────────────────────────────────────────

def test_full_pipeline_regression():
    print("\n--- Test 7: Phase 1-8 gates still pass with Phase 9 changes "
         "(includes test_phase6_gate.py's 247-frame fragmentation+merge "
         "scenario, now exercising isam2 as the default backend) ---")
    import subprocess
    for script in ["test_phase1_gate.py", "test_phase2_gate.py",
                   "test_phase3_gate.py", "test_phase5_gate.py",
                   "test_phase6_gate.py", "test_phase7_gate.py",
                   "test_phase8_gate.py"]:
        result = subprocess.run([sys.executable, script], capture_output=True,
                               text=True, timeout=1800)
        passed = "ALL GATES PASSED" in result.stdout
        print(f"  {script}: {'PASSED' if passed else 'FAILED'}")
        if not passed:
            print("\n".join(result.stdout.splitlines()[-25:]))
        gate(f"{script} still passes with Phase 9 changes", passed)


if __name__ == "__main__":
    print("=" * 64)
    print("  PHASE 9 GATE — real-time incremental iSAM2 back end")
    print("=" * 64)

    test_ground_truth_convergence()
    test_odometry_factor_prevents_disconnected_component()
    test_loop_closure_propagates_through_trajectory()
    test_bad_point_position_is_skipped_not_crashed()
    test_mono_only_first_observation_is_deferred()
    test_absorb_map_bridges_two_backends()
    test_reconciliation_after_partial_commit_failure()
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

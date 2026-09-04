"""
bundle_adjust_gtsam.py — Phase 2: GTSAM replaces the scipy dense-solver BA.

WHY THIS EXISTS: bundle_adjust.py's local_bundle_adjust() uses scipy's
dense least_squares. Cost scales steeply with window size, and Phase 1's
ablation showed this directly -- disabling keyframe culling (which the
original culling bug was, in effect, using as an accidental workaround)
made a 120-frame synthetic test time out. Fixing culling properly (Phase
1) makes larger, healthier local windows possible, which re-exposes this
scaling problem rather than hiding it.

TWO BUGS FOUND AND FIXED DURING THIS FILE'S OWN VERIFICATION (see
test_phase2_gate.py and PROGRESS.md) — recorded here because they are the
kind of thing that would otherwise resurface silently on real data:

  BUG 1 (crash): the free/fixed keyframe split computed an ABSTRACT set of
  "any keyframe observing a window point, not in the free window" and a
  SEPARATE, further-filtered list of keyframes actually instantiated into
  the GTSAM graph (dropping stale/missing/pose-None entries). The
  factor-building loop gated against the abstract set instead of the
  filtered one, so it could add a factor referencing a GTSAM variable that
  was never inserted -- GTSAM correctly refuses this with "inconsistent
  arguments" at solve time. Fixed by computing ONE `available_kf_ids` set
  used everywhere.

  BUG 3 (crash, second distinct root cause behind the same GTSAM error
  message): fixed after Bug 1/2 above, still crashed with "inconsistent
  arguments" on a real (non-synthetic-perfect) map. Cause: a freshly
  created keyframe can have every one of its own points at exactly 1
  observation (itself only), which min_obs_to_optimize correctly excludes
  from the optimized point set. If no OTHER keyframe happens to also
  observe any of that keyframe's points, its pose variable ends up
  inserted into `initial` with ZERO factors ever referencing it -- an
  isolated graph node, which GTSAM's elimination cannot handle. Same
  error message as Bug 1, different mechanism. Fixed by building the
  factor list first and tracking exactly which keyframe/point ids end up
  referenced by at least one factor, inserting only those into `initial`.
  used plain monocular 2D reprojection factors (pixel (u,v) only). Cost
  converged to ~0 while POSE AND POINT ERROR AGAINST GROUND TRUTH GOT
  WORSE after optimization -- the textbook signature of an under-
  constrained system converging to a valid-but-wrong solution. Root
  cause: 2D-only reprojection throws away the metric depth RGB-D actually
  measures, reintroducing monocular SfM's scale/gauge ambiguity that
  RGB-D exists specifically to avoid. Fixed by representing each
  observation with valid depth as a VIRTUAL STEREO measurement
  (u_L, u_R, v) with u_R = u_L - fx*baseline/depth, via GTSAM's
  GenericStereoFactor3D + Cal3_S2Stereo -- the same representation
  ORB-SLAM2 itself uses for RGB-D. Observations lacking valid depth fall
  back to monocular reprojection, which is safe ONLY because at least one
  keyframe is always hard-anchored (see the "if not fixed_kfs" branch).

DROP-IN CONTRACT: matches bundle_adjust.local_bundle_adjust()'s ARGUMENT
signature exactly: (world_map, camera, window, max_iter, verbose,
min_obs_to_optimize, huber_f_scale), plus one new optional kwarg
(`virtual_baseline`). The two functions' RETURN values differ (the scipy
original returns only `n_obs`; this returns a 6-tuple with more
diagnostics) -- checked against run_slam.py directly: the call site never
captures the return value at all, so this difference is safe. If you call
this function anywhere that DOES use the return value, adjust accordingly.

CONVENTION: frame.pose is camera-to-world (Twc), verified against ground
truth to 1e-16 in Phase 1's geo.py. GTSAM's Pose3 also expects world-frame
camera pose, so no convention translation is needed.
"""

import numpy as np
import gtsam
from gtsam import Point3, Pose3, Rot3, Cal3_S2, Cal3_S2Stereo, StereoPoint2
from gtsam.symbol_shorthand import X, L


def _pose3_from_np(T):
    return Pose3(Rot3(T[:3, :3]), Point3(T[:3, 3]))


def _np_from_pose3(pose3):
    T = np.eye(4)
    T[:3, :3] = pose3.rotation().matrix()
    T[:3, 3] = pose3.translation()
    return T


def local_bundle_adjust_gtsam(world_map, camera, window=8, max_iter=10,
                              verbose=False, min_obs_to_optimize=2,
                              huber_f_scale=1.0, virtual_baseline=0.05):
    """
    Returns (n_free_kfs, n_fixed_kfs, n_points, n_obs, cost_before,
    cost_after) to match the existing [BA] log line's fields.
    """
    ordered_kfs = sorted((kf for kf in world_map.keyframes if kf.pose is not None),
                         key=lambda kf: kf.kf_seq)
    if len(ordered_kfs) < 2:
        return 0, 0, 0, 0, 0.0, 0.0

    free_kfs = ordered_kfs[-window:]
    free_ids = {kf.id for kf in free_kfs}

    point_ids_in_window = set()
    for kf in free_kfs:
        for mp_id in kf.map_point_ids:
            if mp_id is not None:
                point_ids_in_window.add(mp_id)

    points = []
    for mp_id in point_ids_in_window:
        mp = world_map.map_points.get(mp_id)
        if mp is None or mp.is_bad:
            continue
        if mp.n_observations() < min_obs_to_optimize:
            continue
        points.append(mp)

    if not points or not free_kfs:
        return len(free_kfs), 0, 0, 0, 0.0, 0.0

    kf_by_id = {kf.id: kf for kf in world_map.keyframes}

    fixed_ids_abstract = set()
    for mp in points:
        for kf_id in mp.observations:
            if kf_id not in free_ids:
                fixed_ids_abstract.add(kf_id)
    fixed_kfs = [kf_by_id[kf_id] for kf_id in fixed_ids_abstract
                if kf_id in kf_by_id and kf_by_id[kf_id].pose is not None]
    # BUG 1 fix: single source of truth for "which keyframe variables
    # actually exist in the GTSAM graph" -- see module docstring.
    available_kf_ids = free_ids | {kf.id for kf in fixed_kfs}

    K_mono = Cal3_S2(camera.fx, camera.fy, 0.0, camera.cx, camera.cy)
    K_stereo = Cal3_S2Stereo(camera.fx, camera.fy, 0.0, camera.cx, camera.cy, virtual_baseline)
    huber = gtsam.noiseModel.mEstimator.Huber.Create(huber_f_scale)
    stereo_noise = gtsam.noiseModel.Robust.Create(
        huber, gtsam.noiseModel.Isotropic.Sigma(3, 1.0))
    mono_noise = gtsam.noiseModel.Robust.Create(
        huber, gtsam.noiseModel.Isotropic.Sigma(2, 1.0))

    # BUG 3 fix (found chasing the "inconsistent arguments" crash a SECOND
    # time, on a real -- not synthetic-perfect -- map): a freshly created
    # keyframe can have every one of its own points still at exactly 1
    # observation (itself only), which the min_obs_to_optimize filter
    # correctly excludes from `points`. If NO other free/fixed keyframe
    # happens to also observe any of that keyframe's points, that
    # keyframe's pose variable ends up in `initial` with ZERO factors
    # ever referencing it -- an isolated node, which GTSAM's elimination
    # ordering cannot handle (same "inconsistent arguments" message as
    # Bug 1, different root cause). Fixed by building the factor list
    # FIRST, tracking exactly which keyframe/point ids end up referenced
    # by at least one factor, and only inserting THOSE into `initial` --
    # rather than assuming every free/fixed keyframe will necessarily be
    # touched by the point set survived by min_obs_to_optimize.
    pending_stereo, pending_mono = [], []
    used_kf_ids, used_point_ids = set(), set()
    n_obs, n_obs_stereo = 0, 0
    for mp in points:
        obs_for_this_point = []
        for kf_id, kp_idx in mp.observations.items():
            if kf_id not in available_kf_ids:
                continue
            kf = kf_by_id[kf_id]
            if kp_idx >= len(kf.keypoints):
                continue
            u, v = kf.keypoints[kp_idx].pt
            depth = kf.depths[kp_idx] if hasattr(kf, "depths") and kp_idx < len(kf.depths) else -1.0
            obs_for_this_point.append((kf_id, u, v, depth))
        if not obs_for_this_point:
            continue
        for kf_id, u, v, depth in obs_for_this_point:
            if depth is not None and depth > 1e-3:
                u_r = u - camera.fx * virtual_baseline / depth
                pending_stereo.append((kf_id, mp.id, StereoPoint2(u, u_r, v)))
                n_obs_stereo += 1
            else:
                pending_mono.append((kf_id, mp.id, gtsam.Point2(u, v)))
            used_kf_ids.add(kf_id)
            n_obs += 1
        used_point_ids.add(mp.id)

    if not pending_stereo and not pending_mono:
        return len(free_kfs), len(fixed_kfs), 0, 0, 0.0, 0.0

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    used_free_kfs = [kf for kf in free_kfs if kf.id in used_kf_ids]
    used_fixed_kfs = [kf for kf in fixed_kfs if kf.id in used_kf_ids]

    for kf in used_free_kfs:
        initial.insert(X(kf.id), _pose3_from_np(kf.pose))
    for kf in used_fixed_kfs:
        pose3 = _pose3_from_np(kf.pose)
        initial.insert(X(kf.id), pose3)
        graph.push_back(gtsam.PriorFactorPose3(
            X(kf.id), pose3, gtsam.noiseModel.Constrained.All(6)))

    if not used_fixed_kfs:
        # Gauge-freedom anchor (see module docstring, Bug 2) -- pick the
        # oldest keyframe that's actually USED, not just the oldest free
        # keyframe, since (per Bug 3 above) the objectively-oldest one
        # could itself be the isolated/unused one in rare cases.
        if not used_free_kfs:
            return len(free_kfs), len(fixed_kfs), 0, 0, 0.0, 0.0
        anchor = min(used_free_kfs, key=lambda kf: kf.kf_seq)
        graph.push_back(gtsam.PriorFactorPose3(
            X(anchor.id), _pose3_from_np(anchor.pose),
            gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3] * 3 + [1e-2] * 3))))

    for mp_id in used_point_ids:
        initial.insert(L(mp_id), Point3(world_map.map_points[mp_id].position))

    for kf_id, mp_id, sp2 in pending_stereo:
        graph.push_back(gtsam.GenericStereoFactor3D(sp2, stereo_noise, X(kf_id), L(mp_id), K_stereo))
    for kf_id, mp_id, pt2 in pending_mono:
        graph.push_back(gtsam.GenericProjectionFactorCal3_S2(pt2, mono_noise, X(kf_id), L(mp_id), K_mono))

    free_kfs = used_free_kfs
    fixed_kfs = used_fixed_kfs

    if graph.size() == 0 or not used_point_ids:
        return len(free_kfs), len(fixed_kfs), 0, 0, 0.0, 0.0

    cost_before = graph.error(initial)

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iter)
    params.setVerbosityLM("SILENT" if not verbose else "SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    cost_after = graph.error(result)

    for kf in free_kfs:
        kf.set_pose(_np_from_pose3(result.atPose3(X(kf.id))))
    for mp in points:
        if mp.id in used_point_ids:
            mp.position = np.array(result.atPoint3(L(mp.id)))

    if verbose:
        print(f"[BA-GTSAM] {len(free_kfs)} free + {len(fixed_kfs)} fixed kfs, "
             f"{len(used_point_ids)} pts, {n_obs} obs ({n_obs_stereo} stereo-depth, "
             f"{n_obs - n_obs_stereo} mono-fallback) | "
             f"cost {cost_before:.1f} -> {cost_after:.1f}")

    return len(free_kfs), len(fixed_kfs), len(used_point_ids), n_obs, cost_before, cost_after

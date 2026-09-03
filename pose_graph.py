"""
pose_graph.py — Phase 5: loop closure CORRECTION.

Maps to: ORB_SLAM3/src/LoopClosing.cc CorrectLoop() + the Essential Graph
optimization (Optimizer::OptimizeEssentialGraph), simplified from Sim3 to
plain SE(3) since this project is RGB-D (metric scale already fixed, so
there's no scale-drift-between-map-segments problem monocular ORB-SLAM3
has to solve for).

SCOPE NOTE: no global bundle adjustment pass after correction (real
ORB-SLAM3 runs one). A full GBA over the whole map is exactly the kind of
expensive, non-bounded call the Phase 0 rewrite of bundle_adjust.py was
built to avoid triggering casually -- see that module's docstring. Skipped
here on purpose; if you want it, call local_bundle_adjust with a large
window after correction and expect it to take a while.

CAUTION (worth repeating from the handoff): this module trusts (a) the
existing consecutive-keyframe relative poses as "odometry" edges, and (b)
one PnP-based measurement at the loop as the "loop" edge. If tracking/BA is
producing bad relative poses somewhere along the path, this will smooth
that error across the graph rather than surface it -- it does not fix a
broken odometry edge, it dilutes it. Validate tracking/BA quality on a
known-good baseline before trusting corrected trajectories as ground truth.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares

from bundle_adjust import _pose_to_vec, _vec_to_pose
from imu import log_so3
import covisibility
import fusion


def compute_loop_edge(current_kf, matched_kf, matcher, camera, world_map,
                      min_inliers=25, reprojection_error=4.0):
    """
    The "measurement" for the loop edge: match current_kf's descriptors
    against matched_kf's descriptors, keep only matches that land on a
    keypoint in matched_kf with an associated MapPoint (giving a real 3D
    world position), then solve PnP. This answers "if current_kf is
    genuinely back at this place, what pose would make it consistent with
    matched_kf's existing map?" -- the gap between that and current_kf's
    actual (drifted) pose is exactly the loop error being corrected.

    Returns (T_rel, n_inliers) where T_rel is the measured matched_kf ->
    current_kf relative transform (camera-to-world composition), or None
    if there isn't a confident enough match.
    """
    matches = matcher.match(current_kf.descriptors, matched_kf.descriptors)
    pts2d, pts3d = [], []
    for m in matches:
        mp_id = matched_kf.map_point_ids[m.trainIdx]
        if mp_id is None:
            continue
        mp = world_map.map_points.get(mp_id)
        if mp is None or mp.is_bad:
            continue
        pts2d.append(current_kf.points_undistorted[m.queryIdx])
        pts3d.append(mp.position)

    if len(pts3d) < min_inliers:
        return None

    pts2d = np.asarray(pts2d, dtype=np.float64)
    pts3d = np.asarray(pts3d, dtype=np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, camera.K, None,
        iterationsCount=300, reprojectionError=reprojection_error,
        confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None

    R, _ = cv2.Rodrigues(rvec)
    pose_cw_meas = np.eye(4)          # world-to-camera, per the OLD/authoritative map
    pose_cw_meas[:3, :3] = R
    pose_cw_meas[:3, 3] = tvec.flatten()
    current_pose_measured = np.linalg.inv(pose_cw_meas)   # camera-to-world

    T_rel = np.linalg.inv(matched_kf.pose) @ current_pose_measured
    return T_rel, len(inliers)


def optimize_pose_graph(world_map, current_kf, matched_kf, T_loop_rel, loop_weight,
                        camera, covis_graph=None, extractor=None,
                        odom_weight=10.0, max_nfev=2000, verbose=False):
    """
    SE(3) pose-graph optimization: redistribute the loop error smoothly
    across every keyframe between the loop pair, using the existing
    consecutive-keyframe relative poses as trusted "odometry" edges (high
    weight) and the single PnP-measured loop transform as the correcting
    edge (weighted by loop_weight, typically scaled by its inlier count).

    The chronologically FIRST keyframe is the fixed gauge anchor -- same
    convention bundle_adjust.py uses.

    Applies corrected poses to every keyframe, rotates velocities by the
    same correction (if IMU is active), corrects every map point via its
    EARLIEST observing keyframe's correction delta (ORB-SLAM3's approach --
    avoids re-triangulating anything), then runs one fusion pass between
    the loop-connected keyframe pair (they were never naturally covisible
    before the loop, so this needs to be forced rather than discovered).

    Returns a stats dict, or None if there weren't enough keyframes to
    optimize.
    """
    kfs = sorted(world_map.keyframes, key=lambda kf: kf.kf_seq)
    if len(kfs) < 3:
        return None

    anchor = kfs[0]
    free_kfs = kfs[1:]
    old_poses = {kf.id: kf.pose.copy() for kf in kfs}

    # odometry: trust the existing consecutive relative poses
    edges = []   # (kf_a, kf_b, T_meas, weight)
    for i in range(1, len(kfs)):
        T_meas = np.linalg.inv(kfs[i - 1].pose) @ kfs[i].pose
        edges.append((kfs[i - 1], kfs[i], T_meas, odom_weight))
    # the loop edge -- this is what actually pulls the path shut
    edges.append((matched_kf, current_kf, T_loop_rel, loop_weight))

    idx = {kf.id: i for i, kf in enumerate(free_kfs)}   # anchor excluded -> None
    x0 = np.concatenate([_pose_to_vec(kf.pose) for kf in free_kfs])

    def pose_of(x, kf):
        if kf.id == anchor.id:
            return anchor.pose
        return _vec_to_pose(x[idx[kf.id] * 6: idx[kf.id] * 6 + 6])

    def residuals(x):
        res = []
        for kf_a, kf_b, T_meas, w in edges:
            T_a, T_b = pose_of(x, kf_a), pose_of(x, kf_b)
            err = np.linalg.inv(T_meas) @ (np.linalg.inv(T_a) @ T_b)
            r_R = log_so3(err[:3, :3])
            r_t = err[:3, 3]
            res.append(np.concatenate([r_R, r_t]) * w)
        return np.concatenate(res)

    before = float(np.sum(residuals(x0) ** 2))
    result = least_squares(residuals, x0, method='trf', loss='huber',
                           f_scale=1.0, max_nfev=max_nfev)
    after = float(np.sum(residuals(result.x) ** 2))

    # apply corrected poses + velocities, compute per-kf correction delta
    delta_by_id = {}
    for kf in free_kfs:
        new_pose = pose_of(result.x, kf)
        delta = new_pose @ np.linalg.inv(old_poses[kf.id])
        delta_by_id[kf.id] = delta
        if getattr(kf, "velocity", None) is not None:
            kf.velocity = delta[:3, :3] @ kf.velocity
        kf.set_pose(new_pose)

    # correct map points via their EARLIEST observing keyframe's delta
    n_pts_corrected = 0
    for mp in world_map.map_points.values():
        if mp.is_bad or not mp.observations:
            continue
        ref_kf_id = min(mp.observations.keys())
        delta = delta_by_id.get(ref_kf_id)
        if delta is None:
            continue   # anchored to the fixed keyframe -- no correction needed
        mp.position = delta[:3, :3] @ mp.position + delta[:3, 3]
        mp.normal_vector = None   # stale after moving; recomputed opportunistically
        n_pts_corrected += 1

    # fuse across the loop seam: these two keyframes were never
    # naturally covisible before the loop closed, so force them as a
    # one-off neighbor pair rather than relying on the graph to find them
    n_levels = extractor.nlevels if extractor else 8
    scale_factor = extractor.scale_factor if extractor else 1.2
    ad_hoc_graph = {matched_kf.id: {current_kf.id: 999}, current_kf.id: {matched_kf.id: 999}}
    n_fused = fusion.search_in_neighbors(current_kf, world_map, ad_hoc_graph, camera,
                                         n_levels=n_levels, scale_factor=scale_factor)

    if covis_graph is not None:
        covisibility.build_covisibility_graph(world_map, graph=covis_graph)

    stats = {
        "n_keyframes_corrected": len(free_kfs),
        "n_points_corrected": n_pts_corrected,
        "n_fused": n_fused,
        "cost_before": before,
        "cost_after": after,
    }
    if verbose:
        print(f"    [PGO] {len(free_kfs)} kfs, {n_pts_corrected} pts corrected, "
              f"{n_fused} fused | cost {before:.2f} -> {after:.2f}")
    return stats

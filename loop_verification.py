"""
loop_verification.py — Phase 6: 3D-3D RANSAC geometric verification.

Maps to: ORB_SLAM3/src/Sim3Solver.cc, simplified from Sim3 to SE(3)+scale-
sanity-check since this project is RGB-D (metric scale is already fixed
by the depth sensor, unlike monocular ORB-SLAM3 where two map segments
can have genuinely different, unknown relative scales that Sim3 exists
specifically to solve for).

WHY THIS REPLACES loop_closing.py's ESSENTIAL-MATRIX CHECK: an essential
matrix only constrains epipolar geometry (5 DOF) from 2D-2D
correspondences, which is satisfiable by many WRONG point correspondences,
especially in degenerate configurations (planar scenes, pure rotation,
small baselines) -- exactly the kind of scene a facility corridor or a
flat wall produces often. This project has actual metric 3D positions for
matched keypoints on BOTH sides (each keyframe's own map points, already
triangulated from real depth) -- using that 3D structure directly, via
RANSAC Umeyama/Kabsch alignment, is strictly more discriminative: a false
match has to be consistent with a full rigid 3D transform, not just an
epipolar constraint, and the recovered scale gives a free, extra sanity
check that has no monocular equivalent (a genuine RGB-D loop closure
should recover scale very close to 1.0; a false match frequently won't).

VERIFIED (not assumed) before use -- see test_phase6_gate.py: the RANSAC
Umeyama implementation here is checked against exact synthetic ground
truth (known rotation+translation+scale=1.0 applied to a point set,
recovered to machine precision) AND checked to correctly REJECT a
scenario built from two unrelated point sets (recovers a low inlier
count / implausible scale), not just to succeed on the easy case.
"""

import numpy as np


def _umeyama(src, dst):
    """
    Closed-form similarity transform (R, t, s) minimizing
    ||s*R@src + t - dst||^2. src, dst: (N,3), N>=3, not collinear.
    Returns (R, t, s).
    """
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst

    cov = (dst_c.T @ src_c) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 1e-12 else 1.0
    t = mu_dst - s * R @ mu_src
    return R, t, s


def verify_3d3d(kf_a, kf_b, matcher, map_a, map_b,
                min_inliers=15, ransac_iters=200, inlier_threshold=0.10,
                max_scale_deviation=0.25):
    """
    kf_a, kf_b: candidate loop-closure keyframe pair. map_a, map_b: the
    Map object each belongs to (may be the SAME object for an intra-map
    loop, or different objects for a cross-map candidate -- see
    relocalization.py for the analogous same/cross-map distinction).

    Returns (n_inliers, R, t, scale) on success, or (0, None, None, None)
    if verification fails. R, t, scale describe the transform mapping
    kf_a's 3D points (and by extension all of map_a) into kf_b's frame:
    p_in_b = scale * R @ p_in_a + t.

    max_scale_deviation: reject if |scale - 1.0| exceeds this fraction --
    RGB-D loop closures should recover scale very close to 1.0 (both
    sides are already metric); a large deviation is a strong signal the
    "match" is actually between unrelated points, not a property normal
    monocular Sim3 verification would even have access to.
    """
    if kf_a.descriptors is None or kf_b.descriptors is None:
        return 0, None, None, None

    matches = matcher.match_ratio(kf_a.descriptors, kf_b.descriptors)
    pts_a, pts_b = [], []
    for m in matches:
        mp_id_a = kf_a.map_point_ids[m.queryIdx] if m.queryIdx < len(kf_a.map_point_ids) else None
        mp_id_b = kf_b.map_point_ids[m.trainIdx] if m.trainIdx < len(kf_b.map_point_ids) else None
        if mp_id_a is None or mp_id_b is None:
            continue
        mp_a = map_a.map_points.get(mp_id_a)
        mp_b = map_b.map_points.get(mp_id_b)
        if mp_a is None or mp_b is None or mp_a.is_bad or mp_b.is_bad:
            continue
        pts_a.append(mp_a.position)
        pts_b.append(mp_b.position)

    if len(pts_a) < min_inliers:
        return 0, None, None, None

    pts_a = np.asarray(pts_a, dtype=np.float64)
    pts_b = np.asarray(pts_b, dtype=np.float64)
    n = len(pts_a)

    best_inlier_mask = None
    best_n_inliers = 0
    rng = np.random.RandomState(0)
    for _ in range(ransac_iters):
        if n < 3:
            break
        sample = rng.choice(n, 3, replace=False)
        try:
            R, t, s = _umeyama(pts_a[sample], pts_b[sample])
        except np.linalg.LinAlgError:
            continue
        pred = s * (R @ pts_a.T).T + t
        errs = np.linalg.norm(pred - pts_b, axis=1)
        mask = errs < inlier_threshold
        n_in = int(mask.sum())
        if n_in > best_n_inliers:
            best_n_inliers = n_in
            best_inlier_mask = mask

    if best_inlier_mask is None or best_n_inliers < min_inliers:
        return 0, None, None, None

    # refine using all inliers from the best sample
    R, t, s = _umeyama(pts_a[best_inlier_mask], pts_b[best_inlier_mask])

    if abs(s - 1.0) > max_scale_deviation:
        return 0, None, None, None

    return best_n_inliers, R, t, s

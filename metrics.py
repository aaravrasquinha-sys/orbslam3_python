"""
metrics.py — quantitative evaluation against ground truth + map-health stats.

Standard SLAM accuracy metrics (ATE, RPE) plus the map-lifecycle statistics
that actually diagnosed the Phase-1 bug (keyframes alive, observation-count
histogram) -- see PROGRESS.md's ablation table. Every future change should
be checked against both categories: a change can improve ATE while quietly
re-breaking the map lifecycle (or vice versa), and only checking one hides
the other kind of regression.
"""

import numpy as np


def align_umeyama(est, gt):
    """
    Umeyama (1991) closed-form similarity alignment (rotation + translation
    + scale) of `est` onto `gt`. Needed before computing ATE for a
    monocular/scale-free run; for RGB-D (metric) runs scale should come out
    ~1.0, and a large deviation from 1.0 is itself a useful diagnostic (it
    means something upstream is silently rescaling the map).

    est, gt: (N,3) arrays of camera centers, same length, corresponding
    frame-for-frame.

    Returns (R, t, s, aligned_est).
    """
    assert est.shape == gt.shape and est.shape[1] == 3
    mu_est, mu_gt = est.mean(axis=0), gt.mean(axis=0)
    est_c, gt_c = est - mu_est, gt - mu_gt

    cov = (gt_c.T @ est_c) / len(est)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    var_est = (est_c ** 2).sum() / len(est)
    s = float(np.trace(np.diag(D) @ S) / var_est) if var_est > 1e-12 else 1.0
    t = mu_gt - s * R @ mu_est

    aligned = (s * (R @ est.T).T) + t
    return R, t, s, aligned


def ate(est_poses, gt_poses, align=True):
    """
    Absolute Trajectory Error: RMSE of camera-center distance after optimal
    alignment (Umeyama). Returns dict with rmse, mean, median, max, scale.

    est_poses, gt_poses: lists of 4x4 camera-to-world matrices, same length
    and frame correspondence. None entries in est_poses are dropped (paired
    gt entry dropped too) so a partial/lost trajectory can still be scored
    on whatever it DID estimate.
    """
    pairs = [(e, g) for e, g in zip(est_poses, gt_poses) if e is not None]
    if len(pairs) < 3:
        return {"rmse": None, "mean": None, "median": None, "max": None,
               "scale": None, "n_valid": len(pairs), "n_total": len(gt_poses)}

    est = np.array([p[:3, 3] for p, _ in pairs])
    gt = np.array([p[:3, 3] for _, p in pairs])

    if align:
        _, _, s, aligned = align_umeyama(est, gt)
    else:
        aligned, s = est, 1.0

    err = np.linalg.norm(aligned - gt, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "max": float(np.max(err)),
        "scale": s,
        "n_valid": len(pairs),
        "n_total": len(gt_poses),
    }


def rpe(est_poses, gt_poses, delta=1):
    """
    Relative Pose Error over a fixed frame delta: measures local drift rate
    rather than global offset. Complements ATE (a trajectory can have low
    ATE from lucky cancellation while having high local drift, or vice
    versa). Returns translational RPE only (rotational RPE is a
    straightforward extension via log_so3 on the same relative transforms,
    added if/when needed).
    """
    pairs = [(e, g) for e, g in zip(est_poses, gt_poses) if e is not None]
    if len(pairs) < delta + 2:
        return {"rmse": None, "n_pairs": 0}

    errs = []
    for i in range(len(pairs) - delta):
        e0, g0 = pairs[i]
        e1, g1 = pairs[i + delta]
        rel_est = np.linalg.inv(e0) @ e1
        rel_gt = np.linalg.inv(g0) @ g1
        errs.append(np.linalg.norm(rel_est[:3, 3] - rel_gt[:3, 3]))

    errs = np.asarray(errs)
    return {"rmse": float(np.sqrt(np.mean(errs ** 2))), "n_pairs": len(errs)}


def map_health(slam_system):
    """
    The statistics that actually diagnosed the Phase-1 bug: not just "does
    it track", but "does the map accumulate mature, multiply-observed
    landmarks, or does it churn". A healthy map should show most points
    with 3+ observations after culling stabilizes; the ORIGINAL
    cull_keyframes() produced a map where NO point ever exceeded 3
    observations (see PROGRESS.md ablation table).
    """
    m = slam_system.atlas.active_map
    obs_counts = [mp.n_observations() for mp in m.good_map_points()]
    import collections
    hist = dict(sorted(collections.Counter(obs_counts).items()))

    return {
        "n_keyframes_alive": m.n_keyframes(),
        "n_keyframes_made": slam_system.stats["keyframes"],
        "n_keyframes_culled": slam_system.stats["keyframes_culled"],
        "n_points_alive": m.n_map_points(),
        "n_points_created": slam_system.stats["points_created"],
        "n_points_culled": slam_system.stats["points_culled"],
        "n_points_fused": slam_system.stats["points_fused"],
        "n_points_obs_ge4": sum(1 for o in obs_counts if o >= 4),
        "max_observations": max(obs_counts) if obs_counts else 0,
        "obs_histogram": hist,
        "n_atlas_maps": slam_system.atlas.n_maps(),
        "n_tracking_losses": slam_system.stats["lost"],
        "n_loop_closures": slam_system.stats["loops"],
    }


def print_report(name, health, ate_result=None, rpe_result=None):
    print(f"\n{'=' * 64}\n  {name}\n{'=' * 64}")
    print(f"Keyframes:  made={health['n_keyframes_made']:4d}  "
         f"culled={health['n_keyframes_culled']:4d}  "
         f"alive={health['n_keyframes_alive']:4d}")
    print(f"Points:     created={health['n_points_created']:5d}  "
         f"culled={health['n_points_culled']:5d}  "
         f"fused={health['n_points_fused']:5d}  "
         f"alive={health['n_points_alive']:5d}")
    print(f"Map health: obs>=4 = {health['n_points_obs_ge4']:5d}   "
         f"max_obs = {health['max_observations']:3d}")
    print(f"            histogram: {health['obs_histogram']}")
    print(f"Tracking:   losses={health['n_tracking_losses']:4d}  "
         f"atlas_maps={health['n_atlas_maps']:3d}  "
         f"loop_closures={health['n_loop_closures']:3d}")
    if ate_result and ate_result["rmse"] is not None:
        print(f"ATE:        rmse={ate_result['rmse']:.4f}m  "
             f"median={ate_result['median']:.4f}m  "
             f"max={ate_result['max']:.4f}m  "
             f"scale={ate_result['scale']:.4f}  "
             f"({ate_result['n_valid']}/{ate_result['n_total']} posed)")
    if rpe_result and rpe_result["rmse"] is not None:
        print(f"RPE(Δ=1):   rmse={rpe_result['rmse']:.4f}m/frame  "
             f"n={rpe_result['n_pairs']}")

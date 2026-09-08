"""
merge_maps.py — Phase 6: actual map merging, not just switching.

WHY THIS EXISTS: relocalization.py (Phase 5) recovers tracking by
switching the Atlas's active map back to a recognized old one, but
EXPLICITLY does not merge -- the just-abandoned fragment's own keyframes
and points stay in their own, now-inactive Map object forever, never
contributing to the map again even though they represent real, valid
structure. This is the actual Phase 6 fix: when a loop is confirmed
BETWEEN TWO DIFFERENT ATLAS MAPS (not just within one, which
pose_graph.py already handles), transform the newer/smaller map's
keyframes and points into the older/surviving map's frame using the
verified similarity transform (loop_verification.py) and fold them in as
one continuous map.

TRANSFORM CONVENTION: matches loop_verification.verify_3d3d's documented
output exactly -- (R, t, s) map points from `absorb_map`'s frame into
`keep_map`'s frame via p_new = s*R@p_old + t. Camera poses transform
consistently: rotation composes as R@old_R (the camera's ORIENTATION
rotates by R), translation (camera center) transforms exactly like any
other point (s*R@center + t) -- this is the standard similarity-
transform composition for SE(3) poses, and matches the same pattern
pose_graph.py's own delta-correction already uses elsewhere in this
project (rotate velocity by the delta's rotation, apply the full delta
to position).

CHRONOLOGICAL ORDERING (PHASE 7 CHANGE): kf_seq is now a GLOBAL,
Atlas-wide, monotonically increasing counter assigned ONCE at keyframe
creation (Map.add_keyframe -> Frame._next_kf_seq -- see frame.py) and
never reassigned afterward. Earlier versions of this function RENUMBERED
every keyframe's kf_seq after a merge (sorted by kf.id, reassigned
0..N-1) because kf_seq used to be a per-Map counter and the two source
maps' numbering wasn't comparable. That renumbering was itself a bug:
every MapPoint created before the merge stores first_keyframe_id in
kf_seq units (see local_mapping.py / map_point.py), and renumbering
silently invalidated all of those, corrupting age-based culling on
every pre-merge point. Now that kf_seq is global and permanent at the
source, the merged keyframe list only needs SORTING (by kf_seq, which
already reflects true creation order across both maps) for iteration
convenience -- nothing is ever reassigned.

DELIBERATELY NOT DONE HERE (documented scope boundary, not an oversight):
a global bundle-adjustment pass over the freshly-merged map. Real
ORB-SLAM3 runs one after every merge; this project's bundle_adjust.py
was specifically built to avoid being triggered on large, unbounded
windows casually (see that module's own docstring). If the merged map
looks locally inconsistent after this function returns, running
local_bundle_adjust with a generously large window afterward is the
expected next step -- not something this function does automatically.
"""

import numpy as np
import fusion
import covisibility


def merge_maps(atlas, keep_map, absorb_map, R, t, s,
               anchor_kf_a=None, anchor_kf_b=None,
               camera=None, matcher=None, extractor=None,
               covis_graph=None, verbose=False):
    """
    Merge absorb_map into keep_map using the similarity transform (R, t, s)
    that maps absorb_map's frame into keep_map's frame (see module
    docstring for the exact convention -- must match
    loop_verification.verify_3d3d's output exactly, not a guessed
    inverse or transpose).

    anchor_kf_a/anchor_kf_b: the two loop-closure keyframes that
    triggered this merge (anchor_kf_a in absorb_map, anchor_kf_b in
    keep_map) -- if given along with camera/matcher/extractor, a fusion
    pass is forced across this pair afterward, since they were never
    naturally covisible before the merge (same pattern pose_graph.py
    already uses for intra-map loop seams).

    Returns a stats dict. Mutates keep_map and atlas in place; absorb_map
    is removed from the Atlas (but the Python object itself is left
    intact, unreferenced by the Atlas -- nothing in this project holds a
    stale reference to it afterward, since Atlas.maps is the only place
    that listed it).
    """
    n_kfs_before = keep_map.n_keyframes()
    n_pts_before = keep_map.n_map_points()

    for kf in absorb_map.keyframes:
        old_pose = kf.pose
        if old_pose is None:
            continue
        new_pose = np.eye(4)
        new_pose[:3, :3] = R @ old_pose[:3, :3]
        new_pose[:3, 3] = s * (R @ old_pose[:3, 3]) + t
        kf.pose = new_pose
        if getattr(kf, "velocity", None) is not None:
            kf.velocity = R @ kf.velocity

    n_pts_transformed = 0
    for mp in absorb_map.map_points.values():
        if mp.is_bad:
            continue
        mp.position = s * (R @ mp.position) + t
        mp.normal_vector = None   # stale after the transform; recomputed opportunistically
        n_pts_transformed += 1

    # PHASE 7: kf_seq is global and was assigned once at creation (see
    # module docstring) -- sort for chronological iteration order only,
    # never reassign. This is what keeps every MapPoint.first_keyframe_id
    # created before this merge still valid afterward.
    all_kfs = sorted(list(keep_map.keyframes) + list(absorb_map.keyframes),
                     key=lambda kf: kf.kf_seq)
    keep_map.keyframes = all_kfs

    keep_map.map_points.update(absorb_map.map_points)   # ids are globally
                                                         # unique (MapPoint
                                                         # ._next_id is a
                                                         # class-level
                                                         # counter) -- no
                                                         # collision risk

    atlas.maps.remove(absorb_map)
    atlas.switch_active_map(keep_map)

    n_fused = 0
    if anchor_kf_a is not None and anchor_kf_b is not None and camera is not None and matcher is not None:
        n_levels = extractor.nlevels if extractor else 8
        scale_factor = extractor.scale_factor if extractor else 1.2
        ad_hoc_graph = {anchor_kf_b.id: {anchor_kf_a.id: 999},
                        anchor_kf_a.id: {anchor_kf_b.id: 999}}
        n_fused = fusion.search_in_neighbors(anchor_kf_b, keep_map, ad_hoc_graph, camera,
                                             n_levels=n_levels, scale_factor=scale_factor)

    if covis_graph is not None:
        covisibility.build_covisibility_graph(keep_map, graph=covis_graph)

    stats = {
        "kept_map_id": keep_map.id,
        "absorbed_map_id": absorb_map.id,
        "n_keyframes_before": n_kfs_before,
        "n_keyframes_after": keep_map.n_keyframes(),
        "n_points_before": n_pts_before,
        "n_points_transformed": n_pts_transformed,
        "n_points_after": keep_map.n_map_points(),
        "n_fused": n_fused,
        "scale_applied": s,
    }
    if verbose:
        print(f"    [MERGE] map {stats['absorbed_map_id']} -> map {stats['kept_map_id']}: "
             f"{n_kfs_before}->{stats['n_keyframes_after']} kfs, "
             f"{n_pts_before}->{stats['n_points_after']} pts, "
             f"{n_fused} fused, scale={s:.4f}")
    return stats

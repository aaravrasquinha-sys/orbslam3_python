"""
fusion.py — SearchInNeighbors: duplicate map point fusion.

Maps to: ORB_SLAM3/src/ORBmatcher.cc Fuse() + LocalMapping.cc SearchInNeighbors()

WHY THIS EXISTS: local_mapping._create_points_from_depth() creates a new
MapPoint for every unmatched keypoint with valid depth, keyframe after
keyframe, with no mechanism to notice "this is the same physical landmark I
already have a point for from three keyframes ago." Without fusion, RGB-D
point creation alone produces a cloud of near-duplicates, almost all with
exactly one observation -- which is also why bundle_adjust.py's ">=2
observations to optimize" rule would otherwise exclude nearly everything.
This module is what actually manufactures multi-observation points.

Algorithm, once per new keyframe:
  1. Take the new keyframe's covisible neighbors (via covisibility.py).
  2. Project the new keyframe's points into each neighbor. Where the
     projection lands within a guided-matching radius of an existing
     keypoint in that neighbor:
       - if that keypoint has no MapPoint yet, adopt it (add observation)
       - if it already holds a DIFFERENT MapPoint, fuse the two: keep
         whichever has more observations, union the observation sets, mark
         the loser bad and remove it from the map
  3. Repeat in the other direction: project each neighbor's points into the
     new keyframe.
"""

import numpy as np

import covisibility

TH_LOW = 50   # ORBmatcher::TH_LOW -- strict Hamming threshold, same as matcher.py


def _hamming(a, b):
    return int(np.count_nonzero(np.unpackbits(np.bitwise_xor(a, b))))


def _fuse_point(world_map, keep_id, absorb_id, kf_by_id):
    """
    Merge the two points, keeping whichever has more observations. Also
    repoints every keyframe's map_point_ids entry that referenced the
    absorbed point to the survivor -- without this, those frames keep a
    dangling reference to an id that clean_bad_points() will delete, which
    silently loses that observation AND permanently blocks a new point
    ever being created at that pixel (local_mapping skips any keypoint
    index whose map_point_ids entry is non-None).
    Returns the surviving point's id, or None if the merge couldn't happen.
    """
    if keep_id == absorb_id:
        return keep_id
    keep = world_map.map_points.get(keep_id)
    absorb = world_map.map_points.get(absorb_id)
    if keep is None or absorb is None or keep.is_bad or absorb.is_bad:
        return None

    # Prefer keeping whichever point has accumulated more observations --
    # it's the more established landmark.
    if absorb.n_observations() > keep.n_observations():
        keep, absorb = absorb, keep

    for kf_id, kp_idx in list(absorb.observations.items()):
        keep.add_observation(kf_id, kp_idx)
        kf = kf_by_id.get(kf_id)
        if kf is not None and 0 <= kp_idx < len(kf.map_point_ids):
            kf.map_point_ids[kp_idx] = keep.id
    keep.increase_visible(absorb.n_visible)
    keep.increase_found(absorb.n_found)
    absorb.set_bad()
    return keep.id


def _project_and_fuse(src_points, target_kf, world_map, camera,
                      n_levels, scale_factor, kf_by_id, radius_px=4.0):
    """
    Project `src_points` (MapPoint ids) into `target_kf`. For each that
    lands in-frustum, guided-match against target_kf's keypoints in a small
    radius; fuse or adopt as appropriate. Returns count of fuse/adopt events.
    """
    n_events = 0
    for mp_id in src_points:
        mp = world_map.map_points.get(mp_id)
        if mp is None or mp.is_bad:
            continue
        if target_kf.id in mp.observations:
            continue   # target already observes this point directly

        visible, u, v, octave = covisibility.is_in_frustum(
            mp, target_kf, camera, n_levels=n_levels, scale_factor=scale_factor)
        if not visible:
            continue

        radius = radius_px * (scale_factor ** octave if octave else 1.0)
        candidates = target_kf.get_features_in_area(
            u, v, radius, min_level=max(0, octave - 1), max_level=octave + 1)
        if not candidates:
            continue

        best_dist, best_idx = TH_LOW + 1, -1
        for kp_i in candidates:
            if target_kf.descriptors is None or kp_i >= len(target_kf.descriptors):
                continue
            d = _hamming(mp.descriptor, target_kf.descriptors[kp_i])
            if d < best_dist:
                best_dist, best_idx = d, kp_i
        if best_idx < 0 or best_dist > TH_LOW:
            continue

        existing_id = target_kf.map_point_ids[best_idx]
        if existing_id is None:
            mp.add_observation(target_kf.id, best_idx)
            target_kf.map_point_ids[best_idx] = mp.id
            n_events += 1
        elif existing_id != mp.id:
            survivor = _fuse_point(world_map, keep_id=mp.id, absorb_id=existing_id,
                                   kf_by_id=kf_by_id)
            if survivor is not None:
                target_kf.map_point_ids[best_idx] = survivor
                n_events += 1
    return n_events


def search_in_neighbors(keyframe, world_map, covis_graph, camera,
                        n_levels=8, scale_factor=1.2,
                        min_weight=15, max_neighbors=10):
    """
    Run fusion between `keyframe` and its covisible neighbors, both
    directions. Call this once per new keyframe, after local mapping has
    created its new points (so there's something for neighbors to fuse
    against) and after the covisibility graph has been rebuilt to include it.

    Returns the number of fuse/adopt events (useful for logging/stats).
    """
    neighbors_dict = covis_graph.get(keyframe.id, {})
    neighbor_ids = [kid for kid, w in sorted(neighbors_dict.items(), key=lambda kv: -kv[1])
                   if w >= min_weight][:max_neighbors]
    kf_by_id = {kf.id: kf for kf in world_map.keyframes}
    neighbors = [kf_by_id[kid] for kid in neighbor_ids if kid in kf_by_id]
    if not neighbors:
        return 0

    my_points = [mp_id for mp_id in keyframe.map_point_ids if mp_id is not None]

    n_events = 0
    # direction 1: this keyframe's points -> each neighbor
    for nb in neighbors:
        n_events += _project_and_fuse(my_points, nb, world_map, camera,
                                      n_levels, scale_factor, kf_by_id)

    # direction 2: each neighbor's points -> this keyframe
    neighbor_points = set()
    for nb in neighbors:
        for mp_id in nb.map_point_ids:
            if mp_id is not None:
                neighbor_points.add(mp_id)
    n_events += _project_and_fuse(neighbor_points, keyframe, world_map, camera,
                                  n_levels, scale_factor, kf_by_id)

    world_map.clean_bad_points()
    return n_events

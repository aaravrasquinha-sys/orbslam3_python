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
        # PHASE 7 BUGFIX: `keep` may ALREADY observe this same kf_id at
        # a DIFFERENT keypoint index (both points can be independently
        # detected from the same keyframe before being fused) -- see
        # map_point.py's add_observation docstring for the full
        # mechanism. Must clear that stale slot or it dangles forever
        # once `keep` is eventually deleted (its own .observations will
        # only remember the NEW index by then).
        stale_idx = keep.add_observation(kf_id, kp_idx)
        kf = kf_by_id.get(kf_id)
        if kf is not None:
            if (stale_idx is not None and 0 <= stale_idx < len(kf.map_point_ids)
                    and kf.map_point_ids[stale_idx] == keep.id):
                kf.map_point_ids[stale_idx] = None
            if 0 <= kp_idx < len(kf.map_point_ids):
                kf.map_point_ids[kp_idx] = keep.id
    # PHASE 7 BUGFIX: found via validate.py surfacing 503 "observations
    # references a frame that isn't a live keyframe anywhere" errors
    # (check_observations_are_keyframes) on a run that forces heavy
    # fusion + relocalization-disabled fragmentation. Root cause: this
    # loop redirects every keyframe absorb used to observe over to
    # `keep` (correct, above) but never cleared absorb.observations
    # ITSELF -- so absorb's dict kept the OLD, now-superseded entries.
    # Moments later, search_in_neighbors() calls clean_bad_points() on
    # this now-is_bad point, which walks absorb.observations (still
    # full of stale entries) and blindly nulls kf.map_point_ids[kp_idx]
    # for each one -- WIPING OUT the correct redirect this loop just
    # made to `keep`, moments after making it. keep's own .observations
    # dict still (correctly) claims that keyframe, but the keyframe's
    # own array no longer agrees -- and once that keyframe is later
    # culled, cull_keyframes() only walks map_point_ids (which no
    # longer lists keep there) to decide what to clean up, so keep's
    # now-orphaned claim survives forever. Clearing absorb's own dict
    # here, immediately after redirecting each entry, is the direct fix
    # -- and map.py's clean_bad_points/erase_map_point also got a
    # defensive ownership check (only clear a slot if it still points
    # at the point actually being deleted) as a second, independent
    # layer against this same failure shape recurring elsewhere.
    absorb.observations.clear()
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
                # BUGFIX (found during Phase 6 verification -- see
                # map.py's own bugfix comment for the sibling half of
                # this issue): _fuse_point's internal redirect loop only
                # updates keyframes found in absorb.observations. But
                # target_kf's own connection to `existing_id` (=absorb)
                # was established via THIS line's map_point_ids
                # assignment -- if THAT prior assignment ever happened
                # through a path that itself skipped add_observation
                # (confirmed: tracking.py's _track_local_map and
                # _solve_pnp both did, before their own fix below),
                # target_kf would never have been in absorb.observations
                # to begin with, so _fuse_point's loop silently never
                # sees it. Unconditionally registering it here as well
                # closes that gap regardless of how the original
                # assignment happened -- add_observation is a plain dict
                # assignment (self.observations[frame_id]=kp_idx),
                # naturally idempotent, so this is always safe to call.
                survivor_mp = world_map.map_points.get(survivor)
                if survivor_mp is not None:
                    # PHASE 7 BUGFIX: same mechanism as _fuse_point above
                    # -- survivor_mp may already observe target_kf at a
                    # different index.
                    stale_idx = survivor_mp.add_observation(target_kf.id, best_idx)
                    if (stale_idx is not None and 0 <= stale_idx < len(target_kf.map_point_ids)
                            and target_kf.map_point_ids[stale_idx] == survivor):
                        target_kf.map_point_ids[stale_idx] = None
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

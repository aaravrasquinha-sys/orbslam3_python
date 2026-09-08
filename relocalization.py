"""
relocalization.py — Phase 5: recover a lost frame's pose by matching
against the Atlas-wide keyframe database, instead of always starting a
fresh map.

Maps to: ORB_SLAM3/src/Tracking.cc's Relocalization()

WHY THIS EXISTS: flagged as a missing capability from the very first
forensic pass on this project ("No Relocalization()" was an explicit
comment already in tracking.py) and again independently by Phase 1's own
verification -- run_slam.py's give-up-and-start-a-new-map path had no
recovery attempt at all before it. Without this, EVERY sustained tracking
loss permanently fragments the Atlas, even when the camera is looking at
a place the system has already mapped perfectly well five seconds ago.

DESIGN, matching tracking.py's existing _solve_pnp() conventions exactly
(same PnP parameters, same pose inversion convention) rather than
introducing a second, untested variant of the same logic: RANSAC
threshold, iteration count, and confidence are identical to _solve_pnp's
values. The one deliberate difference is scope -- this operates against
a CANDIDATE keyframe's own map (which may not be the currently active
one), not `self.map`, since the whole point is finding a match that
ISN'T in the active map.

ON SUCCESS: switches the Atlas's active map to the one containing the
matched keyframe (see atlas.py's switch_active_map()) and RESUMES that
map, rather than attempting to merge the just-abandoned fragment into it.
True map merging (stitching two Atlas maps' keyframes/points together)
is real, separate work -- deferred to Phase 6, matching this project's
existing phase boundaries. Resuming the recognized map is still a large,
real improvement over the alternative (permanent fragmentation), and is
an honest, self-consistent state: the just-abandoned fragment's own
keyframes/points remain valid in their own Map object, simply no longer
active.
"""

import numpy as np
import cv2


def try_relocalize(frame, keyframe_db, atlas, camera, matcher,
                   min_inliers=20, top_n_candidates=5, exclude_ids=None):
    """
    Returns (success, matched_map) on success (frame.pose and
    frame.map_point_ids are set as a side effect, matching _solve_pnp's
    own convention), or (False, None).
    """
    if not keyframe_db.vocab.is_ready():
        return False, None
    if frame.descriptors is None or len(frame.descriptors) == 0:
        return False, None

    bow, _ = keyframe_db.vocab.transform(frame.descriptors)
    if not bow:
        return False, None

    candidates = keyframe_db.query(bow, exclude_ids=exclude_ids, top_n=top_n_candidates)

    for kf_id, score in candidates:
        candidate_kf = keyframe_db.keyframe_refs.get(kf_id)
        if candidate_kf is None or candidate_kf.pose is None:
            continue
        target_map = atlas.map_containing_keyframe(kf_id)
        if target_map is None:
            continue   # candidate keyframe was culled/removed since being indexed

        mp_ids, mp_descs = [], []
        for mp_id in candidate_kf.map_point_ids:
            if mp_id is None:
                continue
            mp = target_map.map_points.get(mp_id)
            if mp is None or mp.is_bad or mp.descriptor is None:
                continue
            mp_ids.append(mp_id)
            mp_descs.append(mp.descriptor)

        if len(mp_ids) < min_inliers:
            continue
        mp_descs = np.asarray(mp_descs, dtype=np.uint8)

        matches = matcher.match_ratio(frame.descriptors, mp_descs)
        matches = matcher.dedupe_by_train_idx(matches)   # PHASE 7 BUGFIX -- see matcher.py
        if len(matches) < min_inliers:
            continue

        pts_2d, pts_3d, kp_indices, matched_mp_ids = [], [], [], []
        for m in matches:
            mp = target_map.map_points.get(mp_ids[m.trainIdx])
            if mp is None or mp.is_bad:
                continue
            pts_2d.append(frame.points_undistorted[m.queryIdx])
            pts_3d.append(mp.position)
            kp_indices.append(m.queryIdx)
            matched_mp_ids.append(mp.id)

        if len(pts_3d) < 6:
            continue

        pts_2d = np.asarray(pts_2d, dtype=np.float64)
        pts_3d = np.asarray(pts_3d, dtype=np.float64)

        # Same PnP parameters as tracking.py's _solve_pnp -- no cold guess
        # available here (that's the whole premise of being lost), so no
        # useExtrinsicGuess branch.
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, camera.K, None,
            iterationsCount=200, reprojectionError=3.0,
            confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)

        if not ok or inliers is None or len(inliers) < min_inliers:
            continue

        R, _ = cv2.Rodrigues(rvec)
        T_wc = np.eye(4)
        T_wc[:3, :3] = R
        T_wc[:3, 3] = tvec.flatten()
        frame.set_pose(np.linalg.inv(T_wc))   # world-to-camera -> camera-to-world

        inlier_set = set(int(i) for i in inliers.flatten())
        for k, (kp_i, mp_id) in enumerate(zip(kp_indices, matched_mp_ids)):
            if k in inlier_set:
                frame.map_point_ids[kp_i] = mp_id
                mp = target_map.map_points.get(mp_id)
                if mp:
                    # PHASE 7 FIX: Phase 6 added add_observation() here,
                    # believing its absence was a bug. It wasn't --
                    # try_relocalize() is called on an ordinary (non-
                    # keyframe) frame by construction (relocalization
                    # only runs while LOST), so registering it into
                    # mp.observations would pollute the keyframe-only
                    # invariant the same way the sibling fixes in
                    # tracking.py did. increase_found() is the correct,
                    # complete bookkeeping here. See tracking.py's
                    # _track_local_map for the full account.
                    mp.increase_found()

        atlas.switch_active_map(target_map)
        return True, target_map

    return False, None


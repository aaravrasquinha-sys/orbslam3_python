"""
map.py — one map: a set of keyframes + a set of 3D points.

Maps to: ORB_SLAM3/src/Map.cc
  AddKeyFrame / AddMapPoint     -> add_keyframe / add_map_point
  EraseKeyFrame / EraseMapPoint -> erase_keyframe / erase_map_point
  GetAllKeyFrames / GetAllMapPoints

Map.cc is ~95% thread-safety locks and disk serialisation. The real logic is
just: hold two collections. We run single-threaded, so no locks needed.
"""

import numpy as np

from frame import Frame


class Map:
    _next_id = 0

    def __init__(self):
        self.id = Map._next_id
        Map._next_id += 1

        self.keyframes = []      # list of Frame objects flagged is_keyframe
        self.map_points = {}     # id -> MapPoint
        self.is_active = True

    # ── keyframes ────────────────────────────────────────────────────────

    def add_keyframe(self, frame):
        frame.is_keyframe = True
        if frame not in self.keyframes:
            # BUGFIX (original, Phase 0): previously nothing tracked "how
            # many keyframes have there been", so local_mapping.py used
            # Frame.id (which counts every processed frame, keyframe or
            # not) to compute a point's age. kf_seq increments once per
            # ACTUAL keyframe.
            #
            # BUGFIX (Phase 7): this used to be `len(self.keyframes)` --
            # a counter local to THIS Map object. Correct within one map,
            # but not comparable across maps, which is exactly what
            # forced merge_maps.py to renumber every kf_seq after a
            # merge (and that renumbering was what silently invalidated
            # every pre-merge MapPoint.first_keyframe_id -- see
            # frame.py's Frame._next_kf_seq docstring for the full
            # account). Using the global, Atlas-wide counter here means
            # kf_seq is assigned once, ever, and merge_maps.py no longer
            # needs to touch it at all.
            if frame.kf_seq is None:
                frame.kf_seq = Frame._next_kf_seq
                Frame._next_kf_seq += 1
            self.keyframes.append(frame)

    def erase_keyframe(self, frame):
        if frame in self.keyframes:
            self.keyframes.remove(frame)

    def n_keyframes(self):
        return len(self.keyframes)

    def get_keyframe_by_id(self, frame_id):
        for kf in self.keyframes:
            if kf.id == frame_id:
                return kf
        return None

    # ── map points ───────────────────────────────────────────────────────

    def add_map_point(self, mp):
        self.map_points[mp.id] = mp

    def erase_map_point(self, mp):
        # BUGFIX (found during Phase 6 verification): this used to only
        # pop the point from self.map_points, leaving every keyframe that
        # observed it holding a DANGLING id in its own map_point_ids
        # array forever. Measured directly on real pipeline output: one
        # keyframe referenced 419 map-point ids, of which only 3 still
        # existed in the map -- 416 stale references. This silently
        # starved loop_verification.py's 3D-3D check of real
        # correspondences (a revisit that should have had ~150+ usable
        # matches had only 3), and would equally have starved any other
        # code that walks a keyframe's map_point_ids expecting live
        # points (local-map tracking, fusion, etc.) -- a pre-existing
        # bug Phase 6's stricter cross-map verification happened to
        # surface, not something Phase 6 introduced.
        kf_by_id = {kf.id: kf for kf in self.keyframes}
        for frame_id, kp_idx in list(mp.observations.items()):
            kf = kf_by_id.get(frame_id)
            # PHASE 7 BUGFIX: added an ownership check (only clear the
            # slot if it STILL points at the point being deleted) as a
            # second, independent safety layer against the same failure
            # shape fusion.py's _fuse_point had (a point's own
            # .observations dict going stale relative to what a
            # keyframe's map_point_ids ACTUALLY holds after some other
            # code already redirected that slot elsewhere). Without
            # this check, deleting a point whose bookkeeping has gone
            # stale for any reason -- including ones not yet found --
            # can silently wipe out a DIFFERENT, still-valid point's
            # correct claim on that same slot.
            if (kf is not None and 0 <= kp_idx < len(kf.map_point_ids)
                    and kf.map_point_ids[kp_idx] == mp.id):
                kf.map_point_ids[kp_idx] = None
        self.map_points.pop(mp.id, None)

    def n_map_points(self):
        return len([mp for mp in self.map_points.values() if not mp.is_bad])

    def good_map_points(self):
        return [mp for mp in self.map_points.values() if not mp.is_bad]

    def all_descriptors(self):
        """
        Returns (ids, descriptors_array) for every good map point.
        Used by tracking.py to match a new frame against the whole map.
        """
        ids, descs = [], []
        for mp in self.map_points.values():
            if mp.is_bad or mp.descriptor is None:
                continue
            ids.append(mp.id)
            descs.append(mp.descriptor)
        if not descs:
            return [], None
        return ids, np.asarray(descs, dtype=np.uint8)

    def clean_bad_points(self):
        """
        Actually remove points flagged bad by MapPointCulling.

        BUGFIX (same root cause as erase_map_point above, found via the
        identical stale-reference symptom): this is local_mapping.py's
        actual, frequently-called cleanup path (erase_map_point above is
        rarely invoked directly) -- so THIS was the real source of the
        419-references/3-valid discrepancy in practice. Now clears each
        removed point's keyframe references the same way.
        """
        bad = [k for k, mp in self.map_points.items() if mp.is_bad]
        if bad:
            kf_by_id = {kf.id: kf for kf in self.keyframes}
            for k in bad:
                mp = self.map_points[k]
                for frame_id, kp_idx in list(mp.observations.items()):
                    kf = kf_by_id.get(frame_id)
                    # PHASE 7 BUGFIX: same ownership check as
                    # erase_map_point above -- see that method's comment.
                    if (kf is not None and 0 <= kp_idx < len(kf.map_point_ids)
                            and kf.map_point_ids[kp_idx] == mp.id):
                        kf.map_point_ids[kp_idx] = None
        for k in bad:
            del self.map_points[k]
        return len(bad)

    def __repr__(self):
        return (f"Map(id={self.id}, kfs={len(self.keyframes)}, "
                f"pts={self.n_map_points()}, active={self.is_active})")

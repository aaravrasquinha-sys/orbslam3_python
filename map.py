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
            # BUGFIX: previously nothing tracked "how many keyframes have
            # there been", so local_mapping.py used Frame.id (which counts
            # every processed frame, keyframe or not) to compute a point's
            # age. With keyframe_max_frames=20, a point could see its age
            # jump by ~20 in a single step -- the >=2 / >=3 probation
            # thresholds in cull_recent_map_points() fired almost the
            # instant a point was created, so the probation logic barely
            # ran. kf_seq increments once per ACTUAL keyframe.
            frame.kf_seq = len(self.keyframes)
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
        """Actually remove points flagged bad by MapPointCulling."""
        bad = [k for k, mp in self.map_points.items() if mp.is_bad]
        for k in bad:
            del self.map_points[k]
        return len(bad)

    def __repr__(self):
        return (f"Map(id={self.id}, kfs={len(self.keyframes)}, "
                f"pts={self.n_map_points()}, active={self.is_active})")
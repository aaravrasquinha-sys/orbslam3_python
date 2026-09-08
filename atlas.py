"""
atlas.py — multi-map container.

Maps to: ORB_SLAM3/src/Atlas.cc
  CreateNewMap / GetCurrentMap -> start_new_map / active_map
  (map MERGING is NOT implemented — see LoopClosing.cc MergeLocal)

The point of Atlas: when tracking dies, don't crash — stash the current map
and start a fresh one. Later, if you revisit a mapped area, merge them back.
We do the stashing; we don't do the merging.
"""

import numpy as np

from map import Map


class Atlas:
    def __init__(self):
        self.maps = [Map()]

    @property
    def active_map(self):
        return self.maps[-1]

    def start_new_map(self):
        """Tracking::CreateMapInAtlas() — give up on the current map, start fresh."""
        self.active_map.is_active = False
        new_map = Map()
        self.maps.append(new_map)
        return new_map

    def switch_active_map(self, target_map):
        """
        Phase 5: relocalization support. Unlike start_new_map(), this
        RESUMES an existing map instead of abandoning it -- the direct
        fix for the original "camera returns to a place the loop
        detector never even looks at" failure mode (the old loop closer
        only ever searched self.map.keyframes, i.e. the currently active
        map; see loop_closing.py). Moves target_map to the end of the
        list (active_map is always self.maps[-1]) rather than mutating
        map identity, so any code elsewhere holding a reference to a Map
        object stays valid.
        """
        if target_map not in self.maps:
            raise ValueError("switch_active_map: target_map is not in this Atlas")
        self.active_map.is_active = False
        self.maps.remove(target_map)
        self.maps.append(target_map)
        target_map.is_active = True
        return target_map

    def map_containing_keyframe(self, kf_id):
        """Phase 5: which map (if any) currently holds this keyframe id --
        needed after a keyframe-database query returns a candidate, since
        the candidate could belong to ANY map, not just the active one."""
        for m in self.maps:
            if any(kf.id == kf_id for kf in m.keyframes):
                return m
        return None

    def n_maps(self):
        return len(self.maps)

    def total_keyframes(self):
        return sum(m.n_keyframes() for m in self.maps)

    def total_map_points(self):
        return sum(m.n_map_points() for m in self.maps)

    def summary(self):
        lines = [f"Atlas: {len(self.maps)} map(s)"]
        for m in self.maps:
            flag = " (active)" if m.is_active else ""
            lines.append(f"  Map {m.id}: {m.n_keyframes()} kfs, "
                         f"{m.n_map_points()} pts{flag}")
        return "\n".join(lines)

    @staticmethod
    def report_loop_drift(kf_a, kf_b):
        if kf_a.pose is None or kf_b.pose is None:
            return None
        return float(np.linalg.norm(kf_a.camera_center() - kf_b.camera_center()))

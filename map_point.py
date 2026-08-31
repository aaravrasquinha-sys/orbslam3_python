"""
map_point.py — one 3D landmark.

Maps to: ORB_SLAM3/src/MapPoint.cc
  SetWorldPos/GetWorldPos      -> position
  AddObservation/Observations  -> observations dict
  ComputeDistinctiveDescriptors-> update_descriptor()
  SetBadFlag / isBad           -> is_bad flag

SIMPLIFICATION: no normal vector, no min/max distance invariance, no
PredictScale. Those feed isInFrustum-style visibility checks we don't do.
"""

import numpy as np


class MapPoint:
    _next_id = 0

    def __init__(self, position_3d, descriptor, ref_keyframe_id=None):
        self.id = MapPoint._next_id
        MapPoint._next_id += 1

        self.position = np.asarray(position_3d, dtype=np.float64).reshape(3)
        self.descriptor = descriptor          # (32,) uint8 — representative fingerprint
        self.ref_keyframe_id = ref_keyframe_id

        # observations[frame_id] = keypoint index in that frame
        self.observations = {}

        # MapPointCulling bookkeeping (LocalMapping.cc)
        self.n_visible = 1   # how many times we EXPECTED to see it
        self.n_found = 1     # how many times we ACTUALLY matched it
        self.is_bad = False
        self.first_keyframe_id = ref_keyframe_id

    def add_observation(self, frame_id, keypoint_idx):
        self.observations[frame_id] = keypoint_idx

    def erase_observation(self, frame_id):
        self.observations.pop(frame_id, None)
        if len(self.observations) <= 2:
            self.is_bad = True

    def n_observations(self):
        return len(self.observations)

    def increase_visible(self, n=1):
        self.n_visible += n

    def increase_found(self, n=1):
        self.n_found += n

    def found_ratio(self):
        """MapPoint::GetFoundRatio() — low ratio means unreliable point."""
        return self.n_found / max(1, self.n_visible)

    def set_bad(self):
        self.is_bad = True

    def update_descriptor(self, all_descriptors):
        """
        MapPoint::ComputeDistinctiveDescriptors().

        Given every descriptor that ever observed this point, pick the one with
        the SMALLEST median Hamming distance to all the others — i.e. the most
        'typical' view of this landmark, discarding outlier observations.
        """
        if all_descriptors is None or len(all_descriptors) == 0:
            return
        descs = np.asarray(all_descriptors, dtype=np.uint8)
        if len(descs) == 1:
            self.descriptor = descs[0]
            return

        n = len(descs)
        dists = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                d = int(np.count_nonzero(
                    np.unpackbits(descs[i] ^ descs[j])))
                dists[i, j] = dists[j, i] = d

        medians = np.median(dists, axis=1)
        self.descriptor = descs[int(np.argmin(medians))]

    def __repr__(self):
        return (f"MapPoint(id={self.id}, pos={np.round(self.position, 3)}, "
                f"obs={len(self.observations)}, bad={self.is_bad})")
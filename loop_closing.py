"""
loop_closing.py â€” detect revisited places.

Maps to: ORB_SLAM3/src/LoopClosing.cc
  NewDetectCommonRegions      -> detect_loop()
  DetectCommonRegionsFromBoW  -> the candidate search below
  Sim3Solver geometric check  -> _geometric_check() (essential matrix instead)
  CorrectLoop / MergeLocal    -> pose_graph.py (Phase 5)

SIMPLIFICATIONS:
  - No Sim3 (we compute a rotation+translation check, not rotation+translation+SCALE)
  - No OptimizeSim3 refinement

CONSISTENCY GATE (Phase 5 addition): a correction is expensive to undo once
map points have moved, so we don't act on the first appearance match. We
require the SAME approximate place (matched keyframe within a small
kf_seq window) to be found on `consistency_checks` consecutive detect_loop
calls before confirming. Since this project only calls detect_loop every
10 keyframes (see run_slam.py), this is a coarser cadence than real
ORB-SLAM3's per-frame check -- it trades a longer confirmation delay for
the same "don't correct on a fluke match" guarantee.
"""

import cv2
import numpy as np


class LoopClosing:
    def __init__(self, camera, matcher, vocabulary,
                 min_keyframe_gap=15,
                 similarity_thresh=0.75,
                 min_geometric_inliers=25,
                 consistency_checks=2,
                 consistency_kf_window=5):
        self.camera = camera
        self.matcher = matcher
        self.vocab = vocabulary
        self.min_keyframe_gap = min_keyframe_gap
        self.similarity_thresh = similarity_thresh
        self.min_geometric_inliers = min_geometric_inliers
        self.consistency_checks = consistency_checks
        self.consistency_kf_window = consistency_kf_window

        self._pending = None      # {'matched_kf_id': int, 'streak': int}
        self.detections = []      # log of every CONFIRMED loop

    def detect_loop(self, current_kf, world_map):
        """
        Returns (matched_keyframe, similarity, n_inliers) once the same
        place has been seen `consistency_checks` times in a row, else None
        (including the first, "pending" sighting of a real loop).
        """
        if not self.vocab.is_ready():
            return None
        if len(world_map.keyframes) < self.min_keyframe_gap + 2:
            return None

        cur_hist = self.vocab.histogram(current_kf.descriptors, current_kf.id)
        if not np.any(cur_hist):
            return None

        # Stage 1: appearance â€” find the most similar old keyframe
        best_kf, best_sim = None, 0.0
        for kf in world_map.keyframes:
            if kf.id == current_kf.id:
                continue
            if current_kf.id - kf.id < self.min_keyframe_gap:
                continue      # too recent â€” that's just normal tracking, not a loop
            sim = self.vocab.similarity(
                cur_hist, self.vocab.histogram(kf.descriptors, kf.id))
            if sim > best_sim:
                best_sim, best_kf = sim, kf

        if best_kf is None or best_sim < self.similarity_thresh:
            self._pending = None   # the trail went cold; don't let a stale streak survive
            return None

        # Stage 2: geometry â€” does it actually hold up?
        n_inliers = self._geometric_check(current_kf, best_kf)
        if n_inliers < self.min_geometric_inliers:
            self._pending = None
            return None

        # Stage 3: temporal consistency â€” same place, seen repeatedly
        if (self._pending is not None and
                abs(self._pending['matched_kf_id'] - best_kf.id) <= self.consistency_kf_window):
            self._pending['streak'] += 1
        else:
            self._pending = {'matched_kf_id': best_kf.id, 'streak': 1}

        if self._pending['streak'] < self.consistency_checks:
            return None   # seen once so far â€” wait for it to reappear

        self._pending = None
        self.detections.append({
            'current_kf': current_kf.id,
            'matched_kf': best_kf.id,
            'similarity': best_sim,
            'inliers': n_inliers,
        })
        return best_kf, best_sim, n_inliers

    def _geometric_check(self, kf_a, kf_b):
        """
        Confirm the appearance match is geometrically consistent.
        Real ORB-SLAM3 uses Sim3Solver (solves for scale too, since two map
        segments can have different arbitrary monocular scales). We just check
        that a consistent essential matrix exists.
        """
        matches = self.matcher.match(kf_a.descriptors, kf_b.descriptors)
        if len(matches) < self.min_geometric_inliers:
            return 0

        pts_a, pts_b = self.matcher.matched_points(
            kf_a.keypoints, kf_b.keypoints, matches)

        E, mask = cv2.findEssentialMat(pts_a, pts_b, self.camera.K,
                                       method=cv2.RANSAC, prob=0.999, threshold=1.5)
        if E is None or mask is None:
            return 0
        return int(mask.sum())

    @staticmethod
    def measure_drift(kf_a, kf_b):
        """
        Two keyframes that SHOULD be at the same physical place â€” how far apart
        do their estimated poses actually claim to be? That gap is the
        accumulated drift the real system would now correct.
        """
        if kf_a.pose is None or kf_b.pose is None:
            return None
        return float(np.linalg.norm(kf_a.camera_center() - kf_b.camera_center()))

"""
loop_closing.py — detect revisited places.

Maps to: ORB_SLAM3/src/LoopClosing.cc
  NewDetectCommonRegions      -> detect_loop()
  DetectCommonRegionsFromBoW  -> the candidate search below
  Sim3Solver geometric check  -> _geometric_check() (essential matrix instead)
  CorrectLoop / MergeLocal    -> NOT IMPLEMENTED

SIMPLIFICATIONS:
  - No 3-consecutive-keyframe confirmation loop (real system requires
    mnLoopNumCoincidences >= 3 before trusting a loop)
  - No Sim3 (we compute a rotation+translation check, not rotation+translation+SCALE)
  - No OptimizeSim3 refinement
  - DETECTION ONLY. We report drift; we never correct it.
"""

import cv2
import numpy as np


class LoopClosing:
    def __init__(self, camera, matcher, vocabulary,
                 min_keyframe_gap=15,
                 similarity_thresh=0.75,
                 min_geometric_inliers=25):
        self.camera = camera
        self.matcher = matcher
        self.vocab = vocabulary
        self.min_keyframe_gap = min_keyframe_gap
        self.similarity_thresh = similarity_thresh
        self.min_geometric_inliers = min_geometric_inliers

        self.detections = []      # log of every confirmed loop

    def detect_loop(self, current_kf, world_map):
        """
        Returns (matched_keyframe, similarity, n_inliers) or None.
        """
        if not self.vocab.is_ready():
            return None
        if len(world_map.keyframes) < self.min_keyframe_gap + 2:
            return None

        cur_hist = self.vocab.histogram(current_kf.descriptors, current_kf.id)
        if not np.any(cur_hist):
            return None

        # Stage 1: appearance — find the most similar old keyframe
        best_kf, best_sim = None, 0.0
        for kf in world_map.keyframes:
            if kf.id == current_kf.id:
                continue
            if current_kf.id - kf.id < self.min_keyframe_gap:
                continue      # too recent — that's just normal tracking, not a loop
            sim = self.vocab.similarity(
                cur_hist, self.vocab.histogram(kf.descriptors, kf.id))
            if sim > best_sim:
                best_sim, best_kf = sim, kf

        if best_kf is None or best_sim < self.similarity_thresh:
            return None

        # Stage 2: geometry — does it actually hold up?
        n_inliers = self._geometric_check(current_kf, best_kf)
        if n_inliers < self.min_geometric_inliers:
            return None

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
        Two keyframes that SHOULD be at the same physical place — how far apart
        do their estimated poses actually claim to be? That gap is the
        accumulated drift the real system would now correct.
        """
        if kf_a.pose is None or kf_b.pose is None:
            return None
        return float(np.linalg.norm(kf_a.camera_center() - kf_b.camera_center()))
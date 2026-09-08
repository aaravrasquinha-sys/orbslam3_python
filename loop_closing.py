"""
loop_closing.py -- detect revisited places.

Maps to: ORB_SLAM3/src/LoopClosing.cc
  NewDetectCommonRegions      -> detect_loop()
  DetectCommonRegionsFromBoW  -> the candidate search below
  Sim3Solver geometric check  -> loop_verification.verify_3d3d() (Phase 6)
  CorrectLoop / MergeLocal    -> pose_graph.py + merge_maps.py, wired in
                                 from run_slam.py (Phase 6)

PHASE 6 CHANGES:
  - Candidate search is now ATLAS-WIDE (via keyframe_db's inverted index),
    not restricted to the current active map. This is the direct fix for
    the architectural gap flagged from this project's earliest forensic
    pass: the original implementation only ever searched
    world_map.keyframes, so a camera returning to a place stored in an
    OLDER, already-abandoned Atlas map could never be recognized by loop
    closing at all, regardless of BoW/geometric quality -- the search
    space itself excluded it by construction.
  - Geometric verification upgraded from an Essential-matrix check
    (2D-2D, 5-DOF epipolar constraint, satisfiable by many wrong
    correspondences in degenerate/planar scenes) to 3D-3D RANSAC
    similarity verification (loop_verification.py) using each keyframe's
    own triangulated map points -- strictly more discriminative, and
    gives the actual (R, t, scale) needed for correction/merging for
    free, rather than needing a separate PnP step to get a metric
    transform. See loop_verification.py's module docstring for the
    verified-before-use methodology.

CONSISTENCY GATE (unchanged from Phase 5): a correction is expensive to
undo once map points have moved, so we don't act on the first appearance
match. We require the SAME approximate place (matched keyframe within a
small kf_seq window) to be found on consistency_checks consecutive
detect_loop calls before confirming.
"""

import numpy as np

import loop_verification


class LoopClosing:
    def __init__(self, camera, matcher, vocabulary, keyframe_db=None, atlas=None,
                 min_keyframe_gap=15,
                 similarity_thresh=0.20,
                 min_geometric_inliers=15,
                 consistency_checks=2,
                 consistency_kf_window=5):
        self.camera = camera
        self.matcher = matcher
        self.vocab = vocabulary
        self.keyframe_db = keyframe_db
        # Phase 6: needed to look up WHICH map a BoW-retrieved candidate
        # keyframe actually belongs to -- the keyframe database itself
        # deliberately has no notion of "map" at all (see
        # keyframe_database.py), so this is the only place that
        # information can come from.
        self.atlas = atlas
        self.min_keyframe_gap = min_keyframe_gap
        # BUGFIX (found during Phase 6 verification, not a Phase 6
        # regression -- this default was 0.75 since before the real
        # vocabulary existed and was never re-validated against it):
        # measured directly on this project's own synthetic test content,
        # a GENUINE overlapping revisit scores ~0.32 against this real
        # ORB-SLAM3 vocabulary (see orb_vocabulary.py), and an unrelated
        # scene scores ~0.13 (see PROGRESS.md's Phase 5 section). A
        # threshold of 0.75 rejects every genuine match by construction --
        # Test 4 of test_phase6_gate.py caught this directly (a hand-
        # constructed, unambiguous same-place revisit across two Atlas
        # maps returned NO candidate at all). 0.20 sits with margin above
        # the unrelated-scene baseline while comfortably below the
        # genuine-match baseline. Real facility photographs (closer to
        # what the vocabulary was actually trained on) should show MORE
        # separation than this project's synthetic rectangles-and-lines
        # texture, not less -- but this has NOT been validated against
        # real hardware, and may need retuning. See RUNBOOK_PHASE6.md.
        self.similarity_thresh = similarity_thresh
        self.min_geometric_inliers = min_geometric_inliers
        self.consistency_checks = consistency_checks
        self.consistency_kf_window = consistency_kf_window

        self._pending = None      # {'matched_kf_id': int, 'streak': int}
        self.detections = []      # log of every CONFIRMED loop

    def _bow(self, kf):
        """BoW vector for a keyframe, preferring the keyframe_db's cache
        over recomputing from scratch."""
        if self.keyframe_db is not None:
            cached = self.keyframe_db.keyframe_bows.get(kf.id)
            if cached is not None:
                return cached
        bow, _ = self.vocab.transform(kf.descriptors)
        return bow

    def detect_loop(self, current_kf, world_map):
        """
        Returns (matched_keyframe, matched_map, similarity, n_inliers,
        R, t, scale) once the same place has been seen
        consistency_checks times in a row, else None. matched_map may be
        a DIFFERENT Map object than world_map (a cross-map candidate) --
        callers must check `matched_map is world_map` to decide between
        intra-map pose-graph correction and cross-map merging.

        R, t, scale: the verified similarity transform mapping
        world_map's frame into matched_map's frame (see
        loop_verification.verify_3d3d's docstring for the exact
        convention) -- provided so a cross-map merge doesn't need to
        redo geometric verification a second time.
        """
        if not self.vocab.is_ready():
            return None
        if self.keyframe_db is not None and self.keyframe_db.n_keyframes() < self.min_keyframe_gap + 2:
            return None
        elif self.keyframe_db is None and len(world_map.keyframes) < self.min_keyframe_gap + 2:
            return None

        cur_bow = self._bow(current_kf)
        if not cur_bow:
            return None

        # Stage 1: appearance -- Atlas-wide candidate retrieval via the
        # inverted index, not a linear scan of one map's keyframes.
        candidates = []
        if self.keyframe_db is not None:
            exclude = self._recent_kf_ids(current_kf, world_map)
            results = self.keyframe_db.query(cur_bow, exclude_ids=exclude, top_n=5)
            candidates = [(self.keyframe_db.keyframe_refs[kf_id], sim)
                         for kf_id, sim in results
                         if kf_id in self.keyframe_db.keyframe_refs]
        else:
            # fallback: current-map-only linear scan (matches pre-Phase-6
            # behavior), used only if constructed without a keyframe_db.
            for kf in world_map.keyframes:
                if kf.id == current_kf.id or current_kf.kf_seq is None or kf.kf_seq is None:
                    continue
                if current_kf.kf_seq - kf.kf_seq < self.min_keyframe_gap:
                    continue
                sim = self.vocab.score(cur_bow, self._bow(kf))
                candidates.append((kf, sim))

        best_kf, best_sim = None, 0.0
        for kf, sim in candidates:
            if sim > best_sim:
                best_sim, best_kf = sim, kf

        if best_kf is None or best_sim < self.similarity_thresh:
            self._pending = None
            return None

        best_map = (self.atlas.map_containing_keyframe(best_kf.id)
                   if self.atlas is not None else world_map)
        if best_map is None:
            self._pending = None
            return None

        # Stage 2: geometry -- 3D-3D verification (Phase 6), not Essential
        # matrix. current_kf is in world_map; best_kf is in best_map
        # (possibly the same object, possibly not).
        n_inliers, R, t, s = loop_verification.verify_3d3d(
            current_kf, best_kf, self.matcher, world_map, best_map,
            min_inliers=self.min_geometric_inliers)
        if n_inliers < self.min_geometric_inliers:
            self._pending = None
            return None

        # Stage 3: temporal consistency -- same place, seen repeatedly
        if (self._pending is not None and
                self._pending['matched_kf_id'] == best_kf.id):
            self._pending['streak'] += 1
        else:
            self._pending = {'matched_kf_id': best_kf.id, 'streak': 1}

        if self._pending['streak'] < self.consistency_checks:
            return None

        self._pending = None
        self.detections.append({
            'current_kf': current_kf.id,
            'matched_kf': best_kf.id,
            'matched_map': best_map.id,
            'similarity': best_sim,
            'inliers': n_inliers,
            'scale': s,
        })
        return best_kf, best_map, best_sim, n_inliers, R, t, s

    def _recent_kf_ids(self, current_kf, world_map):
        """Exclude the current keyframe's own recent covisibility window
        from candidacy -- matching against the keyframe from 2 frames ago
        isn't a loop, it's just normal tracking continuity."""
        exclude = {current_kf.id}
        if current_kf.kf_seq is None:
            return exclude
        for kf in world_map.keyframes:
            if kf.kf_seq is not None and current_kf.kf_seq - kf.kf_seq < self.min_keyframe_gap:
                exclude.add(kf.id)
        return exclude

    @staticmethod
    def measure_drift(kf_a, kf_b):
        """
        Two keyframes that SHOULD be at the same physical place -- how far
        apart do their estimated poses actually claim to be? That gap is
        the accumulated drift the real system would now correct.
        """
        if kf_a.pose is None or kf_b.pose is None:
            return None
        return float(np.linalg.norm(kf_a.camera_center() - kf_b.camera_center()))

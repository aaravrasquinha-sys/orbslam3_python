"""
keyframe_database.py — Phase 5: inverted index over BoW words -> keyframes.

Maps to: ORB_SLAM3/src/KeyFrameDatabase.cc

WHY THIS EXISTS: the original loop_closing.py searched candidates with
`for kf in world_map.keyframes: sim = vocab.similarity(...)` -- brute
force over ONE map's keyframes, recomputing every comparison from
scratch. Two problems this fixes:

1. Speed: with a real ~1,000,000-word vocabulary (orb_vocabulary.py),
   two frames' BoW vectors are sparse (a few hundred nonzero words each
   out of ~1,000,000 possible) -- most keyframe pairs share ZERO words
   and are trivially not candidates. An inverted index (word -> which
   keyframes contain it) finds the keyframes worth comparing directly,
   instead of scoring every keyframe against every other one.

2. Scope: this index is ATLAS-WIDE by construction (add_keyframe takes
   a keyframe reference regardless of which Map object it belongs to).
   This is what makes relocalization.py's job possible at all -- the
   original architectural gap (flagged from the very first forensic
   pass on this project) was that a camera returning to a real, already-
   mapped place after a tracking-loss-induced map fragmentation could
   never be recognized, because the search never looked outside the
   currently active map. This index has no notion of "which map is
   active" at all -- it just tracks word -> keyframe id.
"""

import numpy as np


class KeyFrameDatabase:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inverted_index = {}        # word_id -> set(keyframe_id)
        self.keyframe_bows = {}         # keyframe_id -> bow_vector dict
        self.keyframe_refs = {}         # keyframe_id -> Frame object (for the caller to get pose/descriptors/map)

    def add_keyframe(self, keyframe):
        """
        Computes and stores the keyframe's BoW vector, updating the
        inverted index. Idempotent -- re-adding an already-present
        keyframe id first removes its old entries (handles the case
        where a keyframe's descriptors could theoretically change,
        though in practice keyframes are immutable once created).
        """
        if keyframe.id in self.keyframe_bows:
            self.remove_keyframe(keyframe.id)

        bow, _ = self.vocab.transform(keyframe.descriptors)
        self.keyframe_bows[keyframe.id] = bow
        self.keyframe_refs[keyframe.id] = keyframe
        for word_id in bow:
            self.inverted_index.setdefault(word_id, set()).add(keyframe.id)

    def remove_keyframe(self, kf_id):
        """Call when a keyframe is culled (local_mapping.cull_keyframes())
        -- otherwise the index accumulates dangling entries pointing at
        keyframes no longer in any live Map."""
        bow = self.keyframe_bows.pop(kf_id, None)
        self.keyframe_refs.pop(kf_id, None)
        if bow is None:
            return
        for word_id in bow:
            s = self.inverted_index.get(word_id)
            if s is not None:
                s.discard(kf_id)
                if not s:
                    del self.inverted_index[word_id]

    def query(self, bow_vector, exclude_ids=None, min_shared_words=1, top_n=10):
        """
        Returns up to top_n candidate keyframe ids, ranked by DBoW2 score
        (not just shared-word count -- count is used only as a cheap
        first-pass filter before the more expensive score() call, exactly
        the two-stage design real DBoW2 databases use).

        exclude_ids: keyframe ids to skip entirely (e.g. the querying
        keyframe's own recent covisibility neighbors, so relocalization/
        loop detection isn't just "matching against the frame right
        before this one" -- ORB-SLAM3 excludes covisible keyframes from
        loop/relocalization candidates for the same reason).
        """
        exclude_ids = exclude_ids or set()
        shared_count = {}
        for word_id in bow_vector:
            for kf_id in self.inverted_index.get(word_id, ()):
                if kf_id in exclude_ids:
                    continue
                shared_count[kf_id] = shared_count.get(kf_id, 0) + 1

        candidates = [kf_id for kf_id, c in shared_count.items() if c >= min_shared_words]
        if not candidates:
            return []

        scored = [(kf_id, self.vocab.score(bow_vector, self.keyframe_bows[kf_id]))
                 for kf_id in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def n_keyframes(self):
        return len(self.keyframe_bows)

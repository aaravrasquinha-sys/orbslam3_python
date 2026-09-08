"""
orb_vocabulary.py — Phase 5: real ORBvoc-format vocabulary, replacing
vocabulary.py's KMeans placeholder entirely.

WHY THIS REPLACES vocabulary.py: the KMeans version was explicitly
documented as "the biggest simplification in the project" from Phase 0
onward, and Phase 3's own verification found it could hang for 133
seconds in a single call once fed realistic data (see PROGRESS.md). More
fundamentally, an online-clustered ~256-word vocabulary has nowhere near
the discriminative power of a real vocabulary tree, which is why loop
detection's practical reliability was always going to be poor even after
the Phase 3 stopgap. This is the real fix, not another patch.

WHAT THIS IS: a pure Python/numpy loader and classifier for DBoW2's
actual text vocabulary format (k=10, L=6 hierarchical tree, 1,082,072
nodes, ~1,000,000 of them leaf "words" with real IDF weights) --
specifically ORB-SLAM3's own shipped ORBvoc.txt, the same file real
ORB-SLAM3 uses. See setup_vocabulary.py for how to fetch and cache it;
this module raises a clear error (not a silent fallback) if the cache
is missing, since a facility-mapping run without a real vocabulary
should be a deliberate choice, not something that happens by accident.

WHY PURE PYTHON/NUMPY, NOT A C++ WRAPPER (pyDBoW3 etc.): classifying one
descriptor is L*k = 6*10 = 60 Hamming-distance comparisons between 32-byte
codes -- trivial compute, no real-time requirement in this project. Build
requirements for pyDBoW3-style bindings would need compiling against this
project's exact OpenCV build (5.0.0.93 as of the original environment
audit), which is newer than what such bindings are typically maintained
against -- a real, avoidable source of build breakage for negligible
performance gain. Measured in this file's own verification: classifying
a full frame's ~1000 descriptors takes well under 100ms (see
PROGRESS.md), consistent with this project's per-frame extraction cost
already being 150-250ms -- BoW classification is not the bottleneck.

FORMAT (reverse-engineered and verified against the actual file, not
assumed from documentation -- see PROGRESS.md's Phase 5 section for the
verification methodology, including one mistake caught and fixed):
  Header line: "k L scoring_type weighting_type" (k=10, L=6 for the
  standard ORB-SLAM3 vocabulary: a 10-ary tree, 6 levels deep).
  Each following line: "parent_id is_leaf_flag desc[0..31] weight"
    parent_id: 0 = virtual root; otherwise 1-indexed node id (this row's
               own node id is its line number, 1-indexed).
    is_leaf_flag: 0 for inner nodes, 1 for leaf/word nodes -- NOT a
               unique word identifier (see below).
    desc:      32 bytes (0-255), the node's representative ORB descriptor.
    weight:    IDF weight, meaningful only for leaf/word nodes (0 for
               inner nodes, and also legitimately 0 for very common words).
  MISTAKE CAUGHT DURING VERIFICATION: the second column was initially
  assumed to be a sequential, unique word id (its nonzero-count, 971,814,
  is close to the expected leaf count for this tree, which is what made
  the wrong assumption plausible at first glance). This module's own
  correctness test caught it -- n_words computed as 1 (from taking
  max() of what is actually a 0/1 flag), and every distinct random
  descriptor set scoring identical to every other (since every
  descriptor was being assigned the same "word"). Fixed by using each
  leaf's own node id (row index + 1, already unique by construction) as
  the actual word identifier instead of this column's value.

TREE STRUCTURE, verified empirically before this module was written:
children of a given parent are ALWAYS a contiguous run of rows (checked
across all 1,082,072 rows, zero violations), which is what allows the
array-based child_start/child_count lookup below instead of a slower
reverse-index structure. Greedy nearest-Hamming-distance descent at each
level does NOT always reproduce a leaf's own true ancestor chain exactly
(measured: ~75% exact self-classification on a random sample) -- this
was checked and confirmed to be EXPECTED, not a bug: every mismatch
happens at high Hamming distance (>50 of 256 bits) with several genuine
near-ties, because an ancestor several levels up represents a broad
average over enormous numbers of training descriptors, not a tight match
to any specific descendant. This is inherent to coarse-to-fine
hierarchical search generally (including real DBoW2's own behavior, same
algorithm), not a defect in this implementation.
"""

import os
import numpy as np

_DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "orbvoc_cache.npz")


def _hamming_batch(query, candidates):
    """query: (32,) uint8. candidates: (N,32) uint8. Returns (N,) int distances."""
    return np.unpackbits(np.bitwise_xor(query, candidates), axis=-1).sum(axis=-1)


class ORBVocabulary:
    def __init__(self, cache_path=None, levelsup=4):
        """
        levelsup: DBoW2's standard "feature vector" depth parameter --
        SearchByBoW restricts candidate matches to pairs sharing the same
        node at depth (L - levelsup), a coarser grouping than the full
        leaf level. levelsup=4 is DBoW2's own conventional default.
        """
        self.cache_path = cache_path or _DEFAULT_CACHE_PATH
        self.levelsup = levelsup
        self.k = None
        self.L = None
        self.n_words = 0
        self._loaded = False

    def is_ready(self):
        return self._loaded

    def load(self):
        if not os.path.exists(self.cache_path):
            raise FileNotFoundError(
                f"Vocabulary cache not found at {self.cache_path}. "
                f"Run `python3 setup_vocabulary.py` once to fetch and cache "
                f"the real ORB-SLAM3 vocabulary (~42MB download, ~90s total, "
                f"one-time). This is a deliberate hard requirement, not a "
                f"silent fallback -- a facility-mapping run without a real "
                f"vocabulary should be a conscious choice.")
        d = np.load(self.cache_path)
        self.k = int(d["k"])
        self.L = int(d["L"])
        self.parent = d["parent"]
        self.word_id = d["word_id"]
        self.descs = d["descs"]
        self.weight = d["weight"]
        self.child_start = d["child_start"]
        self.child_count = d["child_count"]
        self.n_words = int(np.count_nonzero(self.word_id))
        # feature-vector level: how many hops from root
        self.fv_level = max(0, self.L - self.levelsup)
        self._loaded = True
        return self

    def _classify_one(self, descriptor):
        """
        Descend the tree greedily by Hamming distance. Returns
        (word_id, weight, fv_node_id) -- fv_node_id is the node visited
        at depth self.fv_level, used for the feature vector (SearchByBoW).

        BUGFIX (found via this module's OWN test giving nonsensical output
        -- n_words=1 and every random descriptor set scoring identical to
        every other): the file's second column was assumed to be a
        sequential, unique word identifier. It is not -- verified
        directly: `word_id.max() == 1` while `count_nonzero(word_id) ==
        971814`. It is a binary IS-LEAF flag (0/1), not an id. The actual
        unique identifier for a leaf/word is its own node id (= row index
        + 1, unique by construction, already used as `node_id` internally
        throughout this class) -- using that instead of the flag value.
        """
        node_id = 0
        fv_node_id = 0
        for level in range(self.L):
            cs, cc = self.child_start[node_id], self.child_count[node_id]
            if cc == 0:
                break
            cand_rows = np.arange(cs, cs + cc)
            dists = _hamming_batch(descriptor, self.descs[cand_rows])
            node_id = int(cand_rows[np.argmin(dists)]) + 1
            if level + 1 == self.fv_level:
                fv_node_id = node_id
        row = node_id - 1
        is_leaf = self.word_id[row] != 0
        word_id = node_id if is_leaf else 0   # BUGFIX: was self.word_id[row] (the 0/1 flag)
        return int(word_id), float(self.weight[row]), fv_node_id

    def transform(self, descriptors):
        """
        descriptors: (M,32) uint8.
        Returns (bow_vector, feature_vector):
          bow_vector: dict {word_id: tf_idf_weight}, L1-normalized (sums
                     to 1.0 across all entries) -- DBoW2's standard
                     representation, used for place-similarity scoring.
          feature_vector: dict {fv_node_id: [descriptor indices]} --
                          descriptors sharing a coarse-level node,
                          used to restrict SearchByBoW's candidate set
                          instead of brute-forcing every descriptor pair.
        """
        if descriptors is None or len(descriptors) == 0:
            return {}, {}

        bow_raw = {}
        feature_vector = {}
        for i, desc in enumerate(descriptors):
            word_id, w, fv_node = self._classify_one(desc)
            if word_id == 0 or w <= 0:
                continue   # inner-node fallback (shouldn't happen) or zero-IDF word
            bow_raw[word_id] = bow_raw.get(word_id, 0.0) + w
            feature_vector.setdefault(fv_node, []).append(i)

        total = sum(bow_raw.values())
        if total > 1e-12:
            bow_vector = {w: v / total for w, v in bow_raw.items()}
        else:
            bow_vector = {}
        return bow_vector, feature_vector

    @staticmethod
    def score(bow_a, bow_b):
        """
        DBoW2's L1-norm score: 1 - 0.5 * sum(|a_i - b_i|) over the union
        of nonzero words, bounded to [0, 1] (1.0 = identical distribution).
        Sparse-dict implementation -- only touches words present in
        EITHER vector, not all ~1,000,000 possible words.
        """
        if not bow_a or not bow_b:
            return 0.0
        keys = set(bow_a.keys()) | set(bow_b.keys())
        l1 = sum(abs(bow_a.get(k, 0.0) - bow_b.get(k, 0.0)) for k in keys)
        return max(0.0, 1.0 - 0.5 * l1)

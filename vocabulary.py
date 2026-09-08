"""
vocabulary.py — DEPRECATED as of Phase 5. Superseded by orb_vocabulary.py.

Kept in the repo (not deleted) for two reasons: it documents the real
bugs found and fixed in it during Phase 3 (the 133-second KMeans hang,
the n_words clamp) which are worth keeping as project history, and
nothing currently imports it -- deleting a file that might still be
referenced by an external script somewhere is a needless risk for zero
benefit. If you're looking for the live vocabulary implementation, it's
orb_vocabulary.py: a real, pre-trained ORBvoc-format vocabulary tree,
not this file's online KMeans placeholder. See that file's module
docstring for why the replacement was necessary, not just an upgrade.

bag-of-visual-words place description.

Stands in for: Thirdparty/DBoW2 + Vocabulary/ORBvoc.txt

SIMPLIFICATION — this is the biggest one in the project. Real DBoW2 uses a
hierarchical vocabulary TREE trained offline on millions of ORB descriptors,
shipped as a data file. We cluster our own descriptors with KMeans as we go.
Consequences: much smaller vocabulary, must be rebuilt periodically, and
matching quality is far below the real thing. Concept demo, not a replacement.
"""

import numpy as np

try:
    from sklearn.cluster import KMeans
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False


class Vocabulary:
    def __init__(self, n_words=64, random_state=0):
        # SAFETY CLAMP (found during Phase 3 verification): n_words=10000
        # was already sitting in config.py from the original repo, but was
        # silently INERT before Phase 0's config-wiring fix (Vocabulary
        # was hardcoded to n_words=64 regardless of what config said). This
        # is the first time it's ever actually been read -- and 10,000
        # clusters via online sklearn KMeans is fundamentally incompatible
        # with any usable sample cap (n_samples must exceed n_clusters, and
        # KMeans cost scales with samples x clusters x dims per iteration
        # regardless of tuning). Real DBoW2-style vocabularies use 10,000+
        # words too -- but from a PRE-TRAINED tree loaded from disk, never
        # clustered at runtime. That's Phase 5 (see module docstring). For
        # this placeholder, clamp to a value verified fast in practice and
        # warn once, rather than let a legitimate future config value
        # silently hang the pipeline against today's implementation.
        self._requested_n_words = n_words
        if n_words > 256:
            print(f"[vocabulary] WARNING: n_words={n_words} requested, but this "
                 f"is the KMeans PLACEHOLDER (see module docstring) -- clamping "
                 f"to 256 for this build. The real Phase 5 ORBvoc loader will "
                 f"support the full {n_words}-word vocabulary from a pre-trained "
                 f"file with no runtime clustering cost.")
            n_words = 256
        self.n_words = n_words
        self.random_state = random_state
        self.kmeans = None
        self._cache = {}          # frame_id -> histogram

    def is_ready(self):
        return self.kmeans is not None

    def build(self, all_descriptors, max_samples=2000):
        """
        Cluster pooled descriptors into n_words 'visual words'.
        all_descriptors: (M,32) uint8

        STOPGAP FIX (found during Phase 3 verification, not a Phase 3
        change itself): with the OLD max_samples=20000 and n_init=4,
        sklearn's KMeans on this much data took 133 SECONDS in a single
        call, hanging the whole pipeline -- discovered because Phase 3's
        extractor rewrite gives every keyframe a much richer, better-
        distributed descriptor pool than the old bare-ORB version did, so
        this pre-existing placeholder (see the module docstring's own
        "biggest simplification in the project" admission) received
        enough real data to go from "slow but tolerable" to catastrophic.
        This is NOT fixed here -- it's tightened enough to stop hanging
        the pipeline while remaining exactly the placeholder it always
        was. The real fix is Phase 5's planned ORBvoc-format loader
        (a pre-trained vocabulary tree, no KMeans at all at runtime).
        Do not raise max_samples back up without re-profiling.
        """
        if not _HAVE_SKLEARN:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        if all_descriptors is None or len(all_descriptors) < self.n_words:
            return False

        max_samples = max(max_samples, self.n_words * 4)
        data = np.asarray(all_descriptors, dtype=np.float32)
        if len(data) > max_samples:
            idx = np.random.RandomState(self.random_state).choice(
                len(data), max_samples, replace=False)
            data = data[idx]

        self.kmeans = KMeans(n_clusters=self.n_words, n_init=1,
                             random_state=self.random_state)
        self.kmeans.fit(data)
        self._cache.clear()
        return True

    def histogram(self, descriptors, frame_id=None):
        """
        One frame's descriptors -> normalised visual-word histogram.
        This is the 'fingerprint of the whole scene' used for place recognition.
        """
        if frame_id is not None and frame_id in self._cache:
            return self._cache[frame_id]

        hist = np.zeros(self.n_words, dtype=np.float64)
        if self.kmeans is None or descriptors is None or len(descriptors) == 0:
            return hist

        words = self.kmeans.predict(np.asarray(descriptors, dtype=np.float32))
        counts = np.bincount(words, minlength=self.n_words).astype(np.float64)
        total = counts.sum()
        hist = counts / total if total > 0 else counts

        if frame_id is not None:
            self._cache[frame_id] = hist
        return hist

    @staticmethod
    def similarity(hist_a, hist_b):
        """Cosine similarity between two histograms. 1.0 = identical scene."""
        na, nb = np.linalg.norm(hist_a), np.linalg.norm(hist_b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(hist_a, hist_b) / (na * nb))

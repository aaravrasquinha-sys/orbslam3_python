"""
vocabulary.py — bag-of-visual-words place description.

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
        self.n_words = n_words
        self.random_state = random_state
        self.kmeans = None
        self._cache = {}          # frame_id -> histogram

    def is_ready(self):
        return self.kmeans is not None

    def build(self, all_descriptors, max_samples=20000):
        """
        Cluster pooled descriptors into n_words 'visual words'.
        all_descriptors: (M,32) uint8
        """
        if not _HAVE_SKLEARN:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        if all_descriptors is None or len(all_descriptors) < self.n_words:
            return False

        data = np.asarray(all_descriptors, dtype=np.float32)
        if len(data) > max_samples:
            idx = np.random.RandomState(self.random_state).choice(
                len(data), max_samples, replace=False)
            data = data[idx]

        self.kmeans = KMeans(n_clusters=self.n_words, n_init=4,
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
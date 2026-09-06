"""
matcher.py — ORB descriptor matching.

Maps to: ORB_SLAM3/src/ORBmatcher.cc
  DescriptorDistance   -> Hamming distance (cv2.NORM_HAMMING does this)
  SearchByBoW /
  SearchByProjection / -> match() (we use ONE general matcher instead of
  SearchForInitialization  ~8 specialised search modes)

SIMPLIFICATIONS:
  - Brute force over all descriptors. Real ORB-SLAM3 narrows candidates first
    using a pixel grid (GetFeaturesInArea) or BoW vocabulary buckets --
    NOTE: the pixel-grid version IS used already, in tracking.py's
    _track_local_map (via frame.get_features_in_area), just not here in
    the generic match()/match_ratio() calls used by initializer.py,
    local_mapping.py, and loop_closing.py.
  - Rotation-consistency histogram (rotation_consistency_filter(), added
    Phase 3) exists now but is only wired into tracking.py's _solve_pnp
    so far -- local_mapping.py's fusion search and loop_closing.py's
    candidate verification don't use it yet (Phase 5/6 territory, where
    loop closing itself becomes reachable).
"""

import cv2
import numpy as np

TH_LOW = 50    # ORBmatcher::TH_LOW  — strict Hamming threshold
TH_HIGH = 100  # ORBmatcher::TH_HIGH — loose Hamming threshold


class Matcher:
    def __init__(self, nn_ratio=0.75, max_distance=TH_HIGH, cross_check=True):
        """
        nn_ratio      Lowe's ratio test threshold (used in knn mode)
        max_distance  reject matches worse than this Hamming distance
        cross_check   require mutual best match (A's best is B AND B's best is A)
        """
        self.nn_ratio = nn_ratio
        self.max_distance = max_distance
        self.cross_check = cross_check
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
        self.bf_knn = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, desc_a, desc_b):
        """
        Two (N,32) uint8 descriptor arrays -> list of cv2.DMatch, sorted best first.
        Each DMatch: .queryIdx (index into a), .trainIdx (index into b), .distance
        """
        if desc_a is None or desc_b is None:
            return []
        if len(desc_a) == 0 or len(desc_b) == 0:
            return []

        matches = self.bf.match(desc_a, desc_b)
        matches = [m for m in matches if m.distance <= self.max_distance]
        return sorted(matches, key=lambda m: m.distance)

    def match_ratio(self, desc_a, desc_b):
        """
        Lowe's ratio test variant — closer to ORBmatcher's mfNNratio logic.
        Keeps a match only if the best is clearly better than the second best.
        """
        if desc_a is None or desc_b is None:
            return []
        if len(desc_a) < 2 or len(desc_b) < 2:
            return []

        knn = self.bf_knn.knnMatch(desc_a, desc_b, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.nn_ratio * n.distance and m.distance <= self.max_distance:
                good.append(m)
        return sorted(good, key=lambda m: m.distance)

    @staticmethod
    def rotation_consistency_filter(matches, kp_a, kp_b, n_bins=30):
        """
        ORBmatcher::ComputeThreeMaxima -- BUGFIX/ADDITION: this did not
        exist at all before (module docstring's own SIMPLIFICATIONS note
        listed it as missing). Real ORB-SLAM3 uses this on every matching
        call, not just loop closing: matched keypoint pairs from two
        views of a genuinely rigid scene should all show roughly the SAME
        relative rotation (kp_b.angle - kp_a.angle), since the whole rBRIEF
        pipeline is built around cancelling out per-keypoint rotation. A
        real correspondence agrees with the dominant rotation; a false
        correspondence (found by a coincidentally-similar-looking but
        wrong descriptor match) usually doesn't. This buckets the angle
        differences into `n_bins`, keeps only matches falling in the 3
        largest bins, and discards the rest as likely-false correspondences
        -- independent of and complementary to the ratio test, which
        filters on descriptor distance alone and has no way to use this
        geometric consistency signal at all.

        matches: list of cv2.DMatch. kp_a/kp_b: the two full keypoint
        lists matches were computed from (angle in degrees, cv2 convention).
        Returns the filtered match list, sorted best-distance-first (same
        convention as match()/match_ratio()).
        """
        if len(matches) < n_bins:
            return matches  # too few matches for histogram voting to be meaningful

        bins = [[] for _ in range(n_bins)]
        for m in matches:
            da = kp_b[m.trainIdx].angle - kp_a[m.queryIdx].angle
            if da < 0:
                da += 360.0
            b = min(n_bins - 1, int(da * n_bins / 360.0))
            bins[b].append(m)

        # top 3 bins by count, matching ORB-SLAM3's HISTO_LENGTH=30 /
        # keep-3-largest convention exactly (ComputeThreeMaxima)
        order = sorted(range(n_bins), key=lambda b: len(bins[b]), reverse=True)
        keep_bins = set(order[:3])
        kept = [m for b in keep_bins for m in bins[b]]
        return sorted(kept, key=lambda m: m.distance)

    @staticmethod
    def matched_points(kp_a, kp_b, matches):
        """DMatch list -> two aligned (N,2) float arrays of pixel coordinates."""
        if not matches:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])
        return pts_a, pts_b


if __name__ == "__main__":
    import sys
    from extractor import Extractor

    if len(sys.argv) < 3:
        print("Usage: python3 matcher.py <img1.png> <img2.png>")
        sys.exit(1)

    ex = Extractor()
    k1, d1 = ex.extract(cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE))
    k2, d2 = ex.extract(cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE))
    m = Matcher()
    matches = m.match(d1, d2)
    print(f"{len(k1)} kps vs {len(k2)} kps -> {len(matches)} matches")
    if matches:
        pa, pb = m.matched_points(k1, k2, matches)
        disp = np.linalg.norm(pa - pb, axis=1)
        print(f"mean pixel displacement: {disp.mean():.2f} px "
              f"(low value = low parallax = bad for triangulation)")

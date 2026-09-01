"""
extractor.py — ORB feature extraction.

Maps to: ORB_SLAM3/src/ORBextractor.cc
  Constructor          -> __init__ (stores config once, reused every frame)
  operator()           -> extract() (pyramid, FAST corners, quadtree spread,
                                     orientation, 256-bit descriptor)

SIMPLIFICATION: OpenCV's ORB takes a single FAST threshold. Real ORB-SLAM3
tries iniThFAST first and falls back to minThFAST per grid cell when a cell
finds nothing. We pass iniThFAST only.
"""

import cv2


class Extractor:
    def __init__(self, nfeatures=1200, scale_factor=1.2, nlevels=8,
                 ini_th_fast=20, min_th_fast=7):
        self.nfeatures = nfeatures
        self.scale_factor = scale_factor
        self.nlevels = nlevels
        self.ini_th_fast = ini_th_fast
        self.min_th_fast = min_th_fast

        # cv2.ORB_create internally does everything ORBextractor.cc does:
        # image pyramid, FAST detection per level, feature distribution,
        # intensity-centroid orientation, rBRIEF descriptor.
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scale_factor,
            nlevels=nlevels,
            fastThreshold=ini_th_fast,
        )

        # Per-level scale factors — same as mvScaleFactor in the C++ constructor.
        # Used for scale-aware checks (e.g. reprojection thresholds).
        self.scale_factors = [1.0]
        for _ in range(1, nlevels):
            self.scale_factors.append(self.scale_factors[-1] * scale_factor)
        self.level_sigma2 = [s * s for s in self.scale_factors]

    def extract(self, image):
        """
        image -> (keypoints, descriptors)

        keypoints:   list of cv2.KeyPoint (.pt = (x,y), .octave = pyramid level,
                     .angle = orientation, .response = corner strength)
        descriptors: (N,32) uint8 — one 256-bit fingerprint per keypoint.
                     descriptors[i] belongs to keypoints[i].
        """
        if image is None:
            return [], None
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        if keypoints is None:
            return [], None
        return list(keypoints), descriptors


if __name__ == "__main__":
    import sys
    import numpy as np

    if len(sys.argv) < 2:
        print("Usage: python3 extractor.py <image.png>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read {sys.argv[1]}")
        sys.exit(1)

    ex = Extractor()
    kps, desc = ex.extract(img)
    print(f"image {img.shape} -> {len(kps)} keypoints")
    print(f"descriptors: {None if desc is None else desc.shape} {None if desc is None else desc.dtype}")
    if kps:
        k = kps[0]
        print(f"kp[0]: pt={k.pt} octave={k.octave} angle={k.angle:.1f} response={k.response:.2f}")
        print(f"desc[0] (32 bytes): {desc[0]}")

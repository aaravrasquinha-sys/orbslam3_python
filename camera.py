"""
camera.py — maps to CameraModels/ (Pinhole.cc) + your settings.yaml / calibration JSON
"""

import json
import numpy as np
import cv2


class Camera:
    def __init__(self, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.K = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]], dtype=np.float64)
        self.dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    @classmethod
    def from_json(cls, path):
        """Load directly from your existing calibration/iphone14pro_4k_1x.json"""
        with open(path) as f:
            c = json.load(f)
        return cls(c['fx'], c['fy'], c['cx'], c['cy'],
                    c.get('k1', 0), c.get('k2', 0),
                    c.get('p1', 0), c.get('p2', 0), c.get('k3', 0))

    def undistort_points(self, pts):
        """pts: (N,2) pixel coords -> undistorted normalized coords (for essential matrix)"""
        pts = pts.reshape(-1, 1, 2).astype(np.float64)
        return cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)
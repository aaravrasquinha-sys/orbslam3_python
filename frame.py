"""
frame.py — everything known about ONE photo.

Maps to: ORB_SLAM3/src/Frame.cc (the monocular + RGB-D constructors)
  ExtractORB           -> runs the extractor in __init__
  UndistortKeyPoints   -> undistorted keypoint coords (was MISSING before, now fixed)
  mvpMapPoints         -> map_point_ids
  SetPose / GetPose    -> pose

RGB-D: if a depth image is supplied, each keypoint also gets a depth value,
which lets us unproject straight to 3D with NO triangulation needed.

SIMPLIFICATIONS: no grid index for fast area lookup (AssignFeaturesToGrid),
no isInFrustum visibility test, no BoW vector stored per frame.
"""

import numpy as np


class Frame:
    _next_id = 0

    def __init__(self, image, timestamp, camera, extractor, depth_image=None):
        self.id = Frame._next_id
        Frame._next_id += 1

        self.timestamp = timestamp
        self.camera = camera
        self.image_shape = None if image is None else image.shape[:2]

        # ORB extraction — Frame.cc calls this inside its constructor too
        self.keypoints, self.descriptors = extractor.extract(image)
        self.n = len(self.keypoints)

        # Undistorted pixel coordinates (Frame::UndistortKeyPoints)
        if self.n > 0 and camera.has_distortion():
            raw = np.float32([kp.pt for kp in self.keypoints])
            self.points_undistorted = camera.undistort_points(raw)
        elif self.n > 0:
            self.points_undistorted = np.float32([kp.pt for kp in self.keypoints])
        else:
            self.points_undistorted = np.zeros((0, 2), np.float32)

        # RGB-D: depth per keypoint, in metres. -1 where invalid.
        self.depths = np.full(self.n, -1.0, dtype=np.float64)
        if depth_image is not None and self.n > 0:
            self._fill_depths(depth_image, camera)

        # map_point_ids[i] = id of the MapPoint that keypoints[i] matches, or None
        self.map_point_ids = [None] * self.n
        self.outliers = [False] * self.n

        self.pose = None          # 4x4 camera-to-world. None until estimated.
        self.is_keyframe = False

    def _fill_depths(self, depth_image, camera):
        h, w = depth_image.shape[:2]
        scale = camera.depth_scale if camera.depth_scale else 0.001
        for i, kp in enumerate(self.keypoints):
            u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
            if 0 <= u < w and 0 <= v < h:
                raw = float(depth_image[v, u])
                if raw > 0:
                    self.depths[i] = raw * scale

    # ── pose helpers ─────────────────────────────────────────────────────

    def set_pose(self, pose_4x4):
        self.pose = np.asarray(pose_4x4, dtype=np.float64).reshape(4, 4)

    def camera_center(self):
        """Position of the camera in world coordinates."""
        return None if self.pose is None else self.pose[:3, 3].copy()

    def rotation(self):
        return None if self.pose is None else self.pose[:3, :3].copy()

    # ── RGB-D unprojection ───────────────────────────────────────────────

    def unproject_keypoint(self, idx):
        """
        Keypoint index -> 3D point in WORLD coordinates, using its depth.
        Returns None if depth invalid or pose unknown.

        This is Frame::UnprojectStereo() — the RGB-D shortcut that skips
        triangulation entirely. This is why depth cameras don't suffer the
        parallax-starvation cold-start problem.
        """
        if self.pose is None:
            return None
        z = self.depths[idx]
        if z <= 0:
            return None
        p_cam = self.camera.unproject(self.points_undistorted[idx], depth=z)
        return self.pose[:3, :3] @ p_cam + self.pose[:3, 3]

    def n_tracked_points(self):
        return sum(1 for x in self.map_point_ids if x is not None)

    def n_valid_depths(self):
        return int(np.count_nonzero(self.depths > 0))

    def __repr__(self):
        return (f"Frame(id={self.id}, t={self.timestamp:.3f}, kps={self.n}, "
                f"tracked={self.n_tracked_points()}, kf={self.is_keyframe}, "
                f"posed={self.pose is not None})")
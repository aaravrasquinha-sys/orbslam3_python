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

        # Frame::AssignFeaturesToGrid() — bucket keypoints into a coarse
        # grid so "find keypoints near pixel (x,y)" is O(few) instead of
        # O(n). Needed for guided matching (covisibility.py's local-map
        # tracking, fusion.py's SearchInNeighbors) instead of brute-forcing
        # every descriptor in the map against every frame descriptor.
        self.grid_cols = 64
        self.grid_rows = 48
        self._grid = None   # built lazily by get_features_in_area()

        self.pose = None          # 4x4 camera-to-world. None until estimated.
        self.is_keyframe = False
        # Sequential keyframe index, assigned by Map.add_keyframe(). None
        # until this frame actually becomes a keyframe. Distinct from `id`
        # (which counts EVERY frame processed) on purpose -- see map.py.
        self.kf_seq = None

        # Visual-inertial state (Phase 3/4). velocity is in the same frame
        # as pose (camera frame at capture time, expressed in world axes --
        # i.e. d(camera_center)/dt). bias_* carried forward from the last
        # inertial BA / initialization (see imu_init.py, bundle_adjust.py).
        # imu_preint links this keyframe to the PREVIOUS keyframe via a
        # imu.Preintegration spanning that interval; None until it becomes
        # a keyframe (see run_slam.py).
        self.velocity = None
        self.bias_gyro = np.zeros(3)
        self.bias_accel = np.zeros(3)
        self.imu_preint = None

    def _build_grid(self):
        h, w = self.image_shape if self.image_shape else (480, 640)
        self._cell_w = max(w / self.grid_cols, 1e-6)
        self._cell_h = max(h / self.grid_rows, 1e-6)
        grid = [[[] for _ in range(self.grid_rows)] for _ in range(self.grid_cols)]
        for i in range(self.n):
            x, y = self.points_undistorted[i]
            gx = int(x / self._cell_w)
            gy = int(y / self._cell_h)
            if 0 <= gx < self.grid_cols and 0 <= gy < self.grid_rows:
                grid[gx][gy].append(i)
        self._grid = grid

    def get_features_in_area(self, x, y, radius, min_level=-1, max_level=-1):
        """
        Frame::GetFeaturesInArea(): keypoint indices within `radius` pixels
        of (x, y), optionally restricted to a pyramid octave range. Backed
        by the grid so this stays cheap regardless of how many keypoints
        the frame has.
        """
        if self._grid is None:
            self._build_grid()
        if self.n == 0:
            return []

        gx_min = max(0, int((x - radius) / self._cell_w))
        gx_max = min(self.grid_cols - 1, int((x + radius) / self._cell_w))
        gy_min = max(0, int((y - radius) / self._cell_h))
        gy_max = min(self.grid_rows - 1, int((y + radius) / self._cell_h))
        if gx_min > gx_max or gy_min > gy_max:
            return []

        result = []
        r2 = radius * radius
        for gx in range(gx_min, gx_max + 1):
            for gy in range(gy_min, gy_max + 1):
                for i in self._grid[gx][gy]:
                    if min_level >= 0 and self.keypoints[i].octave < min_level:
                        continue
                    if max_level >= 0 and self.keypoints[i].octave > max_level:
                        continue
                    px, py = self.points_undistorted[i]
                    if (px - x) ** 2 + (py - y) ** 2 <= r2:
                        result.append(i)
        return result

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
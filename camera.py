"""
camera.py — pinhole camera model + calibration sources.

Maps to: ORB_SLAM3's Pinhole camera model (CameraModels/Pinhole.cc) plus the
calibration-loading bits of System.cc's constructor.

FIXED BUGS (were silently corrupting the map):
  1. from_realsense() used to pull intrinsics from the DEPTH stream profile
     while run_realsense() aligned depth->color and fed the COLOR image to
     the extractor. On a D435I at 640x480 those two streams have very
     different focal lengths (depth fx ~= 385, color fx ~= 615) -- every
     unprojected point was wrong by roughly the ratio of those two focal
     lengths. from_realsense() is kept only for back-compat; prefer
     from_realsense_ir() (see below).
  2. baseline was being read from rs.option.depth_units, which is the
     depth SCALE (metres per raw unit), not the physical stereo baseline.
     Baseline is now read from the real IR1<->IR2 extrinsics.
  3. project() and undistort_points() were called elsewhere in the codebase
     (local_mapping.py, frame.py) but never defined on Camera -- an
     AttributeError waiting to fire the moment distortion or monocular
     triangulation got exercised. Both are implemented below.

RECOMMENDED PIPELINE CHANGE: track on the left infrared image, not RGB.
  - The D435I's IR imagers are global shutter; the RGB imager is rolling
    shutter. Rolling shutter + IMU preintegration injects motion-dependent
    timestamp bias into visual-inertial residuals.
  - Depth is natively registered to the left IR frame, so tracking on IR
    drops rs.align entirely -- no resampling step, no intrinsics ambiguity.
  from_realsense_ir() sets this up: intrinsics come from the SAME profile
  (left IR / "infra1") that record.py saves images from, so there is no
  possible cross-stream mismatch by construction.
"""

import json
import numpy as np
import cv2


class Camera:
    def __init__(self, fx, fy, cx, cy, width, height, baseline=0.05,
                 dist_coeffs=None, depth_scale=0.001):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.baseline = baseline          # metres, real IR1<->IR2 distance
        self.depth_scale = depth_scale    # metres per raw depth unit
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]], dtype=np.float64)
        self.dist_coeffs = (np.array(dist_coeffs, dtype=np.float64)
                            if dist_coeffs is not None
                            else np.zeros(5, dtype=np.float64))

    def has_distortion(self):
        return self.dist_coeffs is not None and np.any(self.dist_coeffs != 0)

    # ── geometry ─────────────────────────────────────────────────────────

    def unproject(self, kp, depth):
        """Pixel + depth (metres) -> 3D point in the CAMERA frame."""
        if depth <= 0:
            return None
        x = (kp[0] - self.cx) * depth / self.fx
        y = (kp[1] - self.cy) * depth / self.fy
        z = depth
        return np.array([x, y, z], dtype=np.float64)

    def project(self, p_cam):
        """
        3D point in the CAMERA frame -> pixel coordinates, or None if the
        point is behind (or at) the camera. Mirrors Pinhole::project().
        This was previously called by local_mapping._check_point() but
        never existed -- pure-monocular triangulation would have crashed.
        """
        p_cam = np.asarray(p_cam, dtype=np.float64)
        if p_cam[2] <= 1e-6:
            return None
        u = self.fx * p_cam[0] / p_cam[2] + self.cx
        v = self.fy * p_cam[1] / p_cam[2] + self.cy
        return np.array([u, v], dtype=np.float64)

    def undistort_points(self, raw_pts):
        """
        (N,2) raw pixel coords -> (N,2) undistorted pixel coords, still in
        pixel units (i.e. re-projected through K after undistorting), which
        is what Frame.UndistortKeyPoints does in the C++ codebase.
        Previously called by frame.py but never defined on Camera.
        """
        raw_pts = np.asarray(raw_pts, dtype=np.float64).reshape(-1, 1, 2)
        if not self.has_distortion():
            return raw_pts.reshape(-1, 2).astype(np.float32)
        undist = cv2.undistortPoints(raw_pts, self.K, self.dist_coeffs, P=self.K)
        return undist.reshape(-1, 2).astype(np.float32)

    # ── construction from RealSense ─────────────────────────────────────

    @classmethod
    def from_realsense_ir(cls, width, height, fps):
        """
        RECOMMENDED constructor for this project. Returns a Camera whose
        intrinsics match the LEFT INFRARED stream -- the same stream
        record.py saves images from and the stream depth is natively
        registered to (no rs.align needed anywhere).

        Also pulls the real stereo baseline from IR1<->IR2 extrinsics
        instead of misreading it off depth_units.
        """
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        profile = pipeline.start(config)
        ir1_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        ir2_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        intrinsics = ir1_profile.get_intrinsics()

        extr = ir1_profile.get_extrinsics_to(ir2_profile)
        baseline = float(np.linalg.norm(np.array(extr.translation)))

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale() if depth_sensor else 0.001

        pipeline.stop()

        return cls(
            fx=intrinsics.fx, fy=intrinsics.fy,
            cx=intrinsics.ppx, cy=intrinsics.ppy,
            width=intrinsics.width, height=intrinsics.height,
            baseline=baseline,
            dist_coeffs=intrinsics.coeffs,   # IR stream: normally all-zero
            depth_scale=depth_scale
        )

    @classmethod
    def from_realsense(cls, width, height, fps):
        """
        LEGACY constructor kept for back-compat with the old RGB-aligned
        pipeline. Intrinsics now correctly come from the COLOR stream
        (matching what run_realsense used to feed the extractor), and
        baseline comes from real extrinsics rather than depth_units.
        Prefer from_realsense_ir() for new work -- see module docstring.
        """
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)

        profile = pipeline.start(config)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()

        try:
            ir1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
            ir2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
            extr = ir1.get_extrinsics_to(ir2)
            baseline = float(np.linalg.norm(np.array(extr.translation)))
        except Exception:
            baseline = 0.05   # D435 nominal baseline fallback

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale() if depth_sensor else 0.001

        pipeline.stop()

        return cls(
            fx=intrinsics.fx, fy=intrinsics.fy,
            cx=intrinsics.ppx, cy=intrinsics.ppy,
            width=intrinsics.width, height=intrinsics.height,
            baseline=baseline,
            dist_coeffs=intrinsics.coeffs,
            depth_scale=depth_scale
        )

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            fx=data['fx'], fy=data['fy'],
            cx=data['cx'], cy=data['cy'],
            width=data['width'], height=data['height'],
            baseline=data.get('baseline', 0.05),
            dist_coeffs=data.get('dist_coeffs', None),
            depth_scale=data.get('depth_scale', 0.001)
        )

    def to_json(self, path):
        data = {
            "fx": self.fx, "fy": self.fy,
            "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height,
            "baseline": self.baseline,
            "dist_coeffs": self.dist_coeffs.tolist() if self.dist_coeffs is not None else [],
            "depth_scale": self.depth_scale
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def __repr__(self):
        return (f"Camera(fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}, "
                f"res={self.width}x{self.height}, baseline={self.baseline:.4f}, "
                f"depth_scale={self.depth_scale})")

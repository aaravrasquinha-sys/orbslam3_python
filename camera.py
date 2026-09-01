import json
import numpy as np

class Camera:
    def __init__(self, fx, fy, cx, cy, width, height, baseline=0.0, dist_coeffs=None, depth_scale=0.001):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.baseline = baseline
        self.depth_scale = depth_scale
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]], dtype=np.float32)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32) if dist_coeffs is not None else np.zeros(5, dtype=np.float32)

    def has_distortion(self):
        return self.dist_coeffs is not None and np.any(self.dist_coeffs != 0)

    def unproject(self, kp, depth):
        if depth <= 0:
            return None
        x = (kp[0] - self.cx) * depth / self.fx
        y = (kp[1] - self.cy) * depth / self.fy
        z = depth
        return np.array([x, y, z], dtype=np.float32)

    @classmethod
    def from_realsense(cls, width, height, fps):
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        profile = pipeline.start(config)
        depth_profile = profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        
        depth_sensor = profile.get_device().first_depth_sensor()
        baseline = depth_sensor.get_option(rs.option.depth_units) if depth_sensor.supports(rs.option.depth_units) else 0.05
        depth_scale = depth_sensor.get_depth_scale() if depth_sensor else 0.001
        
        pipeline.stop()
        
        return cls(
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.ppx,
            cy=intrinsics.ppy,
            width=intrinsics.width,
            height=intrinsics.height,
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
            baseline=data.get('baseline', 0.0),
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
                f"res={self.width}x{self.height}, baseline={self.baseline}, depth_scale={self.depth_scale})")

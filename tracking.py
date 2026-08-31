"""
tracking.py — estimate the camera pose for every incoming frame.

Maps to: ORB_SLAM3/src/Tracking.cc
  Track()                 -> track()
  TrackWithMotionModel    -> _track_with_motion_model()
  TrackReferenceKeyFrame  -> _track_reference_keyframe()
  PoseOptimization        -> solvePnPRansac (scipy-free, OpenCV does it)
  NeedNewKeyFrame         -> needs_new_keyframe()

State machine mirrors the real one: NOT_INITIALIZED -> OK -> RECENTLY_LOST -> LOST

SIMPLIFICATIONS:
  - No Relocalization() (full-database recovery when totally lost)
  - No TrackLocalMap() refinement pass over all nearby map points
  - No isInFrustum visibility filtering before matching
"""

import cv2
import numpy as np


class Tracking:
    def __init__(self, camera, extractor, matcher, world_map,
                 min_matches_for_pose=15,
                 keyframe_min_matches=50,
                 keyframe_min_displacement=0.10,
                 keyframe_max_frames=20):
        self.camera = camera
        self.extractor = extractor
        self.matcher = matcher
        self.map = world_map

        self.state = "NOT_INITIALIZED"
        self.last_frame = None
        self.last_keyframe = None
        self.velocity = None              # 4x4 relative motion, for the motion model
        self.frames_since_keyframe = 0

        self.min_matches_for_pose = min_matches_for_pose
        self.keyframe_min_matches = keyframe_min_matches
        self.keyframe_min_displacement = keyframe_min_displacement
        self.keyframe_max_frames = keyframe_max_frames

    def set_map(self, world_map):
        """Called when Atlas switches to a new active map."""
        self.map = world_map

    # ── main entry point ─────────────────────────────────────────────────

    def track(self, frame):
        """Estimate frame.pose. Returns True on success."""
        if self.state == "NOT_INITIALIZED":
            return False

        ok = False
        # Strategy 1: motion model (fast) — assume constant velocity
        if self.velocity is not None and self.last_frame is not None:
            ok = self._track_with_motion_model(frame)

        # Strategy 2: fall back to matching against the whole map
        if not ok:
            ok = self._track_reference_keyframe(frame)

        if ok:
            self.state = "OK"
            # update motion model: velocity = current * inverse(last)
            if self.last_frame is not None and self.last_frame.pose is not None:
                self.velocity = frame.pose @ np.linalg.inv(self.last_frame.pose)
            self.last_frame = frame
            self.frames_since_keyframe += 1
        else:
            self.state = "RECENTLY_LOST" if self.state == "OK" else "LOST"
            self.velocity = None

        return ok

    # ── strategies ───────────────────────────────────────────────────────

    def _track_with_motion_model(self, frame):
        """
        Predict the pose from constant velocity, then verify with PnP against
        the map points the LAST frame was tracking (a small, likely-visible set).
        """
        if self.last_frame is None or self.last_frame.pose is None:
            return False

        # Candidate points: only those the previous frame saw
        cand_ids, cand_descs = [], []
        for mp_id in self.last_frame.map_point_ids:
            if mp_id is None:
                continue
            mp = self.map.map_points.get(mp_id)
            if mp is None or mp.is_bad or mp.descriptor is None:
                continue
            cand_ids.append(mp_id)
            cand_descs.append(mp.descriptor)

        if len(cand_ids) < self.min_matches_for_pose:
            return False

        cand_descs = np.asarray(cand_descs, dtype=np.uint8)
        return self._solve_pnp(frame, cand_ids, cand_descs)

    def _track_reference_keyframe(self, frame):
        """Match against every good map point. Slower, but recovers better."""
        ids, descs = self.map.all_descriptors()
        if descs is None or len(ids) < self.min_matches_for_pose:
            return False
        return self._solve_pnp(frame, ids, descs)

    def _solve_pnp(self, frame, mp_ids, mp_descs):
        """
        Match frame descriptors to candidate map point descriptors, build
        2D-3D correspondences, solve for pose with PnP + RANSAC.
        """
        if frame.descriptors is None or len(frame.descriptors) == 0:
            return False

        matches = self.matcher.match(frame.descriptors, mp_descs)
        if len(matches) < self.min_matches_for_pose:
            return False

        pts_2d, pts_3d, kp_indices, matched_mp_ids = [], [], [], []
        for m in matches:
            mp = self.map.map_points.get(mp_ids[m.trainIdx])
            if mp is None or mp.is_bad:
                continue
            pts_2d.append(frame.points_undistorted[m.queryIdx])
            pts_3d.append(mp.position)
            kp_indices.append(m.queryIdx)
            matched_mp_ids.append(mp.id)
            mp.increase_visible()

        if len(pts_3d) < 6:
            return False

        pts_2d = np.asarray(pts_2d, dtype=np.float64)
        pts_3d = np.asarray(pts_3d, dtype=np.float64)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, self.camera.K, None,
            iterationsCount=200, reprojectionError=3.0,
            confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)

        if not ok or inliers is None or len(inliers) < self.min_matches_for_pose:
            return False

        # solvePnP gives world-to-camera; invert for camera-to-world
        R, _ = cv2.Rodrigues(rvec)
        T_wc = np.eye(4)
        T_wc[:3, :3] = R
        T_wc[:3, 3] = tvec.flatten()
        frame.set_pose(np.linalg.inv(T_wc))

        # Record only the inlier associations
        inlier_set = set(int(i) for i in inliers.flatten())
        for k, (kp_i, mp_id) in enumerate(zip(kp_indices, matched_mp_ids)):
            if k in inlier_set:
                frame.map_point_ids[kp_i] = mp_id
                mp = self.map.map_points.get(mp_id)
                if mp:
                    mp.increase_found()

        return True

    # ── keyframe decision ────────────────────────────────────────────────

    def needs_new_keyframe(self, frame):
        """
        Tracking::NeedNewKeyFrame(), simplified.
        Insert a keyframe if we've drifted far enough, waited long enough,
        or tracking quality has dropped.
        """
        if self.last_keyframe is None:
            return True
        if frame.pose is None or self.last_keyframe.pose is None:
            return False

        n_tracked = frame.n_tracked_points()
        displacement = float(np.linalg.norm(
            frame.camera_center() - self.last_keyframe.camera_center()))

        c1 = self.frames_since_keyframe >= self.keyframe_max_frames
        c2 = displacement > self.keyframe_min_displacement
        c3 = n_tracked < self.keyframe_min_matches

        return (c1 or c2 or c3) and n_tracked >= 15

    def mark_keyframe(self, frame):
        self.last_keyframe = frame
        self.frames_since_keyframe = 0
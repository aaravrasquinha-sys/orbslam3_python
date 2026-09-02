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

import covisibility

TH_LOW = 50   # ORBmatcher::TH_LOW — same strict threshold matcher.py uses


def _hamming(a, b):
    return int(np.count_nonzero(np.unpackbits(np.bitwise_xor(a, b))))


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
        self.velocity = None            # 4x4 relative motion, for the motion model
        self.frames_since_keyframe = 0

        # Shared reference to LocalMapping's covisibility graph (see
        # covisibility.py). Set by SLAMSystem right after constructing
        # both objects. None until the first keyframe exists, in which
        # case tracking falls back to the old whole-map match (there's
        # nothing to be local ABOUT yet).
        self.covis_graph = None

        # Visual-inertial state (Phase 3/4). Set by SLAMSystem once
        # imu_init.py succeeds. Before that, IMU prediction is skipped and
        # tracking behaves exactly as it did visual-only.
        self.imu_initialized = False
        self.gravity = None

        self.min_matches_for_pose = min_matches_for_pose
        self.keyframe_min_matches = keyframe_min_matches
        self.keyframe_min_displacement = keyframe_min_displacement
        self.keyframe_max_frames = keyframe_max_frames

    def set_map(self, world_map):
        """Called when Atlas switches to a new active map."""
        self.map = world_map

    # ── main entry point ─────────────────────────────────────────────────

    def track(self, frame, imu_preint=None):
        """
        Estimate frame.pose. Returns True on success.

        imu_preint: the RUNNING imu.Preintegration accumulator since
        self.last_keyframe (see run_slam.py) -- a valid PARTIAL
        preintegration even before the segment is finalized at the next
        keyframe. Used for IMU-predicted pose ONLY after imu_init.py has
        succeeded (self.imu_initialized); before that this is ignored and
        tracking behaves exactly as it did visual-only.
        """
        if self.state == "NOT_INITIALIZED":
            return False

        ok = False
        predicted_pose, predicted_vel = None, None
        if (self.imu_initialized and imu_preint is not None and
                self.last_keyframe is not None and self.last_keyframe.velocity is not None):
            predicted_pose, predicted_vel = self._predict_pose_imu(self.last_keyframe, imu_preint)

        # Strategy 0: IMU-predicted pose, verified with PnP. Large
        # robustness win through fast rotation / motion blur -- exactly the
        # failure mode that used to trip consecutive_lost with only a
        # constant-velocity model to fall back on.
        if predicted_pose is not None:
            ok = self._track_with_motion_model(frame, initial_pose=predicted_pose)
            if ok:
                frame.velocity = predicted_vel

        # Strategy 1: constant-velocity motion model (fast)
        if not ok and self.velocity is not None and self.last_frame is not None:
            ok = self._track_with_motion_model(frame)

        # Strategy 2: fall back to matching against the local map
        if not ok:
            ok = self._track_reference_keyframe(frame)

        if ok:
            # TrackLocalMap: now that we have an initial pose, expand
            # matches against the WIDER local map (K1+K2 covisible
            # keyframes) using frustum culling + guided matching, instead
            # of stopping at whatever the fast initial estimate found. This
            # is what used to be missing entirely -- the old code had no
            # refinement pass, so tracking quality was capped by whichever
            # of the two cheap strategies fired.
            n_expanded = self._track_local_map(frame)

            self.state = "OK"
            # update motion model: velocity = current * inverse(last)
            if self.last_frame is not None and self.last_frame.pose is not None:
                self.velocity = frame.pose @ np.linalg.inv(self.last_frame.pose)
                # Diagnostic: Print out frame-to-frame motion delta
                trans_vector = self.velocity[:3, 3]
                trans_dist = np.linalg.norm(trans_vector)
                print(f"[TRACK] Frame {frame.id} Motion -> Trans: [{trans_vector[0]:.3f}, {trans_vector[1]:.3f}, {trans_vector[2]:.3f}] (Dist: {trans_dist:.3f}m) | +{n_expanded} local-map matches")

            self.last_frame = frame
            self.frames_since_keyframe += 1
        elif predicted_pose is not None:
            # RECENTLY_LOST + IMU initialized: carry the pose forward via
            # IMU integration alone rather than dropping it entirely. This
            # is what lets a facility-length walk survive a few seconds of
            # bad visual conditions (motion blur, a blank wall) instead of
            # abandoning the map -- see run_slam.py's recently_lost handling
            # for how long this is allowed to continue before giving up.
            frame.set_pose(predicted_pose)
            frame.velocity = predicted_vel
            self.state = "RECENTLY_LOST"
            self.last_frame = frame
        else:
            self.state = "RECENTLY_LOST" if self.state == "OK" else "LOST"
            self.velocity = None

        return ok

    def _predict_pose_imu(self, last_keyframe, preint):
        """
        Standard IMU state propagation from the last keyframe's (pose,
        velocity) using the running preintegration (see imu.py). Both
        states live directly in the camera frame (see imu.py's docstring
        on the lever-arm simplification), so no extrinsic conversion is
        needed here.
        """
        R_i = last_keyframe.pose[:3, :3]
        p_i = last_keyframe.pose[:3, 3]
        v_i = last_keyframe.velocity
        dt = preint.dt

        R_j = R_i @ preint.dR
        v_j = v_i + self.gravity * dt + R_i @ preint.dv
        p_j = p_i + v_i * dt + 0.5 * self.gravity * dt ** 2 + R_i @ preint.dp

        pose = np.eye(4)
        pose[:3, :3] = R_j
        pose[:3, 3] = p_j
        return pose, v_j

    # ── strategies ───────────────────────────────────────────────────────

    def _track_with_motion_model(self, frame, initial_pose=None):
        """
        Predict the pose (from IMU if `initial_pose` is given, else from
        constant velocity), then verify with PnP against the map points the
        LAST frame was tracking (a small, likely-visible set).
        """
        if initial_pose is None:
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
        return self._solve_pnp(frame, cand_ids, cand_descs, initial_pose=initial_pose)

    def _track_reference_keyframe(self, frame):
        """
        Match against the LOCAL map, not the whole map.

        BUGFIX: this used to call self.map.all_descriptors() -- every good
        point in the ENTIRE map, growing without bound (a 1200x30000
        Hamming match at 30k points, EVERY frame this strategy fired).
        Replaced with covisibility.py's K1+K2 local map: the reference
        keyframe's strongly-covisible neighbors plus their neighbors, which
        stays roughly constant-sized as the facility grows because it's
        bounded by local connectivity, not total map size.
        """
        if self.last_keyframe is None or self.covis_graph is None:
            # No keyframes yet to be local ABOUT (e.g. right after init) --
            # fall back to whatever the map currently has.
            ids, descs = self.map.all_descriptors()
            if descs is None or len(ids) < self.min_matches_for_pose:
                return False
            return self._solve_pnp(frame, ids, descs)

        local_kfs = covisibility.local_keyframes(self.last_keyframe, self.covis_graph, self.map)
        local_pt_ids = covisibility.local_map_points(local_kfs, self.map)
        ids, descs = [], []
        for mp_id in local_pt_ids:
            mp = self.map.map_points.get(mp_id)
            if mp is not None and not mp.is_bad and mp.descriptor is not None:
                ids.append(mp_id)
                descs.append(mp.descriptor)
        if len(ids) < self.min_matches_for_pose:
            return False
        return self._solve_pnp(frame, ids, np.asarray(descs, dtype=np.uint8))

    def _track_local_map(self, frame):
        """
        Tracking::TrackLocalMap(): once an initial pose exists (from either
        the motion model or _track_reference_keyframe), expand the match
        set against the WIDER local map using frustum culling + guided
        matching, rather than accepting whatever the fast initial strategy
        found. This is what turns per-frame matching from O(map size) into
        O(local map size) end-to-end, and is what lets tracking stay
        reliable as the facility (and total map) gets larger. Returns the
        number of newly added matches.
        """
        if self.last_keyframe is None or self.covis_graph is None or frame.pose is None:
            return 0

        local_kfs = covisibility.local_keyframes(self.last_keyframe, self.covis_graph, self.map)
        local_pt_ids = covisibility.local_map_points(local_kfs, self.map)
        already_tracked = set(mp_id for mp_id in frame.map_point_ids if mp_id is not None)

        n_levels = self.extractor.nlevels
        scale_factor = self.extractor.scale_factor
        n_new = 0

        for mp_id in local_pt_ids:
            if mp_id in already_tracked:
                continue
            mp = self.map.map_points.get(mp_id)
            if mp is None or mp.is_bad or mp.descriptor is None:
                continue

            visible, u, v, octave = covisibility.is_in_frustum(
                mp, frame, self.camera, n_levels=n_levels, scale_factor=scale_factor)
            if not visible:
                continue
            mp.increase_visible()

            radius = 4.0 * (scale_factor ** octave if octave else 1.0)
            candidates = frame.get_features_in_area(
                u, v, radius, min_level=max(0, octave - 1), max_level=octave + 1)
            if not candidates:
                continue

            best_dist, best_idx = TH_LOW + 1, -1
            for kp_i in candidates:
                if frame.map_point_ids[kp_i] is not None:
                    continue   # slot already claimed by another match this frame
                d = _hamming(mp.descriptor, frame.descriptors[kp_i])
                if d < best_dist:
                    best_dist, best_idx = d, kp_i
            if best_idx < 0 or best_dist > TH_LOW:
                continue

            frame.map_point_ids[best_idx] = mp.id
            mp.increase_found()
            already_tracked.add(mp.id)
            n_new += 1

        return n_new

    def _solve_pnp(self, frame, mp_ids, mp_descs, initial_pose=None):
        """
        Match frame descriptors to candidate map point descriptors, build
        2D-3D correspondences, solve for pose with PnP + RANSAC.

        initial_pose: optional camera-to-world 4x4 (from IMU prediction) fed
        to solvePnPRansac via useExtrinsicGuess=True -- lets RANSAC start
        from a good guess instead of searching cold, which matters most
        exactly when motion is fast/blurred (the case IMU prediction is
        for in the first place).
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

        if initial_pose is not None:
            pose_cw_guess = np.linalg.inv(initial_pose)
            rvec0, _ = cv2.Rodrigues(pose_cw_guess[:3, :3])
            tvec0 = pose_cw_guess[:3, 3].reshape(3, 1)
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d, pts_2d, self.camera.K, None,
                rvec0.copy(), tvec0.copy(), useExtrinsicGuess=True,
                iterationsCount=200, reprojectionError=3.0,
                confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d, pts_2d, self.camera.K, None,
                iterationsCount=200, reprojectionError=3.0,
                confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)

        inlier_count = len(inliers) if (inliers is not None) else 0
        print(f"[TRACK] Frame {frame.id}: Raw Matches={len(matches)}, RANSAC Inliers={inlier_count}")

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

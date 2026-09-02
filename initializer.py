"""
initializer.py — bootstrap the very first map.

Maps to: ORB_SLAM3/src/TwoViewReconstruction.cc + Tracking::MonocularInitialization
  Reconstruct / ReconstructF -> init_monocular()
  CheckRT parallax gate      -> the parallax check below

Two paths:
  init_rgbd()       — depth available: unproject frame 0 directly. Instant,
                      metric, no parallax needed. Maps to StereoInitialization().
  init_monocular()  — no depth: essential matrix + triangulation, needs parallax.

SIMPLIFICATION (monocular): real ORB-SLAM3 tries BOTH a Homography (flat scene)
and a Fundamental matrix (general 3D scene) in parallel and picks whichever
scores better. We only do the Fundamental/Essential path, so we're weaker on
planar scenes (staring at a wall or floor).
"""

import cv2
import numpy as np

from map_point import MapPoint


def init_rgbd(frame, world_map, min_points=100):
    """
    RGB-D initialisation: one frame is enough.
    Sets frame.pose to identity (world origin) and creates map points directly
    from depth. Returns (success, new_points).
    """
    if frame.n_valid_depths() < min_points:
        return False, []

    frame.set_pose(np.eye(4))
    world_map.add_keyframe(frame)   # assigns frame.kf_seq before points reference it

    new_points = []
    for i in range(frame.n):
        p3d = frame.unproject_keypoint(i)
        if p3d is None:
            continue
        mp = MapPoint(p3d, frame.descriptors[i], ref_keyframe_id=frame.kf_seq)
        mp.add_observation(frame.id, i)
        frame.map_point_ids[i] = mp.id
        world_map.add_map_point(mp)
        new_points.append(mp)

    if len(new_points) < min_points:
        world_map.erase_keyframe(frame)
        for mp in new_points:
            world_map.erase_map_point(mp)
        return False, []

    return True, new_points


def init_monocular(frame_a, frame_b, matcher, camera, world_map,
                   min_matches=100, min_parallax_px=2.0):
    """
    Monocular two-view initialisation.
    frame_a becomes the world origin; frame_b's pose is estimated relative to it.
    Returns (success, new_points).
    """
    matches = matcher.match(frame_a.descriptors, frame_b.descriptors)
    if len(matches) < min_matches:
        return False, []

    pts_a, pts_b = matcher.matched_points(frame_a.keypoints, frame_b.keypoints, matches)

    # Parallax gate — the single most important check in this file.
    # TwoViewReconstruction::CheckRT computes a proper 3D parallax ANGLE;
    # we approximate with mean pixel displacement. Same purpose: refuse to
    # triangulate when the two views are too similar to trust.
    avg_disp = float(np.mean(np.linalg.norm(pts_a - pts_b, axis=1)))
    if avg_disp < min_parallax_px:
        return False, []

    E, mask = cv2.findEssentialMat(pts_a, pts_b, camera.K,
                                   method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        return False, []

    n_in, R, t, pose_mask = cv2.recoverPose(E, pts_a, pts_b, camera.K, mask=mask)
    if n_in < min_matches * 0.5:
        return False, []

    # frame_a = world origin. frame_b's camera-to-world pose:
    # recoverPose gives world-to-camera (R,t), so invert for camera-to-world.
    T_ba = np.eye(4)
    T_ba[:3, :3] = R
    T_ba[:3, 3] = t.flatten()
    frame_a.set_pose(np.eye(4))
    frame_b.set_pose(np.linalg.inv(T_ba))
    world_map.add_keyframe(frame_a)   # assigns kf_seq before points reference it
    world_map.add_keyframe(frame_b)

    # Triangulate: projection matrices are world-to-camera
    P_a = camera.K @ np.eye(3, 4)
    P_b = camera.K @ T_ba[:3, :]
    pts4d = cv2.triangulatePoints(P_a, P_b, pts_a.T, pts_b.T)
    with np.errstate(divide='ignore', invalid='ignore'):
        pts3d = (pts4d[:3] / pts4d[3]).T

    new_points = []
    for i, m in enumerate(matches):
        if pose_mask is not None and pose_mask[i] == 0:
            continue
        p = pts3d[i]
        if not np.all(np.isfinite(p)):
            continue
        if p[2] <= 0:          # must be in front of the first camera
            continue

        mp = MapPoint(p, frame_a.descriptors[m.queryIdx], ref_keyframe_id=frame_a.kf_seq)
        mp.add_observation(frame_a.id, m.queryIdx)
        mp.add_observation(frame_b.id, m.trainIdx)
        frame_a.map_point_ids[m.queryIdx] = mp.id
        frame_b.map_point_ids[m.trainIdx] = mp.id
        world_map.add_map_point(mp)
        new_points.append(mp)

    if len(new_points) < min_matches * 0.3:
        world_map.erase_keyframe(frame_a)
        world_map.erase_keyframe(frame_b)
        for mp in new_points:
            world_map.erase_map_point(mp)
        return False, []

    # Normalise scale so median depth = 1.0 (CreateInitialMapMonocular does this).
    # Monocular scale is arbitrary — this just picks a consistent reference.
    depths = np.array([np.linalg.norm(mp.position) for mp in new_points])
    median_depth = float(np.median(depths))
    if median_depth > 1e-6:
        s = 1.0 / median_depth
        for mp in new_points:
            mp.position *= s
        pose_b = frame_b.pose.copy()
        pose_b[:3, 3] *= s
        frame_b.set_pose(pose_b)

    return True, new_points
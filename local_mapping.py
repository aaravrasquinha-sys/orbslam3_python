"""
local_mapping.py — grow and clean the map when a new keyframe arrives.

Maps to: ORB_SLAM3/src/LocalMapping.cc
  ProcessNewKeyFrame  -> process_new_keyframe() step 1
  MapPointCulling     -> cull_recent_map_points()
  CreateNewMapPoints  -> _triangulate_new_points() / _create_points_from_depth()
  KeyFrameCulling     -> (NOT implemented — flagged simplification)
  SearchInNeighbors   -> (NOT implemented — no duplicate fusion)
"""

import cv2
import numpy as np

from map_point import MapPoint


class LocalMapping:
    def __init__(self, camera, matcher, world_map,
                 min_parallax_px=2.0, max_neighbors=5,
                 culling_found_ratio=0.25, culling_min_obs=3):
        self.camera = camera
        self.matcher = matcher
        self.map = world_map
        self.min_parallax_px = min_parallax_px
        self.max_neighbors = max_neighbors
        self.culling_found_ratio = culling_found_ratio
        self.culling_min_obs = culling_min_obs

        self.recent_points = []   # mlpRecentAddedMapPoints — points on probation

    def set_map(self, world_map):
        self.map = world_map
        self.recent_points = []

    def process_new_keyframe(self, keyframe, use_depth=False):
        """Full LocalMapping::Run() body for one keyframe."""
        self.map.add_keyframe(keyframe)

        # 1. link existing observations
        for i, mp_id in enumerate(keyframe.map_point_ids):
            if mp_id is None:
                continue
            mp = self.map.map_points.get(mp_id)
            if mp and not mp.is_bad:
                mp.add_observation(keyframe.id, i)

        # 2. cull unreliable recently-created points
        n_culled = self.cull_recent_map_points(keyframe)

        # 3. create new points
        if use_depth:
            new_points = self._create_points_from_depth(keyframe)
        else:
            new_points = self._triangulate_new_points(keyframe)

        for mp in new_points:
            self.map.add_map_point(mp)
            self.recent_points.append(mp)

        return new_points, n_culled

    # ── MapPointCulling ──────────────────────────────────────────────────

    def cull_recent_map_points(self, current_keyframe):
        """
        Points created recently are on probation. Kill them if they're rarely
        re-detected, or if they never accumulated enough observations.
        """
        survivors, n_culled = [], 0
        for mp in self.recent_points:
            if mp.is_bad:
                n_culled += 1
                continue

            age = current_keyframe.id - (mp.first_keyframe_id or 0)

            if mp.found_ratio() < self.culling_found_ratio:
                mp.set_bad()
                n_culled += 1
            elif age >= 2 and mp.n_observations() <= self.culling_min_obs:
                mp.set_bad()
                n_culled += 1
            elif age >= 3:
                pass          # graduated — trusted from now on, stop watching
            else:
                survivors.append(mp)

        self.recent_points = survivors
        self.map.clean_bad_points()
        return n_culled

    # ── RGB-D point creation ─────────────────────────────────────────────

    def _create_points_from_depth(self, keyframe):
        """Depth available: unproject unmatched keypoints directly. No parallax needed."""
        new_points = []
        for i in range(keyframe.n):
            if keyframe.map_point_ids[i] is not None:
                continue
            p3d = keyframe.unproject_keypoint(i)
            if p3d is None:
                continue
            mp = MapPoint(p3d, keyframe.descriptors[i], ref_keyframe_id=keyframe.id)
            mp.add_observation(keyframe.id, i)
            keyframe.map_point_ids[i] = mp.id
            new_points.append(mp)
        return new_points

    # ── monocular triangulation ──────────────────────────────────────────

    def _triangulate_new_points(self, keyframe):
        """
        Match this keyframe's UNMATCHED features against recent keyframes'
        unmatched features, and triangulate whatever passes the checks.
        """
        if len(self.map.keyframes) < 2 or keyframe.pose is None:
            return []

        neighbors = [kf for kf in self.map.keyframes[-(self.max_neighbors + 1):-1]
                     if kf.pose is not None and kf.id != keyframe.id]
        if not neighbors:
            return []

        new_points = []
        cur_unmatched = [i for i, m in enumerate(keyframe.map_point_ids) if m is None]
        if not cur_unmatched:
            return []

        for neigh in neighbors:
            neigh_unmatched = [i for i, m in enumerate(neigh.map_point_ids) if m is None]
            if not neigh_unmatched:
                continue

            desc_cur = keyframe.descriptors[cur_unmatched]
            desc_nei = neigh.descriptors[neigh_unmatched]
            matches = self.matcher.match(desc_cur, desc_nei)
            if len(matches) < 8:
                continue

            pts_cur = np.float32([keyframe.points_undistorted[cur_unmatched[m.queryIdx]]
                                  for m in matches])
            pts_nei = np.float32([neigh.points_undistorted[neigh_unmatched[m.trainIdx]]
                                  for m in matches])

            # parallax gate — same idea as initializer.py
            if float(np.mean(np.linalg.norm(pts_cur - pts_nei, axis=1))) < self.min_parallax_px:
                continue

            # projection matrices are world-to-camera = inverse of stored pose
            P_cur = self.camera.K @ np.linalg.inv(keyframe.pose)[:3, :]
            P_nei = self.camera.K @ np.linalg.inv(neigh.pose)[:3, :]

            pts4d = cv2.triangulatePoints(P_cur, P_nei, pts_cur.T, pts_nei.T)
            with np.errstate(divide='ignore', invalid='ignore'):
                pts3d = (pts4d[:3] / pts4d[3]).T

            for k, m in enumerate(matches):
                p = pts3d[k]
                if not np.all(np.isfinite(p)):
                    continue
                if not self._check_point(p, keyframe, neigh,
                                         pts_cur[k], pts_nei[k]):
                    continue

                ci = cur_unmatched[m.queryIdx]
                ni = neigh_unmatched[m.trainIdx]
                if keyframe.map_point_ids[ci] is not None:
                    continue

                mp = MapPoint(p, keyframe.descriptors[ci], ref_keyframe_id=keyframe.id)
                mp.add_observation(keyframe.id, ci)
                mp.add_observation(neigh.id, ni)
                keyframe.map_point_ids[ci] = mp.id
                neigh.map_point_ids[ni] = mp.id
                new_points.append(mp)

            cur_unmatched = [i for i, m in enumerate(keyframe.map_point_ids) if m is None]
            if not cur_unmatched:
                break

        return new_points

    def _check_point(self, p_world, kf_a, kf_b, px_a, px_b,
                     max_reproj_err=4.0):
        """
        LocalMapping::CreateNewMapPoints()'s validity checks:
        positive depth in both cameras + acceptable reprojection error in both.
        """
        for kf, px in ((kf_a, px_a), (kf_b, px_b)):
            T_wc = np.linalg.inv(kf.pose)
            p_cam = T_wc[:3, :3] @ p_world + T_wc[:3, 3]
            if p_cam[2] <= 0:
                return False
            proj = self.camera.project(p_cam)
            if proj is None:
                return False
            if float(np.linalg.norm(proj - px)) > max_reproj_err:
                return False
        return True
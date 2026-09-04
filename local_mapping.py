"""
local_mapping.py — grow and clean the map when a new keyframe arrives.

Maps to: ORB_SLAM3/src/LocalMapping.cc
  ProcessNewKeyFrame  -> process_new_keyframe() step 1
  MapPointCulling     -> cull_recent_map_points()
  CreateNewMapPoints  -> _triangulate_new_points() / _create_points_from_depth()
  KeyFrameCulling     -> cull_keyframes()
  SearchInNeighbors   -> delegated to fusion.search_in_neighbors()
"""

import cv2
import numpy as np

from map_point import MapPoint
import covisibility
import fusion


class LocalMapping:
    def __init__(self, camera, matcher, world_map, extractor=None,
                 min_parallax_px=2.0, max_neighbors=5,
                 culling_found_ratio=0.25, culling_min_obs=3,
                 max_new_points_per_kf=100,
                 depth_min=0.3, depth_max=3.5,
                 depth_patch_radius=2, depth_rel_std_max=0.02,
                 kf_culling_redundancy=0.9, kf_culling_min_obs=3,
                 kf_culling_min_map_size=5, kf_culling_protect_recent=3):
        self.camera = camera
        self.matcher = matcher
        self.map = world_map
        self.extractor = extractor   # needed for scale_factor/n_levels (frustum + normals)
        self.min_parallax_px = min_parallax_px
        self.max_neighbors = max_neighbors
        self.culling_found_ratio = culling_found_ratio
        self.culling_min_obs = culling_min_obs

        # ORB-SLAM3's rule for RGB-D point creation: cap how many new points
        # a single keyframe can spawn, prioritise the closest (most
        # reliable) depth, gate to a sane working range, and reject points
        # sitting on a depth discontinuity (object edges -> flying points).
        self.max_new_points_per_kf = max_new_points_per_kf
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_patch_radius = depth_patch_radius
        self.depth_rel_std_max = depth_rel_std_max

        # KeyFrameCulling thresholds (LocalMapping::KeyFrameCulling)
        self.kf_culling_redundancy = kf_culling_redundancy
        self.kf_culling_min_obs = kf_culling_min_obs
        self.kf_culling_min_map_size = kf_culling_min_map_size
        self.kf_culling_protect_recent = kf_culling_protect_recent

        # Covisibility graph: kf_id -> {neighbor_kf_id: shared_point_count}.
        # Rebuilt in place (see covisibility.py) so Tracking can hold the
        # same dict reference and always see the latest version.
        self.covis_graph = {}

        self.recent_points = []   # mlpRecentAddedMapPoints — points on probation

    def set_map(self, world_map):
        self.map = world_map
        self.recent_points = []
        self.covis_graph = {}

    def process_new_keyframe(self, keyframe, use_depth=False, depth_image=None):
        """
        Full LocalMapping::Run() body for one keyframe.

        depth_image: raw 16-bit depth frame for THIS keyframe, used only for
        the local-patch variance gate in _create_points_from_depth. Passed
        through rather than stored on Frame, so we don't retain a ~600KB
        array per frame for the lifetime of a facility-length run.
        """
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
            new_points = self._create_points_from_depth(keyframe, depth_image=depth_image)
        else:
            new_points = self._triangulate_new_points(keyframe)

        for mp in new_points:
            self.map.add_map_point(mp)
            self.recent_points.append(mp)

        # 4. covisibility graph (needed by fusion below AND by Tracking's
        #    local-map matching -- see covisibility.py / tracking.py).
        #    Built from the state right after new points were added; fusion
        #    below will change observations again, so we rebuild once more
        #    at the end to hand Tracking a fully up-to-date graph.
        covisibility.build_covisibility_graph(self.map, graph=self.covis_graph)

        # 5. SearchInNeighbors: fuse duplicate points against covisible
        #    neighbors. This is what turns the RGB-D duplicate-cloud problem
        #    into a proper landmark set, and is what manufactures the
        #    >=2-observation points bundle_adjust.py's windowed BA needs.
        n_levels = self.extractor.nlevels if self.extractor else 8
        scale_factor = self.extractor.scale_factor if self.extractor else 1.2
        n_fused = fusion.search_in_neighbors(
            keyframe, self.map, self.covis_graph, self.camera,
            n_levels=n_levels, scale_factor=scale_factor)

        # 6. refresh normal/depth invariance for points this keyframe (and
        #    its neighbors) touched, and rebuild the graph one more time so
        #    it reflects the fused observations.
        kf_by_id = {kf.id: kf for kf in self.map.keyframes}
        scale_factors = self.extractor.scale_factors if self.extractor else \
            [scale_factor ** i for i in range(n_levels)]
        touched_ids = set(mp_id for mp_id in keyframe.map_point_ids if mp_id is not None)
        for mp_id in touched_ids:
            mp = self.map.map_points.get(mp_id)
            if mp is not None and not mp.is_bad:
                mp.update_normal_and_depth(kf_by_id, scale_factors)
        covisibility.build_covisibility_graph(self.map, graph=self.covis_graph)

        return new_points, n_culled, n_fused

    # ── MapPointCulling ──────────────────────────────────────────────────

    def cull_recent_map_points(self, current_keyframe):
        """
        Points created recently are on probation. Kill them if they're rarely
        re-detected, or if they never accumulated enough observations.
        """
        survivors, n_culled = [], 0
        # BUGFIX: age must count KEYFRAMES, not frames. current_keyframe.id
        # and mp.first_keyframe_id used to be Frame.id, which increments on
        # every processed frame -- with keyframe_max_frames=20 a point's
        # "age" could leap straight past the >=2 / >=3 thresholds the
        # moment the NEXT keyframe was created, so probation barely ran.
        # kf_seq (assigned in Map.add_keyframe) counts actual keyframes.
        cur_seq = current_keyframe.kf_seq if current_keyframe.kf_seq is not None else 0
        for mp in self.recent_points:
            if mp.is_bad:
                n_culled += 1
                continue

            age = cur_seq - (mp.first_keyframe_id or 0)

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

    # ── KeyFrameCulling ──────────────────────────────────────────────────

    def cull_keyframes(self):
        """
        LocalMapping::KeyFrameCulling(), corrected against three bugs found
        during the Phase-1 forensic audit (see PROGRESS.md for the ablation
        numbers): on a noise-free 60-frame synthetic sequence with zero
        tracking losses, the ORIGINAL version of this function culled 14 of
        20 keyframes (70%) and left ZERO map points in the entire map with
        >=4 observations -- every landmark was destroyed before it could
        mature, which is what produced the empty local maps, the "5 free +
        1 fixed kfs" BA windows, and the eventual tracking collapse under
        rotation. The three bugs, all biased toward over-culling:

          1. Redundancy was judged against the ENTIRE map
             (self.map.keyframes), not the keyframe's own covisible
             neighbors. A point can look "well covered" globally while
             being poorly covered locally -- ORB-SLAM2 only ever compares a
             keyframe against ITS OWN covisibility neighbors.
          2. n_observations() counts the keyframe being judged too, so
             kf_culling_min_obs=3 actually required only 2 OTHER observers
             to call a point redundant -- one keyframe more than intended.
             ORB-SLAM2's default is nObs>3 measured over OTHER keyframes,
             i.e. >=4 other observers minimum.
          3. No scale check: a point should only count as "redundant" via
             another observer if that observer sees it at the same or a
             FINER pyramid octave (a coarser-octave observation is a worse
             view and doesn't actually make the point disposable at this
             scale). Approximated here by comparing keypoint octaves.

        A keyframe is redundant if >= kf_culling_redundancy of its good
        points are each observed, at comparable-or-better scale, by >=
        kf_culling_min_obs OTHER keyframes drawn from its own covisibility
        neighbors. The most recent few keyframes stay protected.
        """
        if self.map.n_keyframes() < self.kf_culling_min_map_size:
            return 0

        ordered = sorted((kf for kf in self.map.keyframes if kf.kf_seq is not None),
                         key=lambda kf: kf.kf_seq)
        protected_ids = {kf.id for kf in ordered[-self.kf_culling_protect_recent:]}
        kf_by_id = {kf.id: kf for kf in self.map.keyframes}

        to_cull = []
        for kf in ordered:
            if kf.id in protected_ids:
                continue

            # BUGFIX #1: neighbors, not the whole map.
            neighbor_ids = set(self.covis_graph.get(kf.id, {}).keys())
            if not neighbor_ids:
                continue   # isolated keyframe -- nothing to compare against, keep it

            obs_points = [(kp_i, mp_id) for kp_i, mp_id in enumerate(kf.map_point_ids)
                         if mp_id is not None]
            good = [(kp_i, self.map.map_points[mp_id]) for kp_i, mp_id in obs_points
                   if mp_id in self.map.map_points and not self.map.map_points[mp_id].is_bad]
            if len(good) < 20:      # too few points to judge either way -- keep it
                continue

            n_redundant = 0
            for kp_i, mp in good:
                my_octave = kf.keypoints[kp_i].octave if kp_i < len(kf.keypoints) else 0
                # BUGFIX #2: count OTHER observers only, restricted to this
                # keyframe's own covisibility neighbors (BUGFIX #1).
                n_other = 0
                for obs_kf_id, obs_kp_i in mp.observations.items():
                    if obs_kf_id == kf.id or obs_kf_id not in neighbor_ids:
                        continue
                    obs_kf = kf_by_id.get(obs_kf_id)
                    if obs_kf is None:
                        continue
                    # BUGFIX #3: only counts if seen at the same-or-finer scale.
                    obs_octave = (obs_kf.keypoints[obs_kp_i].octave
                                 if obs_kp_i < len(obs_kf.keypoints) else 0)
                    if obs_octave <= my_octave:
                        n_other += 1
                        if n_other >= self.kf_culling_min_obs:
                            break
                if n_other >= self.kf_culling_min_obs:
                    n_redundant += 1

            if n_redundant / len(good) >= self.kf_culling_redundancy:
                to_cull.append(kf)

        for kf in to_cull:
            for mp_id in kf.map_point_ids:
                if mp_id is None:
                    continue
                mp = self.map.map_points.get(mp_id)
                if mp is not None:
                    mp.erase_observation(kf.id)
            self.map.erase_keyframe(kf)

        if to_cull:
            self.map.clean_bad_points()
            covisibility.build_covisibility_graph(self.map, graph=self.covis_graph)
        return len(to_cull)

    def _create_points_from_depth(self, keyframe, depth_image=None):
        """
        Depth available: unproject unmatched keypoints directly. No parallax
        needed.

        Was previously unbounded -- ~1000 new points created per keyframe,
        forever, with no fusion against existing points (SearchInNeighbors
        is a separate, not-yet-implemented piece), so the same physical
        landmark got recreated at every keyframe and the map grew without
        bound. That flood of near-duplicate, mostly-single-observation
        points was also the single biggest contributor to the BA hang: a
        point seen once contributes 2 residuals but 3 free unknowns, so
        thousands of them made the old dense solver wander through a
        rank-deficient null space.

        Fix, following ORB-SLAM3's own rule: cap new points per keyframe,
        create closest-depth-first, gate to a working depth range, and
        reject points sitting on a depth discontinuity (a median/std check
        over a small patch -- this is what kills "flying points" at object
        edges, the classic RGB-D artifact).
        """
        candidates = []
        for i in range(keyframe.n):
            if keyframe.map_point_ids[i] is not None:
                continue
            z = keyframe.depths[i]
            if z <= 0 or z < self.depth_min or z > self.depth_max:
                continue
            if depth_image is not None and not self._depth_patch_ok(keyframe, depth_image, i, z):
                continue
            candidates.append((z, i))

        # closest first -- nearer depth is more reliable on this sensor
        candidates.sort(key=lambda t: t[0])
        candidates = candidates[:self.max_new_points_per_kf]

        new_points = []
        for _, i in candidates:
            p3d = keyframe.unproject_keypoint(i)
            if p3d is None:
                continue
            mp = MapPoint(p3d, keyframe.descriptors[i], ref_keyframe_id=keyframe.kf_seq)
            mp.add_observation(keyframe.id, i)
            keyframe.map_point_ids[i] = mp.id
            new_points.append(mp)
        return new_points

    def _depth_patch_ok(self, keyframe, depth_image, kp_idx, center_depth):
        """
        Sample a small window around the keypoint in the raw depth image and
        reject if local depth variance is high relative to the depth itself
        -- i.e. the keypoint sits on a depth discontinuity (an object edge)
        rather than a flat surface, which is where RGB-D "flying points"
        come from.
        """
        kp = keyframe.keypoints[kp_idx]
        u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
        r = self.depth_patch_radius
        h, w = depth_image.shape[:2]
        u0, u1 = max(0, u - r), min(w, u + r + 1)
        v0, v1 = max(0, v - r), min(h, v + r + 1)
        patch = depth_image[v0:v1, u0:u1].astype(np.float64) * self.camera.depth_scale
        valid = patch[patch > 0]
        if valid.size < 3:
            return False
        rel_std = float(np.std(valid)) / max(center_depth, 1e-6)
        return rel_std <= self.depth_rel_std_max

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

                mp = MapPoint(p, keyframe.descriptors[ci], ref_keyframe_id=keyframe.kf_seq)
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

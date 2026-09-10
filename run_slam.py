"""
run_slam.py — the orchestrator.

Replaces: mono_video.cc + System.cc's thread launching.
Real ORB-SLAM3 runs Tracking / LocalMapping / LoopClosing on three parallel
threads. We run them sequentially in one loop — simpler to follow, and the
result is identical for offline processing.

Two input modes:
  --realsense       live D435 (RGB-D, uses depth for instant init)
  --frames <dir>    replay pre-extracted PNG frames (monocular)

Usage:
  python3 run_slam.py --realsense --seconds 30
  python3 run_slam.py --frames ~/orb_scratch/IMG_1112 --calib calibration/x.json
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from camera import Camera
from extractor import Extractor
from matcher import Matcher
from frame import Frame
from atlas import Atlas
from tracking import Tracking
from local_mapping import LocalMapping
from orb_vocabulary import ORBVocabulary
from keyframe_database import KeyFrameDatabase
import relocalization
import pose_graph
import merge_maps
import fusion
import covisibility
from loop_closing import LoopClosing
import initializer
from bundle_adjust import local_bundle_adjust, local_inertial_bundle_adjust, pose_only_optimize
# Phase 2: GTSAM backend is optional -- only imported if config actually
# requests it, so `gtsam` is not a hard dependency for anyone still on the
# scipy backend. See bundle_adjust_gtsam.py's docstring for why this
# exists and PROGRESS.md for the three bugs found verifying it.
try:
    from bundle_adjust_gtsam import local_bundle_adjust_gtsam
except ImportError:
    local_bundle_adjust_gtsam = None
import imu
import imu_init
from isam2_backend import Isam2Backend
from config import load_config


class SLAMSystem:
    def __init__(self, camera, use_depth=False, verbose=True, use_imu=False, config_path=None):
        self.camera = camera
        self.use_depth = use_depth
        self.verbose = verbose
        self.use_imu = use_imu
        self.cfg = load_config(config_path)

        # BUGFIX: config.py was loaded (self.cfg) but every module below
        # used to be constructed with hardcoded defaults instead of cfg's
        # values -- e.g. cfg["loop_closing"]["vocab_words"]=10000 existed on
        # disk but Vocabulary(n_words=64) below ignored it completely, and
        # nothing in cfg["extractor"]/["tracking"]/["local_mapping"] had any
        # effect at all. Wired through properly now so config.yaml edits
        # actually change behavior. See PROGRESS.md.
        self.extractor = Extractor(**self.cfg["extractor"])
        self.matcher = Matcher()
        self.atlas = Atlas()
        self.tracking = Tracking(camera, self.extractor, self.matcher,
                                 self.atlas.active_map, **self.cfg["tracking"])
        self.local_mapping = LocalMapping(camera, self.matcher,
                                          self.atlas.active_map,
                                          extractor=self.extractor,
                                          **self.cfg["local_mapping"])
        # Shared covisibility graph: LocalMapping rebuilds this dict IN
        # PLACE (see covisibility.py) after every keyframe, so Tracking
        # always sees the latest version through this one reference.
        self.tracking.covis_graph = self.local_mapping.covis_graph
        # Phase 5: real ORBvoc-format vocabulary replaces the KMeans
        # placeholder entirely -- see orb_vocabulary.py's module docstring
        # for why, and setup_vocabulary.py for the one-time fetch script.
        # Loading is NOT wrapped in a silent try/except: a facility-
        # mapping run without a real vocabulary (BoW is an explicit hard
        # requirement for this project) should fail loudly and tell the
        # person exactly what to run, not silently degrade to something
        # worse. If you deliberately want to run without it for a quick
        # test, catch FileNotFoundError yourself around SLAMSystem(...).
        self.vocab = ORBVocabulary()
        self.vocab.load()
        self.keyframe_db = KeyFrameDatabase(self.vocab)
        self.loop_closer = LoopClosing(camera, self.matcher, self.vocab,
                                       keyframe_db=self.keyframe_db,
                                       atlas=self.atlas,
                                       min_keyframe_gap=self.cfg["loop_closing"]["min_keyframe_gap"],
                                       consistency_checks=self.cfg["loop_closing"]["consistency_checks"])

        # ── Visual-inertial state (Phase 3/4) ───────────────────────────
        # imu_preint: the RUNNING preintegration accumulator since the last
        # keyframe. Created once the first keyframe exists, reset every
        # time a new keyframe is marked (see process() below). bias_* are
        # carried forward as constants once imu_init.py succeeds (see that
        # module's docstring for why bias isn't re-estimated online here).
        self.imu_preint = None
        self.imu_raw_buffer = []     # accumulated synchronized [t,gx,gy,gz,ax,ay,az] rows
        self._last_imu_t = None
        self.bias_gyro = np.zeros(3)
        self.bias_accel = np.zeros(3)
        self.last_ok_timestamp = None
        # PHASE 8: raw per-stream buffers for the LIVE realsense capture
        # path (run_realsense()) -- accel and gyro arrive as separate
        # async streams at different native rates, so they're buffered
        # here and synchronized (imu.synchronize()) once per process()
        # call rather than per-sample. Unused/empty for any other input
        # path (run_dataset.py already hands over pre-synchronized rows
        # loaded straight from a recording's imu.csv).
        self._live_accel_buf = []
        self._live_gyro_buf = []

        self.frames = []
        self.init_candidate = None
        # PHASE 9: one Isam2Backend per Atlas map (mirrors the existing
        # per-map tracking.set_map()/local_mapping.set_map() pattern),
        # created lazily on that map's first keyframe. Only used for the
        # VISUAL-ONLY path (backend config == "isam2" and IMU not yet
        # initialized on this run) -- see the per-keyframe block in
        # process() and this class's docstring-level notes on why IMU
        # integration into this same graph is explicitly OUT of scope
        # for this phase (Phase 8's local_inertial_bundle_adjust keeps
        # handling the IMU-active case, unintegrated, until a later
        # phase joins them).
        self.isam2_backends = {}   # Map.id -> Isam2Backend
        self.consecutive_lost = 0
        self.stats = {'tracked': 0, 'lost': 0, 'keyframes': 0,
                      'points_created': 0, 'points_culled': 0, 'loops': 0,
                      'points_fused': 0, 'keyframes_culled': 0,
                      'imu_init_attempts': 0, 'relocalizations': 0}

    def process(self, image, timestamp, depth_image=None, imu_samples=None):
        """
        imu_samples: optional (K,7) array of synchronized [t,gx,gy,gz,ax,ay,az]
        rows (see imu.py) covering the interval since the previous process()
        call. Already rotated into the camera frame. Ignored entirely
        unless use_imu=True.
        """
        frame = Frame(image, timestamp, self.camera, self.extractor,
                      depth_image=depth_image)
        
        # ── Diagnostic output immediately after creating each Frame ──
        # BUGFIX: Frame has no `valid_depth_count` attribute -- this used to
        # silently print 0 forever via getattr's fallback. n_valid_depths()
        # is the real accessor (see frame.py).
        if self.verbose:
            print(f"[RGBD] Frame {frame.id}: keypoints={len(frame.keypoints)}, "
                  f"valid_depth={frame.n_valid_depths()}")

        self.frames.append(frame)
        world_map = self.atlas.active_map

        if self.use_imu and imu_samples is not None and len(imu_samples) > 0:
            self._integrate_imu(imu_samples)

        # ── initialisation ───────────────────────────────────────────────
        if self.tracking.state == "NOT_INITIALIZED":
            if self.use_depth:
                ok, new_pts, fail_reason = initializer.init_rgbd(
                    frame, world_map, min_points=self.cfg["initializer"]["rgbd_min_points"])
                if ok:
                    self.tracking.state = "OK"
                    self.tracking.last_frame = frame
                    self.tracking.mark_keyframe(frame)
                    self.local_mapping.recent_points.extend(new_pts)
                    self.stats['keyframes'] += 1
                    self.stats['points_created'] += len(new_pts)
                    self.last_ok_timestamp = timestamp
                    self.keyframe_db.add_keyframe(frame)   # Phase 5
                    if self.use_imu:
                        self._start_imu_segment()
                    # PHASE 9 BUGFIX: the init keyframe used to never
                    # reach isam2_backend.py at all -- this whole branch
                    # `return`s below, before the ordinary per-keyframe
                    # block (where add_keyframe() normally gets called)
                    # is ever reached. That meant the FIRST keyframe --
                    # which local_mapping/init_rgbd anchors most of the
                    # map's early points to -- was silently absent from
                    # the graph, so whichever keyframe reached
                    # add_keyframe() FIRST became the gauge anchor
                    # instead, anchored at ITS OWN (not the true origin)
                    # pose, with every point the true first keyframe
                    # created inserted fresh, single-factor, in one large
                    # batch. Found via a real IndeterminantLinearSystem-
                    # Exception on a realistic multi-fragment synthetic
                    # sequence -- see PROGRESS.md's Phase 9 section.
                    if self.cfg["bundle_adjust"].get("backend", "scipy") == "isam2":
                        self._get_isam2_backend(world_map).add_keyframe(frame, world_map)
                    self._log(f"[init] RGB-D success at frame {frame.id} "
                              f"({len(new_pts)} points)")
                else:
                    self._log(f"[init] RGB-D failed at frame {frame.id}: {fail_reason}")
            else:
                if self.init_candidate is None:
                    self.init_candidate = frame
                    return frame
                ok, new_pts = initializer.init_monocular(
                    self.init_candidate, frame, self.matcher,
                    self.camera, world_map)
                if ok:
                    self.tracking.state = "OK"
                    self.tracking.last_frame = frame
                    self.tracking.mark_keyframe(frame)
                    self.local_mapping.recent_points.extend(new_pts)
                    self.stats['keyframes'] += 2
                    self.stats['points_created'] += len(new_pts)
                    self.last_ok_timestamp = timestamp
                    self.keyframe_db.add_keyframe(self.init_candidate)   # Phase 5
                    self.keyframe_db.add_keyframe(frame)                 # Phase 5
                    if self.use_imu:
                        self._start_imu_segment()
                    # PHASE 9 BUGFIX: same gap as the RGB-D init branch
                    # above -- both init keyframes must be added, in
                    # chronological order, so the FIRST one (not
                    # whichever keyframe happens to reach the ordinary
                    # per-keyframe block first) becomes the gauge anchor.
                    if self.cfg["bundle_adjust"].get("backend", "scipy") == "isam2":
                        isam2 = self._get_isam2_backend(world_map)
                        isam2.add_keyframe(self.init_candidate, world_map)
                        isam2.add_keyframe(frame, world_map)
                    self._log(f"[init] mono success at frame {frame.id} "
                              f"({len(new_pts)} points)")
                else:
                    self.init_candidate = frame   # slide the window forward
                    if frame.id % 30 == 0:
                        self._log(f"[init] waiting for parallax... frame {frame.id}")
            return frame

        # ── tracking ─────────────────────────────────────────────────────
        ok = self.tracking.track(frame, imu_preint=self.imu_preint if self.use_imu else None)
        if ok:
            self.last_ok_timestamp = timestamp
        if not ok:
            self.stats['lost'] += 1
            self.consecutive_lost += 1
            self._log(f"[track] LOST at frame {frame.id} "
                      f"(consecutive: {self.consecutive_lost})")

            # Atlas behaviour: after sustained loss, abandon and start fresh.
            # With IMU initialized, give the pipeline a TIME-based grace
            # window instead of a fixed frame count -- tracking.py already
            # keeps propagating frame.pose via IMU alone during this
            # window (state RECENTLY_LOST), so a few seconds of bad visual
            # conditions (motion blur, a blank wall) can recover instead of
            # immediately abandoning the map.
            elapsed_lost = timestamp - self.last_ok_timestamp if self.last_ok_timestamp is not None else 0.0
            if self.use_imu and self.tracking.imu_initialized:
                give_up = (elapsed_lost >= self.cfg["imu"]["recently_lost_max_seconds"]
                           and self.consecutive_lost >= 3)
            else:
                give_up = self.consecutive_lost >= 10

            # Phase 5: attempt relocalization BEFORE giving up. This is
            # the direct fix for the original "no Relocalization()"
            # gap flagged from this project's very first forensic pass --
            # every sustained tracking loss used to permanently fragment
            # the Atlas with zero recovery attempt. Tried on every LOST
            # frame, not just once give_up triggers: a successful match
            # here means we never fragment in the first place, whether
            # the recognized place is in the CURRENTLY active map (no
            # fragmentation ever happens) or an older, already-abandoned
            # one (Atlas resumes it instead of starting yet another
            # fragment -- see relocalization.py's docstring for why this
            # switches maps rather than attempting a full merge, which is
            # real, separate work deferred to Phase 6).
            exclude_ids = {self.tracking.last_keyframe.id} if self.tracking.last_keyframe else None
            relocalized, matched_map = relocalization.try_relocalize(
                frame, self.keyframe_db, self.atlas, self.camera, self.matcher,
                min_inliers=self.cfg["relocalization"]["min_inliers"],
                top_n_candidates=self.cfg["relocalization"]["top_n_candidates"],
                exclude_ids=exclude_ids)

            if relocalized:
                self._log(f"[reloc] recovered at frame {frame.id} into "
                          f"map {matched_map.id} after {self.consecutive_lost} lost frames")
                self.tracking.set_map(matched_map)
                self.local_mapping.set_map(matched_map)
                self.tracking.covis_graph = self.local_mapping.covis_graph
                self.tracking.state = "OK"
                self.tracking.last_frame = frame
                self.tracking.velocity = None   # motion model has no basis to trust yet
                self.consecutive_lost = 0
                self.last_ok_timestamp = timestamp
                self.stats['relocalizations'] = self.stats.get('relocalizations', 0) + 1
                return frame

            if give_up:
                self._log(f"[atlas] starting NEW MAP after "
                          f"{self.consecutive_lost} lost frames ({elapsed_lost:.2f}s)")
                new_map = self.atlas.start_new_map()
                self.tracking.set_map(new_map)
                self.local_mapping.set_map(new_map)
                self.tracking.covis_graph = self.local_mapping.covis_graph
                self.tracking.state = "NOT_INITIALIZED"
                self.tracking.last_frame = None
                self.tracking.last_keyframe = None
                self.tracking.velocity = None
                self.tracking.imu_initialized = False
                self.init_candidate = None
                self.consecutive_lost = 0
                self.imu_preint = None
            return frame

        self.consecutive_lost = 0
        self.stats['tracked'] += 1
        pose_only_optimize(frame, world_map, self.camera)

        # ── keyframe -> local mapping ────────────────────────────────────
        if self.tracking.needs_new_keyframe(frame):
            new_pts, n_culled, n_fused = self.local_mapping.process_new_keyframe(
                frame, use_depth=self.use_depth, depth_image=depth_image)
            self.tracking.mark_keyframe(frame)
            self.stats['keyframes'] += 1
            self.stats['points_created'] += len(new_pts)
            self.stats['points_culled'] += n_culled
            self.stats['points_fused'] += n_fused
            self.keyframe_db.add_keyframe(frame)   # Phase 5

            if self.use_imu:
                self._finalize_imu_segment(frame)

            # KeyFrameCulling: drop redundant keyframes so covisibility
            # rebuilds / BA windows / local-map matching don't grow forever
            # over a facility-length walk. Cheap enough to run every
            # keyframe; internally protects the most recent few.
            #
            # SKIPPED when IMU is active: each keyframe's imu_preint spans
            # from its IMMEDIATE predecessor at recording time. If that
            # predecessor gets culled, the chain imu_init.py and
            # local_inertial_bundle_adjust walk (consecutive keyframes in
            # world_map.keyframes) would silently desync from what each
            # segment actually covers. Fixing that properly means re-
            # concatenating preintegration segments across a cull, which
            # is real work deferred for now -- see PROGRESS notes.
            if not self.use_imu:
                n_kf_culled, culled_ids = self.local_mapping.cull_keyframes()
                self.stats['keyframes_culled'] += n_kf_culled
                for kf_id in culled_ids:   # Phase 5: keep the inverted index consistent
                    self.keyframe_db.remove_keyframe(kf_id)

            # PHASE 9: real-time back end. The windowed local_bundle_
            # adjust_gtsam()/local_bundle_adjust() batch calls below are
            # RETIRED from this live per-keyframe path for the visual-
            # only case -- replaced by one continuous incremental iSAM2
            # graph (isam2_backend.py), which is what the real-time
            # architecture reframe (PHASE7_ARCHITECTURE_V2_REALTIME.md)
            # calls for and is now this project's default. They're kept
            # in the codebase (not deleted) for comparison/fallback --
            # select the old path via config["bundle_adjust"]["backend"]
            # = "gtsam" or "scipy" instead of "isam2" if you need it.
            #
            # SCOPE BOUNDARY (explicit, not an oversight): IMU factors
            # joining this SAME graph is deliberately NOT done this
            # phase -- see PHASE7_ARCHITECTURE_V2_REALTIME.md's
            # Workstream E, still future work. While IMU is active on a
            # given run, Phase 8's local_inertial_bundle_adjust keeps
            # handling optimization instead, exactly as it already did --
            # the iSAM2 backend simply stops receiving new keyframes for
            # that map once IMU takes over, so the two never fight over
            # the same pose variables.
            n_kf = world_map.n_keyframes()
            if self.use_imu and not self.tracking.imu_initialized:
                self._try_imu_init(world_map)

            backend = self.cfg["bundle_adjust"].get("backend", "scipy")
            use_isam2 = (backend == "isam2" and
                        not (self.use_imu and self.tracking.imu_initialized))

            if self.use_imu and self.tracking.imu_initialized:
                if n_kf >= 3:
                    local_inertial_bundle_adjust(
                        world_map, self.camera, gravity=self.tracking.gravity,
                        bias_gyro=self.bias_gyro, bias_accel=self.bias_accel,
                        window=self.cfg["imu"]["ba_window"], verbose=self.verbose)
            elif use_isam2:
                isam2 = self._get_isam2_backend(world_map)
                n_pts, n_obs, dt = isam2.add_keyframe(frame, world_map)
                if self.verbose:
                    print(f"[iSAM2] kf {frame.id}: +{n_pts} new pts, "
                         f"+{n_obs} obs, update {dt*1000:.1f}ms "
                         f"({isam2.n_updates} total updates)")
            elif n_kf >= 3:
                # gtsam/scipy fallback comparison path (backend != "isam2").
                if backend == "gtsam" and local_bundle_adjust_gtsam is not None:
                    local_bundle_adjust_gtsam(world_map, self.camera,
                                              window=self.cfg["bundle_adjust"]["window"],
                                              max_iter=self.cfg["bundle_adjust"]["max_iter"],
                                              min_obs_to_optimize=self.cfg["bundle_adjust"]["min_obs_to_optimize"],
                                              huber_f_scale=self.cfg["bundle_adjust"]["huber_f_scale"],
                                              verbose=self.verbose)
                else:
                    if backend == "gtsam" and self.verbose:
                        print("[BA] config requested gtsam backend but gtsam "
                             "is not installed -- falling back to scipy")
                    local_bundle_adjust(world_map, self.camera,
                                        window=self.cfg["bundle_adjust"]["window"],
                                        max_iter=self.cfg["bundle_adjust"]["max_iter"],
                                        min_obs_to_optimize=self.cfg["bundle_adjust"]["min_obs_to_optimize"],
                                        huber_f_scale=self.cfg["bundle_adjust"]["huber_f_scale"],
                                        verbose=self.verbose)

            # ── loop closing ─────────────────────────────────────────────
            # Phase 6: loop closing is now Atlas-wide (keyframe_database.py)
            # with 3D-3D geometric verification (loop_verification.py),
            # and a confirmed loop is ACTUALLY CORRECTED, not just logged
            # -- the "NOT corrected" era ends here. Two distinct cases,
            # branched on whether the matched keyframe is in the SAME map
            # (intra-map: pose_graph.py redistributes the loop error
            # across the path) or a DIFFERENT, previously-fragmented map
            # (cross-map: merge_maps.py actually folds it back in, which
            # relocalization.py's map-switching alone never did).
            #
            # Extracted into _handle_loop_result() (see below) rather than
            # inlined here: test_phase6_gate.py's end-to-end test needs to
            # invoke this exact logic directly, independent of whether the
            # n_kf%10==0 cadence happens to land on the right keyframe in
            # a finite synthetic sequence -- the cadence itself is a
            # legitimate production design choice (bounding how often an
            # expensive Atlas-wide check runs), not something a
            # correctness test should be at the mercy of.
            # PHASE 9: cadence is now configurable
            # (loop_closing.check_every_n_kf), defaulting to 1 (every
            # keyframe) -- the real-time reframe's own justification
            # (PHASE7_ARCHITECTURE_V2_REALTIME.md sec 3/7): a slow-moving
            # robot's keyframes are seconds apart, not the tight-real-
            # time cadence the old hardcoded "every 10th" was designed
            # for, so checking every keyframe is affordable. The
            # `n_kf >= 20` warm-up guard stays -- no point checking
            # before there's a meaningfully large map to match against.
            check_every = self.cfg["loop_closing"].get("check_every_n_kf", 1)
            if n_kf % check_every == 0 and n_kf >= 20:
                result = self.loop_closer.detect_loop(frame, world_map)
                if result is not None:
                    world_map = self._handle_loop_result(frame, world_map, result)
        return frame

    # ── IMU orchestration (Phase 3/4) ───────────────────────────────────

    def _integrate_imu(self, imu_samples):
        """Append raw synchronized samples to the running buffer (used by
        imu_init's observability gate) and integrate them into the RUNNING
        preintegration segment since the last keyframe, if one exists yet."""
        self.imu_raw_buffer.extend(imu_samples.tolist())
        if self.imu_preint is None:
            return
        t_prev = self._last_imu_t
        for row in imu_samples:
            t, gx, gy, gz, ax, ay, az = row
            if t_prev is not None:
                self.imu_preint.integrate_sample([gx, gy, gz], [ax, ay, az], t - t_prev)
            t_prev = t
        self._last_imu_t = t_prev

    def _get_isam2_backend(self, world_map):
        """PHASE 9: get-or-create this map's Isam2Backend instance."""
        backend = self.isam2_backends.get(world_map.id)
        if backend is None:
            isam2_cfg = self.cfg.get("isam2", {})
            backend = Isam2Backend(
                self.camera,
                virtual_baseline=self.cfg["bundle_adjust"].get("virtual_baseline", 0.05),
                huber_f_scale=self.cfg["bundle_adjust"]["huber_f_scale"],
                min_obs_to_optimize=self.cfg["bundle_adjust"]["min_obs_to_optimize"],
                relinearize_threshold=isam2_cfg.get("relinearize_threshold", 0.1),
                relinearize_skip=isam2_cfg.get("relinearize_skip", 1),
                anchor_sigma_pos=isam2_cfg.get("anchor_sigma_pos", 1e-3),
                anchor_sigma_rot=isam2_cfg.get("anchor_sigma_rot", 1e-2),
                odom_sigma_pos=isam2_cfg.get("odom_sigma_pos", 0.05),
                odom_sigma_rot=isam2_cfg.get("odom_sigma_rot", 0.05),
                # POST-PHASE-9 FIX: see isam2_backend.py's module
                # docstring -- these five are new, tuned against the
                # first real D435i session rather than only synthetic
                # data. level_sigma2 comes straight from this SLAM
                # system's own extractor pyramid, so octave-scaled
                # observation noise always matches whatever nfeatures/
                # scale_factor/nlevels config is actually in use.
                base_pixel_sigma=isam2_cfg.get("base_pixel_sigma", 1.5),
                level_sigma2=self.extractor.level_sigma2,
                odom_confidence_ref_points=isam2_cfg.get("odom_confidence_ref_points", 100),
                odom_confidence_max_scale=isam2_cfg.get("odom_confidence_max_scale", 5.0),
                reset_after_n_consecutive_failures=isam2_cfg.get(
                    "reset_after_n_consecutive_failures", 3))
            self.isam2_backends[world_map.id] = backend
        return backend

    def _start_imu_segment(self):
        """Begin accumulating a fresh preintegration segment from the
        just-created (first) keyframe forward."""
        self.imu_preint = imu.Preintegration(
            self.bias_gyro, self.bias_accel,
            noise_gyro=self.cfg["imu"]["noise_gyro"],
            noise_accel=self.cfg["imu"]["noise_accel"])
        self._last_imu_t = None

    def _finalize_imu_segment(self, keyframe):
        """Attach the just-completed segment to the new keyframe and start
        the next one."""
        if self.imu_preint is None:
            self._start_imu_segment()
            return
        keyframe.imu_preint = self.imu_preint
        keyframe.bias_gyro = self.bias_gyro.copy()
        keyframe.bias_accel = self.bias_accel.copy()
        self._start_imu_segment()

    def _handle_loop_result(self, frame, world_map, result):
        """
        Phase 6: apply a CONFIRMED loop-closure result -- either intra-map
        pose-graph correction or cross-map merging. Extracted from
        process()'s inline loop-closing block specifically so
        test_phase6_gate.py can invoke this exact logic directly,
        independent of whether the production n_kf%10==0 cadence happens
        to land on the right keyframe within a finite synthetic test
        sequence (see that file's test_end_to_end_merge for the full
        reasoning). Returns the (possibly updated, if a merge happened)
        world_map -- callers MUST use the returned value, not assume the
        one they passed in is still current.
        """
        matched_kf, matched_map, sim, inliers, R, t, s = result
        drift = self.loop_closer.measure_drift(frame, matched_kf)
        self.stats['loops'] += 1

        # PHASE 9: same "is iSAM2 the active real-time back end right
        # now" check used in process()'s per-keyframe block -- IMU-active
        # runs still fall back to pose_graph.py's scipy correction (see
        # that block's SCOPE BOUNDARY comment for why).
        use_isam2 = (self.cfg["bundle_adjust"].get("backend", "scipy") == "isam2" and
                    not (self.use_imu and self.tracking.imu_initialized))

        if matched_map is world_map:
            loop_edge = pose_graph.compute_loop_edge(
                frame, matched_kf, self.matcher, self.camera, world_map)
            if loop_edge is not None:
                T_rel, n_pnp_inliers = loop_edge
                if use_isam2:
                    isam2 = self._get_isam2_backend(world_map)
                    dt = isam2.add_loop_factor(
                        matched_kf, frame, T_rel, loop_weight=min(100.0, n_pnp_inliers),
                        world_map=world_map)
                    # Loop-seam fusion, same as pose_graph.py always did
                    # (these two keyframes were never naturally covisible
                    # before the loop closed) -- iSAM2 corrects POSES via
                    # the factor above, but doesn't discover duplicate
                    # points across the seam by itself.
                    n_levels = self.extractor.nlevels
                    scale_factor = self.extractor.scale_factor
                    ad_hoc_graph = {matched_kf.id: {frame.id: 999}, frame.id: {matched_kf.id: 999}}
                    n_fused = fusion.search_in_neighbors(frame, world_map, ad_hoc_graph, self.camera,
                                                         n_levels=n_levels, scale_factor=scale_factor)
                    covisibility.build_covisibility_graph(world_map, graph=self.local_mapping.covis_graph)
                    self._log(f"[LOOP] kf {frame.id} <-> kf {matched_kf.id} "
                              f"(same map) | sim={sim:.3f} inliers={inliers} | "
                              f"drift={drift:.3f}m -> "
                              f"{'CORRECTED via iSAM2 (' + str(round(dt*1000,1)) + 'ms, ' + str(n_fused) + ' fused)' if dt is not None else 'iSAM2 update FAILED (see [iSAM2] WARNING above) -- pose left uncorrected this round'}")
                else:
                    pg_stats = pose_graph.optimize_pose_graph(
                        world_map, frame, matched_kf, T_rel,
                        loop_weight=min(100.0, n_pnp_inliers),
                        camera=self.camera, covis_graph=self.local_mapping.covis_graph,
                        extractor=self.extractor, verbose=self.verbose)
                    self._log(f"[LOOP] kf {frame.id} <-> kf {matched_kf.id} "
                              f"(same map) | sim={sim:.3f} inliers={inliers} | "
                              f"drift={drift:.3f}m -> CORRECTED "
                              f"({pg_stats['n_keyframes_corrected'] if pg_stats else 0} kfs, "
                              f"{pg_stats['n_points_corrected'] if pg_stats else 0} pts)")
            else:
                self._log(f"[LOOP] kf {frame.id} <-> kf {matched_kf.id} "
                          f"(same map) | drift={drift:.3f}m -> correction "
                          f"attempted but compute_loop_edge found too few "
                          f"inliers, skipped")
            return world_map

        merge_stats = merge_maps.merge_maps(
            self.atlas, matched_map, world_map, R, t, s,
            anchor_kf_a=frame, anchor_kf_b=matched_kf,
            camera=self.camera, matcher=self.matcher,
            extractor=self.extractor,
            covis_graph=self.local_mapping.covis_graph,
            verbose=self.verbose)
        # world_map (the just-absorbed fragment) no longer exists as a
        # separate map -- everything downstream of this point needs to
        # operate on matched_map, the survivor. Mirrors relocalization.py's
        # own post-switch bookkeeping exactly.
        absorbed_kfs = list(world_map.keyframes)   # merge_maps leaves this list intact -- see its docstring
        world_map = matched_map
        self.tracking.set_map(world_map)
        self.local_mapping.set_map(world_map)
        self.tracking.covis_graph = self.local_mapping.covis_graph

        if use_isam2:
            # PHASE 9: bring the absorbed map's keyframes into the
            # SURVIVOR'S iSAM2 graph -- absorb_map() connects the first
            # one via bridge_kf (matched_kf, already in this graph) so
            # it's never left as a disconnected component (see
            # isam2_backend.py's absorb_map() docstring), then this
            # explicit add_loop_factor() call afterward adds the REAL,
            # PnP-measured transform between the actual matched pair as
            # a tighter constraint -- the bridge only prevents a crash,
            # this is what actually pulls the seam tight.
            isam2 = self._get_isam2_backend(world_map)
            isam2.absorb_map(absorbed_kfs, world_map, bridge_kf=matched_kf)
            # anchor_kf_a (frame) and anchor_kf_b (matched_kf) both now
            # live in world_map's coordinate frame post-merge.
            T_rel_bridge = np.linalg.inv(matched_kf.pose) @ frame.pose
            isam2.add_loop_factor(matched_kf, frame, T_rel_bridge,
                                  loop_weight=min(100.0, inliers), world_map=world_map)

        self._log(f"[LOOP] kf {frame.id} <-> kf {matched_kf.id} "
                  f"(CROSS-MAP) | sim={sim:.3f} inliers={inliers} | "
                  f"drift={drift:.3f}m -> MERGED "
                  f"({merge_stats['n_keyframes_after']} kfs, "
                  f"{merge_stats['n_points_after']} pts, "
                  f"scale={merge_stats['scale_applied']:.3f})")
        return world_map

    def _try_imu_init(self, world_map):
        """
        PHASE 8 BUGFIX: this method's docstring and body existed in the
        code, but the `def _try_imu_init(self, world_map):` signature line
        itself was MISSING -- the entire block was dead, dangling code
        sitting after _handle_loop_result's own `return`, not a real
        method. `hasattr(SLAMSystem, '_try_imu_init')` returned False.
        The call site in process() (`if self.use_imu and not self.
        tracking.imu_initialized: self._try_imu_init(world_map)`) would
        raise AttributeError on the very first keyframe processed with
        use_imu=True -- meaning IMU initialization has never successfully
        run, in any test, on any recording, ever, in this project's
        history. Nothing caught this because no gate test and no field
        log ever actually set use_imu=True end-to-end (see PROGRESS.md's
        Phase 8 section for the full account and test_phase8_gate.py for
        the regression test that exercises this directly).

        Attempt staged inertial initialization (imu_init.py) once enough
        keyframes with attached preintegration segments exist. On success,
        assigns velocities to those keyframes, stores gravity/bias, and
        flips tracking.imu_initialized so future frames get IMU prediction
        and future keyframes get inertial BA.
        """
        kfs = [kf for kf in world_map.keyframes if kf.imu_preint is not None or kf.kf_seq == 0]
        kfs = sorted(kfs, key=lambda kf: kf.kf_seq)
        min_kf = self.cfg["imu"]["init_min_keyframes"]
        if len(kfs) < min_kf:
            return
        if kfs[-1].timestamp - kfs[0].timestamp < self.cfg["imu"]["init_window_seconds"]:
            return

        self.stats['imu_init_attempts'] += 1
        sync_samples = np.asarray(self.imu_raw_buffer) if self.imu_raw_buffer else np.zeros((0, 7))
        result = imu_init.initialize(
            kfs, sync_samples,
            gravity_mag=self.cfg["imu"]["gravity_magnitude"],
            min_gyro_std=self.cfg["imu"]["observability_min_gyro_std"],
            min_accel_std=self.cfg["imu"]["observability_min_accel_std"])

        if not result["success"]:
            self._log(f"[imu-init] not ready yet: {result['reason']}")
            return

        for kf, v in zip(kfs, result["velocities"]):
            kf.velocity = v
        self.bias_gyro = result["bias_gyro"]
        self.bias_accel = result["bias_accel"]
        self.tracking.gravity = result["gravity"]
        self.tracking.imu_initialized = True
        self._log(f"[imu-init] SUCCESS: gravity={np.round(result['gravity'], 3)} "
                  f"|g|={np.linalg.norm(result['gravity']):.3f} "
                  f"bias_gyro={np.round(result['bias_gyro'], 4)} "
                  f"bias_accel={np.round(result['bias_accel'], 4)}")

        # Fold the initialization straight into the map once (stands in for
        # ORB-SLAM3's VIBA1/VIBA2 re-refinement passes -- see
        # imu_init.py's docstring for why those aren't implemented here).
        local_inertial_bundle_adjust(world_map, self.camera, gravity=self.tracking.gravity,
                                     bias_gyro=self.bias_gyro, bias_accel=self.bias_accel,
                                     window=len(kfs), verbose=self.verbose)

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ── outputs ──────────────────────────────────────────────────────────

    def save_trajectory_plot(self, path="trajectory_python.png"):
        poses = [f.pose for f in self.frames if f.pose is not None]
        if len(poses) < 2:
            print("Not enough poses to plot.")
            return None

        c = np.array([p[:3, 3] for p in poses])
        kf_c = np.array([kf.camera_center() for m in self.atlas.maps
                            for kf in m.keyframes if kf.pose is not None])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (i, j), (xl, yl), title in (
                (axes[0], (0, 2), ("X (m)", "Z (m)"), "Bird's-eye (X-Z)"),
                (axes[1], (0, 1), ("X (m)", "Y (m)"), "Front (X-Y)")):
            ax.plot(c[:, i], c[:, j], '-', lw=1.2, color='tab:orange', label='trajectory')
            if len(kf_c):
                ax.scatter(kf_c[:, i], kf_c[:, j], s=14, color='tab:blue',
                           zorder=4, label='keyframes')
            ax.scatter([c[0, i]], [c[0, j]], s=70, c='green', marker='o',
                       edgecolors='k', zorder=5, label='start')
            ax.scatter([c[-1, i]], [c[-1, j]], s=70, c='red', marker='s',
                       edgecolors='k', zorder=5, label='end')
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(alpha=0.3); ax.legend(fontsize=8)

        fig.suptitle("Python ORB-SLAM3 reimplementation")
        fig.tight_layout()
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {path}")
        return path

    def save_map_ply(self, path="map_points.ply"):
        pts = [mp.position for m in self.atlas.maps
               for mp in m.good_map_points()]
        if not pts:
            return None
        pts = np.asarray(pts)
        with open(path, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for p in pts:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        print(f"Saved {path} ({len(pts)} points)")
        return path

    def print_summary(self):
        n_posed = sum(1 for f in self.frames if f.pose is not None)
        total = len(self.frames)
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        print(f"Frames processed:    {total}")
        print(f"Frames with pose:    {n_posed} ({100.0 * n_posed / max(1, total):.1f}%)")
        print(f"Tracking successes:  {self.stats['tracked']}")
        print(f"Tracking losses:     {self.stats['lost']}")
        print(f"Keyframes:           {self.stats['keyframes']}")
        print(f"Keyframes culled:    {self.stats['keyframes_culled']}")
        print(f"Map points created:  {self.stats['points_created']}")
        print(f"Map points culled:   {self.stats['points_culled']}")
        print(f"Map points fused:    {self.stats['points_fused']}")
        print(f"Loop closures found: {self.stats['loops']}")
        print()
        print(self.atlas.summary())

        poses = [f.pose for f in self.frames if f.pose is not None]
        if len(poses) >= 2:
            c = np.array([p[:3, 3] for p in poses])
            path_len = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
            net = float(np.linalg.norm(c[-1] - c[0]))
            print(f"\nPath length:      {path_len:.3f}"
                  f"{' m' if self.use_depth else ' (arbitrary units)'}")
            print(f"Net displacement: {net:.3f}"
                  f"{' m' if self.use_depth else ' (arbitrary units)'}")
            if not self.use_depth:
                print("  NOTE: monocular = no absolute scale. Distances are "
                      "relative only.")
        print("=" * 60)


# ── runners ──────────────────────────────────────────────────────────────

def run_realsense(args):
    """
    Live capture. Default path tracks on the COLOR image (fixed: intrinsics
    now genuinely come from the color stream that gets aligned-to and fed
    to the extractor, and baseline comes from real extrinsics -- see
    camera.py's docstring for what was wrong before).

    --ir switches to the recommended pipeline: track on the left infrared
    image instead. Global-shutter IR pairs better with future IMU
    preintegration than the rolling-shutter RGB sensor, and depth is
    natively registered to it (no rs.align needed at all). Only worth it if
    the facility has enough natural texture for emitter-off IR frames to be
    usable -- see the architecture notes. Uses emitter_on_off alternating
    mode: even frames keep the projector pattern (for depth), odd frames
    are clean for ORB (real-time isn't a priority, so halving the rate is fine).

    --imu (PHASE 8): enables the D435i's accel (250Hz) + gyro (200Hz)
    motion streams and feeds synchronized samples into slam.process() every
    iteration. Uses the exact same technique record.py's proven offline
    recorder already uses (motion frames arrive interleaved in the same
    `pipeline.wait_for_frames()` frameset as video/depth when enabled on
    one pipeline -- iterate the frameset and pick out `is_motion_frame()`
    entries, no separate thread or callback needed) -- see record.py's
    docstring for why this works and PROGRESS.md's Phase 8 section for
    why this was never actually wired here before, despite the tracking/
    initialization code that CONSUMES this data having existed since
    before Phase 6.
    """
    import pyrealsense2 as rs

    if args.ir:
        camera = Camera.from_realsense_ir(args.width, args.height, args.fps)
    else:
        camera = Camera.from_realsense(args.width, args.height, args.fps)
    print(camera)
    os.makedirs("calibration", exist_ok=True)
    camera.to_json("calibration/realsense_d435.json")

    slam = SLAMSystem(camera, use_depth=not args.mono, verbose=not args.quiet,
                      use_imu=args.imu)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.ir:
        config.enable_stream(rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps)
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    else:
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        align = rs.align(rs.stream.color)

    if args.imu:
        # PHASE 8: native rate (0 = sensor's own rate -- 250Hz accel /
        # 200Hz gyro on the D435i's BMI085, matching record.py exactly).
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 0)
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 0)

    profile = pipeline.start(config)

    # PHASE 8: put every sensor's timestamps on ONE clock -- critical for
    # IMU (video and motion frames come from physically different sensors
    # with independent internal clocks otherwise). record.py has done this
    # unconditionally for a while; this path only ever did it inside the
    # --ir branch, which meant a plain RGB + --imu run never got it at
    # all. Doing it unconditionally, same as record.py.
    for sensor in profile.get_device().sensors:
        if sensor.supports(rs.option.global_time_enabled):
            sensor.set_option(rs.option.global_time_enabled, 1)

    if args.ir:
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_on_off):
            depth_sensor.set_option(rs.option.emitter_on_off, 1)
            depth_sensor.set_option(rs.option.emitter_enabled, 1)

    if args.imu:
        R_cam_imu = imu.load_T_cam_imu(args.t_cam_imu) if args.t_cam_imu else np.eye(3)
        if args.t_cam_imu is None:
            print("[imu] no --t_cam_imu given -- using identity rotation "
                 "(fine for a quick test; for real use, record a "
                 "T_cam_imu.txt via record.py --imu and pass it here)")

    total_target_frames = args.fps * args.seconds
    print(f"\nStreaming at {args.fps} FPS. Capturing approximately {total_target_frames} target frames ({args.fps} per second over {args.seconds} seconds).\n")

    t0 = time.time()
    processed_count = 0
    frame_counter = 0
    pending_ir_clean = None      # holds the last emitter-off IR frame while we wait for the paired depth
    pending_ir_clean_ts = None   # POST-PHASE-9 FIX: that frame's own device timestamp, paired with it
    last_process_t_raw = None    # POST-PHASE-9 FIX: RAW device-clock cursor for IMU slicing (see below)
    device_t0 = None             # POST-PHASE-9 FIX: device-clock baseline, set from the first processed frame

    try:
        while processed_count < total_target_frames:
            frames = pipeline.wait_for_frames()

            # PHASE 8: pull out any motion samples riding along in this
            # frameset BEFORE the video-frame branching below (which uses
            # `continue` in several places -- IMU samples must not be
            # dropped just because this particular frameset didn't yield
            # a usable video frame this iteration).
            if args.imu:
                for f in frames:
                    if f.is_motion_frame():
                        mf = f.as_motion_frame()
                        md = mf.get_motion_data()
                        ts = mf.get_timestamp() / 1000.0   # ms -> s, RAW device/global clock
                        stream = "accel" if mf.get_profile().stream_type() == rs.stream.accel else "gyro"
                        raw = np.array([ts, md.x, md.y, md.z])
                        (slam._live_accel_buf if stream == "accel"
                         else slam._live_gyro_buf).append(raw)

            if args.ir:
                ir = frames.get_infrared_frame(1)
                depth = frames.get_depth_frame()
                if not ir:
                    continue
                try:
                    emitter_on = bool(ir.get_frame_metadata(
                        rs.frame_metadata_value.frame_laser_power_mode))
                except Exception:
                    emitter_on = (frame_counter % 2 == 0)

                if emitter_on:
                    if pending_ir_clean is not None and depth:
                        color_img = pending_ir_clean
                        # POST-PHASE-9 FIX: use the CLEAN frame's own
                        # device timestamp (captured when it was stored
                        # below), not this later emitter-on frame's --
                        # this is the frame actually being processed.
                        frame_ts_raw = pending_ir_clean_ts
                        depth_img = np.asanyarray(depth.get_data()) if not args.mono else None
                        pending_ir_clean = None
                        pending_ir_clean_ts = None
                    else:
                        frame_counter += 1
                        continue
                else:
                    pending_ir_clean = np.asanyarray(ir.get_data())
                    pending_ir_clean_ts = ir.get_timestamp() / 1000.0
                    frame_counter += 1
                    continue
            else:
                frames = align.process(frames)
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if not color:
                    continue
                frame_counter += 1
                color_img = np.asanyarray(color.get_data())
                frame_ts_raw = color.get_timestamp() / 1000.0
                depth_img = np.asanyarray(depth.get_data()) if (depth and not args.mono) else None

            processed_count += 1
            print(f"Processing target frame {processed_count}/{total_target_frames}")

            # POST-PHASE-9 FIX (found from a real hardware session, not
            # synthetic): this used to be `this_t = time.time() - t0` --
            # the HOST's own wall clock, started at pipeline setup. IMU
            # motion-frame timestamps, both here and in record.py, come
            # from `frame.get_timestamp()` -- the DEVICE's own clock
            # (global_time_enabled puts every sensor on one shared clock,
            # but that clock is the device's, not `time.time()`'s). Those
            # two clocks are NOT the same epoch, so `imu.slice_between()`
            # comparing a tiny host-relative `this_t` against large
            # device-clock IMU timestamps was silently selecting zero (or
            # near-zero) rows on almost every call -- confirmed directly:
            # a 30+ second live session with `--imu` never accumulated
            # enough samples to pass imu_init.py's observability gate,
            # for the ENTIRE session, regardless of how long it ran.
            # record.py never had this bug because it uses
            # `frame.get_timestamp()` for the SAVED video timestamp too
            # (see that file) -- this now matches that proven pattern:
            # both video and IMU timestamps come from the same device
            # clock. `device_t0` is just a cosmetic offset (first frame's
            # raw device timestamp) so the numbers handed to
            # slam.process() and printed in logs start near zero, same as
            # before -- it's applied uniformly, so it doesn't change any
            # elapsed-time math anywhere downstream.
            if device_t0 is None:
                device_t0 = frame_ts_raw
            this_t = frame_ts_raw - device_t0

            imu_samples = None
            if args.imu:
                # PHASE 8: synchronize (interpolate accel onto gyro
                # timestamps, rotate into camera frame -- imu.py's
                # synchronize()) whatever's accumulated since the last
                # process() call, then slice to exactly this interval.
                # Buffers keep everything seen so far rather than being
                # cleared every iteration -- interpolation needs a little
                # context on both sides of the window to avoid edge
                # artifacts, and imu.slice_between already does the exact
                # windowing we need on the synchronized result.
                #
                # POST-PHASE-9 FIX: slicing bounds are now in the SAME
                # raw device-clock terms as the buffered IMU samples
                # themselves (see above) -- `lo_raw`/`frame_ts_raw`, not
                # the device_t0-shifted `this_t`.
                sync = imu.synchronize(
                    {"accel": np.asarray(slam._live_accel_buf) if slam._live_accel_buf else np.zeros((0, 4)),
                     "gyro": np.asarray(slam._live_gyro_buf) if slam._live_gyro_buf else np.zeros((0, 4))},
                    R_cam_imu=R_cam_imu)
                lo_raw = last_process_t_raw if last_process_t_raw is not None else frame_ts_raw - (1.0 / args.fps)
                imu_samples = imu.slice_between(sync, lo_raw, frame_ts_raw)
                # Trim buffers so they don't grow for the whole session --
                # keep a small tail before `lo_raw` for the next interpolation.
                keep_from = lo_raw - 0.05
                slam._live_accel_buf = [r for r in slam._live_accel_buf if r[0] >= keep_from]
                slam._live_gyro_buf = [r for r in slam._live_gyro_buf if r[0] >= keep_from]
                last_process_t_raw = frame_ts_raw

            slam.process(color_img, this_t, depth_image=depth_img, imu_samples=imu_samples)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        pipeline.stop()

    return slam


def run_frames(args):
    if args.calib:
        camera = Camera.from_json(args.calib)
    else:
        print("ERROR: --frames needs --calib <calibration.json>")
        sys.exit(1)
    print(camera)

    paths = sorted(glob.glob(os.path.join(args.frames, "*.png")) +
                   glob.glob(os.path.join(args.frames, "*.jpg")))
    if args.max_frames:
        paths = paths[:args.max_frames]
    if not paths:
        print(f"No images found in {args.frames}")
        sys.exit(1)
    print(f"Found {len(paths)} frames\n")

    slam = SLAMSystem(camera, use_depth=False, verbose=not args.quiet)
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        slam.process(img, i / float(args.fps))
    return slam


def main():
    ap = argparse.ArgumentParser(description="Python ORB-SLAM3 reimplementation")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--realsense", action="store_true", help="live D435 input")
    src.add_argument("--frames", type=str, help="directory of image frames")

    ap.add_argument("--calib", type=str, help="calibration JSON (for --frames)")
    ap.add_argument("--seconds", type=int, default=30, help="capture duration")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mono", action="store_true",
                    help="ignore depth even on RealSense (monocular mode)")
    ap.add_argument("--ir", action="store_true",
                    help="track on left IR (global shutter, natively depth-"
                         "registered) instead of RGB. See camera.py docstring. "
                         "Only worth it if the facility has enough natural "
                         "texture for emitter-off IR frames to be usable.")
    ap.add_argument("--imu", action="store_true",
                    help="PHASE 8: enable the D435i's accel+gyro motion "
                         "streams and feed them into the pipeline for "
                         "IMU-predicted pose (tracking robustness through "
                         "fast rotation / motion blur) and IMU-coasted "
                         "recovery during brief tracking loss. --realsense "
                         "only -- --frames has no IMU source.")
    ap.add_argument("--t_cam_imu", type=str, default=None,
                    help="path to a T_cam_imu.txt (4x4) from record.py's "
                         "--imu extrinsics dump. Only the rotation block is "
                         "used (imu.py's lever-arm simplification). Falls "
                         "back to identity if omitted -- fine for a quick "
                         "test, but a real T_cam_imu measurement matters "
                         "for prediction accuracy.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    slam = run_realsense(args) if args.realsense else run_frames(args)

    slam.print_summary()
    slam.save_trajectory_plot()
    slam.save_map_ply()


if __name__ == "__main__":
    main()

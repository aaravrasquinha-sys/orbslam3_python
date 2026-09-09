"""
isam2_backend.py — Phase 9: unified real-time incremental back end.

Replaces TWO separate optimizers that existed before this phase:
  bundle_adjust_gtsam.py's local_bundle_adjust_gtsam()  (windowed local BA)
  pose_graph.py's optimize_pose_graph()                  (scipy loop correction)

WHY: see PHASE7_ARCHITECTURE_V2_REALTIME.md section 2 for the full
reasoning. In short -- the batch/windowed design assumed you could always
stop, look at the whole map, and re-optimize; the real-time reframe
(robot testing live, not offline reconstruction) removes that
assumption. GTSAM's ISAM2 is the standard real-time incremental back end
in the visual(-inertial) SLAM literature for exactly this reason: it
maintains one continuously-updated factor graph and Bayes tree, and its
own relinearization logic figures out what actually needs re-solving on
each update rather than requiring a full batch re-solve. This also
removes an entire recurring bug class this project hit three separate
times (Phase 2's isolated-node crash, the free/fixed keyframe split bug,
pose_graph.py's own reference-keyframe lookup bug) -- all three came
from THIS PROJECT'S OWN hand-built bookkeeping for "which keyframes/
points are in this batch window," built three times for three separate
optimizers. ISAM2 owns the whole graph continuously; there's no window
boundary left to get wrong.

DESIGN, kept deliberately simple for a first working version (see
PHASE7_ARCHITECTURE_V2_REALTIME.md section 2.4's "build the
straightforward version first, measure, add complexity only if
needed"):
  - ONE Isam2Backend instance per ACTIVE Atlas map (mirrors the existing
    per-map architecture -- tracking.set_map()/local_mapping.set_map()
    already work this way for map switching). A fresh map (Atlas.
    start_new_map()) gets a fresh backend.
  - UNBOUNDED: every keyframe added stays a live variable forever. No
    fixed-lag marginalization. This is the documented, deliberate
    starting point -- see the module docstring section "on scaling"
    below for the measured numbers this phase's own benchmark produced
    and what to do if they stop being good enough.
  - Factors are added ONCE, when a keyframe is first given to
    add_keyframe() -- observations of EXISTING map points reuse that
    point's existing landmark variable; observations of NEW map points
    insert a fresh landmark. A later re-linking of an EXISTING keyframe
    to a DIFFERENT point (e.g. via fusion, after this keyframe was
    already added) does NOT retroactively add a new factor -- this
    mirrors how real incremental visual(-inertial) SLAM back ends
    generally work (fresh observations from NEW keyframes are what
    corrects a redirected point, not retroactive edits to old factors)
    and keeps the design simple. Documented as a known simplification,
    not silently assumed.
  - Loop closure becomes ONE MORE factor (or a small batch of them) added
    via the exact same incremental update() call, not a separate
    optimizer with separate bookkeeping.
  - Map merges: handled by replaying the absorbed map's keyframes
    through the SAME add_keyframe() method, one at a time, in
    chronological (kf_seq) order, right after merge_maps.py combines the
    Map objects at the Python level. Not a special code path.

ON THE ODOMETRY FACTOR (found during this phase's own testing, not
anticipated in the original design): the first version of this module
constrained each new keyframe ONLY through factors to the map points it
observes. That's fine as long as a new keyframe always shares at least
one already-graphed point with something earlier in the graph -- but
it's not guaranteed. A synthetic test with a circular trajectory and
realistic per-point visibility angles (a point only visible from a
limited arc, not from every keyframe) hit a keyframe that shared ZERO
points with anything processed so far -- its whole local sub-cluster
(that keyframe's pose plus every brand-new point it observes) had no
path back to the gauge-anchored first keyframe at all. GTSAM correctly
raised IndeterminantLinearSystemException: a disconnected component with
no absolute reference genuinely IS underdetermined, not a numerical
fluke to paper over. Fixed the way every real incremental visual(-
inertial) SLAM back end actually does this: add_keyframe() now ALSO adds
a BetweenFactorPose3 to the immediately-preceding keyframe, using the
relative pose tracking already estimated between them, every time,
regardless of whether they share any map points. This guarantees the
graph is always one connected component and is standard practice, not a
workaround specific to this bug -- it was simply missing from the first
draft because the batch windowed BA this module replaces never needed
it (a whole window is typically densely co-visible by construction, so
this gap rarely if ever shows up there).

ON SCALING (measured this phase, see test_phase9_gate.py's benchmark):
  incremental isam.update() calls on a synthetic multi-hundred-keyframe
  trajectory averaged low tens of milliseconds per keyframe on this
  sandbox's CPU (see PROGRESS.md's Phase 9 section for the exact
  number) -- consistent with the literature figures cited in
  PHASE7_ARCHITECTURE_V2_REALTIME.md (iSAM2-based visual SLAM running on
  a single core of a comparable-or-weaker CPU). If a real facility-scale
  session on the actual Z440 shows unbounded growth becoming a genuine
  problem, the documented lever is fixed-lag marginalization
  (converting old keyframes to a prior via Schur complement rather than
  keeping them as live variables) -- NOT built here, per the "measure
  before building" plan; add it only if the bring-up numbers on real
  hardware say so.

CONVENTIONS (unchanged from bundle_adjust_gtsam.py): frame.pose is
camera-to-world (Twc); GTSAM's Pose3 expects the same, so no convention
translation is needed. Uses points_undistorted (NOT keypoints[i].pt,
which is the raw, possibly-distorted pixel location) for factor
measurements -- bundle_adjust_gtsam.py uses the raw one, a latent
inaccuracy on any camera with real distortion that never surfaced on
this project's distortion-free synthetic tests; noted here rather than
silently repeated.
"""

import numpy as np
import gtsam
from gtsam import Point3, Pose3, Rot3, Cal3_S2, Cal3_S2Stereo, StereoPoint2
from gtsam.symbol_shorthand import X, L


def _pose3_from_np(T):
    return Pose3(Rot3(T[:3, :3]), Point3(T[:3, 3]))


def _np_from_pose3(pose3):
    T = np.eye(4)
    T[:3, :3] = pose3.rotation().matrix()
    T[:3, 3] = pose3.translation()
    return T


class Isam2Backend:
    def __init__(self, camera, virtual_baseline=0.05, huber_f_scale=2.0,
                min_obs_to_optimize=2, relinearize_threshold=0.1,
                relinearize_skip=1, anchor_sigma_pos=1e-3, anchor_sigma_rot=1e-2,
                odom_sigma_pos=0.05, odom_sigma_rot=0.05):
        self.camera = camera
        self.min_obs_to_optimize = min_obs_to_optimize

        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(relinearize_threshold)
        params.relinearizeSkip = relinearize_skip
        self.isam = gtsam.ISAM2(params)

        self.K_stereo = Cal3_S2Stereo(camera.fx, camera.fy, 0.0, camera.cx, camera.cy,
                                      virtual_baseline)
        self.K_mono = Cal3_S2(camera.fx, camera.fy, 0.0, camera.cx, camera.cy)
        self.virtual_baseline = virtual_baseline
        huber = gtsam.noiseModel.mEstimator.Huber.Create(huber_f_scale)
        self.stereo_noise = gtsam.noiseModel.Robust.Create(
            huber, gtsam.noiseModel.Isotropic.Sigma(3, 1.0))
        self.mono_noise = gtsam.noiseModel.Robust.Create(
            huber, gtsam.noiseModel.Isotropic.Sigma(2, 1.0))
        self.anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([anchor_sigma_pos] * 3 + [anchor_sigma_rot] * 3))
        # PHASE 9 BUGFIX (found, not anticipated -- see module docstring's
        # "on the odometry factor" section): odometry noise between
        # consecutive keyframes.
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([odom_sigma_rot] * 3 + [odom_sigma_pos] * 3))

        self.kf_ids_in_graph = set()
        self.point_ids_in_graph = set()
        self._kf_objects = {}     # id -> Frame, so _sync_back never needs to scan world_map.keyframes
        self._prev_kf_id = None   # chronologically-last keyframe added -- see add_keyframe
        self._has_anchor = False
        self.n_updates = 0
        self.last_update_seconds = 0.0

    def add_keyframe(self, keyframe, world_map, extra_factors=None):
        """
        Incrementally add ONE new keyframe (must not already be in the
        graph) plus factors for its current observations of map points
        with >= min_obs_to_optimize observations. New points it observes
        get a fresh landmark variable; points already in the graph reuse
        their existing one. Returns (n_points_added, n_obs_added,
        elapsed_seconds).

        extra_factors: optional list of already-built gtsam factors to
        add in the SAME update() call (used for loop-closure factors --
        see add_loop_factor()) -- batching them into one call is why that
        method takes this argument rather than calling update() twice.
        """
        import time
        t_start = time.time()

        if keyframe.id in self.kf_ids_in_graph:
            raise ValueError(f"keyframe {keyframe.id} already in the iSAM2 graph -- "
                            f"add_keyframe() adds a keyframe exactly once, ever")

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        pose3 = _pose3_from_np(keyframe.pose)
        initial.insert(X(keyframe.id), pose3)
        if not self._has_anchor:
            # PHASE 9: gauge-fixing anchor, same convention every earlier
            # optimizer in this project used (chronologically-first
            # keyframe hard-anchored). Only ever added ONCE per backend
            # instance (i.e. once per Atlas map) -- every later keyframe
            # is fully free, constrained only by its observation and
            # (once added) loop-closure factors.
            graph.push_back(gtsam.PriorFactorPose3(X(keyframe.id), pose3, self.anchor_noise))
            self._has_anchor = True
        elif self._prev_kf_id is not None and self._prev_kf_id in self._kf_objects:
            # PHASE 9 BUGFIX: odometry factor to the immediately-
            # preceding keyframe, ALWAYS added, regardless of whether the
            # two keyframes share any map points -- see module docstring
            # "on the odometry factor" for why this is required, not
            # optional. Measurement is the tracking-estimated relative
            # pose (same quantity pose_graph.py's old scipy version
            # trusted as an "odometry edge").
            prev_kf = self._kf_objects[self._prev_kf_id]
            T_rel = np.linalg.inv(prev_kf.pose) @ keyframe.pose
            graph.push_back(gtsam.BetweenFactorPose3(
                X(self._prev_kf_id), X(keyframe.id), _pose3_from_np(T_rel), self.odom_noise))

        n_points_added, n_obs_added = 0, 0
        newly_added_point_ids = set()   # PHASE 9: committed to self.point_ids_in_graph
                                        # only AFTER isam.update() succeeds -- see below
        fx, fy = self.camera.fx, self.camera.fy
        for kp_idx, mp_id in enumerate(keyframe.map_point_ids):
            if mp_id is None:
                continue
            mp = world_map.map_points.get(mp_id)
            if mp is None or mp.is_bad:
                continue
            if mp.n_observations() < self.min_obs_to_optimize:
                continue
            if kp_idx >= len(keyframe.points_undistorted):
                continue

            u, v = keyframe.points_undistorted[kp_idx]
            depth = keyframe.depths[kp_idx] if kp_idx < len(keyframe.depths) else -1.0
            u_r = u - fx * self.virtual_baseline / depth if (depth is not None and depth > 1e-3) else None
            # PHASE 9 defensive check: a valid depth reading can still
            # produce a near-degenerate virtual-stereo disparity for a
            # very distant point relative to the (small, ~5cm) baseline.
            # 0.1px is far below any real disparity this sensor should
            # report for an in-range point (see local_mapping.py's
            # depth_max/th_depth for the sane working range).
            is_stereo = u_r is not None and (u - u_r) > 0.1

            if mp_id not in self.point_ids_in_graph and mp_id not in newly_added_point_ids:
                # A brand-new landmark's FIRST-EVER factor must be
                # stereo, never mono -- a single mono (2D) observation
                # gives only 2 constraints for 3 unknowns, genuinely
                # underdetermined on its own regardless of how well-
                # connected the observing keyframe otherwise is (the
                # odometry factor above constrains the KEYFRAME pose,
                # not this landmark's position at all). Skipping here
                # defers insertion to a later keyframe with a valid
                # stereo view of the same point -- not lost, just not
                # yet, exactly like the position-sanity skip above.
                if not is_stereo:
                    continue
                pos = mp.position
                if pos is None or not np.all(np.isfinite(pos)) or float(np.max(np.abs(pos))) > 500.0:
                    continue
                initial.insert(L(mp_id), Point3(pos))
                newly_added_point_ids.add(mp_id)
                n_points_added += 1

            if is_stereo:
                graph.push_back(gtsam.GenericStereoFactor3D(
                    StereoPoint2(u, u_r, v), self.stereo_noise,
                    X(keyframe.id), L(mp_id), self.K_stereo))
            else:
                graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
                    gtsam.Point2(u, v), self.mono_noise,
                    X(keyframe.id), L(mp_id), self.K_mono))
            n_obs_added += 1

        if extra_factors:
            for f in extra_factors:
                graph.push_back(f)

        # PHASE 9 defense-in-depth: the checks above catch the failure
        # modes actually found and diagnosed this phase (a brand-new
        # landmark's first factor being mono-only; a wildly-out-of-range
        # position from bad triangulation during degraded tracking; a
        # near-zero virtual-stereo disparity). This try/except is the
        # final backstop for whatever THIS list doesn't cover -- a real,
        # live pipeline should not go down because one keyframe's update
        # hit a numerical edge case the filtering didn't anticipate.
        #
        # RECONCILIATION (found the hard way, via a real cascading
        # failure on a plain 100-frame synthetic run -- NOT a
        # hypothetical): ISAM2.update() is NOT transactional. An earlier
        # draft assumed a failed call left the graph in its exact
        # previous state and therefore left self.point_ids_in_graph
        # un-mutated on failure (staging new ids locally, only merging
        # them in after a confirmed success -- see newly_added_point_ids
        # above). That assumption is wrong: GTSAM can partially commit
        # variables internally before the elimination step throws. The
        # observed consequence: keyframe 42 hit a genuine
        # IndeterminantLinearSystemException; every SUBSEQUENT keyframe
        # that happened to reference one of THAT failed call's landmark
        # ids then crashed with "key already exists" (IndexError, a
        # DIFFERENT exception type than the RuntimeError this block
        # originally only caught) -- my bookkeeping said "not yet
        # added, safe to insert," GTSAM said "already have it." A single
        # early failure was silently cascading into permanent, repeated
        # failures for every later keyframe touching those points.
        #
        # Fixed by not trusting either side's bookkeeping after a
        # failure -- ask GTSAM directly (valueExists()) which of THIS
        # attempt's ids actually made it in, and reconcile
        # self.kf_ids_in_graph / self.point_ids_in_graph / _kf_objects /
        # _prev_kf_id to match that ground truth, whatever it turns out
        # to be, rather than assuming either "nothing committed" or
        # "everything committed."
        try:
            self.isam.update(graph, initial)
        except (RuntimeError, IndexError) as e:
            self.last_update_seconds = time.time() - t_start
            kf_committed = self.isam.valueExists(X(keyframe.id))
            if kf_committed:
                self.kf_ids_in_graph.add(keyframe.id)
                self._kf_objects[keyframe.id] = keyframe
                self._prev_kf_id = keyframe.id
            actually_added = {mp_id for mp_id in newly_added_point_ids
                             if self.isam.valueExists(L(mp_id))}
            self.point_ids_in_graph |= actually_added
            print(f"[iSAM2] WARNING: add_keyframe({keyframe.id}) update failed "
                 f"({type(e).__name__}); reconciled against actual GTSAM state "
                 f"(kf committed={kf_committed}, {len(actually_added)}/"
                 f"{len(newly_added_point_ids)} new points committed): {str(e)[:150]}")
            # Whether or not the keyframe itself got committed, this
            # attempt is done -- either way, do NOT fall through to the
            # unconditional success-path merge below (it would re-add
            # every id in newly_added_point_ids regardless of what
            # valueExists() actually confirmed, undoing the
            # reconciliation just performed above).
            if kf_committed:
                self._sync_back(world_map)
            return n_points_added, n_obs_added, None

        self.point_ids_in_graph |= newly_added_point_ids
        self.kf_ids_in_graph.add(keyframe.id)
        self._kf_objects[keyframe.id] = keyframe
        self._prev_kf_id = keyframe.id
        self.n_updates += 1
        self.last_update_seconds = time.time() - t_start

        self._sync_back(world_map)
        return n_points_added, n_obs_added, self.last_update_seconds

    def add_loop_factor(self, kf_a, kf_b, T_rel, loop_weight, world_map,
                        sigma_pos=None, sigma_rot=None):
        """
        PHASE 9: this is the entire "loop closure correction" step now --
        one BetweenFactorPose3 added via the SAME incremental update()
        pathway everything else uses, replacing pose_graph.py's separate
        scipy optimizer entirely. iSAM2's own relinearization figures out
        how far the correction needs to propagate; there's no separate
        "walk every keyframe between the loop pair" bookkeeping to get
        wrong (see this project's history of bugs in exactly that kind
        of bookkeeping, in the module docstring above).

        T_rel: measured kf_a -> kf_b relative transform (same convention
        pose_graph.compute_loop_edge already produces -- drop-in).
        loop_weight: higher = more trusted; converted to a noise sigma
        the same rough way the old scipy version used it (loop_weight
        scaled by inlier count there too).

        Both kf_a and kf_b MUST already be in this graph (call
        add_keyframe() for both first -- true by construction in the
        normal flow, since loop closing only runs after the current
        keyframe's own add_keyframe() call has already happened).
        """
        if kf_a.id not in self.kf_ids_in_graph or kf_b.id not in self.kf_ids_in_graph:
            raise ValueError("add_loop_factor: both keyframes must already be in the graph")

        s_pos = sigma_pos if sigma_pos is not None else max(1e-3, 1.0 / max(loop_weight, 1e-6))
        s_rot = sigma_rot if sigma_rot is not None else max(1e-3, 1.0 / max(loop_weight, 1e-6))
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([s_rot] * 3 + [s_pos] * 3))

        T_rel_pose3 = _pose3_from_np(T_rel)
        factor = gtsam.BetweenFactorPose3(X(kf_a.id), X(kf_b.id), T_rel_pose3, noise)

        graph = gtsam.NonlinearFactorGraph()
        graph.push_back(factor)
        import time
        t_start = time.time()
        # PHASE 9: same defensive backstop as add_keyframe -- a loop
        # factor adds no NEW variables (both endpoints already exist),
        # so there's no bookkeeping to reconcile on failure, but the
        # call should still never take down the whole pipeline over one
        # bad loop-closure measurement (e.g. a plausible-looking but
        # ultimately geometrically-inconsistent PnP result that RANSAC's
        # own inlier count didn't fully catch).
        try:
            self.isam.update(graph, gtsam.Values())
        except (RuntimeError, IndexError) as e:
            self.last_update_seconds = time.time() - t_start
            print(f"[iSAM2] WARNING: add_loop_factor({kf_a.id}, {kf_b.id}) "
                 f"failed, loop correction skipped this time: "
                 f"{type(e).__name__}: {str(e)[:150]}")
            return None
        self.n_updates += 1
        self.last_update_seconds = time.time() - t_start
        self._sync_back(world_map)
        return self.last_update_seconds

    def absorb_map(self, absorbed_kfs, world_map, bridge_kf=None,
                   bridge_sigma_pos=0.2, bridge_sigma_rot=0.2):
        """
        PHASE 9: map-merge integration. `absorbed_kfs` are keyframes from
        a map that just got merged INTO the map this backend serves
        (merge_maps.py already combined the Python-level Map objects by
        the time this is called -- see run_slam.py's _handle_loop_result)
        but were never added to THIS specific ISAM2 instance, since they
        belonged to a different map (and therefore a different backend)
        until the merge. Not a special code path -- just replays
        add_keyframe() for each one, in chronological order, exactly as
        if they were being created live.

        Resets the odometry chain (_prev_kf_id) before absorbing, so the
        first absorbed keyframe does NOT get a spurious BetweenFactorPose3
        to whatever this backend's own last keyframe happened to be --
        see the earlier version of this docstring for why that's wrong
        (treats "both now exist in the same post-merge coordinate frame"
        as if it were a real tracked measurement).

        bridge_kf: a keyframe ALREADY in this graph, used instead to
        connect the chronologically-FIRST absorbed keyframe -- found
        necessary via test_phase9_gate.py: resetting the odometry chain
        alone isn't sufficient, because that first keyframe in general
        shares NO map points with the surviving graph either (that's
        exactly the scenario a genuine merge produces -- two previously
        separate, independently-mapped areas). Without any connection at
        all, that keyframe is a disconnected, gauge-ambiguous component,
        the identical failure shape add_keyframe()'s own odometry factor
        exists to prevent, just at the map-merge seam instead of an
        ordinary keyframe boundary. Uses the GEOMETRIC relative pose
        between bridge_kf and the first absorbed keyframe (both already
        expressed in the same coordinate frame by merge_maps.py's own
        transform application) with a deliberately loose noise model --
        this is NOT a substitute for the real, PnP-measured loop-closure
        transform between the actual matched keyframe pair, which the
        caller should still add separately via add_loop_factor right
        after this call returns (a much tighter constraint, reflecting a
        genuine verified measurement -- this bridge only prevents a
        crash, add_loop_factor is what actually pulls the seam tight).
        """
        self._prev_kf_id = None
        sorted_kfs = sorted(absorbed_kfs, key=lambda k: k.kf_seq)
        bridge_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([bridge_sigma_rot] * 3 + [bridge_sigma_pos] * 3))
        bridged = False
        for kf in sorted_kfs:
            if kf.id in self.kf_ids_in_graph or kf.pose is None:
                continue
            extra = None
            if not bridged and bridge_kf is not None and bridge_kf.id in self.kf_ids_in_graph:
                T_rel = np.linalg.inv(bridge_kf.pose) @ kf.pose
                extra = [gtsam.BetweenFactorPose3(
                    X(bridge_kf.id), X(kf.id), _pose3_from_np(T_rel), bridge_noise)]
                bridged = True
            self.add_keyframe(kf, world_map, extra_factors=extra)

    def _sync_back(self, world_map):
        """Write the current iSAM2 estimate back into the actual Frame/
        MapPoint objects everything else in the codebase reads."""
        estimate = self.isam.calculateEstimate()
        for kf_id in self.kf_ids_in_graph:
            kf = self._kf_objects.get(kf_id)
            if kf is None:
                continue
            if estimate.exists(X(kf_id)):
                pose = _np_from_pose3(estimate.atPose3(X(kf_id)))
                velocity = getattr(kf, "velocity", None)
                if velocity is not None:
                    delta_R = pose[:3, :3] @ kf.pose[:3, :3].T
                    kf.velocity = delta_R @ velocity
                kf.set_pose(pose)
        for mp_id in self.point_ids_in_graph:
            mp = world_map.map_points.get(mp_id)
            if mp is None:
                continue
            if estimate.exists(L(mp_id)):
                mp.position = np.array(estimate.atPoint3(L(mp_id)))

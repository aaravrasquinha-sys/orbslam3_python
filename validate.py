"""
validate.py — Phase 7: Atlas-wide data-model invariant checks.

WHY THIS EXISTS: this project has now found the SAME underlying bug
family three separate times, each time by accident, each time after it
had already silently corrupted results for a while:
  - Phase 0: frame.id vs kf_seq unit confusion in update_normal_and_depth.
  - Phase 5: the identical frame.id vs kf_seq confusion in loop_closing's
    keyframe-gap check.
  - Phase 6: MapPoint.observations silently holding dangling references
    after point deletion (map.py/local_mapping.py), found via a 99.3%
    stale-reference rate on one real keyframe -- and the FIX for that bug
    then introduced a NEW instance of a related problem (observations
    polluted with non-keyframe frame ids -- see tracking.py's Phase 7
    comments), because there was no code anywhere that stated "here is
    what a valid Atlas looks like" for the fix to be checked against.

This module is that statement, made checkable. It doesn't fix anything;
it walks an Atlas and reports every place the documented invariants
(map_point.py's docstring, frame.py's docstring, this project's
PROGRESS.md) don't actually hold, with the specific offending ids -- not
just a count, so a violation is immediately actionable rather than
"something's wrong somewhere."

USAGE:
    import validate
    report = validate.validate_atlas(atlas)
    if not report.ok():
        print(report)              # human-readable, one line per violation
        raise SystemExit(1)        # or just log it and keep going

Call this at every pass/session boundary (after init, after every N
keyframes in --validate-often mode, before/after a merge, at shutdown).
Cheap enough (single pass over keyframes/points, no O(n^2) work) to run
far more often than "only when something looks wrong."
"""

import numpy as np


class ValidationReport:
    def __init__(self):
        self.errors = []      # invariant violations -- data IS corrupt
        self.warnings = []    # suspicious but not necessarily wrong

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def ok(self):
        return len(self.errors) == 0

    def __repr__(self):
        lines = [f"ValidationReport: {len(self.errors)} error(s), "
                f"{len(self.warnings)} warning(s)"]
        for e in self.errors:
            lines.append(f"  [ERROR] {e}")
        for w in self.warnings:
            lines.append(f"  [warn]  {w}")
        return "\n".join(lines)


class AtlasValidationError(RuntimeError):
    pass


def _all_keyframes(atlas):
    for m in atlas.maps:
        for kf in m.keyframes:
            yield m, kf


def _all_map_points(atlas):
    for m in atlas.maps:
        for mp in m.map_points.values():
            yield m, mp


# ── individual checks ───────────────────────────────────────────────────

def check_observations_are_keyframes(atlas, report):
    """
    Every id in mp.observations must be the .id of a keyframe that
    actually exists in SOME map of this Atlas right now. This is the
    invariant map_point.py's docstring documents and the one the Phase 7
    tracking.py/relocalization.py fixes exist to protect -- see this
    module's own docstring for the history of why it was violated.
    """
    all_kf_ids = {kf.id for _, kf in _all_keyframes(atlas)}
    for m, mp in _all_map_points(atlas):
        if mp.is_bad:
            continue
        for obs_frame_id in mp.observations.keys():
            if obs_frame_id not in all_kf_ids:
                report.error(
                    f"map_point {mp.id} (map {m.id}): observations "
                    f"references frame {obs_frame_id}, which is not a "
                    f"live keyframe in any map (either a non-keyframe "
                    f"frame -- observations pollution -- or a keyframe "
                    f"that was culled without cleaning up this point)")


def check_map_point_id_consistency(atlas, report):
    """
    Every non-None kf.map_point_ids[i] must refer to a live (not bad,
    not deleted) point in THAT KEYFRAME'S OWN map. This is the direction
    the Phase 6 map.py/erase_map_point fix protects -- a dangling
    reference here is exactly the "419 references, 3 valid" bug.
    """
    for m, kf in _all_keyframes(atlas):
        for kp_i, mp_id in enumerate(kf.map_point_ids):
            if mp_id is None:
                continue
            mp = m.map_points.get(mp_id)
            if mp is None:
                report.error(
                    f"keyframe {kf.id} (map {m.id}): map_point_ids[{kp_i}] "
                    f"= {mp_id}, which does not exist in this keyframe's "
                    f"own map at all (dangling reference)")
            elif mp.is_bad:
                report.error(
                    f"keyframe {kf.id} (map {m.id}): map_point_ids[{kp_i}] "
                    f"= {mp_id}, which exists but is flagged bad (should "
                    f"have been cleared by clean_bad_points())")


def check_bidirectional_consistency(atlas, report):
    """
    Where both sides reference each other, they must agree on the
    keypoint index: mp.observations[kf.id] == i  <=>  kf.map_point_ids[i]
    == mp.id. A one-directional link (only one side knows about the
    connection) is itself an error under checks above; THIS check is
    for the case where both sides link to each other but DISAGREE on
    which keypoint index -- a sign something did a partial update.
    """
    for m, kf in _all_keyframes(atlas):
        for kp_i, mp_id in enumerate(kf.map_point_ids):
            if mp_id is None:
                continue
            mp = m.map_points.get(mp_id)
            if mp is None or mp.is_bad:
                continue   # already reported by check_map_point_id_consistency
            recorded_idx = mp.observations.get(kf.id)
            if recorded_idx is not None and recorded_idx != kp_i:
                report.error(
                    f"keyframe {kf.id} / map_point {mp.id}: keyframe says "
                    f"keypoint {kp_i}, but the point's own observations "
                    f"dict says keyframe {kf.id} observes keypoint "
                    f"{recorded_idx} -- indices disagree")


def check_kf_seq_global_monotonic(atlas, report):
    """
    kf_seq must be unique Atlas-wide (no two keyframes anywhere, in any
    map, share a kf_seq) and every keyframe must have one assigned. This
    is the invariant the Phase 7 global-counter fix (frame.py's
    Frame._next_kf_seq) is supposed to guarantee -- see merge_maps.py's
    module docstring for why a per-map counter used to violate this
    across a merge.
    """
    seen = {}
    for m, kf in _all_keyframes(atlas):
        if kf.kf_seq is None:
            report.error(f"keyframe {kf.id} (map {m.id}): kf_seq is None")
            continue
        if kf.kf_seq in seen:
            other_id, other_map = seen[kf.kf_seq]
            report.error(
                f"kf_seq {kf.kf_seq} is shared by keyframe {kf.id} "
                f"(map {m.id}) and keyframe {other_id} (map {other_map}) "
                f"-- kf_seq must be globally unique")
        else:
            seen[kf.kf_seq] = (kf.id, m.id)


def check_poses_well_formed(atlas, report, rtol=1e-3, atol=1e-4):
    """
    Every keyframe with a pose must have a finite, valid SE(3) matrix:
    finite entries, orthonormal rotation block (R @ R.T == I), det(R) ==
    +1 (not a reflection), bottom row == [0,0,0,1].
    """
    for m, kf in _all_keyframes(atlas):
        if kf.pose is None:
            continue
        T = kf.pose
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            report.error(f"keyframe {kf.id} (map {m.id}): pose is not a "
                        f"finite 4x4 matrix")
            continue
        R = T[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), rtol=rtol, atol=atol):
            report.error(f"keyframe {kf.id} (map {m.id}): rotation block "
                        f"is not orthonormal (R @ R.T != I)")
        det = np.linalg.det(R)
        if det < 0:
            report.error(f"keyframe {kf.id} (map {m.id}): rotation block "
                        f"has determinant {det:.4f} < 0 (reflection, not "
                        f"a valid rotation)")
        if not np.allclose(T[3, :], [0, 0, 0, 1], atol=atol):
            report.error(f"keyframe {kf.id} (map {m.id}): bottom row is "
                        f"{T[3, :]}, expected [0,0,0,1]")


def check_no_orphans(atlas, report, min_obs_warn=1):
    """
    Every live map point should be observed by at least one keyframe;
    every keyframe should carry at least a handful of tracked points.
    These are WARNINGS, not errors -- a brand-new keyframe or a point on
    probation can legitimately be sparse for a moment, so this is a
    signal to look, not proof of corruption.
    """
    for m, mp in _all_map_points(atlas):
        if mp.is_bad:
            continue
        if mp.n_observations() < min_obs_warn:
            report.warn(f"map_point {mp.id} (map {m.id}): only "
                       f"{mp.n_observations()} observation(s)")
    for m, kf in _all_keyframes(atlas):
        n_tracked = sum(1 for x in kf.map_point_ids if x is not None)
        if n_tracked == 0:
            report.warn(f"keyframe {kf.id} (map {m.id}): tracks zero "
                       f"map points")


def check_positions_finite_bounded(atlas, report, world_bound_m=500.0):
    """
    Every point position must be finite and within a sane world bound.
    Catches numerical blow-ups (e.g. a bad triangulation, a division by
    a near-zero disparity) before they silently poison BA / loop
    verification with an extreme outlier.
    """
    for m, mp in _all_map_points(atlas):
        if mp.is_bad:
            continue
        p = mp.position
        if p is None or not np.all(np.isfinite(p)):
            report.error(f"map_point {mp.id} (map {m.id}): position is "
                        f"not finite: {p}")
            continue
        if float(np.max(np.abs(p))) > world_bound_m:
            report.warn(f"map_point {mp.id} (map {m.id}): position {p} "
                       f"exceeds the {world_bound_m}m sanity bound")


def check_first_keyframe_id_sane(atlas, report):
    """
    MapPoint.first_keyframe_id is stored in kf_seq units (see
    map_point.py). It must be a non-negative int not exceeding the
    current global kf_seq counter -- it does NOT need to refer to a
    keyframe that still exists (that keyframe may have been culled since;
    first_keyframe_id is only used for age arithmetic, not lookup).
    """
    from frame import Frame
    ceiling = Frame._next_kf_seq
    for m, mp in _all_map_points(atlas):
        if mp.is_bad:
            continue
        fid = mp.first_keyframe_id
        if fid is None:
            report.warn(f"map_point {mp.id} (map {m.id}): "
                       f"first_keyframe_id is None")
        elif fid < 0 or fid >= ceiling:
            report.error(f"map_point {mp.id} (map {m.id}): "
                        f"first_keyframe_id={fid} is out of the valid "
                        f"kf_seq range [0, {ceiling})")


# ── entry point ──────────────────────────────────────────────────────────

_ALL_CHECKS = [
    check_observations_are_keyframes,
    check_map_point_id_consistency,
    check_bidirectional_consistency,
    check_kf_seq_global_monotonic,
    check_poses_well_formed,
    check_no_orphans,
    check_positions_finite_bounded,
    check_first_keyframe_id_sane,
]


def validate_atlas(atlas, world_bound_m=500.0, raise_on_error=False):
    """
    Run every invariant check against the current Atlas state. Returns a
    ValidationReport (always -- even a clean Atlas gets one, with empty
    error/warning lists) so callers can log or assert on it uniformly.
    """
    report = ValidationReport()
    for check in _ALL_CHECKS:
        if check is check_positions_finite_bounded:
            check(atlas, report, world_bound_m=world_bound_m)
        else:
            check(atlas, report)
    if raise_on_error and not report.ok():
        raise AtlasValidationError(str(report))
    return report


def reference_validity_rate(atlas):
    """
    The specific diagnostic FIELD_LOG_PHASE6.md's Sec.2 asked for by
    hand -- promoted to code so it's always run the same way. Returns
    (n_valid, n_total, pct). A reference is "valid" if the keyframe's
    map_point_ids entry points at a live point in that keyframe's own
    map -- i.e. exactly what check_map_point_id_consistency checks, but
    returned as a ratio for quick reporting rather than an error list.
    """
    n_total, n_valid = 0, 0
    for m, kf in _all_keyframes(atlas):
        for mp_id in kf.map_point_ids:
            if mp_id is None:
                continue
            n_total += 1
            mp = m.map_points.get(mp_id)
            if mp is not None and not mp.is_bad:
                n_valid += 1
    pct = (100.0 * n_valid / n_total) if n_total else 100.0
    return n_valid, n_total, pct

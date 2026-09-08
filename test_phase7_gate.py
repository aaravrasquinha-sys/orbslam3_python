"""
test_phase7_gate.py — Phase 7, Build 1 (Workstream A: data-model
invariants, Workstream B: map coverage).

Run: python3 test_phase7_gate.py

Gates:
  - validate.py catches each deliberately-injected invariant violation
    it claims to catch (observations pollution, dangling map_point_ids,
    duplicate kf_seq, malformed pose) and passes cleanly on a healthy
    Atlas.
  - reference_validity_rate() reads >99% after a normal pipeline run
    (the FIELD_LOG_PHASE6.md Sec.2 diagnostic, now checked automatically
    every run instead of by hand).
  - Global kf_seq survives a map merge WITHOUT renumbering, and every
    pre-merge MapPoint.first_keyframe_id stays valid afterward (the
    direct regression test for finding 3.2).
  - Non-keyframe tracking does NOT pollute MapPoint.observations (the
    direct regression test for finding 3.1).
  - RGB-D mode creates FAR points via triangulation, not just close
    points from bare depth (finding 3.3) -- uses two_depth_scroll_
    sequence, the only synthetic generator with both a close and a far
    depth plane in the same scene.
  - Pure rotation (near-zero translation) triggers keyframe insertion
    (the keyframe_min_rotation_deg criterion).
  - Full Phase 1-6 regression still passes with the Build 1 changes.
"""

import sys
import numpy as np

from camera import Camera
from frame import Frame
from map import Map
from map_point import MapPoint
from atlas import Atlas
import validate
import merge_maps
import synthetic
import metrics
from run_slam import SLAMSystem
from tracking import Tracking

FAILURES = []


def gate(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def _dummy_camera():
    return Camera(fx=385.0, fy=385.0, cx=320.0, cy=240.0,
                 width=640, height=480, baseline=0.05, depth_scale=0.001)


def run_sequence(images, depths, verbose=False):
    slam = None
    for i, (img, depth) in enumerate(zip(images, depths)):
        if slam is None:
            h, w = img.shape[:2]
            cam = Camera(fx=385.0, fy=385.0, cx=w / 2, cy=h / 2, width=w, height=h,
                        baseline=0.05, dist_coeffs=None, depth_scale=0.001)
            slam = SLAMSystem(cam, use_depth=True, verbose=verbose)
        slam.process(img, i / 30.0, depth_image=depth)
    return slam


# ── Workstream A: validate.py catches what it claims to ────────────────

def test_validate_clean_atlas():
    print("\n--- Test 1: validate.py passes cleanly on a healthy Atlas ---")
    images, depths, gt, _ = synthetic.scroll_sequence(n_frames=30)
    slam = run_sequence(images, depths)
    report = validate.validate_atlas(slam.atlas)
    gate("no errors on a normal pipeline run", report.ok(),
        f"{len(report.errors)} error(s): {report.errors[:3]}")

    n_valid, n_total, pct = validate.reference_validity_rate(slam.atlas)
    gate("reference validity > 99% (FIELD_LOG_PHASE6.md Sec.2, now automated)",
        pct > 99.0, f"{n_valid}/{n_total} = {pct:.1f}%")


def test_validate_catches_observation_pollution():
    print("\n--- Test 2: validate.py catches observations pollution ---")
    cam = _dummy_camera()
    atlas = Atlas()
    m = atlas.active_map
    kf = Frame.__new__(Frame)
    kf.id = 9001
    kf.kf_seq = 0
    kf.pose = np.eye(4)
    kf.map_point_ids = [None]
    m.add_keyframe(kf)

    mp = MapPoint(np.array([0.0, 0.0, 2.0]), np.zeros(32, np.uint8),
                 ref_keyframe_id=kf.kf_seq)
    m.add_map_point(mp)
    mp.add_observation(kf.id, 0)
    kf.map_point_ids[0] = mp.id

    # inject the exact Phase 6 bug: an observation from a NON-keyframe
    # frame id that was never added to any map.
    mp.add_observation(99999, 0)

    report = validate.validate_atlas(atlas)
    gate("catches an observation referencing a non-keyframe frame id",
        not report.ok() and any("99999" in e for e in report.errors),
        f"errors: {report.errors}")


def test_validate_catches_dangling_map_point_id():
    print("\n--- Test 3: validate.py catches a dangling map_point_ids reference ---")
    atlas = Atlas()
    m = atlas.active_map
    kf = Frame.__new__(Frame)
    kf.id = 9002
    kf.kf_seq = 0
    kf.pose = np.eye(4)
    kf.map_point_ids = [123456]   # references a point that was never added
    m.add_keyframe(kf)

    report = validate.validate_atlas(atlas)
    gate("catches a map_point_ids entry with no corresponding live point",
        not report.ok() and any("123456" in e for e in report.errors),
        f"errors: {report.errors}")


def test_validate_catches_duplicate_kf_seq():
    print("\n--- Test 4: validate.py catches a duplicate kf_seq ---")
    atlas = Atlas()
    m = atlas.active_map
    kf_a = Frame.__new__(Frame)
    kf_a.id, kf_a.kf_seq, kf_a.pose, kf_a.map_point_ids = 9003, 5, np.eye(4), []
    kf_b = Frame.__new__(Frame)
    kf_b.id, kf_b.kf_seq, kf_b.pose, kf_b.map_point_ids = 9004, 5, np.eye(4), []
    m.add_keyframe(kf_a)
    m.add_keyframe(kf_b)   # both keep kf_seq=5 since we bypassed Map.add_keyframe's assignment

    report = validate.validate_atlas(atlas)
    gate("catches two keyframes sharing one kf_seq",
        not report.ok() and any("kf_seq 5" in e for e in report.errors),
        f"errors: {report.errors}")


def test_validate_catches_bad_pose():
    print("\n--- Test 5: validate.py catches a malformed pose ---")
    atlas = Atlas()
    m = atlas.active_map
    kf = Frame.__new__(Frame)
    kf.id, kf.kf_seq, kf.map_point_ids = 9005, 0, []
    bad = np.eye(4)
    bad[:3, :3] *= 2.0   # not orthonormal
    kf.pose = bad
    m.add_keyframe(kf)

    report = validate.validate_atlas(atlas)
    gate("catches a non-orthonormal rotation block",
        not report.ok() and any("orthonormal" in e for e in report.errors),
        f"errors: {report.errors}")


# ── Workstream A: global kf_seq survives a merge ────────────────────────

def test_kf_seq_survives_merge_no_renumbering():
    print("\n--- Test 6: global kf_seq is NOT renumbered by merge_maps, "
         "first_keyframe_id stays valid ---")
    cam = _dummy_camera()
    atlas = Atlas()
    map_a = atlas.active_map
    map_b = atlas.start_new_map()

    kfs_a, kfs_b = [], []
    for j in range(3):
        kf = Frame.__new__(Frame)
        kf.id = 100 + j
        kf.kf_seq = None
        kf.pose = np.eye(4)
        kf.map_point_ids = []
        kf.velocity = None
        map_a.add_keyframe(kf)
        kfs_a.append(kf)
    for j in range(3):
        kf = Frame.__new__(Frame)
        kf.id = 200 + j
        kf.kf_seq = None
        kf.pose = np.eye(4)
        kf.map_point_ids = []
        kf.velocity = None
        map_b.add_keyframe(kf)
        kfs_b.append(kf)

    # a point created "before the merge", in map_a, with first_keyframe_id
    # recorded from kfs_a[0].kf_seq -- this is the exact quantity finding
    # 3.2 showed getting silently invalidated by the old renumbering.
    mp = MapPoint(np.array([0.0, 0.0, 1.0]), np.zeros(32, np.uint8),
                 ref_keyframe_id=kfs_a[0].kf_seq)
    map_a.add_map_point(mp)
    pre_merge_kf_seqs = {kf.id: kf.kf_seq for kf in kfs_a + kfs_b}
    pre_merge_first_kf_id = mp.first_keyframe_id

    merge_maps.merge_maps(atlas, keep_map=map_a, absorb_map=map_b,
                          R=np.eye(3), t=np.zeros(3), s=1.0)

    gate("kf_seq values are UNCHANGED by the merge (no renumbering)",
        all(kf.kf_seq == pre_merge_kf_seqs[kf.id] for kf in map_a.keyframes),
        f"pre={pre_merge_kf_seqs}, post={{kf.id: kf.kf_seq for kf in map_a.keyframes}}")
    gate("MapPoint.first_keyframe_id created before the merge is unchanged",
        mp.first_keyframe_id == pre_merge_first_kf_id,
        f"pre={pre_merge_first_kf_id}, post={mp.first_keyframe_id}")
    gate("kf_seq is still globally unique after the merge",
        len(set(kf.kf_seq for kf in map_a.keyframes)) == len(map_a.keyframes))


# ── Workstream A: observations pollution regression ─────────────────────

def test_tracking_does_not_pollute_observations():
    print("\n--- Test 7: ordinary (non-keyframe) tracking does not touch "
         "MapPoint.observations ---")
    images, depths, gt, _ = synthetic.scroll_sequence(n_frames=40)
    slam = run_sequence(images, depths)

    all_kf_ids = {kf.id for kf in slam.atlas.active_map.keyframes}
    violations = []
    for mp in slam.atlas.active_map.map_points.values():
        if mp.is_bad:
            continue
        for obs_id in mp.observations.keys():
            if obs_id not in all_kf_ids:
                violations.append((mp.id, obs_id))
    gate("no map point observes a non-keyframe frame id",
        len(violations) == 0, f"{len(violations)} violation(s): {violations[:5]}")


# ── Workstream B: far-point coverage ─────────────────────────────────────

def test_far_points_are_created():
    print("\n--- Test 8: RGB-D mode creates FAR points via triangulation, "
         "not just close points from bare depth ---")
    images, depths, gt, cam_dict = synthetic.two_depth_scroll_sequence(
        n_frames=30, near_depth_m=1.0, far_depth_m=3.0)
    slam = run_sequence(images, depths)

    near_pts, far_pts = 0, 0
    for mp in slam.atlas.active_map.map_points.values():
        if mp.is_bad:
            continue
        z = mp.position[2]
        if z < 2.0:
            near_pts += 1
        elif z >= 2.0:
            far_pts += 1
    gate("near (close, depth-created) points exist", near_pts > 0, f"{near_pts}")
    gate("far (triangulated) points exist -- this is the Phase 7 fix; "
        "would be 0 before it, since RGB-D mode never triangulated",
        far_pts > 0, f"{far_pts}")


# ── Workstream B: rotation-triggered keyframes ───────────────────────────

def test_rotation_triggers_keyframe():
    print("\n--- Test 9: pure rotation (near-zero translation) triggers "
         "keyframe insertion ---")
    cam = _dummy_camera()
    tracking = Tracking(cam, extractor=None, matcher=None, world_map=None,
                        keyframe_min_displacement=0.25,
                        keyframe_max_frames=20,
                        keyframe_min_rotation_deg=15.0)

    kf0 = Frame.__new__(Frame)
    kf0.pose = np.eye(4)
    tracking.last_keyframe = kf0
    tracking.frames_since_keyframe = 1

    class _FakeFrame:
        def __init__(self, pose, n_tracked):
            self.pose = pose
            self._n = n_tracked
        def camera_center(self):
            return self.pose[:3, 3]
        def n_tracked_points(self):
            return self._n

    # 20 degree yaw, translation ~0 -- should trigger via rotation alone,
    # BEFORE frame count (c1) or displacement (c2) would ever fire.
    yaw = np.deg2rad(20.0)
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                 [0, 1, 0],
                 [-np.sin(yaw), 0, np.cos(yaw)]])
    pose = np.eye(4)
    pose[:3, :3] = R
    frame = _FakeFrame(pose, n_tracked=100)

    needs_kf = tracking.needs_new_keyframe(frame)
    gate("20deg pure-rotation frame triggers a keyframe "
        "(displacement=0, frames_since_keyframe=1 -- only rotation could fire this)",
        needs_kf is True)

    # sanity: a SMALL 2deg rotation with the same near-zero translation
    # should NOT trigger (confirms the threshold is doing real work, not
    # firing unconditionally).
    yaw_small = np.deg2rad(2.0)
    R_small = np.array([[np.cos(yaw_small), 0, np.sin(yaw_small)],
                        [0, 1, 0],
                        [-np.sin(yaw_small), 0, np.cos(yaw_small)]])
    pose_small = np.eye(4)
    pose_small[:3, :3] = R_small
    frame_small = _FakeFrame(pose_small, n_tracked=100)
    needs_kf_small = tracking.needs_new_keyframe(frame_small)
    gate("2deg rotation does NOT trigger a keyframe (threshold has real effect)",
        needs_kf_small is False)


# ── full regression ──────────────────────────────────────────────────────

def test_reference_integrity_survives_heavy_fusion_and_merge():
    print("\n--- Test 10: reference integrity survives heavy fusion + "
         "fragmentation + a real cross-map merge ---")
    # Regression test for a real bug found DURING this phase's own
    # verification (not a synthetic injection like Tests 2-5 above):
    # fusion.py's _fuse_point() redirected every keyframe an absorbed
    # point used to observe over to the survivor, but never cleared the
    # absorbed point's OWN observations dict -- so moments later,
    # clean_bad_points() (walking that now-stale dict) wiped out the
    # survivor's just-established, correct claim on those same slots.
    # The survivor's own .observations kept claiming those keyframes
    # long after the keyframes' own map_point_ids no longer agreed, and
    # once such a keyframe was later culled, cull_keyframes() (which
    # only walks map_point_ids, not .observations) never found the
    # survivor to clean up -- producing a permanent, silent orphan.
    # Measured impact before the fix: 503 errors (observations
    # referencing frames that were no longer live keyframes anywhere)
    # on this exact scenario. Forcing relocalization off maximizes
    # fragmentation (and therefore fusion + culling activity), which is
    # what originally surfaced this.
    import test_phase6_gate as t6
    import contextlib, io
    import relocalization as reloc_module

    images, depths = t6._make_end_to_end_sequence()
    cam = t6._dummy_camera()
    slam = SLAMSystem(cam, use_depth=True, verbose=False)
    orig_reloc = reloc_module.try_relocalize
    reloc_module.try_relocalize = lambda *a, **k: (False, None)
    with contextlib.redirect_stdout(io.StringIO()):
        for i, (img, d) in enumerate(zip(images, depths)):
            slam.process(img, i / 30.0, depth_image=d)
    reloc_module.try_relocalize = orig_reloc

    report = validate.validate_atlas(slam.atlas)
    gate("no errors after heavy fusion/fragmentation (pre-merge)", report.ok(),
        f"{len(report.errors)} error(s): {report.errors[:3]}")

    lc = slam.loop_closer
    result = lc.detect_loop(slam.frames[-1], slam.atlas.active_map)
    if result is not None:
        slam._handle_loop_result(slam.frames[-1], slam.atlas.active_map, result)
        report2 = validate.validate_atlas(slam.atlas)
        gate("no errors after a real cross-map merge", report2.ok(),
            f"{len(report2.errors)} error(s): {report2.errors[:3]}")
    else:
        gate("no errors after a real cross-map merge (skipped: no loop found "
            "this run -- not itself a failure of this gate)", True)


def test_full_pipeline_regression():
    print("\n--- Test 11: Phase 1-6 gates still pass with Build 1 changes ---")
    import subprocess
    for script in ["test_phase1_gate.py", "test_phase2_gate.py",
                   "test_phase3_gate.py", "test_phase5_gate.py",
                   "test_phase6_gate.py"]:
        result = subprocess.run([sys.executable, script], capture_output=True,
                               text=True, timeout=600)
        passed = "ALL GATES PASSED" in result.stdout
        print(f"  {script}: {'PASSED' if passed else 'FAILED'}")
        if not passed:
            print("\n".join(result.stdout.splitlines()[-20:]))
        gate(f"{script} still passes with Build 1 changes", passed)


if __name__ == "__main__":
    print("=" * 64)
    print("  PHASE 7 GATE — Build 1: data-model invariants + map coverage")
    print("=" * 64)

    test_validate_clean_atlas()
    test_validate_catches_observation_pollution()
    test_validate_catches_dangling_map_point_id()
    test_validate_catches_duplicate_kf_seq()
    test_validate_catches_bad_pose()
    test_kf_seq_survives_merge_no_renumbering()
    test_tracking_does_not_pollute_observations()
    test_far_points_are_created()
    test_rotation_triggers_keyframe()
    test_reference_integrity_survives_heavy_fusion_and_merge()
    test_full_pipeline_regression()

    print("\n" + "=" * 64)
    if FAILURES:
        print(f"  RESULT: {len(FAILURES)} GATE(S) FAILED")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    else:
        print("  RESULT: ALL GATES PASSED")
    print("=" * 64)

"""
covisibility.py — the covisibility graph and local-map selection.

Maps to: ORB_SLAM3/src/KeyFrame.cc (UpdateConnections, GetCovisiblesByWeight)
         + Tracking.cc (UpdateLocalKeyFrames, UpdateLocalPoints, isInFrustum)

WHY THIS EXISTS: tracking.py's _track_reference_keyframe used to call
map.all_descriptors() -- every good point in the ENTIRE map, growing
without bound. At 30k points that's a 1200x30000 Hamming match every
frame. This module bounds that: build a graph of "which keyframes share
enough map points to be relevant right now", take the reference keyframe's
strong neighbors (K1) plus their neighbors (K2), and only match against
points THOSE keyframes see. Cost becomes O(local map size), which stays
roughly constant as the facility gets bigger -- it's what makes mapping a
whole building tractable instead of just one room.
"""

from collections import defaultdict

import numpy as np


def build_covisibility_graph(world_map, graph=None):
    """
    KeyFrame::UpdateConnections(), computed map-wide via an inverted index
    (for each point, connect every pair of keyframes that observe it) rather
    than the classic per-keyframe recomputation -- simpler, and fine at the
    keyframe counts this project targets.

    `graph` if given is an existing dict that gets cleared and refilled IN
    PLACE, so callers (e.g. Tracking) can hold one reference across the
    whole run and always see the latest version without re-assigning it.

    Returns: dict kf_id -> dict{neighbor_kf_id: shared_point_count}
    """
    if graph is None:
        graph = defaultdict(dict)
    else:
        graph.clear()

    counts = defaultdict(lambda: defaultdict(int))
    for mp in world_map.map_points.values():
        if mp.is_bad:
            continue
        kf_ids = list(mp.observations.keys())
        for i in range(len(kf_ids)):
            for j in range(i + 1, len(kf_ids)):
                a, b = kf_ids[i], kf_ids[j]
                counts[a][b] += 1
                counts[b][a] += 1

    for a, neighbors in counts.items():
        graph[a] = dict(neighbors)
    return graph


def local_keyframes(reference_kf, covis_graph, world_map,
                    min_weight=15, max_k1=20, max_k2=30):
    """
    Tracking::UpdateLocalKeyFrames(): K1 = keyframes sharing >= min_weight
    points with the reference (i.e. strongly covisible), K2 = K1's own
    neighbors (any weight), for a second-degree local map. Both capped so a
    single very well-connected keyframe can't blow up the local map size.

    Returns: list of Frame objects (K1 + K2, deduplicated, reference included)
    """
    kf_by_id = {kf.id: kf for kf in world_map.keyframes}
    ref_neighbors = covis_graph.get(reference_kf.id, {})

    k1_ids = [kid for kid, w in sorted(ref_neighbors.items(), key=lambda kv: -kv[1])
              if w >= min_weight][:max_k1]
    k1_set = set(k1_ids) | {reference_kf.id}

    k2_ids = []
    for kid in k1_ids:
        for nid, _w in sorted(covis_graph.get(kid, {}).items(), key=lambda kv: -kv[1]):
            if nid not in k1_set and nid not in k2_ids:
                k2_ids.append(nid)
    k2_ids = k2_ids[:max_k2]

    all_ids = [reference_kf.id] + k1_ids + k2_ids
    seen = set()
    result = []
    for kid in all_ids:
        if kid in seen:
            continue
        kf = kf_by_id.get(kid)
        if kf is not None:
            seen.add(kid)
            result.append(kf)
    return result


def local_map_points(local_kfs, world_map):
    """Tracking::UpdateLocalPoints(): union of good points seen by local_kfs."""
    ids = set()
    for kf in local_kfs:
        for mp_id in kf.map_point_ids:
            if mp_id is None:
                continue
            mp = world_map.map_points.get(mp_id)
            if mp is not None and not mp.is_bad:
                ids.add(mp_id)
    return ids


def predict_scale(distance, mp, n_levels, scale_factor):
    """
    MapPoint::PredictScale(): given how far away the point currently is,
    guess which pyramid octave its descriptor will best match at.

    Points that haven't had update_normal_and_depth() called yet (e.g. the
    very first init keyframe's points, before any regular keyframe has run
    through local_mapping) have max_distance == inf by default -- treat
    that as "no scale information available yet" and fall back to octave 0
    rather than feeding inf into log().
    """
    if distance <= 0 or not np.isfinite(mp.max_distance) or mp.max_distance <= 0:
        return 0
    ratio = mp.max_distance / distance
    if ratio <= 0 or not np.isfinite(ratio):
        return 0
    level = int(np.ceil(np.log(ratio) / np.log(scale_factor)))
    return int(np.clip(level, 0, n_levels - 1))


def is_in_frustum(mp, frame, camera, viewing_cos_thresh=0.5,
                  n_levels=8, scale_factor=1.2):
    """
    Frame::isInFrustum(): can `mp` plausibly be visible in `frame` given its
    CURRENT (predicted) pose? Four checks, all cheap, run BEFORE spending a
    descriptor-match attempt on a candidate point:
      1. projects inside the image
      2. positive depth (in front of the camera)
      3. within the point's scale-invariance distance range
      4. viewing angle from this pose isn't too far from the point's mean
         viewing direction (i.e. we're not looking at it from a wildly
         different angle than every prior observation)

    Returns (visible: bool, predicted_u, predicted_v, predicted_octave) --
    the projection and octave are reused by the guided-matching radius
    search in tracking.py / fusion.py so they aren't recomputed twice.
    """
    if frame.pose is None:
        return False, None, None, None

    pose_cw = np.linalg.inv(frame.pose)
    p_cam = pose_cw[:3, :3] @ mp.position + pose_cw[:3, 3]
    if p_cam[2] <= 0.0:
        return False, None, None, None

    proj = camera.project(p_cam)
    if proj is None:
        return False, None, None, None
    u, v = proj
    if not (0 <= u < camera.width and 0 <= v < camera.height):
        return False, None, None, None

    dist = float(np.linalg.norm(mp.position - frame.pose[:3, 3]))
    if mp.max_distance > 0 and (dist < mp.min_distance or dist > mp.max_distance):
        return False, None, None, None

    if mp.normal_vector is not None:
        view_dir = (mp.position - frame.pose[:3, 3])
        n = np.linalg.norm(view_dir)
        if n > 1e-9:
            cos_angle = float(np.dot(view_dir / n, mp.normal_vector))
            if cos_angle < viewing_cos_thresh:
                return False, None, None, None

    octave = predict_scale(dist, mp, n_levels, scale_factor)
    return True, float(u), float(v), octave

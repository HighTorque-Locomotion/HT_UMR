"""Sole-based ground alignment, swing clearance, and per-joint smoothing."""
from __future__ import annotations

import fnmatch
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation


SIDES = ("left", "right")


def positive(value, name, allow_zero=False):
    value = float(value)
    if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def smplx_foot_vertex_ids(faces, vertex_count):
    from retarget_body_segment_surface import load_smplx17_face_parts

    if vertex_count != 10475:
        raise ValueError("Sole ground alignment/clearance currently requires SMPL-X topology.")
    face_parts, names, _, _ = load_smplx17_face_parts()
    if len(face_parts) != len(faces):
        raise ValueError("SMPL-X face segmentation does not match the source mesh.")
    by_name = {name: part_id for part_id, name in names.items()}
    return [np.unique(faces[face_parts == by_name[side + "Foot"]]) for side in SIDES]


def source_sole_tracks(vertices, foot_ids):
    heights = np.stack([vertices[:, ids, 2].min(axis=1) for ids in foot_ids], axis=1)
    centers = np.stack([vertices[:, ids].mean(axis=1) for ids in foot_ids], axis=1)
    return heights, centers


def source_sole_contact_speeds(vertices, foot_ids, fps, band=.003):
    """Horizontal material-point speed near the sole, in robot metres/second.

    A low percentile within the contact patch preserves a stationary toe/heel
    during pivots, where the foot centre itself can move considerably.
    """
    positive(fps, "fps")
    positive(band, "source_contact_band")
    speeds = np.zeros((len(vertices), 2), dtype=np.float64)
    if len(vertices) < 2:
        return speeds
    for side, ids in enumerate(foot_ids):
        points = vertices[:, ids]
        lowest = points[:, :, 2].min(axis=1)
        values = np.linalg.norm(np.diff(points[:, :, :2], axis=0), axis=2) * fps
        patch = points[1:, :, 2] <= lowest[1:, None] + band
        values[~patch] = np.nan
        speeds[1:, side] = np.nanpercentile(values, 10, axis=1)
        speeds[0, side] = speeds[1, side]
    return speeds


def bvh_foot_phase(path, frame_ids, scale, unit_scale=.01, expected_frames=None, expected_fps=None, include_markers=False):
    """Use original ankle/toe motion for contact timing before SMPL-X bone transfer."""
    from nr_source import _bvh_local_tracks, _parse_bvh

    nodes, values, frame_time = _parse_bvh(Path(path))
    positive(frame_time, "BVH frame time")
    positive(scale, "robot scale")
    positive(unit_scale, "BVH unit scale")
    if expected_frames is not None and len(values) != int(expected_frames):
        raise ValueError("BVH/SMPL-X frame counts differ; native contact phases require matching source frames")
    if expected_fps is not None and not np.isclose(1 / frame_time, expected_fps):
        raise ValueError("BVH/SMPL-X frame rates differ; native contact phases require matching source timing")
    ids = np.asarray(frame_ids, dtype=int)
    if ids.ndim != 1 or not len(ids) or ids.min() < 0 or ids.max() >= len(values):
        raise ValueError("Invalid BVH contact frame indices")
    tracks = _bvh_local_tracks(nodes, values, np.arange(len(values)))
    rotations = np.empty((len(values), len(nodes), 3, 3))
    positions = np.empty((len(values), len(nodes), 3))
    for i, node in enumerate(nodes):
        if node["channels"]:
            local, quat = tracks[node["name"]]
            rotation = Rotation.from_quat(quat).as_matrix()
        else:
            local = np.broadcast_to(node["offset"], (len(values), 3))
            rotation = np.eye(3)
        parent = node["parent"]
        if parent < 0:
            rotations[:, i], positions[:, i] = rotation, local
        else:
            rotations[:, i] = rotations[:, parent] @ rotation
            positions[:, i] = positions[:, parent] + np.einsum("nij,nj->ni", rotations[:, parent], local)
    names = {node["name"]: i for i, node in enumerate(nodes)}
    required = [[side + marker for marker in ("Foot", "Toe")] for side in ("Left", "Right")]
    if any(name not in names for group in required for name in group):
        raise ValueError("BVH contact inference requires Left/RightFoot and Left/RightToe markers")
    marker_ids = [[names[name] for name in group] for group in required]
    points = positions[:, marker_ids] * float(unit_scale)
    velocities = np.zeros(points.shape[:-1])
    if len(points) > 1:
        velocities[1:] = np.linalg.norm(np.diff(points, axis=0), axis=3) / frame_time
        velocities[0] = velocities[1]
    heights = points[:, :, :, 1].copy()
    marker_floors = np.zeros((2, 2))
    for side in range(2):
        for marker in range(2):
            low = heights[:, side, marker] <= np.percentile(heights[:, side, marker], 20)
            stable = low & (velocities[:, side, marker] < .15)
            selected = heights[stable, side, marker] if stable.sum() >= 5 else heights[low, side, marker]
            marker_floors[side, marker] = np.median(selected)
            heights[:, side, marker] -= marker_floors[side, marker]
    centers = points.mean(axis=2)[:, :, [0, 2, 1]] * scale
    centers[:, :, 1] *= -1
    info = {"bvh": str(Path(path).resolve()), "marker_ground_source_m": marker_floors.tolist()}
    if include_markers:
        floor = float(np.min(marker_floors[:, 1]))
        joint_positions = positions[ids] * unit_scale
        joint_positions = joint_positions[:, :, [0, 2, 1]]
        joint_positions[:, :, 1] *= -1
        joint_positions[:, :, 2] -= floor
        info["tracks"] = {"marker_height": heights[ids], "marker_speed": velocities[ids],
                          "joints": {name: joint_positions[:, index] for name, index in names.items()}}
    return heights.min(axis=2)[ids] * scale, velocities.min(axis=2)[ids] * scale, centers[ids], info


def align_source_soles(vertices, joints, foot_ids, fps, config=None):
    """Estimate one constant floor from low, slowly translating source soles."""
    config = config or {}
    positive(fps, "fps")
    low_percentile = float(config.get("low_height_percentile", 20.0))
    if not 0 < low_percentile <= 50:
        raise ValueError("low_height_percentile must be in (0, 50]")
    speed_limit = positive(config.get("stationary_speed", .15), "stationary_speed")
    heights, centers = source_sole_tracks(vertices, foot_ids)
    if not np.isfinite(heights).all() or not np.isfinite(centers).all():
        raise ValueError("Source sole tracks must be finite.")
    speed = np.zeros_like(heights)
    if len(heights) > 1:
        speed = np.linalg.norm(np.gradient(centers[:, :, :2], 1.0 / fps, axis=0), axis=2)
    low = heights <= np.percentile(heights, low_percentile)
    candidates = heights[low & (speed <= speed_limit)]
    fallback = len(candidates) < min(5, heights.size)
    if fallback:
        candidates = heights[low]
    ground = float(np.median(candidates))
    vertices = np.asarray(vertices, dtype=np.float32).copy()
    joints = np.asarray(joints, dtype=np.float32).copy()
    vertices[:, :, 2] -= ground
    joints[:, :, 2] -= ground
    info = {"mode": "foot_surface", "ground_z_source_m": ground,
            "stationary_samples": len(candidates), "low_height_fallback": fallback,
            "low_height_percentile": low_percentile, "stationary_speed_source_m_s": speed_limit}
    return vertices, joints, ground, info


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x ** 3 * (10 + x * (-15 + 6 * x))


def plan_swing_clearance(sole_heights, fps, config=None, contact_speeds=None, foot_centers=None, phase_heights=None,
                         timing_exclusion=None):
    """Plan absolute peak clearance in robot metres, using the unmodified source.

    Complete lifts use height hysteresis and a minimum duration/peak to reject
    contact jitter and tiptoe pivots. Partial clips retain their original lift.
    A quintic envelope adds height without reducing an already high kick.
    """
    config = config or {}
    mode = config.get("mode", "peak")
    if config.get("transport_timing", {}).get("enabled", False) and mode != "motion_aware":
        raise ValueError("transport_timing requires mode=motion_aware")
    if mode == "motion_aware":
        return plan_motion_clearance(sole_heights, fps, config, contact_speeds, foot_centers, phase_heights, timing_exclusion)
    if mode != "peak":
        raise ValueError(f"Unknown foot clearance mode: {mode}")
    positive(fps, "fps")
    raw = np.asarray(sole_heights, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 2 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("sole_heights must be finite with shape (frames, 2)")
    minimum_peak = positive(config.get("min_peak_height", .035), "min_peak_height", True)
    enter = positive(config.get("swing_enter_height", .006), "swing_enter_height")
    leave = positive(config.get("swing_exit_height", .002), "swing_exit_height", True)
    detect_peak = positive(config.get("min_detected_peak", .008), "min_detected_peak")
    duration = positive(config.get("min_swing_duration", .08), "min_swing_duration")
    if leave >= enter or detect_peak < enter:
        raise ValueError("Require swing_exit_height < swing_enter_height <= min_detected_peak")
    heights = np.maximum(median_filter(raw, size=(3, 1), mode="nearest"), 0)
    boost = np.zeros_like(heights)
    swing = np.zeros_like(heights, dtype=bool)
    events = []
    min_frames = max(3, int(np.ceil(duration * fps)))
    for side in range(2):
        active = np.flatnonzero(heights[:, side] > leave)
        runs = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1)
        for run in runs:
            if not len(run):
                continue
            a, b = int(run[0]), int(run[-1])
            detected_peak = float(heights[a:b + 1, side].max())
            peak_index = a + int(np.argmax(raw[a:b + 1, side]))
            peak = float(max(0.0, raw[peak_index, side]))
            if detected_peak < enter:
                continue
            swing[a:b + 1, side] = True
            start, end = max(0, a - 1), min(len(raw) - 1, b + 1)
            complete = a > 0 and b < len(raw) - 1
            accepted = complete and len(run) >= min_frames and detected_peak >= detect_peak
            amount = max(0.0, minimum_peak - peak) if accepted else 0.0
            if amount > 0:
                ids = np.arange(start, end + 1)
                rise = _smoothstep((ids - start) / max(1, peak_index - start))
                fall = _smoothstep((end - ids) / max(1, end - peak_index))
                boost[ids, side] = amount * np.minimum(rise, fall)
            events.append({"side": SIDES[side], "side_index": side, "start": start, "end": end,
                           "peak_frame": peak_index, "complete": complete, "boosted": amount > 0,
                           "source_peak_m": peak, "target_peak_m": peak + amount})
    stance = (~swing) & (heights <= enter)
    target = np.maximum(raw, 0) + boost
    target[stance] = 0
    return {"source_heights": raw, "filtered_heights": heights, "target_heights": target,
            "boost": boost, "swing": swing, "stance": stance, "events": events}


def _runs(mask):
    ids = np.flatnonzero(mask)
    return [run for run in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1) if len(run)]


def plan_motion_clearance(sole_heights, fps, config, contact_speeds, foot_centers, phase_heights=None, timing_exclusion=None):
    """Separate low stationary contacts from moving feet, then clear transport.

    Clear the floor for the duration of meaningful horizontal transport,
    including lift preparation and landing ramps. Peak-only compensation is
    retained for vertical gestures and high kicks. This is geometric contact
    inference; ambiguous low double-foot translation keeps one support foot.
    """
    positive(fps, "fps")
    raw = np.asarray(sole_heights, dtype=np.float64)
    speeds = np.asarray(contact_speeds, dtype=np.float64)
    centers = np.asarray(foot_centers, dtype=np.float64)
    if (raw.ndim != 2 or raw.shape[1] != 2 or not len(raw) or speeds.shape != raw.shape
            or centers.shape != (*raw.shape, 3)
            or not all(np.isfinite(v).all() for v in (raw, speeds, centers))):
        raise ValueError("motion_aware clearance requires finite heights/speeds (N,2) and centres (N,2,3)")
    legacy_config = dict(config, mode="peak", transport_timing={})
    # Validate the peak/duration parameters using the legacy planner.
    plan_swing_clearance(raw[:1], fps, legacy_config)
    phase_raw = raw if phase_heights is None else np.asarray(phase_heights, dtype=np.float64)
    if phase_raw.shape != raw.shape or not np.isfinite(phase_raw).all():
        raise ValueError("Contact phase heights must be finite with shape (frames, 2)")
    heights = np.maximum(median_filter(phase_raw, size=(3, 1), mode="nearest"), 0)
    speeds = median_filter(speeds, size=(3, 1), mode="nearest")
    if np.any(speeds < 0):
        raise ValueError("Contact speeds must be nonnegative")
    vertical = np.gradient(heights, 1 / fps, axis=0) if len(raw) > 1 else np.zeros_like(raw)
    contact_h = positive(config.get("contact_height", .015), "contact_height")
    contact_v = positive(config.get("contact_speed", .10), "contact_speed")
    contact_vz = positive(config.get("contact_vertical_speed", .12), "contact_vertical_speed")
    move_on = positive(config.get("travel_speed_on", .12), "travel_speed_on")
    move_off = positive(config.get("travel_speed_off", .06), "travel_speed_off")
    if move_off >= move_on:
        raise ValueError("Require travel_speed_off < travel_speed_on")
    min_distance = positive(config.get("min_travel_distance", .01), "min_travel_distance")
    slow_distance = positive(config.get("slow_travel_distance", .04), "slow_travel_distance")
    # Zero disables the transport height floor while retaining contact phases.
    travel_h = positive(config.get("min_travel_height", .03), "min_travel_height", allow_zero=True)
    minimum_peak = positive(config.get("min_peak_height", .035), "min_peak_height", True)
    min_frames = max(3, int(np.ceil(float(config.get("min_swing_duration", .08)) * fps)))
    contact_frames = max(2, int(np.ceil(positive(config.get("contact_confirm_time", .10), "contact_confirm_time") * fps)))
    lead = max(1, int(np.ceil(positive(config.get("lift_lead_time", .10), "lift_lead_time") * fps)))
    tail = max(1, int(np.ceil(positive(config.get("landing_time", .10), "landing_time") * fps)))
    contact = np.zeros_like(raw, dtype=bool)
    transport = np.zeros_like(raw, dtype=bool)
    for side in range(2):
        candidate = ((heights[:, side] <= contact_h + 1e-7) & (speeds[:, side] <= contact_v + 1e-7)
                     & (np.abs(vertical[:, side]) <= contact_vz + 1e-7))
        for run in _runs(candidate):
            if len(run) >= contact_frames:
                contact[run, side] = True
        for run in _runs(~contact[:, side]):
            a, b = int(run[0]), int(run[-1])
            if (a > 0 and b < len(raw) - 1 and len(run) <= contact_frames
                    and contact[a - 1, side] and contact[b + 1, side]
                    and heights[run, side].max() <= contact_h and speeds[run, side].max() < move_on):
                contact[run, side] = True
        for run in _runs(speeds[:, side] > move_off):
            distance = float(np.linalg.norm(np.ptp(centers[run, side, :2], axis=0)))
            significant = speeds[run, side].max() >= move_on or distance >= slow_distance
            if len(run) >= min_frames and distance >= min_distance and significant:
                transport[run, side] = True
    # A confirmed low, slowly moving contact splits a transport episode; do not
    # merge successive steps merely because a small residual speed stays nonzero.
    transport &= ~contact
    for side in range(2):
        for run in _runs(transport[:, side]):
            if len(run) < min_frames:
                transport[run, side] = False
    forced_support = np.zeros_like(contact)
    if bool(config.get("preserve_low_support", True)):
        ambiguous = (transport.all(axis=1) & (heights < contact_h).all(axis=1)
                     & (np.abs(vertical) < contact_vz).all(axis=1))
        for run in _runs(ambiguous):
            a = int(run[0])
            previous = transport[a - 1] if a > 0 else np.zeros(2, dtype=bool)
            if previous.sum() == 1:
                support = int(np.flatnonzero(~previous)[0])
            else:
                support = int(np.argmin((speeds[run] + 5 * heights[run]).mean(axis=0)))
            transport[run, support] = False
            contact[run, support] = True
            forced_support[run, support] = True
    envelope = np.zeros_like(raw)
    for side in range(2):
        # Never consume a confirmed support interval to create an early lift.
        # Use short ramps within the available flight window, then a plateau.
        for run in _runs(~contact[:, side]):
            a, b = int(run[0]), int(run[-1])
            eligible = run[transport[run, side] | (heights[run, side] > contact_h)]
            if not len(eligible) or len(run) < 3:
                continue
            start, end = max(a, int(eligible[0]) - lead), min(b, int(eligible[-1]) + tail)
            if end - start < 2:
                continue
            rise_frames = min(lead, max(1, (end - start) // 2))
            fall_frames = min(tail, max(1, (end - start) // 2))
            ids = np.arange(start, end + 1)
            rise = _smoothstep((ids - start) / rise_frames)
            fall = _smoothstep((end - ids) / fall_frames)
            envelope[ids, side] = np.maximum(envelope[ids, side], np.minimum(rise, fall))
    stance = contact
    swing = ~stance
    target = np.maximum(raw, 0)
    target[contact] = 0
    target = np.maximum(target, travel_h * envelope)
    events = []
    for side in range(2):
        for run in _runs(swing[:, side]):
            a, b = int(run[0]), int(run[-1])
            start, end = max(0, a - 1), min(len(raw) - 1, b + 1)
            peak_index = a + int(np.argmax(raw[a:b + 1, side]))
            peak = max(0.0, float(raw[peak_index, side]))
            complete = a > 0 and b < len(raw) - 1
            significant = (heights[run, side].max() >= float(config.get("min_detected_peak", .008))
                           or transport[run, side].any())
            amount = max(0.0, minimum_peak - peak) if complete and len(run) >= min_frames and significant else 0.0
            if amount:
                # An artificial minimum peak belongs inside the flight window;
                # a flat/noisy source may put its argmax on the first frame.
                peak_index = (a + b) // 2
                ids = np.arange(start, end + 1)
                rise = _smoothstep((ids - start) / max(1, peak_index - start))
                fall = _smoothstep((end - ids) / max(1, end - peak_index))
                target[ids, side] = np.maximum(target[ids, side], np.maximum(raw[ids, side], 0) + amount * np.minimum(rise, fall))
            events.append({"side": SIDES[side], "side_index": side, "start": start, "end": end,
                           "peak_frame": peak_index, "complete": complete,
                           "boosted": bool(amount or (target[run, side] > np.maximum(raw[run, side], 0) + 1e-6).any()),
                           "source_peak_m": peak, "target_peak_m": float(target[run, side].max())})
    target[stance] = 0
    plan = {"source_heights": raw, "filtered_heights": heights, "target_heights": target,
            "boost": np.maximum(target - np.maximum(raw, 0), 0), "swing": swing, "stance": stance,
            "events": events, "contact_speeds": speeds, "transport": transport,
            "transport_envelope": envelope, "forced_support": forced_support, "phase_heights": phase_raw}
    timing = config.get("transport_timing", {})
    if timing_exclusion is not None:
        exclusion = np.asarray(timing_exclusion, dtype=bool)
        if exclusion.shape != (len(raw),):
            raise ValueError("timing_exclusion must have one flag per output frame")
        plan["timing_exclusion"] = exclusion
    if timing.get("enabled", False):
        plan_transport_timing(plan, fps, config)
    return plan


def plan_transport_timing(plan, fps, config):
    """Lift, traverse the original foot path, brake, then lower within each step.

    Contact boundaries stay fixed. Only complete flights with horizontal
    transport are retimed; source endpoints and the intervening path are kept.
    Continuous quintic phase maps also work for short, sparsely sampled steps.
    """
    timing = config.get("transport_timing", {})
    lift = positive(timing.get("lift_time", .08), "transport_timing.lift_time") * fps
    land = positive(timing.get("landing_time", .10), "transport_timing.landing_time") * fps
    fraction = positive(timing.get("max_phase_fraction", .25), "transport_timing.max_phase_fraction")
    if fraction >= .5:
        raise ValueError("transport_timing.max_phase_fraction must be below 0.5")
    clearance = positive(timing.get("move_clearance", .015), "transport_timing.move_clearance")
    full_height = positive(timing.get("free_motion_height", .028), "transport_timing.free_motion_height")
    travel_height = positive(config.get("min_travel_height", .03), "min_travel_height")
    if not clearance < full_height <= travel_height:
        raise ValueError("Require move_clearance < free_motion_height <= min_travel_height")
    low_speed = positive(timing.get("near_ground_speed", .03), "transport_timing.near_ground_speed", True)
    high_speed = positive(timing.get("free_motion_speed", 2.0), "transport_timing.free_motion_speed")
    if low_speed > high_speed:
        raise ValueError("near_ground_speed must not exceed free_motion_speed")
    positive(timing.get("velocity_slack_cost", 1e6), "transport_timing.velocity_slack_cost")
    count = len(plan["target_heights"])
    times = np.repeat(np.arange(count, dtype=float)[:, None], 2, axis=1)
    active = np.zeros((count, 2), dtype=bool)
    timing_events = []
    for side in range(2):
        for run in _runs(~plan["stance"][:, side]):
            a, b = int(run[0]) - 1, int(run[-1]) + 1
            if a < 0 or b >= count or not plan["transport"][run, side].any():
                continue
            if "timing_exclusion" in plan and plan["timing_exclusion"][a:b + 1].any():
                continue
            # Reserve distinct samples for lift, transport and braking.
            if b - a < 4:
                continue
            rise, fall = min(lift, fraction * (b - a)), min(land, fraction * (b - a))
            begin, finish = a + rise, min(b - fall, b - 2.0)
            ids = np.arange(a, b + 1)
            phase = _smoothstep((ids - begin) / (finish - begin))
            times[ids, side] = a + (b - a) * phase
            envelope = np.minimum(_smoothstep((ids - a) / rise), _smoothstep((b - ids) / fall))
            plan["target_heights"][ids, side] = np.maximum(plan["target_heights"][ids, side], travel_height * envelope)
            plan["transport_envelope"][ids, side] = envelope
            active[ids, side] = True
            timing_events.append({"side": SIDES[side], "side_index": side, "start": a, "end": b,
                                  "move_start": float(begin), "move_end": float(finish)})
    plan["target_heights"][plan["stance"]] = 0
    plan["boost"] = np.maximum(plan["target_heights"] - np.maximum(plan["source_heights"], 0), 0)
    plan["transport_sample_times"] = times
    plan["transport_timing_active"] = active
    # The last high frame is already a braking frame; it must not accelerate
    # toward an outdated horizontal target immediately before descent.
    height = plan["target_heights"]
    plan["transport_speed_height"] = np.minimum.reduce(
        [height, np.vstack([height[:1], height[:-1]]), np.vstack([height[1:], height[-1:]])])
    plan["transport_timing_events"] = timing_events
    for event in plan["events"]:
        side = event["side_index"]
        a, b = event["start"], event["end"]
        peak = a + int(np.argmax(height[a:b + 1, side]))
        event.update(peak_frame=peak, target_peak_m=float(height[peak, side]),
                     boosted=bool((plan["boost"][a:b + 1, side] > 1e-6).any()))


def sample_track(values, times):
    values = np.asarray(values)
    lower = np.floor(times).astype(int)
    upper = np.minimum(lower + 1, len(values) - 1)
    alpha = (times - lower).reshape((-1,) + (1,) * (values.ndim - 1))
    return (1 - alpha) * values[lower] + alpha * values[upper]


def retime_foot_vectors(vectors, plan, groups):
    """Retarget foot orientation and position from the same source instants."""
    if "transport_sample_times" not in plan:
        return vectors
    result = vectors.copy()
    for side, ids in enumerate(groups):
        values = sample_track(vectors[:, ids], plan["transport_sample_times"][:, side])
        result[:, ids] = values / np.maximum(np.linalg.norm(values, axis=2, keepdims=True), 1e-12)
    return result


def foot_slot_groups(part_ids, part_name_to_id):
    groups = [np.flatnonzero(part_ids == part_name_to_id[side + "Foot"]) for side in SIDES]
    if any(not len(ids) for ids in groups):
        raise ValueError("Both feet require source correspondence slots for clearance control.")
    return groups


def shift_foot_targets(source_slots, plan, groups):
    result = source_slots.copy()
    for side, ids in enumerate(groups):
        source_height = plan["source_heights"][:, side]
        if "transport_sample_times" in plan:
            times = plan["transport_sample_times"][:, side]
            result[:, ids] = sample_track(source_slots[:, ids], times)
            source_height = sample_track(source_height, times)
        delta = plan["target_heights"][:, side] - source_height
        result[:, ids, 2] += delta[:, None]
    return result


def sole_horizontal_velocity_rows(model, data, sole, previous_points, speed, fps, max_points=16):
    """Bound displacement of low material points, including toe/heel rotation.

    Each row represents an octagonal approximation to a horizontal speed
    limit. Low points at both frame endpoints are included, so changing the
    identity of the lowest vertex cannot evade the constraint.
    """
    points = sole_world_points(data, sole)
    low = np.minimum(points[:, 2] - points[:, 2].min(), previous_points[:, 2] - previous_points[:, 2].min())
    selected = list(np.argsort(low, kind="stable")[:max_points])
    selected.extend([int(np.argmin(points[:, 2])), int(np.argmin(previous_points[:, 2]))])
    patch = np.flatnonzero(low <= .003)
    for angle in np.arange(8) * np.pi / 4:
        direction = np.array([np.cos(angle), np.sin(angle)])
        selected.append(int(patch[np.argmax(sole["points"][patch, :2] @ direction)]))
    ids = np.unique(selected)
    points = points[ids]
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    body = sole["body_id"]
    mujoco.mj_jac(model, data, jp, jr, data.xpos[body], body)
    relative = points - data.xpos[body]
    jx = jp[0] + relative[:, 2, None] * jr[1] - relative[:, 1, None] * jr[2]
    jy = jp[1] + relative[:, 0, None] * jr[2] - relative[:, 2, None] * jr[0]
    displacement = points[:, :2] - previous_points[ids, :2]
    rows, bounds = [], []
    bound = speed / fps * np.cos(np.pi / 8)
    for angle in np.arange(8) * np.pi / 4:
        direction = np.array([np.cos(angle), np.sin(angle)])
        rows.extend(direction[0] * jx + direction[1] * jy)
        bounds.extend(bound - displacement @ direction)
    return np.asarray(rows), np.asarray(bounds)


def build_robot_soles(model, foot_bodies):
    """Exact minimum of each foot's visual convex hull in every orientation."""
    from mujoco_geom_surface import geom_local_mesh

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    soles = []
    for side in SIDES:
        name = foot_bodies.get(side)
        if not name:
            raise ValueError(f"robot.foot_bodies.{side} is required for foot clearance")
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
        if body <= 0:
            raise ValueError(f"Unknown foot body: {name}")
        geoms = np.flatnonzero(model.geom_bodyid == body)
        mesh_geoms = [g for g in geoms if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        selected = mesh_geoms or list(geoms)
        if not selected:
            raise ValueError(f"Foot body has no geometry: {name}")
        parts = []
        for geom in selected:
            vertices, _ = geom_local_mesh(model, geom)
            world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            parts.append((world - data.xpos[body]) @ data.xmat[body].reshape(3, 3))
        points = np.unique(np.concatenate(parts), axis=0)
        try:
            points = points[ConvexHull(points).vertices]
        except QhullError:
            pass
        soles.append({"body_id": body, "body_name": str(name), "points": points})
    return soles


def sole_world_points(data, sole):
    body = sole["body_id"]
    return sole["points"] @ data.xmat[body].reshape(3, 3).T + data.xpos[body]


def build_support_patches(model, soles, band=.001):
    """Find the sole face in the robot's neutral, flat-foot reference pose.

    Keep these material vertices fixed throughout the motion. Selecting the
    current lowest points instead would accept a tilted toe-only contact.
    """
    positive(band, "support_patch_band")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for sole in soles:
        points = sole_world_points(data, sole)
        ids = np.flatnonzero(points[:, 2] <= points[:, 2].min() + band)
        patch = points[ids]
        if len(ids) < 3 or np.linalg.matrix_rank(patch[:, :2] - patch[:, :2].mean(axis=0), tol=1e-6) < 2:
            raise ValueError(f"No flat sole face in neutral pose for {sole['body_name']}")
        sole["support_indices"] = ids


def support_patch_trajectory(model, qpos, soles):
    """Height of the highest sole-face vertex and heel/toe/edge height span."""
    data = mujoco.MjData(model)
    maximum, span = np.zeros((2, len(qpos), len(soles)))
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        for side, sole in enumerate(soles):
            heights = sole_world_points(data, sole)[sole["support_indices"], 2]
            maximum[frame, side], span[frame, side] = heights.max(), np.ptp(heights)
    return maximum, span


def sole_height_kinematics(model, data, sole):
    points = sole_world_points(data, sole)
    body = sole["body_id"]
    jac_pos, jac_rot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jac_pos, jac_rot, data.xpos[body], body)
    relative = points - data.xpos[body]
    rows = jac_pos[2] + relative[:, 1, None] * jac_rot[0] - relative[:, 0, None] * jac_rot[1]
    return points, rows


def update_stance_anchors(data, soles, stance, anchors, band=.003):
    """Anchor a material point in the current low contact patch per stance."""
    for side, sole in enumerate(soles):
        if not stance[side]:
            anchors.pop(side, None)
        elif side not in anchors:
            points = sole_world_points(data, sole)
            patch = points[:, 2] <= points[:, 2].min() + band
            local = sole["points"][patch].mean(axis=0)
            world = local @ data.xmat[sole["body_id"]].reshape(3, 3).T + data.xpos[sole["body_id"]]
            anchors[side] = {"local_point": local, "world_xy": world[:2].copy()}


def robot_sole_trajectory(model, qpos, soles, fps=None):
    data = mujoco.MjData(model)
    heights = np.zeros((len(qpos), 2))
    centers = np.zeros((len(qpos), 2, 3))
    speeds = np.zeros((len(qpos), 2))
    previous_points = [None, None]
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        for side, sole in enumerate(soles):
            points = sole_world_points(data, sole)
            lowest = int(np.argmin(points[:, 2]))
            heights[frame, side] = points[lowest, 2]
            centers[frame, side] = data.xpos[sole["body_id"]]
            if fps is not None and previous_points[side] is not None:
                speeds[frame, side] = np.linalg.norm(points[lowest, :2] - previous_points[side][lowest, :2]) * fps
            previous_points[side] = points
    return (heights, centers) if fps is None else (heights, centers, speeds)


def clearance_summary(source, target, actual, swing, stance, events, contact_speeds=None):
    """Report soft target tracking and stance quality without claiming exact tracking."""
    source, target, actual = (np.asarray(v) for v in (source, target, actual))
    swing, stance = np.asarray(swing, dtype=bool), np.asarray(stance, dtype=bool)
    if not all(v.shape == actual.shape for v in (source, target, swing, stance)):
        raise ValueError("Foot diagnostics must have matching (frames, 2) shapes")
    if actual.ndim != 2 or actual.shape[1] != 2 or not all(np.isfinite(v).all() for v in (source, target, actual)):
        raise ValueError("Foot diagnostic heights must be finite with shape (frames, 2)")
    event_rows = []
    for event in events:
        a, b, k = event["start"], event["end"], event["side_index"]
        row = dict(event)
        row["actual_peak_m"] = float(actual[a:b + 1, k].max())
        row["target_peak_m"] = float(target[a:b + 1, k].max())
        row["peak_shortfall_m"] = max(0.0, row["target_peak_m"] - row["actual_peak_m"])
        event_rows.append(row)
    result = {
        "min_robot_sole_height_m": float(actual.min()),
        "penetrating_frame_count_1mm": int(np.count_nonzero(actual.min(axis=1) < -.001)),
        "worst_ground_frame_index": int(np.argmin(actual.min(axis=1))),
        "stance_height_rms_m": float(np.sqrt(np.mean(actual[stance] ** 2))) if stance.any() else None,
        "stance_height_p95_m": float(np.percentile(np.abs(actual[stance]), 95)) if stance.any() else None,
        "swing_height_error_rms_m": float(np.sqrt(np.mean((actual[swing] - target[swing]) ** 2))) if swing.any() else None,
        "boosted_swing_count": sum(e["boosted"] for e in events), "events": event_rows,
    }
    if contact_speeds is not None:
        stable = stance.copy()
        stable[0] = False
        stable[1:] &= stance[:-1]
        values = np.asarray(contact_speeds)[stable]
        result["stance_lowest_point_xy_speed_p95_m_s"] = float(np.percentile(values, 95)) if values.size else None
    return result


def joint_smoothing_weights(joint_names, smooth_cost, temporal_cost, groups):
    """Explicit glob groups override the two global fallback weights."""
    smooth = np.full(len(joint_names), positive(smooth_cost, "smooth_cost", True))
    temporal = np.full(len(joint_names), positive(temporal_cost, "temporal_smooth_cost", True))
    assigned = set()
    audit = {}
    for group_name, group in groups.items():
        patterns = group.get("joints", [])
        if not isinstance(patterns, list) or not patterns or not all(isinstance(p, str) for p in patterns):
            raise ValueError(f"smoothing group {group_name!r} needs a nonempty list of joint patterns")
        matched = set()
        for pattern in patterns:
            ids = {i for i, name in enumerate(joint_names) if fnmatch.fnmatchcase(name, pattern)}
            if not ids:
                raise ValueError(f"Smoothing pattern {pattern!r} matches no scalar joints")
            matched.update(ids)
        if matched & assigned:
            raise ValueError(f"Overlapping joint smoothing groups: {group_name}")
        assigned.update(matched)
        ids = sorted(matched)
        if "smooth_cost" in group:
            smooth[ids] = positive(group["smooth_cost"], "smooth_cost", True)
        if "temporal_smooth_cost" in group:
            temporal[ids] = positive(group["temporal_smooth_cost"], "temporal_smooth_cost", True)
        audit[group_name] = [str(joint_names[i]) for i in ids]
    return smooth, temporal, audit

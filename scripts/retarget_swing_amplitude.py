"""Amplify source swing-leg targets before the existing retarget solve.

Closed-form two-link geometry increases source knee flexion, transports thigh
and shin positions/normals coherently, and translates the foot rigidly. There
is no second robot IK pass, qpos postprocessing, or absolute clearance floor.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

import retarget_foot_clearance as foot


def validate_config(config, clearance):
    gain = foot.positive(config.get("gain", 1.2), "swing_amplitude.gain")
    if gain < 1:
        raise ValueError("swing_amplitude.gain must be >= 1")
    foot.positive(config.get("min_swing_duration", .12), "swing_amplitude.min_swing_duration")
    if not clearance.get("enabled", False) or clearance.get("mode") != "motion_aware":
        raise ValueError("swing_amplitude requires enabled motion_aware foot_clearance")
    if clearance.get("min_travel_height", .03) != 0 or clearance.get("min_peak_height", .035) != 0:
        raise ValueError("swing_amplitude requires min_travel_height=min_peak_height=0")
    if clearance.get("transport_timing", {}).get("enabled", False):
        raise ValueError("swing_amplitude cannot be combined with transport_timing")
    return gain


def phase_envelope(plan, fps, config):
    """A C2 bump across complete flights, zero at adjoining support samples."""
    foot.positive(fps, "fps")
    minimum = foot.positive(config.get("min_swing_duration", .12), "swing_amplitude.min_swing_duration")
    swing = np.asarray(plan["swing"], dtype=bool)
    envelope = np.zeros(swing.shape, dtype=float)
    events = []
    for side in range(2):
        for run in foot._runs(swing[:, side]):
            a, b = int(run[0]), int(run[-1])
            complete = a > 0 and b < len(swing) - 1
            significant = bool(np.any(plan["transport"][run, side])
                               or np.max(plan["phase_heights"][run, side]) >= .008)
            accepted = complete and len(run) >= max(3, int(np.ceil(minimum * fps))) and significant
            if accepted:
                phase = (run - (a - 1)) / (b + 1 - (a - 1))
                envelope[run, side] = 64 * phase**3 * (1 - phase)**3
            events.append({"side": foot.SIDES[side], "side_index": side, "start": a, "end": b,
                           "complete": complete, "accepted": accepted})
    return envelope, events


def rotation_between(before, after):
    before = before / np.maximum(np.linalg.norm(before, axis=-1, keepdims=True), 1e-12)
    after = after / np.maximum(np.linalg.norm(after, axis=-1, keepdims=True), 1e-12)
    cross = np.cross(before, after)
    quat = np.concatenate([cross, (1 + np.sum(before * after, axis=-1))[:, None]], axis=-1)
    if np.any(np.linalg.norm(quat, axis=-1) < 1e-8):
        raise ValueError("Swing amplification cannot reverse a bone by 180 degrees")
    return Rotation.from_quat(quat).as_matrix()


def prepare(joints, joint_ids, plan, fps, config):
    """Compute rigid segment transforms in robot-scale world coordinates."""
    gain = foot.positive(config.get("gain", 1.2), "swing_amplitude.gain")
    if gain < 1:
        raise ValueError("swing_amplitude.gain must be >= 1")
    joints = np.asarray(joints, dtype=float)
    if joints.ndim != 3 or joints.shape[-1] != 3 or not np.isfinite(joints).all():
        raise ValueError("Source joints must be finite (frames, joints, 3)")
    envelope, events = phase_envelope(plan, fps, config)
    if envelope.shape != (len(joints), 2):
        raise ValueError("Swing plan and source joint frames differ")
    transforms = {}
    shifts = np.zeros((len(joints), 2, 3))
    angles = np.zeros((len(joints), 2))
    target_angles = np.zeros_like(angles)
    for side, (name, prefix) in enumerate(zip(foot.SIDES, ("L", "R"))):
        hip, knee, ankle = [joints[:, joint_ids[prefix + "_" + joint]] for joint in ("Hip", "Knee", "Ankle")]
        l1 = np.linalg.norm(knee - hip, axis=1)
        l2 = np.linalg.norm(ankle - knee, axis=1)
        radius = np.linalg.norm(ankle - hip, axis=1)
        if np.any(np.minimum(np.minimum(l1, l2), radius) < 1e-6):
            raise ValueError("Degenerate source leg geometry")
        direction = (ankle - hip) / radius[:, None]
        along = (l1*l1 - l2*l2 + radius*radius) / (2*radius)
        bend = knee - hip - along[:, None]*direction
        bend_norm = np.linalg.norm(bend, axis=1)
        angle = np.arccos(np.clip((radius*radius - l1*l1 - l2*l2) / (2*l1*l2), -1, 1))
        # Skip a whole event if its geometry cannot be amplified safely;
        # do not clip gain independently at each frame.
        for event in events:
            if event["side_index"] != side or not event["accepted"]:
                continue
            a, b = event["start"], event["end"] + 1
            if np.any(ankle[a:b, 2] >= hip[a:b, 2]) or np.any(bend_norm[a:b] < 1e-7):
                envelope[a:b, side] = 0
                event.update(accepted=False, reason="inverted_or_straight_leg")
                continue
            requested_angles = angle[a:b] * (1 + (gain - 1)*envelope[a:b, side])
            if gain > 1 and np.any(requested_angles >= np.pi - .01):
                envelope[a:b, side] = 0
                event.update(accepted=False, reason="excessive_knee_flexion",
                             max_requested_knee_flexion_deg=float(np.rad2deg(requested_angles).max()))
        effective = 1 + (gain - 1)*envelope[:, side]
        new_angle = angle * effective
        new_radius = np.sqrt(l1*l1 + l2*l2 + 2*l1*l2*np.cos(new_angle))
        new_along = (l1*l1 - l2*l2 + new_radius*new_radius) / (2*new_radius)
        new_bend = np.sqrt(np.maximum(l1*l1 - new_along*new_along, 0))
        new_ankle = hip + new_radius[:, None]*direction
        new_knee = hip + new_along[:, None]*direction + new_bend[:, None]*bend/np.maximum(bend_norm[:, None], 1e-12)
        active = (envelope[:, side] > 0) & (gain > 1)
        new_ankle[~active] = ankle[~active]
        new_knee[~active] = knee[~active]
        upper_rotation = rotation_between(knee-hip, new_knee-hip)
        lower_rotation = rotation_between(ankle-knee, new_ankle-new_knee)
        upper_rotation[~active] = np.eye(3)
        lower_rotation[~active] = np.eye(3)
        transforms[name + "UpLeg"] = dict(rotation=upper_rotation, origin=hip, target_origin=hip, active=active)
        transforms[name + "Leg"] = dict(rotation=lower_rotation, origin=knee, target_origin=new_knee, active=active)
        transforms[name + "Foot"] = dict(rotation=np.broadcast_to(np.eye(3), upper_rotation.shape),
                                          origin=ankle, target_origin=new_ankle, active=active)
        shifts[:, side] = new_ankle - ankle
        angles[:, side], target_angles[:, side] = angle, new_angle
    updated = dict(plan)
    updated["target_heights"] = plan["target_heights"] + shifts[:, :, 2]
    updated["boost"] = np.maximum(updated["target_heights"] - np.maximum(plan["source_heights"], 0), 0)
    updated["events"] = [dict(event) for event in plan["events"]]
    for event in updated["events"]:
        rows = slice(event["start"], event["end"] + 1)
        side = event["side_index"]
        event["target_peak_m"] = float(updated["target_heights"][rows, side].max())
        event["boosted"] = bool((updated["boost"][rows, side] > 1e-6).any())
    payload = {"swing_amplitude_envelope": envelope.astype(np.float32),
               "swing_amplitude_gain": (1+(gain-1)*envelope).astype(np.float32),
               "swing_amplitude_foot_translation": shifts.astype(np.float32),
               "swing_amplitude_source_knee_flexion": angles.astype(np.float32),
               "swing_amplitude_target_knee_flexion": target_angles.astype(np.float32)}
    report = {"config": config, "method": "source_leg_targets_before_single_retarget_solve", "events": events,
              "boosted_events": sum(event["accepted"] and gain > 1 for event in events),
              "skipped_excessive_flexion_events": sum(event.get("reason") == "excessive_knee_flexion" for event in events),
              "max_vertical_target_lift_m": float(shifts[:, :, 2].max(initial=0)),
              "max_source_knee_flexion_increase_deg": float(np.rad2deg(target_angles-angles).max(initial=0))}
    return transforms, updated, payload, report


def transform_slots(values, part_ids, part_name_to_id, transforms, normals=False):
    """Transport source point or normal targets, leaving other segments exact."""
    result = values.copy()
    for name, transform in transforms.items():
        slots = np.flatnonzero(part_ids == part_name_to_id[name])
        frames = np.flatnonzero(transform["active"])
        if not len(slots) or not len(frames):
            continue
        points = np.asarray(values[np.ix_(frames, slots)], dtype=float)
        if not normals:
            points -= transform["origin"][frames, None]
        rotated = np.einsum("nij,nkj->nki", transform["rotation"][frames], points)
        if not normals:
            rotated += transform["target_origin"][frames, None]
        result[np.ix_(frames, slots)] = rotated
    return result

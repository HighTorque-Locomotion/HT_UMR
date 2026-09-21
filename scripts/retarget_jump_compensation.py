"""Conservative BVH jump events and fixed-time robot COM compensation."""
from __future__ import annotations

import copy
import numpy as np
import mujoco
from scipy.ndimage import median_filter

from retarget_foot_clearance import SIDES, _runs, positive, sole_world_points


def detect_jumps(tracks, fps, config):
    """Classify observable supporting feet, not unmeasured contact forces.

    All thresholds here use unscaled human metres. Low fast points remain
    uncertain; flight requires positive clearance for both feet.
    """
    positive(fps, "fps")
    h = median_filter(np.asarray(tracks["marker_height"], float), size=(3, 1, 1), mode="nearest")
    speed = np.asarray(tracks["marker_speed"], float)
    joints = tracks["joints"]
    if h.ndim != 3 or h.shape[1:] != (2, 2) or speed.shape != h.shape:
        raise ValueError("Jump detection requires (frames, feet, markers) heights and speeds")
    if not np.isfinite(h).all() or not np.isfinite(speed).all():
        raise ValueError("Jump source tracks must be finite")
    contact_h = positive(config.get("contact_height", .02), "jump.contact_height")
    contact_v = positive(config.get("contact_speed", .30), "jump.contact_speed")
    release = positive(config.get("release_height", .006), "jump.release_height")
    enter = positive(config.get("air_height", .018), "jump.air_height")
    if release >= enter:
        raise ValueError("jump.release_height must be below air_height")
    min_duration = positive(config.get("min_flight_time", .10), "jump.min_flight_time")
    max_duration = positive(config.get("max_flight_time", .80), "jump.max_flight_time")
    if min_duration >= max_duration:
        raise ValueError("Invalid jump flight duration range")
    candidate_support = ((h <= contact_h) & (speed <= contact_v)).any(axis=2)
    support = np.zeros_like(candidate_support)
    for side in range(2):
        for run in _runs(candidate_support[:, side]):
            if len(run) >= 2:
                support[run, side] = True
    foot_h = h.min(axis=2)
    low = foot_h <= release
    boundary_contact = low | support
    hip = np.asarray(joints["Hips"], float)
    look = max(3, int(np.ceil(float(config.get("support_window", .16)) * fps)))
    sync = max(1, int(round(float(config.get("support_sync_time", .067)) * fps)))
    prep = max(2, int(round(positive(config.get("prepare_time", .16), "jump.prepare_time") * fps)))
    settle = max(2, int(round(positive(config.get("settle_time", .20), "jump.settle_time") * fps)))
    min_rise = positive(config.get("min_hip_rise", .025), "jump.min_hip_rise")
    min_hip = positive(config.get("min_hip_height", .45), "jump.min_hip_height")
    min_confidence = float(config.get("min_confidence", .80))
    if not 0 <= min_confidence <= 1:
        raise ValueError("jump.min_confidence must be in [0,1]")
    events, rejected = [], []
    for run in _runs((foot_h > release).all(axis=1) & ~support.any(axis=1)):
        a, b = int(run[0]) - 1, int(run[-1]) + 1
        if a < 0 or b >= len(h):
            rejected.append({"start": max(0, a), "end": min(len(h)-1, b), "reason": "partial_flight"}); continue
        duration = (b-a) / fps
        if not min_duration <= duration <= max_duration or len(run) < 3 or foot_h[run].min(axis=1).max() < enter:
            continue
        reason = None
        before = np.arange(max(0, a-look), a+1)
        after = np.arange(b, min(len(h), b+look+1))
        leave, land = {}, {}
        for side in range(2):
            if support[before, side].sum() >= 2 and boundary_contact[before, side].any():
                leave[side] = int(before[boundary_contact[before, side]][-1])
            if support[after, side].sum() >= 2 and boundary_contact[after, side].any():
                land[side] = int(after[boundary_contact[after, side]][0])
        takeoff = [side for side, frame in leave.items() if a-frame <= sync]
        landing = [side for side, frame in land.items() if frame-b <= sync]
        if not takeoff or not landing:
            reason = "uncertain_support_feet"
        t = np.arange(len(run)) / fps
        coef = np.polyfit(t, hip[run, 2], 2)
        error = float(np.sqrt(np.mean((np.polyval(coef, t)-hip[run, 2])**2)))
        rise = float(hip[run, 2].max() - max(hip[a, 2], hip[b, 2]))
        if hip[run, 2].min() < min_hip:
            reason = "low_body_or_nonfoot_support"
        if rise < min_rise or coef[1] <= .1 or 2*coef[0]*t[-1]+coef[1] >= -.1 or not 2 < -2*coef[0] < 20:
            reason = "no_clear_rise_and_fall"
        if error > max(.015, .25 * max(rise, 0)):
            reason = "nonballistic_hip_motion"
        for name in ("LeftHand", "RightHand", "LeftLeg", "RightLeg"):
            if name not in joints: continue
            points = np.asarray(joints[name], float)
            velocity = np.linalg.norm(np.gradient(points, 1/fps, axis=0), axis=1)
            if ((points[run, 2] < .08) & (velocity[run] < .30)).sum() >= 2:
                reason = "possible_hand_or_knee_support"
        for side in range(2):
            stationary = (speed[run, side].min(axis=1) < .06) & (foot_h[run, side] > enter)
            if any(len(segment) >= max(4, int(np.ceil(.12*fps))) for segment in _runs(stationary)):
                reason = "possible_elevated_support"
        alternating = len(takeoff) == len(landing) == 1 and takeoff != landing
        if alternating and (rise < float(config.get("min_leap_rise", .12)) or duration < .25):
            reason = "alternating_gait_or_uncertain_leap"
        horizontal_speed = float(np.median(np.linalg.norm(np.gradient(hip[run, :2], 1/fps, axis=0), axis=1)))
        if (not bool(config.get("allow_running_like", False)) and duration < .4 and rise < .08
                and horizontal_speed > 1.0):
            reason = "fast_low_flight_or_running"
        confidence = float(.85 + .15 * max(0, 1-error/.02))
        if confidence < min_confidence: reason = "low_confidence"
        event = {"start": a, "end": b, "duration_s": duration, "hip_rise_source_m": rise,
                 "hip_fit_gravity_source_m_s2": float(-2*coef[0]), "hip_fit_rms_source_m": error,
                 "takeoff_feet": takeoff, "landing_feet": landing,
                 "takeoff_contact_frames": {str(s): leave[s] for s in takeoff},
                 "landing_contact_frames": {str(s): land[s] for s in landing},
                 "confidence": confidence, "confidence_kind": "heuristic_score_not_probability"}
        event["hip_horizontal_speed_source_m_s"] = horizontal_speed
        if reason:
            rejected.append(dict(event, reason=reason)); continue
        label = ("double" if len(takeoff) == 2 else SIDES[takeoff[0]]) + "_to_" + ("double" if len(landing) == 2 else SIDES[landing[0]])
        event.update(label=label, prepare_start=max(0, a-prep), settle_end=min(len(h)-1, b+settle))
        events.append(event)
    exclusion = np.zeros(len(h), dtype=bool)
    override = np.full((len(h), 2), -1, dtype=np.int8)
    for event in events:
        p, a, b, q = event["prepare_start"], event["start"], event["end"], event["settle_end"]
        exclusion[p:q+1] = True
        override[p:q+1] = support[p:q+1]
    # Fill all native windows before enforcing event boundaries. Otherwise
    # the next jump's overlapping preparation can erase a prior landing.
    for event in events:
        p, a, b, q = event["prepare_start"], event["start"], event["end"], event["settle_end"]
        for side in event["takeoff_feet"]:
            stop = event["takeoff_contact_frames"][str(side)]
            # Keep only the native low contact stretch ending at push-off.
            first = stop
            while first > p and boundary_contact[first-1, side]: first -= 1
            override[first:stop+1, side] = 1
        for side in event["landing_feet"]:
            first = event["landing_contact_frames"][str(side)]
            stop = first
            while stop < q and boundary_contact[stop+1, side]: stop += 1
            override[first:stop+1, side] = 1
    # Flight overrides win when adjacent preparation/settling windows overlap.
    for event in events:
        override[event["start"]+1:event["end"]] = 0
    return {"events": events, "rejected": rejected, "exclusion": exclusion,
            "support_override": override, "native_support": support, "source_foot_height": foot_h}


def apply_contact_plan(foot_plan, analysis):
    override = analysis["support_override"]
    active = override >= 0
    foot_plan["stance"][active] = override[active].astype(bool)
    foot_plan["swing"][active] = ~foot_plan["stance"][active]
    airborne = active & foot_plan["swing"]
    foot_plan["target_heights"][airborne] = np.maximum(foot_plan["target_heights"][airborne], foot_plan["source_heights"][airborne])
    foot_plan["target_heights"][foot_plan["stance"]] = 0
    foot_plan["boost"] = np.maximum(foot_plan["target_heights"]-np.maximum(foot_plan["source_heights"], 0), 0)


def robot_com_trajectory(model, qpos):
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1: raise ValueError("Jump COM compensation requires one free-root robot")
    body = int(model.jnt_bodyid[free[0]])
    data = mujoco.MjData(model)
    com = np.empty((len(qpos), 3))
    for i, pose in enumerate(qpos):
        data.qpos[:] = pose; mujoco.mj_forward(model, data); com[i] = data.subtree_com[body]
    return com, body


def hermite(z0, v0, z1, v1, duration, u):
    return ((2*u**3-3*u**2+1)*z0 + (u**3-2*u**2+u)*duration*v0
            + (-2*u**3+3*u**2)*z1 + (u**3-u**2)*duration*v1)


def build_com_plan(com, analysis, fps, config):
    """Fit exact-gravity flight with its original endpoints/time, C1 bridges."""
    z = np.asarray(com[:, 2], float)
    velocity = np.gradient(z, 1/fps) if len(z) > 1 else np.zeros_like(z)
    gravity = positive(config.get("gravity", 9.81), "jump.gravity")
    max_extra = positive(config.get("max_extra_height", .20), "jump.max_extra_height")
    max_rise = positive(config.get("max_com_rise", .30), "jump.max_com_rise")
    min_com = positive(config.get("min_com_height", .20), "jump.min_com_height")
    events, rejected = [], []
    for event in analysis["events"]:
        a, b = event["start"], event["end"]
        T = (b-a) / fps
        t = np.arange(b-a+1) / fps
        target = z[a] + (z[b]-z[a])*t/T + .5*gravity*t*(T-t)
        vz0 = (z[b]-z[a])/T + .5*gravity*T
        vz1 = (z[b]-z[a])/T - .5*gravity*T
        if vz0 <= 0 or vz1 >= 0:
            rejected.append(dict(event, compensation_reason="incompatible_endpoint_heights")); continue
        if (target-z[a:b+1]).max() > max_extra or vz0**2/(2*gravity) > max_rise:
            rejected.append(dict(event, compensation_reason="configured_height_limit")); continue
        events.append(dict(event, takeoff_com_vz=vz0, landing_com_vz=vz1,
                           takeoff_com_z=float(z[a]), landing_com_z=float(z[b]),
                           ballistic_rise_m=float(vz0**2/(2*gravity))))
    while True:
        target = z.copy(); active = np.zeros(len(z), dtype=bool)
        invalid = set()
        for i, event in enumerate(events):
            p, a, b, q = event["prepare_start"], event["start"], event["end"], event["settle_end"]
            ids = np.arange(a, b+1); t = (ids-a)/fps; T = (b-a)/fps
            target[ids] = z[a]+(z[b]-z[a])*t/T+.5*gravity*t*(T-t)
            active[a:b+1] = True
            if i == 0 or events[i-1]["settle_end"] < p:
                ids = np.arange(p, a+1)
                if a > p: target[ids] = hermite(z[p], velocity[p], z[a], event["takeoff_com_vz"], (a-p)/fps, (ids-p)/(a-p))
                active[p:a+1] = True
            else:
                prev = events[i-1]; first = prev["end"]
                if first >= a:
                    invalid.update([i-1, i]); continue
                ids = np.arange(first, a+1)
                target[ids] = hermite(z[first], prev["landing_com_vz"], z[a], event["takeoff_com_vz"], (a-first)/fps, (ids-first)/(a-first))
                active[ids] = True
                if target[ids].min() < min_com: invalid.update([i-1, i])
            if i == len(events)-1 or events[i+1]["prepare_start"] > q:
                ids = np.arange(b, q+1)
                if q > b: target[ids] = hermite(z[b], event["landing_com_vz"], z[q], velocity[q], (q-b)/fps, (ids-b)/(q-b))
                active[b:q+1] = True
            if target[p:q+1].min() < min_com: invalid.add(i)
        if not invalid: break
        rejected.extend(dict(events[i], compensation_reason="configured_crouch_height_limit") for i in sorted(invalid))
        events = [event for i, event in enumerate(events) if i not in invalid]
    desired_velocity = np.r_[0, np.diff(target)*fps]
    return {"target_z": target, "target_vz": desired_velocity, "delta_z": target-z,
            "active": active, "events": events, "rejected": rejected, "baseline_com": com}


def compensate_targets(source_slots, foot_plan, com_plan, groups):
    result = source_slots.copy()
    delta = com_plan["delta_z"]
    result[:, :, 2] += delta[:, None]
    updated = copy.deepcopy(foot_plan)
    target = np.maximum(foot_plan["target_heights"] + delta[:, None], 0)
    # A raised, non-supporting foot must not acquire an unintended contact
    # just because the body crouches deeper during the support phase.
    target = np.maximum(target, np.minimum(foot_plan["target_heights"], .012))
    target[foot_plan["stance"]] = 0
    for side, ids in enumerate(groups):
        result[:, ids, 2] += (target[:, side]-foot_plan["target_heights"][:, side]-delta)[:, None]
    updated["target_heights"] = target
    updated["boost"] = np.maximum(target-np.maximum(updated["source_heights"], 0), 0)
    for event in updated["events"]:
        a, b, side = event["start"], event["end"], event["side_index"]
        peak = a+int(np.argmax(target[a:b+1, side]))
        event.update(peak_frame=peak, target_peak_m=float(target[peak, side]))
    return result, updated


def landing_anchor_overrides(model, qpos, soles, foot_plan, active):
    data = mujoco.MjData(model)
    overrides = {}
    for side, sole in enumerate(soles):
        for run in _runs(foot_plan["stance"][:, side] & active):
            frame = int(run[0]); data.qpos[:] = qpos[frame]; mujoco.mj_forward(model, data)
            points = sole_world_points(data, sole)
            local = sole["points"][points[:, 2] <= points[:, 2].min()+.003].mean(axis=0)
            world = local @ data.xmat[sole["body_id"]].reshape(3, 3).T + data.xpos[sole["body_id"]]
            overrides.setdefault(frame, {})[side] = {"local_point": local, "world_xy": world[:2]}
    return overrides

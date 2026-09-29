"""Anatomical ankle/toe directions for surface-based motion retargeting."""
from __future__ import annotations

import mujoco
import numpy as np

from retarget_foot_clearance import SIDES, sample_track


def unit_directions(vectors):
    vectors = np.asarray(vectors, dtype=float)
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(lengths < 1e-8):
        raise ValueError("Ankle and toe landmarks must be finite and distinct")
    return vectors / lengths


def build_robot_landmarks(model, config):
    """Bind explicit body-local points; the toe moves with the actual foot."""
    specs = []
    for side in SIDES:
        spec = config.get(side)
        if not isinstance(spec, dict):
            raise ValueError(f"robot.foot_landmarks.{side} is required")
        bound = {"side": side}
        for key in ("ankle", "toe"):
            name = spec.get(key + "_body", "")
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
            if body <= 0:
                raise ValueError(f"Unknown {side} {key} body: {name!r}")
            point = np.asarray(spec.get(key + "_local_pos", [0., 0., 0.]), dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError(f"Invalid {side} {key}_local_pos; expected a finite 3-vector")
            bound[key + "_body_id"] = body
            bound[key + "_local_pos"] = point
        ancestor = bound["toe_body_id"]
        while ancestor and ancestor != bound["ankle_body_id"]:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor != bound["ankle_body_id"]:
            raise ValueError(f"{side} toe body must descend from its ankle body")
        specs.append(bound)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for spec in specs:
        points = landmark_points(data, spec)
        unit_directions(points[1] - points[0])
    return specs


def landmark_points(data, spec):
    return np.asarray([
        data.xpos[spec[key + "_body_id"]]
        + data.xmat[spec[key + "_body_id"]].reshape(3, 3) @ spec[key + "_local_pos"]
        for key in ("ankle", "toe")
    ])


def direction_kinematics(model, data, spec):
    """Differentiate both endpoints, including the intervening ankle roll joint."""
    points = landmark_points(data, spec)
    delta = points[1] - points[0]
    direction = unit_directions(delta)
    jacobians = []
    for key, point in zip(("ankle", "toe"), points):
        jac = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jac, None, point, spec[key + "_body_id"])
        jacobians.append(jac)
    jac = (np.eye(3) - np.outer(direction, direction)) @ (jacobians[1] - jacobians[0]) / np.linalg.norm(delta)
    return direction, jac


def source_directions(joints, joint_ids, foot_plan=None):
    """Keep the source bone direction in the same world frame/time as surfaces.

    Normalize lengths only: no angle clamp, gain, or contact-dependent
    flattening. Ground/clearance and lateral translations cancel in the vector.
    """
    directions = []
    for side, prefix in enumerate(("L", "R")):
        delta = np.asarray(joints[:, joint_ids[prefix + "_Foot"]], float) - joints[:, joint_ids[prefix + "_Ankle"]]
        if foot_plan is not None and "transport_sample_times" in foot_plan:
            delta = sample_track(delta, foot_plan["transport_sample_times"][:, side])
        directions.append(unit_directions(delta))
    return np.stack(directions, axis=1)


def trajectory(model, qpos, specs):
    data = mujoco.MjData(model)
    points = np.empty((len(qpos), len(specs), 2, 3))
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        for side, spec in enumerate(specs):
            points[frame, side] = landmark_points(data, spec)
    directions = unit_directions(points[:, :, 1] - points[:, :, 0])
    return points, directions


def direction_summary(actual, target):
    error = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(actual, target), axis=-1),
                                 np.einsum("fsc,fsc->fs", actual, target)))
    return {"direction_error_mean_deg": error.mean(axis=0).tolist(),
            "direction_error_p95_deg": np.percentile(error, 95, axis=0).tolist(),
            "direction_error_max_deg": error.max(axis=0).tolist()}

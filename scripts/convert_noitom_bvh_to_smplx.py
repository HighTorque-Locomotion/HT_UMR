#!/usr/bin/env python3
"""Convert a Y-up, T-pose Noitom BVH to neutral SMPL-X for surface retargeting.

This is rotation transfer with rest-bone alignment, not a fitted body scan.
Body shape is neutral (zero betas); face and finger poses are zero. The named
Noitom skeleton is required so unsupported BVH layouts fail explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from nr_source import _bvh_local_tracks, _parse_bvh


# SMPL-X body order. Spine2 in BVH is collapsed into SMPL-X spine3 by using
# global rotations, preserving all four source spine rotations.
BODY_JOINT_NAMES = (
    "Hips", "LeftUpperLeg", "RightUpperLeg", "Spine",
    "LeftLowerLeg", "RightLowerLeg", "Spine1", "LeftFoot", "RightFoot",
    "Spine3", "LeftToe", "RightToe", "Neck", "LeftShoulder",
    "RightShoulder", "Head", "LeftUpperArm", "RightUpperArm",
    "LeftLowerArm", "RightLowerArm", "LeftHand", "RightHand",
)
BODY_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
)


def bvh_body_transforms(nodes, values, unit_scale=0.01, joint_names=BODY_JOINT_NAMES):
    """Return mapped world rotations, rest joints, and animated root position."""
    names = [node["name"] for node in nodes]
    if len(set(names)) != len(names):
        raise ValueError("BVH joint names must be unique.")
    missing = sorted(set(joint_names) - set(names))
    if missing:
        raise ValueError(f"Unsupported BVH skeleton; missing joints: {missing}")
    if nodes[0]["name"] != "Hips" or nodes[0]["parent"] != -1:
        raise ValueError("Expected Hips as the BVH root.")
    if not np.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError("unit_scale must be finite and positive.")
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("BVH motion must contain finite, nonempty frames.")
    tracks = _bvh_local_tracks(nodes, values, np.arange(len(values)))
    rotations = np.zeros((len(values), len(nodes), 3, 3), dtype=np.float64)
    rest = np.zeros((len(nodes), 3), dtype=np.float64)
    for index, node in enumerate(nodes):
        parent = node["parent"]
        if parent >= index:
            raise ValueError("BVH hierarchy must list parents before children.")
        if parent >= 0 and any(c.endswith("position") for c in node["channels"]):
            raise ValueError(f"Animated non-root translations are unsupported: {node['name']}")
        local = Rotation.from_quat(tracks[node["name"]][1]).as_matrix() if node["channels"] else np.eye(3)
        rotations[:, index] = local if parent < 0 else rotations[:, parent] @ local
        rest[index] = node["offset"] + (rest[parent] if parent >= 0 else 0)
    indices = [names.index(name) for name in joint_names]
    # Every mapped SMPL-X edge must correspond to an ancestor path in BVH.
    for index, parent in enumerate(BODY_PARENTS):
        if parent < 0:
            continue
        ancestor = nodes[indices[index]]["parent"]
        while ancestor >= 0 and ancestor != indices[parent]:
            ancestor = nodes[ancestor]["parent"]
        if ancestor < 0:
            raise ValueError(f"Incompatible BVH hierarchy at {joint_names[index]}")
    # Noitom's root position channels are absolute: adding OFFSET again would
    # raise zoo_3 by 92.2 cm. Non-root offsets still define the rest skeleton.
    root_position = tracks["Hips"][0] * unit_scale
    return rotations[:, indices], rest[indices] * unit_scale, root_position


def rest_alignment(source_rest, target_rest):
    """Map SMPL-X rest-bone directions into the source T-pose frame."""
    correction = np.repeat(np.eye(3)[None], len(BODY_PARENTS), axis=0)
    for joint in range(1, len(BODY_PARENTS)):
        children = np.flatnonzero(BODY_PARENTS == joint)
        if not len(children):
            continue
        source = source_rest[children] - source_rest[joint]
        target = target_rest[children] - target_rest[joint]
        source_lengths = np.linalg.norm(source, axis=-1, keepdims=True)
        target_lengths = np.linalg.norm(target, axis=-1, keepdims=True)
        if np.any(source_lengths < 1e-8) or np.any(target_lengths < 1e-8):
            raise ValueError(f"Zero-length rest bone below {BODY_JOINT_NAMES[joint]}")
        source = source / source_lengths
        target = target / target_lengths
        correction[joint] = Rotation.align_vectors(source, target)[0].as_matrix()
    return correction


def transfer_body_rotations(source_rotations, source_rest, target_rest, root_position):
    correction = rest_alignment(source_rest, target_rest)
    world = source_rotations @ correction[None]
    local = world.copy()
    for joint, parent in enumerate(BODY_PARENTS):
        if parent >= 0:
            local[:, joint] = np.swapaxes(world[:, parent], -1, -2) @ world[:, joint]
    poses = np.zeros((len(world), 165), dtype=np.float32)
    poses[:, :66] = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 66)
    # SMPL-X rotates around its rest pelvis; transl is a displacement, not the
    # world pelvis position. Subtract the rest pelvis once, without rotating it.
    trans = (root_position - target_rest[0]).astype(np.float32)
    return poses, trans, correction


def convert(input_path: Path, output_path: Path, model_dir: Path, unit_scale=0.01):
    import torch
    from smplx_model_loader import build_smplx_model

    nodes, values, frame_time = _parse_bvh(input_path)
    if not np.isfinite(frame_time) or frame_time <= 0:
        raise ValueError("BVH Frame Time must be finite and positive.")
    world, rest, root_position = bvh_body_transforms(nodes, values, unit_scale)
    model = build_smplx_model(model_dir, "neutral", 1)
    if not np.array_equal(model.parents[:22].cpu().numpy(), BODY_PARENTS):
        raise ValueError("Unexpected SMPL-X body joint order.")
    with torch.no_grad():
        target_rest = model().joints[0, :22].cpu().numpy().astype(np.float64)
    poses, trans, correction = transfer_body_rotations(world, rest, target_rest, root_position)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        poses=poses,
        trans=trans,
        betas=np.zeros(10, dtype=np.float32),
        gender=np.asarray("neutral"),
        model_type=np.asarray("smplx"),
        mocap_framerate=np.asarray(1.0 / frame_time),
        output_up=np.asarray("y"),
        source_bvh=np.asarray(str(input_path.resolve())),
        source_joint_names=np.asarray(BODY_JOINT_NAMES),
        source_unit_scale=np.asarray(unit_scale),
        rest_alignment=correction.astype(np.float32),
    )
    summary = {
        "source_bvh": str(input_path.resolve()),
        "smplx_model_dir": str(model_dir.resolve()),
        "output": str(output_path.resolve()),
        "frames": len(values),
        "fps": 1.0 / frame_time,
        "duration_seconds": len(values) * frame_time,
        "unit_scale": unit_scale,
        "output_up": "y",
        "method": "global_rotation_transfer_with_rest_bone_alignment",
        "shape": "neutral_zero_betas",
        "joint_mapping": dict(enumerate(BODY_JOINT_NAMES)),
        "notes": "Root position channels are absolute; face/fingers zero; no body-shape or joint-position fitting.",
    }
    output_path.with_suffix(".conversion.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[BVHToSMPLX] saved {output_path}: frames={len(values)} fps={1.0 / frame_time:.6f} poses={poses.shape}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smplx-model-dir", type=Path, default=Path("smpl"))
    parser.add_argument("--unit-scale", type=float, default=0.01, help="BVH units to metres (default: centimetres).")
    args = parser.parse_args()
    if args.out.suffix.lower() != ".npz":
        parser.error("--out must end with .npz")
    convert(args.input, args.out, args.smplx_model_dir, args.unit_scale)


if __name__ == "__main__":
    main()

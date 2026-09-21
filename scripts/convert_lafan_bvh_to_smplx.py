#!/usr/bin/env python3
"""Convert LAFAN's Y-up, centimetre BVH with joint-local bone axes to SMPL-X.

LAFAN offsets mostly point along local +X; their zero rotations are not a
human T-pose. Canonical joint bases remove that convention before applying
the same rest-bone alignment used by the Noitom converter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from convert_noitom_bvh_to_smplx import (
    BODY_PARENTS, bvh_body_transforms, transfer_body_rotations,
)
from nr_source import _parse_bvh


LAFAN_BODY_JOINT_NAMES = (
    "Hips", "LeftUpLeg", "RightUpLeg", "Spine",
    "LeftLeg", "RightLeg", "Spine1", "LeftFoot", "RightFoot",
    "Spine2", "LeftToe", "RightToe", "Neck", "LeftShoulder",
    "RightShoulder", "Head", "LeftArm", "RightArm",
    "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
)


def lafan_joint_bases():
    """Map canonical SMPL-X (left, up, forward) axes into each BVH joint.

    Torso: X up, Y forward. Legs: X down, Y forward.
    Arms: X towards the hand, Y backward, Z up on the left/down on the right.
    Feet/toes: X towards the toe, Y up. These bases also define leaf rotations,
    where LAFAN's zero-length End Sites cannot provide bone directions.
    """
    torso = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    leg = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
    foot = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    left_arm = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    right_arm = [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]
    return np.asarray([
        torso, leg, leg, torso, leg, leg, torso, foot, foot, torso,
        foot, foot, torso, left_arm, right_arm, torso,
        left_arm, right_arm, left_arm, right_arm, left_arm, right_arm,
    ], dtype=np.float64)


def lafan_body_transforms(nodes, values, unit_scale=0.01):
    world, _, root = bvh_body_transforms(
        nodes, values, unit_scale, joint_names=LAFAN_BODY_JOINT_NAMES,
    )
    by_name = {node["name"]: index for index, node in enumerate(nodes)}
    bases = lafan_joint_bases()
    rest = np.zeros((len(BODY_PARENTS), 3), dtype=np.float64)
    for joint, parent in enumerate(BODY_PARENTS):
        if parent < 0:
            continue
        node = nodes[by_name[LAFAN_BODY_JOINT_NAMES[joint]]]
        if node["parent"] != by_name[LAFAN_BODY_JOINT_NAMES[parent]]:
            raise ValueError(f"Expected direct LAFAN parent at {node['name']}")
        rest[joint] = rest[parent] + bases[parent].T @ node["offset"] * unit_scale
    return world @ bases[None], rest, root


def convert(input_path: Path, output_path: Path, model_dir: Path, unit_scale=0.01):
    import torch
    from smplx_model_loader import build_smplx_model

    nodes, values, frame_time = _parse_bvh(input_path)
    if not np.isfinite(frame_time) or frame_time <= 0:
        raise ValueError("BVH Frame Time must be finite and positive.")
    world, rest, root = lafan_body_transforms(nodes, values, unit_scale)
    model = build_smplx_model(model_dir, "neutral", 1)
    if not np.array_equal(model.parents[:22].cpu().numpy(), BODY_PARENTS):
        raise ValueError("Unexpected SMPL-X body joint order.")
    with torch.no_grad():
        target_rest = model().joints[0, :22].cpu().numpy().astype(np.float64)
    poses, trans, correction = transfer_body_rotations(world, rest, target_rest, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, poses=poses, trans=trans, betas=np.zeros(10, dtype=np.float32),
        gender=np.asarray("neutral"), model_type=np.asarray("smplx"),
        mocap_framerate=np.asarray(1.0 / frame_time), output_up=np.asarray("y"),
        source_bvh=np.asarray(str(input_path.resolve())), source_format=np.asarray("lafan"),
        source_joint_names=np.asarray(LAFAN_BODY_JOINT_NAMES),
        source_unit_scale=np.asarray(unit_scale),
        source_joint_bases=lafan_joint_bases().astype(np.float32),
        rest_alignment=correction.astype(np.float32),
    )
    summary = {
        "source_bvh": str(input_path.resolve()), "source_format": "lafan",
        "smplx_model_dir": str(model_dir.resolve()), "output": str(output_path.resolve()),
        "frames": len(values), "fps": 1.0 / frame_time,
        "duration_seconds": len(values) * frame_time, "unit_scale": unit_scale,
        "output_up": "y", "method": "joint_basis_and_rest_bone_aligned_global_rotation_transfer",
        "shape": "neutral_zero_betas", "joint_mapping": dict(enumerate(LAFAN_BODY_JOINT_NAMES)),
        "notes": "Absolute root position channels; face/fingers zero; no body-shape or joint-position fitting.",
    }
    output_path.with_suffix(".conversion.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[LAFANToSMPLX] saved {output_path}: frames={len(values)} fps={1.0 / frame_time:.6f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smplx-model-dir", type=Path, default=Path("smpl"))
    parser.add_argument("--unit-scale", type=float, default=0.01)
    args = parser.parse_args()
    if args.out.suffix.lower() != ".npz":
        parser.error("--out must end with .npz")
    convert(args.input, args.out, args.smplx_model_dir, args.unit_scale)


if __name__ == "__main__":
    main()

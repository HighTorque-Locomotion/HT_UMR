#!/usr/bin/env python3
"""Add legacy UMR BVH metadata to recommended 16-beta LAFAN SMPL-X motions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from convert_lafan_bvh_to_smplx import BODY_PARENTS, lafan_body_transforms
from nr_source import _parse_bvh


def prepare_metadata(source: Path, bvh: Path, output: Path) -> dict:
    if source.resolve() == output.resolve():
        raise ValueError("Prepared output must differ from the original converted file")
    with np.load(source, allow_pickle=False) as data:
        payload = {key: data[key].copy() for key in data.files}
    nodes, values, _ = _parse_bvh(bvh)
    count = len(values)
    poses = np.asarray(payload["poses"]).reshape(len(payload["poses"]), -1)
    if poses.shape != (count, 165) or np.asarray(payload["betas"]).size != 16:
        raise ValueError("Expected one SMPL-X pose per BVH frame and exactly 16 betas")
    if str(payload.get("output_up", "z")) != "z":
        raise ValueError("Recommended converter input must use Z-up")
    if not all(np.isfinite(payload[key]).all() for key in ("poses", "trans", "betas")):
        raise ValueError("Motion arrays must be finite")
    canonical, _, _ = lafan_body_transforms(nodes, values)
    local = Rotation.from_rotvec(poses[:, :66].reshape(-1, 3)).as_matrix().reshape(count, 22, 3, 3)
    world = local.copy()
    for joint, parent in enumerate(BODY_PARENTS):
        if parent >= 0:
            world[:, joint] = world[:, parent] @ local[:, joint]
    to_z_up = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    mappings = canonical.swapaxes(-1, -2) @ (to_z_up.T @ world)
    alignment = np.stack([Rotation.from_matrix(mappings[:, joint]).mean().as_matrix() for joint in range(22)])
    error = float(np.max(np.abs(to_z_up @ (canonical @ alignment) - world)))
    spread = float(Rotation.from_matrix((mappings @ alignment.swapaxes(-1, -2)).reshape(-1, 3, 3)).magnitude().max())
    if error >= 2e-6 or spread >= 2e-6:
        raise ValueError(f"No constant BVH/SMPL-X rest mapping: error={error}, spread={spread}")
    payload.update(source_bvh=np.asarray(str(bvh.resolve())), source_format=np.asarray("lafan"),
                   source_unit_scale=np.asarray(.01), output_up=np.asarray("z"),
                   rest_alignment=alignment.astype(np.float32),
                   source_converter=np.asarray("jaraujo98/lafan_to_smplx + num_betas=16 compatibility argument"))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    with np.load(source, allow_pickle=False) as original, np.load(output, allow_pickle=False) as prepared:
        for key in original.files:
            np.testing.assert_array_equal(original[key], prepared[key])
    return {"frames": count, "num_betas": 16, "original_arrays_unchanged": True,
            "max_rotation_matrix_error": error, "max_alignment_spread_rad": spread}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-bvh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_metadata(args.input, args.source_bvh, args.out))


if __name__ == "__main__":
    main()

"""Remove BVH-to-SMPL-X rest alignment from surface orientation targets.

The converted BVH rest pose is generally NOT SMPL-X's zero pose. Calibrate
reference vectors before transport and optionally match lateral reference
positions to the robot. Neither operation imposes a world-space foot pose.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from convert_noitom_bvh_to_smplx import BODY_PARENTS
from retarget_body_segment_surface import triangle_frames


def reference_body_pose(rest_alignment):
    """SMPL-X local rotations for a canonical BVH rest pose, not frame zero."""
    world = np.asarray(rest_alignment, dtype=np.float64)
    if world.shape != (len(BODY_PARENTS), 3, 3) or not np.isfinite(world).all():
        raise ValueError("BVH rest_alignment must be finite with shape (22, 3, 3)")
    if (not np.allclose(world @ world.swapaxes(-1, -2), np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(world), 1, atol=1e-5)):
        raise ValueError("BVH rest_alignment must contain proper rotation matrices")
    local = world.copy()
    for joint, parent in enumerate(BODY_PARENTS):
        if parent >= 0:
            local[joint] = world[parent].T @ world[joint]
    return Rotation.from_matrix(local).as_rotvec().reshape(1, 66).astype(np.float32)


def lafan_reference_alignment(rest_alignment, nodes, values):
    """Account for LAFAN's fixed ankle-to-toe frame offset.

    LAFAN's foot X axis follows the ankle-to-toe bone, not the sole plane.
    Its fixed leaf toe frame supplies the sole reference. Only a constant
    local transform is used, never the first frame's world-space foot pose.
    """
    from convert_lafan_bvh_to_smplx import lafan_joint_bases
    from nr_source import _bvh_local_tracks

    bases = lafan_joint_bases()
    tracks = _bvh_local_tracks(nodes, values, np.arange(len(values)))
    world = np.asarray(rest_alignment, dtype=np.float64).copy()
    audit = {}
    for foot, toe, name in ((7, 10, "LeftToe"), (8, 11, "RightToe")):
        node = next(n for n in nodes if n["name"] == name)
        if nodes[node["parent"]]["name"] != name.replace("Toe", "Foot"):
            raise ValueError(f"Unexpected LAFAN parent for {name}")
        local = bases[foot].T @ Rotation.from_quat(tracks[name][1]).as_matrix() @ bases[toe]
        spread = Rotation.from_matrix(local @ local[0].T).magnitude().max()
        if spread > 1e-4:
            raise ValueError(f"LAFAN {name} is animated; a fixed toe reference cannot be inferred")
        fixed = Rotation.from_matrix(local).mean().as_matrix()
        world[foot] = fixed.T @ world[foot]
        audit[name] = {"fixed_local_rotation_deg": Rotation.from_matrix(fixed).as_euler("xyz", degrees=True).tolist(),
                       "max_variation_rad": float(spread)}
    return world, audit


def bvh_rest_geometry(sequence, model_dir):
    """Reconstruct the reference using metadata saved by both BVH converters."""
    import torch
    from smplx_model_loader import build_smplx_model

    source = Path(sequence["source_file"])
    with np.load(source, allow_pickle=False) as metadata:
        if "rest_alignment" not in metadata or "source_bvh" not in metadata:
            raise ValueError("bvh_rest_calibration requires a converted BVH NPZ with rest_alignment")
        alignment = metadata["rest_alignment"]
        reference_body_pose(alignment)  # Validate before composing any transforms.
        audit = {"reference": "converter_rest_alignment"}
        if str(metadata.get("source_format", "")) == "lafan":
            from nr_source import _parse_bvh
            nodes, values, _ = _parse_bvh(Path(str(metadata["source_bvh"])))
            alignment, toe_audit = lafan_reference_alignment(alignment, nodes, values)
            audit["lafan_fixed_toe_frames"] = toe_audit
        pose = reference_body_pose(alignment)
    betas = np.asarray(sequence.get("beta", np.zeros(10)), dtype=np.float32).reshape(-1)[:10]
    betas = np.pad(betas, (0, 10 - len(betas)))
    model = build_smplx_model(model_dir, str(sequence.get("gender", "neutral")), 1)
    with torch.no_grad():
        output = model(global_orient=torch.from_numpy(pose[:, :3]),
                       body_pose=torch.from_numpy(pose[:, 3:]),
                       betas=torch.from_numpy(betas[None]))
    return (output.vertices[0].cpu().numpy().astype(np.float32),
            output.joints[0].cpu().numpy().astype(np.float32), audit)


def calibrate_lateral_positions(source_slots, source_reference_slots, robot_reference_slots,
                                groups, selected_slot_ids, lateral_directions, sample_times=None):
    """Match each segment's reference lateral centre, preserving motion and Z.

    Reference slots use SMPL coordinates (X left). Lateral directions are
    the animated pelvis heading's left axis in world XY. Constant offsets
    follow that heading, with the same optional foot retiming as positions.
    There is no minimum stance width, contact test or frame-zero pose fit.
    """
    from retarget_foot_clearance import sample_track

    result = np.asarray(source_slots).copy()
    directions = np.asarray(lateral_directions)
    if directions.shape != (len(result), 2):
        raise ValueError("Lateral reference directions must have shape (frames, 2)")
    audit = {}
    for name, ids in groups.items():
        selected = np.intersect1d(ids, selected_slot_ids)
        if not len(selected):
            raise ValueError(f"No selected surface slots for lateral calibration: {name}")
        offset = float(np.mean(robot_reference_slots[selected, 0] - source_reference_slots[selected, 0]))
        displacement = offset * directions
        if sample_times is not None and name in sample_times:
            displacement = sample_track(displacement, sample_times[name])
        result[:, ids, :2] += displacement[:, None]
        audit[name] = offset
    return result, audit


def calibrate_reference_normals(normals, binding, template_vertices, faces,
                                reference_vertices, slot_mask):
    """Rebase selected reference vectors without changing their motion.

    Existing transport is F_motion @ F_smpl.T @ normal. Pre-rotating the
    vector by F_smpl @ F_bvh.T makes it F_motion @ F_bvh.T @ normal. In the
    BVH rest pose this gives the robot reference normal; a genuine foot
    rotation is still transported in full, including during support.
    """
    result = np.asarray(normals, dtype=np.float32).copy()
    mask = np.asarray(slot_mask, dtype=bool)
    if mask.shape != (len(result),):
        raise ValueError("BVH calibration mask must have one entry per surface slot")
    if np.asarray(reference_vertices).shape != np.asarray(template_vertices).shape:
        raise ValueError("BVH reference and SMPL-X template must have the same topology")
    slot_faces = np.asarray(faces)[np.asarray(binding["face_ids"])[mask]]
    if len(slot_faces):
        source_basis = triangle_frames(template_vertices, slot_faces)
        reference_basis = triangle_frames(reference_vertices, slot_faces)
        correction = source_basis @ reference_basis.swapaxes(-1, -2)
        result[mask] = np.einsum("nij,nj->ni", correction, result[mask])
        result[mask] /= np.maximum(np.linalg.norm(result[mask], axis=1, keepdims=True), 1e-12)
    return result

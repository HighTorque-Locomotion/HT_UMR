#!/usr/bin/env python3
"""Render a SMPL-X motion as a moving triangle mesh for comparison videos."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "2")

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from view_smpl_mujoco import compute_vertex_normals, update_dynamic_mesh
import smpl_surface_retarget_common as common


def build_scene(vertices, faces, width, height, ground_reflectance=.1):
    values = lambda array: " ".join(map(str, np.asarray(array).reshape(-1)))
    normals = compute_vertex_normals(vertices, faces)
    xml = f"""<mujoco>
      <visual><global offwidth="{width}" offheight="{height}"/></visual>
      <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"
          rgb1="0.16 0.22 0.28" rgb2="0.24 0.32 0.40"/>
        <material name="floor" texture="grid" texrepeat="20 20" reflectance="{ground_reflectance}"/>
        <mesh name="human" vertex="{values(vertices)}" face="{values(faces)}"
          normal="{values(normals)}"/>
      </asset>
      <worldbody>
        <light pos="0 -3 5" dir="0 1 -1" directional="true"/>
        <geom type="plane" size="0 0 0.1" material="floor"/>
        <body name="human" mocap="true">
          <geom type="mesh" mesh="human" rgba="0.25 0.65 0.95 1"
            contype="0" conaffinity="0"/>
        </body>
      </worldbody>
    </mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--smplx-model-dir", type=Path, default=Path("smpl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=int, default=300)
    parser.add_argument("--end", type=int, default=900, help="Exclusive source frame index")
    parser.add_argument("--stride", type=int, default=1, help="Source frame step")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--scale-from-result", type=Path, help="Use saved smpl_scale for display only")
    parser.add_argument("--ground-align", choices=("global_min", "none", "result"), default="global_min",
                        help="global_min applies a display floor shift; none preserves input Z; result reuses ground_z from --scale-from-result")
    parser.add_argument("--camera-distance", type=float, default=1.5)
    parser.add_argument("--camera-azimuth", type=float, default=-135)
    parser.add_argument("--camera-elevation", type=float, default=-18)
    parser.add_argument("--camera-lookat-height", type=float,
                        help="Fixed camera target height in robot/world metres; otherwise follows the pelvis Z")
    parser.add_argument("--camera-follow-heading", action="store_true",
                        help="Treat azimuth as an offset from the saved robot root heading; requires --scale-from-result")
    parser.add_argument("--ground-reflectance", type=float, default=.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Check model/mesh without rendering or writing files")
    args = parser.parse_args()
    if args.ground_align == "result" and args.scale_from_result is None:
        parser.error("--ground-align result requires --scale-from-result")
    if args.camera_follow_heading and args.scale_from_result is None:
        parser.error("--camera-follow-heading requires --scale-from-result")
    if not np.isfinite([args.camera_distance, args.camera_azimuth, args.camera_elevation]).all() or args.camera_distance <= 0:
        parser.error("Camera angles must be finite and distance must be positive")
    if args.camera_lookat_height is not None and not np.isfinite(args.camera_lookat_height):
        parser.error("--camera-lookat-height must be finite")
    if not 0 <= args.ground_reflectance <= 1:
        parser.error("--ground-reflectance must be in [0, 1]")
    sequence = common.load_smplx_npz_motion(args.data)
    if not 0 <= args.start < args.end <= len(sequence["pose_aa"]):
        parser.error("Frame range must lie inside the input motion")
    if args.fps <= 0 or min(args.width, args.height) <= 0 or args.stride <= 0:
        parser.error("FPS, image dimensions and stride must be positive")
    ids = np.arange(args.start, args.end, args.stride)
    if args.dry_run:
        ids = ids[:2]
    vertices, joints, faces = common.smplx_motion_vertices_joints(
        sequence, ids, args.smplx_model_dir, device=args.device,
        smplx_batch_size=64, smplx_batch_size_max=64,
    )
    vertices = common.source_points_to_retarget_frame(vertices, "smplx", sequence["output_up"])
    joints = common.source_points_to_retarget_frame(joints, "smplx", sequence["output_up"])
    scale = 1.0
    result_ground = None
    saved_sole = None
    camera_heading = None
    if args.scale_from_result:
        with np.load(args.scale_from_result, allow_pickle=False) as result:
            scale = float(result["smpl_scale"].reshape(-1)[0])
            if args.camera_follow_heading:
                result_ids = np.asarray(result["frame_ids"], dtype=int).reshape(-1)
                indices = np.searchsorted(result_ids, ids)
                if np.any(indices >= len(result_ids)) or not np.array_equal(result_ids[indices], ids):
                    raise ValueError("Camera reference does not contain the selected source frames")
                from scipy.spatial.transform import Rotation
                q = np.asarray(result["qpos"])[indices, 3:7]
                forward = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()[:, :, 0]
                camera_heading = np.rad2deg(np.arctan2(forward[:, 1], forward[:, 0]))
            if "smplx_num_betas" in result:
                expected_count = int(np.asarray(result["smplx_num_betas"]).reshape(-1)[0])
                if common.smplx_num_betas(args.smplx_model_dir) != expected_count:
                    raise ValueError(f"Source renderer model must use num_betas={expected_count}")
            if args.ground_align == "result":
                result_ground = float(np.asarray(result["ground_z"]).reshape(-1)[0])
                if "source_sole_height" in result and "frame_ids" in result:
                    result_ids = np.asarray(result["frame_ids"], dtype=int).reshape(-1)
                    if len(result_ids) == 0 or np.any(np.diff(result_ids) <= 0):
                        raise ValueError("Saved result frame_ids must be strictly increasing")
                    indices = np.searchsorted(result_ids, ids)
                    if np.any(indices >= len(result_ids)) or not np.array_equal(result_ids[indices], ids):
                        raise ValueError("Selected source frames are not present in the scale/reference result")
                    saved_sole = np.asarray(result["source_sole_height"])[indices]
                    sides = np.asarray(result.get("foot_sides", ["left", "right"])).astype(str).reshape(-1).tolist()
                    if sorted(sides) != ["left", "right"]:
                        raise ValueError(f"Unsupported saved foot order: {sides}")
                    saved_sole = saved_sole[:, [sides.index("left"), sides.index("right")]]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Display scale must be finite and positive")
    # One fixed floor offset for the entire selected clip; no per-frame IK.
    floor_offset = float(vertices[:, :, 2].min()) if args.ground_align == "global_min" else 0.0
    if result_ground is not None:
        if not np.isfinite(result_ground):
            raise ValueError("Saved source ground offset must be finite")
        floor_offset = result_ground
    vertices[:, :, 2] -= floor_offset
    joints[:, :, 2] -= floor_offset
    vertices *= scale
    joints *= scale
    source_height_error = None
    if saved_sole is not None:
        from retarget_foot_clearance import smplx_foot_vertex_ids, source_sole_tracks
        sole_ids = smplx_foot_vertex_ids(faces, vertices.shape[1])
        actual_sole, _ = source_sole_tracks(vertices, sole_ids)
        if actual_sole.shape != saved_sole.shape or not np.isfinite(saved_sole).all() or not np.isfinite(actual_sole).all():
            raise ValueError("Invalid source sole-height reference")
        source_height_error = float(np.max(np.abs(actual_sole - saved_sole)))
        if source_height_error > 2e-6:
            raise ValueError(f"Reconstructed source heights differ from saved reference by {source_height_error:g} m; check the SMPL-X model and shape settings")
        print(f"[SMPLXVideo] source-height verification error={source_height_error:.3g} m")
    local_vertices = vertices - joints[:, :1]
    model = build_scene(local_vertices[0], faces, args.width, args.height, args.ground_reflectance)
    mesh_id = model.mesh("human").id
    if model.mesh_vertnum[mesh_id] != vertices.shape[1]:
        raise ValueError("MuJoCo changed the source mesh vertex count")
    data = mujoco.MjData(model)
    update_dynamic_mesh(model, mesh_id, local_vertices[-1], faces)
    data.mocap_pos[0] = joints[-1, 0]
    mujoco.mj_forward(model, data)
    if args.dry_run:
        print(f"[SMPLXVideo] dry run OK: vertices={vertices.shape[1]}, faces={len(faces)}, scale={scale}")
        return
    camera = mujoco.MjvCamera()
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        with imageio.get_writer(str(args.out), fps=args.fps, macro_block_size=1) as writer:
            for index in range(len(ids)):
                update_dynamic_mesh(model, mesh_id, local_vertices[index], faces)
                data.mocap_pos[0] = joints[index, 0]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = joints[index, 0]
                if args.camera_lookat_height is not None:
                    camera.lookat[2] = args.camera_lookat_height
                if camera_heading is not None:
                    camera.azimuth = camera_heading[index] + args.camera_azimuth
                renderer.update_scene(data, camera=camera)
                mujoco.mjr_uploadMesh(model, renderer._mjr_context, mesh_id)
                writer.append_data(renderer.render())
                if index == 0 or (index + 1) % 100 == 0:
                    print(f"[SMPLXVideo] {index + 1}/{len(ids)}", flush=True)
    args.out.with_suffix(".json").write_text(json.dumps({
        "source": str(args.data.resolve()), "model_dir": str(args.smplx_model_dir.resolve()),
        "start": args.start, "end_exclusive": args.end, "fps": args.fps,
        "stride": args.stride, "source_frame_ids": ids.tolist(),
        "display_scale": scale, "display_floor_offset_before_scale": floor_offset,
        "display_ground_alignment": args.ground_align,
        "source_height_reference_error_m": source_height_error,
        "camera": {"distance": args.camera_distance, "azimuth": args.camera_azimuth,
                   "elevation": args.camera_elevation, "lookat_height": args.camera_lookat_height,
                   "follow_heading": args.camera_follow_heading, "ground_reflectance": args.ground_reflectance},
    }, indent=2) + "\n")
    print(f"[SMPLXVideo] saved {args.out}")


if __name__ == "__main__":
    main()

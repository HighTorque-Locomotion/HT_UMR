"""Production local-region adapter; does not replace the upstream solver."""

from pathlib import Path
import argparse
import sys
import json
import inspect
import os
import numpy as np
import mujoco

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--config", type=Path, required=True)
p.add_argument("--prepared", type=Path, required=True)
p.add_argument("--audit-dir", type=Path, required=True)
p.add_argument("--soma-assets", type=Path, required=True)
args = p.parse_args()
TEMPLATE = args.prepared.resolve()
OUT = args.audit_dir.resolve()
OUT.mkdir(parents=True, exist_ok=False)
UP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(UP / "scripts"))
sys.path.insert(0, str(UP))
os.environ["UMR_SOMA_SMPLX_ASSETS_PATH"] = str(args.soma_assets.resolve())
os.environ["UMR_SOMA_ASSETS_PATH"] = str(args.soma_assets.resolve())
import retarget_smpl_to_humanoid_surface_vector as rt

original = rt.bind_robot_slots
prepared = np.load(TEMPLATE / "regions.npz")
# Uniform SOMA shares one template and slot identity. Freeze its calibrated
# segmentation before sampling/cost construction, avoiding numerical relabeling
# of a few boundary points when the same template is recomputed in a new batch.
source_original = rt.bind_source_slots_with_normals


def stable_source_binding(*a, **kw):
    result = list(source_original(*a, **kw))
    labels = result[2]
    expected = prepared["source_part_ids"]
    assert labels.shape == expected.shape
    changed = np.flatnonzero(labels != expected)
    if len(changed) > 10:
        raise ValueError(
            "Unexpected template/segmentation change; not a boundary precision issue"
        )
    result[2] = expected.copy()
    (OUT / "source_binding_audit.json").write_text(
        json.dumps(
            dict(
                calibrated_source_labels=True,
                boundary_points_restored=changed.tolist(),
                count=len(changed),
            ),
            indent=2,
        )
    )
    print("FROZEN_UNIFORM_TEMPLATE_LABELS", len(changed), flush=True)
    return tuple(result)


import retarget_body_segment_surface as body_segment_module

body_segment_module.bind_source_slots_with_normals = stable_source_binding
rt.bind_source_slots_with_normals = stable_source_binding


def bind(model, args, *a, **kw):
    result = original(model, args, *a, **kw)
    v = inspect.currentframe().f_back.f_locals
    np.testing.assert_array_equal(
        v["source_slot_part_ids"], prepared["source_part_ids"]
    )
    print(
        "UNCONSTRAINED_LABEL_BOUNDARY_DIFFERENCES",
        int(np.count_nonzero(v["source_slot_part_ids"] != prepared["source_part_ids"])),
        flush=True,
    )
    d = mujoco.MjData(model)
    pose = rt.robot_sample_qpos_for_pose(
        args.config_data,
        rt.robot_sample_pose_for_source(
            args.config_data, v["source_model_type"], v["template_cfg"]
        ),
    )
    rt.apply_joint_qpos(model, d, pose, required=False)
    mujoco.mj_forward(model, d)
    for i in np.flatnonzero(prepared["constrained"]):
        b = model.body(str(prepared["body_names"][i])).id
        g = np.flatnonzero(model.geom_bodyid == b)[prepared["geom_ordinal"][i]]
        result["geom_ids"][i] = g
        result["local_pos"][i] = prepared["local_pos"][i]
        result["local_normals"][i] = prepared["local_normals"][i]
        result["root_points"][i] = (
            prepared["robot_points"][i] - d.xpos[model.body("torso_link").id]
        )
    points = np.array(
        [
            d.geom_xmat[g].reshape(3, 3) @ p + d.geom_xpos[g]
            for g, p in zip(result["geom_ids"], result["local_pos"])
        ]
    )
    error = np.linalg.norm(
        points[prepared["constrained"]]
        - prepared["robot_points"][prepared["constrained"]],
        axis=1,
    )
    assert error.max() < 1e-6, error.max()
    binding = v["source_binding"]
    src = (
        v["template_vertices"][v["faces"][binding["face_ids"]]]
        * binding["bary"][..., None]
    ).sum(1)
    selected = np.zeros(len(src), bool)
    selected[v["selected_slot_ids"]] = True
    bn = np.array(
        [model.body(int(b)).name for b in model.geom_bodyid[result["geom_ids"]]]
    )
    np.savez_compressed(
        OUT / "mapping.npz",
        source_points=src,
        source_joints=v["template_joints"],
        source_joint_names=v["source_joint_names"],
        source_vertices=v["template_vertices"],
        source_faces=v["faces"],
        source_part_ids=v["source_slot_part_ids"],
        robot_points=points,
        robot_geom_ids=result["geom_ids"],
        robot_body_names=bn,
        robot_qpos=d.qpos,
        robot_xml=str(args.robot_xml),
        selected=selected,
        source_scale=v["smpl_scale"],
    )
    groups = {}
    for name, pid in rt.SMPLX_PART_IDS.items():
        mask = v["source_slot_part_ids"] == pid
        b, c = np.unique(bn[mask], return_counts=True)
        groups[name] = dict(
            part_id=int(pid),
            slots=int(mask.sum()),
            selected_slots=int(selected[mask].sum()),
            target_bodies=dict(zip(b.tolist(), c.tolist())),
        )
    (OUT / "mapping.json").write_text(
        json.dumps(
            dict(
                total_slots=len(src),
                selected_slots=int(selected.sum()),
                groups=groups,
                maximum_region_projection_error_m=float(error.max()),
            ),
            indent=2,
        )
    )
    print(
        "LOCAL_REGION_BINDINGS_OK",
        int(prepared["constrained"].sum()),
        "max_error_m",
        error.max(),
        flush=True,
    )
    return result


rt.bind_robot_slots = bind
sys.argv = [
    "retarget",
    "--config",
    str(args.config.resolve()),
    "--slots",
    str(TEMPLATE / "constrained_slots.npz"),
]
rt.main()

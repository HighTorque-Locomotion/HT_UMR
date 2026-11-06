"""Capture source slot labels and robot bindings without solving a trajectory."""

from pathlib import Path
import sys
import json
import inspect
import argparse
import numpy as np
import mujoco

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--config", type=Path, required=True)
p.add_argument("--slots", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
cli = p.parse_args()
OUT = cli.out.resolve()
OUT.mkdir(parents=True, exist_ok=False)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import retarget_smpl_to_humanoid_surface_vector as rt

original = rt.bind_robot_slots


class Captured(Exception):
    pass


def capture(model, args, *a, **kw):
    result = original(model, args, *a, **kw)
    v = inspect.currentframe().f_back.f_locals
    data = mujoco.MjData(model)
    pose = rt.robot_sample_qpos_for_pose(
        args.config_data,
        rt.robot_sample_pose_for_source(
            args.config_data, v["source_model_type"], v["template_cfg"]
        ),
    )
    rt.apply_joint_qpos(model, data, pose, required=False)
    mujoco.mj_forward(model, data)
    points = []
    for gid, local in zip(result["geom_ids"], result["local_pos"]):
        points.append(data.geom_xmat[gid].reshape(3, 3) @ local + data.geom_xpos[gid])
    bind = v["source_binding"]
    tri = v["template_vertices"][v["faces"][bind["face_ids"]]]
    source_points = (tri * bind["bary"][..., None]).sum(axis=1)
    selected = np.zeros(len(points), bool)
    selected[v["selected_slot_ids"]] = True
    body_names = np.array(
        [model.body(int(i)).name for i in model.geom_bodyid[result["geom_ids"]]]
    )
    np.savez_compressed(
        OUT / "mapping.npz",
        source_points=source_points,
        source_joints=v["template_joints"],
        source_joint_names=v["source_joint_names"],
        source_vertices=v["template_vertices"],
        source_faces=v["faces"],
        source_part_ids=v["source_slot_part_ids"],
        source_binding_face_ids=bind["face_ids"],
        source_binding_bary=bind["bary"],
        robot_points=points,
        robot_geom_ids=result["geom_ids"],
        robot_body_ids=model.geom_bodyid[result["geom_ids"]],
        robot_body_names=body_names,
        robot_qpos=data.qpos,
        robot_xml=str(args.robot_xml),
        selected=selected,
        source_scale=v["smpl_scale"],
    )
    groups = {}
    for name, pid in rt.SMPLX_PART_IDS.items():
        ids = np.flatnonzero(v["source_slot_part_ids"] == pid)
        bn, counts = np.unique(body_names[ids], return_counts=True)
        groups[name] = dict(
            part_id=int(pid),
            slots=len(ids),
            selected_slots=int(selected[ids].sum()),
            target_bodies=dict(zip(bn.tolist(), counts.astype(int).tolist())),
        )
    report = dict(
        total_slots=len(points),
        selected_slots=int(selected.sum()),
        groups=groups,
        source_segmentation="official SMPLX-SOMA correspondence",
        source_joint_names=list(v["source_joint_names"]),
    )
    (OUT / "mapping.json").write_text(json.dumps(report, indent=2))
    print("CAPTURED", len(points), selected.sum(), flush=True)
    raise Captured()


rt.bind_robot_slots = capture
sys.argv = [
    "capture",
    "--config",
    str(cli.config.resolve()),
    "--slots",
    str(cli.slots.resolve()),
]
try:
    rt.main()
except Captured:
    pass

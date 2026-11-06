"""Restrict learned correspondence slots to approved robot surface regions."""

from pathlib import Path
import argparse
import sys
import json
import numpy as np
import mujoco
import trimesh
from clipped_surface import clipped_mesh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from mujoco_geom_surface import geom_local_mesh

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--mapping", type=Path, required=True)
p.add_argument("--mapping-report", type=Path, required=True)
p.add_argument("--slots", type=Path, required=True)
p.add_argument(
    "--profile",
    type=Path,
    default=REPO / "robot_configs/local_regions/piplus_40v_0908.json",
)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
OUT = args.out.resolve()
OUT.mkdir(parents=True, exist_ok=False)
z = np.load(args.mapping)
md = json.loads(args.mapping_report.read_text())
reference = json.loads(args.profile.read_text())
if len(z["robot_points"]) != 4096:
    raise ValueError("This calibrated profile expects 4096 source slots")
m = mujoco.MjModel.from_xml_path(str(z["robot_xml"]))
d = mujoco.MjData(m)
d.qpos[:] = z["robot_qpos"]
mujoco.mj_forward(m, d)
points = z["robot_points"].copy()
gids = z["robot_geom_ids"].copy()
normals = np.zeros_like(points)
constrained = np.zeros(len(points), bool)
audit = {}
cache = {
    g: (v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g], f)
    for g in range(m.ngeom)
    if m.geom_group[g] == 1
    for v, f in [geom_local_mesh(m, g)]
}
for name, group in md["groups"].items():
    if not any(name.endswith(n) for n in ["Arm", "Hand", "UpLeg", "Leg", "Foot"]):
        continue
    reg = reference["regions"][name]
    lo, hi = np.array(reg["target_region_bounds"])
    bodies = reg["allowed_bodies"]
    vv = []
    ff = []
    gg = []
    offset = 0
    for g, (v, f) in cache.items():
        if m.body(int(m.geom_bodyid[g])).name not in bodies:
            continue
        vc, fc, gc = clipped_mesh(v, f, g, lo, hi)
        vv.append(vc)
        ff.append(fc + offset)
        gg.append(gc)
        offset += len(vc)
    if not vv or not any(len(f) for f in ff):
        raise ValueError(f"No target surface for {name}; check robot assets and T-pose")
    rv, rf, rg = np.concatenate(vv), np.concatenate(ff), np.concatenate(gg)
    mesh = trimesh.Trimesh(vertices=rv, faces=rf, process=False)
    mask = z["source_part_ids"] == group["part_id"]
    if not np.any(mask):
        raise ValueError(
            f"No source slots for {name}; check calibrated source template"
        )
    old = points[mask].copy()
    rlo, rhi = rv.min(0), rv.max(0)
    # Keep the learned ordering within the region, while relocating its extent
    # to the anatomically valid target surface. This is inference-time adaptation.
    t = (old - old.min(0)) / np.maximum(np.ptp(old, axis=0), 1e-8)
    query = rlo + t * (rhi - rlo)
    p, dist, face = trimesh.proximity.closest_point(mesh, query)
    points[mask] = p
    gids[mask] = rg[face]
    normals[mask] = mesh.face_normals[face]
    constrained[mask] = True
    assert np.all(p >= lo - 1e-6) and np.all(p <= hi + 1e-6)
    audit[name] = dict(
        slots=int(mask.sum()),
        allowed_bodies=bodies,
        bounds=[lo.tolist(), hi.tolist()],
        mean_adjustment_mm=float(np.linalg.norm(p - old, axis=1).mean() * 1000),
        max_projection_mm=float(dist.max() * 1000),
    )
local = np.zeros_like(points)
localn = np.zeros_like(points)
body = []
ordinal = []
for i, g in enumerate(gids):
    rot = d.geom_xmat[g].reshape(3, 3)
    local[i] = (points[i] - d.geom_xpos[g]) @ rot
    localn[i] = normals[i] @ rot
    b = int(m.geom_bodyid[g])
    body.append(m.body(b).name)
    ordinal.append(list(np.flatnonzero(m.geom_bodyid == b)).index(int(g)))
np.savez_compressed(
    OUT / "regions.npz",
    robot_points=points,
    local_pos=local,
    local_normals=localn,
    body_names=body,
    geom_ordinal=ordinal,
    constrained=constrained,
    source_part_ids=z["source_part_ids"],
)
orig = np.load(args.slots)
pairs = orig["reconstructed_slots"].copy()
pairs[1] = (points - d.xpos[m.body("torso_link").id])[:, [1, 2, 0]]
np.savez_compressed(
    OUT / "constrained_slots.npz",
    names=orig["names"],
    center_modes=orig["center_modes"],
    reconstructed_slots=pairs,
    target_points=pairs,
    slot_index=orig["slot_index"],
    mapping_method="learned4096_then_region_extent_and_clipped_surface_projection",
)
(OUT / "region_audit.json").write_text(
    json.dumps(
        dict(
            method="custom inference-time domain restriction; official source labels, slots and quotas unchanged; region-wise affine placement preserves learned intra-region ordering",
            regions=audit,
            unrestricted=["head", "hips", "torso", "leftShoulder", "rightShoulder"],
        ),
        indent=2,
    )
)
print("PREPARED", int(constrained.sum()), "of", len(points))

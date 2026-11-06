"""Generate a local config for bundled PiPlus 22-DOF and external human assets."""

from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--robot-xml",
    type=Path,
    default=ROOT / "assets/piplus_40v_0908/piplus_40v_260908_no_hyperextension.xml",
    help="Defaults to the bundled PiPlus 40V 0908 22-DOF model.",
)
for name in ["bvh", "soma-assets", "soma-usd", "smpl-models", "out"]:
    p.add_argument("--" + name, type=Path, required=True)
p.add_argument("--max-frames", type=int, default=0)
a = p.parse_args()
for name in ["robot_xml", "bvh", "soma_assets", "soma_usd", "smpl_models"]:
    if not getattr(a, name).exists():
        raise FileNotFoundError(getattr(a, name))
out = a.out.resolve()
out.mkdir(parents=True, exist_ok=True)
c = json.loads(
    (ROOT / "robot_configs/local_regions/piplus_40v_0908_settings.json").read_text()
)
c["extends"] = str(ROOT / "humanoid_retarget_defaults_bones_seed.json")
c["robot"]["xml"] = str(a.robot_xml.resolve())
c["motion"].update(
    data=str(a.bvh.resolve().parent), seq_key=a.bvh.stem, max_frames=a.max_frames
)
c["smplx_model_dir"] = str(a.smpl_models.resolve())
c["soma_usd_path"] = str(a.soma_usd.resolve())
c["smpl_template"] = {
    "soma_usd_path": c["soma_usd_path"],
    "soma_assets_path": str(a.soma_assets.resolve()),
}
c["correspondence"]["dataset"]["out"] = str(out / "correspondence_dataset.npz")
c["correspondence"]["train"]["out_dir"] = str(out / "correspondence_slots")
c["retarget"]["out"] = str(out / "motion.npz")
config = out / "config.json"
with config.open("x") as f:
    json.dump(c, f, indent=2)
print(config)

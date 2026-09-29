#!/usr/bin/env python3
"""Batch the validated recommended 16-beta converter and legacy PiPlus solver.

Keeps the existing trained correspondence and runs the single-clip retargeter.
Successful clips with matching input/config/code/output hashes are reused.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

from humanoid_retarget_config import load_config, resolve_path
from humanoid_retarget_pipeline import correspondence_slots_compatible, smpl_template_config
from prepare_lafan_smplx_metadata import prepare_metadata
from retarget_noitom_bvh_batch import result_relative_path, validate_result
from retarget_smpl_to_humanoid_surface_vector import absolutize_asset_paths, model_has_freejoint
from smplx_model_loader import smplx_num_betas

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "output/piplus_walk1_betas16_legacy"
UPSTREAM = ROOT / "output/upstream_g1_baseline"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def process(source, args, base_config, recipe):
    started = time.monotonic()
    relative = source.relative_to(args.source_root).with_suffix(".npz")
    paths = {name: args.output_root / name / relative.with_suffix(suffix)
             for name, suffix in (("raw_smplx", ".npz"), ("smplx", ".npz"), ("retarget", ".npz"),
                                  ("configs", ".json"), ("logs", ".log"), ("validation", ".json"))}
    result_relative = result_relative_path(relative, args.output_name_template,
                                          args.output_date, args.output_robot_name)
    paths["retarget"] = args.output_root / "retarget" / result_relative
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    signature = fingerprint({"recipe": recipe, "source": digest(source), "relative": str(relative),
                             "result_relative": str(result_relative)})
    record = {"source": str(source), "result": str(paths["retarget"]), "log": str(paths["logs"]),
              "fingerprint": signature, "status": "running", "reused": False}
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.cpu_threads)
    converter_env = dict(env, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(args.gmr_root))
    try:
        if not args.force and paths["validation"].is_file():
            previous = json.loads(paths["validation"].read_text())
            if (previous.get("status") == "ok" and previous.get("fingerprint") == signature
                    and all(paths[key].is_file() and digest(paths[key]) == previous.get(key + "_sha256")
                            for key in ("retarget", "smplx", "raw_smplx"))):
                return dict(previous, reused=True, elapsed_seconds=time.monotonic() - started)
        config = json.loads(json.dumps(base_config))
        config["motion"].update(data=str(paths["smplx"]), seq_key=source.stem, seq_index=0,
                                start=0, end=-1, stride=1, max_frames=0)
        config["retarget"]["out"] = str(paths["retarget"])
        config["view"]["enabled"] = False
        with paths["logs"].open("w") as log:
            command = [str(args.converter_python), "-u", str(args.converter), "--bvh_file", str(source),
                       "--smplx_model_path", str(args.converter_model_root), "--model_type", "smplx",
                       "--output_file", str(paths["raw_smplx"])]
            log.write(json.dumps(command) + "\n"); log.flush()
            subprocess.run(command, cwd=ROOT, env=converter_env, stdout=log, stderr=subprocess.STDOUT, check=True)
            preparation = prepare_metadata(paths["raw_smplx"], source, paths["smplx"])
            log.write(json.dumps(preparation) + "\n"); log.flush()
            config["_config_dir"] = str(paths["configs"].parent)
            if smpl_template_config(config)["num_betas"] != 16:
                raise ValueError("Expected a 16-beta source template")
            if not correspondence_slots_compatible(args.slots, config):
                raise ValueError("Existing slots do not match this source shape and robot")
            config.pop("_config_dir", None)
            write_json(paths["configs"], config)
            command = [sys.executable, "-u", str(ROOT / "scripts/retarget_smpl_to_humanoid_surface_vector.py"),
                       "--config", str(paths["configs"]), "--slots", str(args.slots), "--force-no-floating-root"]
            log.write(json.dumps(command) + "\n"); log.flush()
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        with np.load(paths["smplx"], allow_pickle=False) as motion:
            fps = float(np.asarray(motion["mocap_frame_rate"]).reshape(-1)[0])
            betas = motion["betas"]
        with np.load(paths["retarget"], allow_pickle=False) as result:
            if int(result["smplx_num_betas"]) != 16:
                raise ValueError("Retarget result did not retain 16 betas")
            np.testing.assert_array_equal(result["smplx_betas"], betas)
        audit = validate_result(paths["retarget"], config, preparation["frames"], fps)
        record.update(status="ok", preparation=preparation, audit=audit,
                      **{key + "_sha256": digest(paths[key]) for key in ("retarget", "smplx", "raw_smplx")})
    except Exception as error:
        record.update(status="error", error=str(error))
        if isinstance(error, subprocess.CalledProcessError) and paths["logs"].is_file():
            lines = paths["logs"].read_text(errors="replace").splitlines()
            record["error_detail"] = next((line.strip() for line in reversed(lines) if line.strip()), str(error))
    record["elapsed_seconds"] = time.monotonic() - started
    write_json(paths["validation"], record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pattern", default="*.bvh")
    parser.add_argument("--config", type=Path, default=EXPERIMENT / "config.json")
    parser.add_argument("--slots", type=Path, default=EXPERIMENT / "correspondence_slots/correspondence_slots_final.npz")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-name-template", default="{name}",
                        help="Result stem, supporting {name}, {date}, {robot}; .npz is appended automatically")
    parser.add_argument("--output-date", default=datetime.now().strftime("%y%m%d"),
                        help="Value for {date}; defaults to today's YYMMDD. Pin this for later resume runs")
    parser.add_argument("--converter-python", type=Path, default=Path(sys.executable).resolve().parents[2] / "gmr/bin/python")
    parser.add_argument("--converter", type=Path, default=UPSTREAM / "input_adapter/lafan_to_smplx_compat.py")
    parser.add_argument("--gmr-root", type=Path, default=UPSTREAM / "repos/GMR")
    parser.add_argument("--converter-model-root", type=Path, default=Path("/data/download/models"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.cpu_threads < 1 or args.limit < 0:
        parser.error("workers/cpu-threads must be positive; limit must be nonnegative")
    for name in ("source_root", "config", "slots", "output_root", "converter_python", "converter", "gmr_root", "converter_model_root"):
        path = getattr(args, name).expanduser()
        # A venv Python may symlink to its base interpreter; keep the venv path.
        setattr(args, name, path.absolute() if name == "converter_python" else path.resolve())
    sources = sorted(path for path in args.source_root.rglob(args.pattern) if path.is_file() and path.suffix.lower() == ".bvh")
    if args.limit:
        sources = sources[:args.limit]
    if not sources:
        parser.error("No BVH files matched")
    for path in (args.config, args.slots, args.converter_python, args.converter, args.gmr_root / "general_motion_retargeting/utils/lafan1.py"):
        if not path.is_file():
            parser.error(f"Required dependency missing: {path}")
    original = load_config(args.config)
    config = {k: v for k, v in original.items() if not k.startswith("_")}
    config["smplx_model_dir"] = str(resolve_path(config["smplx_model_dir"], original))
    config["robot"]["xml"] = str(resolve_path(config["robot"]["xml"], original))
    robot_xml = Path(config["robot"]["xml"])
    robot_tree = ET.parse(robot_xml)
    args.output_robot_name = (robot_tree.getroot().get("model") or robot_xml.stem).removesuffix("_limit")
    if not model_has_freejoint(robot_tree.getroot()):
        parser.error("The configured robot XML must already have a free root joint")
    absolutize_asset_paths(robot_tree.getroot(), robot_xml)
    if smplx_num_betas(Path(config["smplx_model_dir"])) != 16:
        parser.error("Config must select a 16-beta SMPL-X model directory")
    config["smpl_template"].update(source="motion", type="smplx", use_betas=True, use_gender=True, name="auto")
    config["correspondence"]["smpl_name"] = "auto"
    for spec in config["correspondence"]["dataset"].get("smpl_models", []):
        spec["dir"] = str(resolve_path(spec["dir"], original))
    try:
        outputs = [result_relative_path(source.relative_to(args.source_root), args.output_name_template,
                                        args.output_date, args.output_robot_name) for source in sources]
    except (KeyError, ValueError, IndexError, AttributeError) as error:
        parser.error(f"Invalid --output-name-template: {error}")
    if len(set(outputs)) != len(outputs):
        parser.error("--output-name-template produces duplicate result paths")
    if args.dry_run:
        for source, output in zip(sources, outputs):
            print(f"[LAFAN16] {source.relative_to(args.source_root)} -> {args.output_root / 'retarget' / output}")
        print(f"[LAFAN16] dry-run clips={len(sources)} workers={args.workers} num_betas=16")
        return 0
    # Hash dependencies once; each job adds the selected BVH content hash.
    model_dir = Path(config["smplx_model_dir"])
    manifest_path = model_dir / "umr_smplx_overlay.json"
    manifest = json.loads(manifest_path.read_text())
    body_model = (model_dir / manifest["base_model_dir"]).resolve()
    if body_model.is_dir():
        candidates = [body_model / ("SMPLX_NEUTRAL." + ext) for ext in ("pkl", "npz")]
        body_model = next((p for p in candidates if p.is_file()), candidates[0])
    dependencies = [args.converter, args.slots, manifest_path, body_model, Path(config["robot"]["xml"]),
                    args.converter_model_root / "smplx/SMPLX_NEUTRAL.npz"]
    dependencies += [Path(item.get("file")) for item in robot_tree.getroot().findall("./asset/*") if item.get("file")]
    dependencies += [ROOT / "assets/smplx_parts_segm.pkl"]
    dependencies += sorted((ROOT / "scripts").glob("*.py"))
    dependencies += sorted((args.gmr_root / "general_motion_retargeting").rglob("*.py"))
    naming = {"output_name_template": args.output_name_template,
              "output_date": args.output_date if "{date" in args.output_name_template else None,
              "output_robot_name": args.output_robot_name}
    recipe = fingerprint({"config": config, **naming, "dependencies": {str(p): digest(p) for p in dependencies}})
    write_json(args.output_root / "resolved_base_config.json", config)
    records = []
    print(f"[LAFAN16] clips={len(sources)} workers={args.workers} output={args.output_root}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, source, args, config, recipe) for source in sources]
        for future in as_completed(futures):
            record = future.result(); records.append(record)
            write_json(args.output_root / "batch_summary.json", {"source_root": str(args.source_root),
                       "config": str(args.config), **naming, "clips": len(sources),
                       "results": sorted(records, key=lambda r: r["source"])})
            print(f"[LAFAN16] {len(records)}/{len(sources)} {record['status']} {Path(record['source']).name} reused={record['reused']}", flush=True)
            if record["status"] == "error":
                print(f"[LAFAN16]   {record.get('error_detail', record['error'])}\n"
                      f"[LAFAN16]   log={record['log']}", flush=True)
    return int(any(record["status"] != "ok" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Convert selected LAFAN BVHs and run the original UMR single-motion pipeline.

Correspondence learning and all solver settings come from --config. This entry
does not apply the specialized Noitom batch runner's solver overrides/audits.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from humanoid_retarget_config import load_config, resolve_path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pattern", default="*.bvh")
    parser.add_argument("--config", type=Path, default=ROOT / "demo_configs/umr_piplus_s_40v_kid_original.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output/lafan_kid_original")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0, help="0 retains every frame; use a separate output root for previews.")
    parser.add_argument("--prepare-only", action="store_true", help="Convert BVH and write configs without training/retargeting.")
    parser.add_argument("--force", action="store_true", help="Rebuild correspondence as well as retargeting.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.max_frames < 0:
        parser.error("--cpu-threads must be positive and --max-frames nonnegative")
    source_root = args.source_root.expanduser().resolve()
    sources = sorted(p for p in source_root.rglob(args.pattern) if p.is_file() and p.suffix.lower() == ".bvh")
    if not sources:
        parser.error(f"No BVH matched {args.pattern!r} under {source_root}")
    config = load_config(args.config)
    model_dir = resolve_path(config["smplx_model_dir"], config)
    output_root = args.output_root.expanduser().resolve()
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.cpu_threads)
    for source in sources:
        relative = source.relative_to(source_root).with_suffix(".npz")
        motion_path = output_root / "smplx" / relative
        config_path = (output_root / "configs" / relative).with_suffix(".json")
        result_path = output_root / "retarget" / relative
        print(f"[LAFANOriginalUMR] {source} -> {result_path}", flush=True)
        if args.dry_run:
            continue
        clip = {
            "extends": str(args.config.resolve()),
            "motion": {"data": str(motion_path), "seq_key": source.stem,
                       "start": 0, "end": -1, "stride": 1, "max_frames": args.max_frames},
            "correspondence": {
                "dataset": {"out": str(output_root / "correspondence/dataset.npz")},
                "train": {"out_dir": str(output_root / "correspondence/slots")},
            },
            "retarget": {"out": str(result_path)},
            "view": {"enabled": False},
        }
        # The config loader resolves paths relative to the final child config.
        # Make inherited asset paths absolute before relocating it to output/.
        clip["smplx_model_dir"] = str(model_dir)
        clip["robot"] = {"xml": str(resolve_path(config["robot"]["xml"], config))}
        specs = json.loads(json.dumps(config["correspondence"]["dataset"]["smpl_models"]))
        for spec in specs:
            spec["dir"] = str(resolve_path(spec["dir"], config))
        clip["correspondence"]["dataset"]["smpl_models"] = specs
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(clip, indent=2) + "\n")
        subprocess.run([
            sys.executable, "-u", str(ROOT / "scripts/convert_lafan_bvh_to_smplx.py"),
            "--input", str(source), "--out", str(motion_path), "--smplx-model-dir", str(model_dir),
        ], cwd=ROOT, env=env, check=True)
        if args.prepare_only:
            print(f"[LAFANOriginalUMR] prepared {config_path}", flush=True)
            continue
        command = [sys.executable, "-u", str(ROOT / "scripts/humanoid_retarget_pipeline.py"),
                   "--config", str(config_path), "--skip-view", "--force-retarget"]
        if args.force:
            command += ["--force-build", "--force-train"]
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

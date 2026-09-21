#!/usr/bin/env python3
"""Recursively convert Noitom/LAFAN BVH, retarget with the single-clip solver, and audit.

Outputs retain the input directory structure. A completed clip is reused only
when its input, configuration, dependencies and saved result still match.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from humanoid_retarget_config import load_config, resolve_path
from nr_source import _bvh_metadata

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def detect_source_format(source):
    from convert_lafan_bvh_to_smplx import LAFAN_BODY_JOINT_NAMES
    from convert_noitom_bvh_to_smplx import BODY_JOINT_NAMES

    names = set()
    with Path(source).open() as stream:
        for line in stream:
            parts = line.split()
            if parts and parts[0] == "MOTION":
                break
            if parts and parts[0] in ("ROOT", "JOINT"):
                names.add(parts[1])
    if set(BODY_JOINT_NAMES) <= names:
        return "noitom"
    if set(LAFAN_BODY_JOINT_NAMES) <= names:
        return "lafan"
    raise ValueError(f"Unsupported Noitom/LAFAN joint names: {source}")


def result_relative_path(relative, template, date, robot):
    relative = Path(relative)
    name = template.format(name=relative.stem, date=date, robot=robot)
    if not name or name in (".", "..") or any(c in name for c in ("/", "\\", "\0")):
        raise ValueError("Output name template must produce a filename without directories.")
    return relative.with_name(name + ".npz")


def matches_source_pattern(source, pattern):
    """Accept filename keywords while retaining explicit path/glob matching."""
    source = Path(source)
    if any(char in pattern for char in "*?[/\\") or pattern.lower().endswith(".bvh"):
        return source.match(pattern)
    return pattern.casefold() in source.stem.casefold()


def load_batch_config(args):
    """Apply CLI overrides before saving clip configs and hashing the recipe."""
    original = load_config(args.config)
    config = {k: v for k, v in original.items() if not k.startswith("_")}
    if args.preserve_foot_timing:
        # Apply to the whole selected clip: jump detection alone misses held
        # legs and preparation, which can otherwise become one very long step.
        clearance = config.setdefault("retarget", {}).setdefault("foot_clearance", {})
        clearance.setdefault("transport_timing", {})["enabled"] = False
    return config, original


def read_source_file_list(path, source_root):
    """Read exact BVH paths relative to the source root, never glob patterns."""
    source_root = Path(source_root).resolve()
    selected = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name:
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"File list entries must be relative to --source-root: {name}")
        source = source_root / relative
        if not source.resolve().is_relative_to(source_root):
            raise ValueError(f"File list entry leaves --source-root: {name}")
        if source.suffix.lower() != ".bvh" or not source.is_file():
            raise ValueError(f"File list entry is not an existing BVH: {name}")
        selected.add(relative.as_posix())
    if not selected:
        raise ValueError("The file list is empty; no clips were selected")
    return selected


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def validate_result(path, config, frames, fps):
    import mujoco
    import smpl_surface_retarget_common as common

    with np.load(path, allow_pickle=True) as result:
        qpos = np.asarray(result["qpos"], dtype=np.float64)
        frame_ids = result["frame_ids"]
        saved_fps = float(np.asarray(result["fps"]).reshape(-1)[0])
        model = mujoco.MjModel.from_xml_path(str(result["robot_xml"]))
    if qpos.shape != (frames, model.nq) or not np.isfinite(qpos).all():
        raise ValueError(f"Invalid qpos: shape={qpos.shape}, expected={(frames, model.nq)}")
    if not np.array_equal(frame_ids, np.arange(frames)) or not np.isclose(saved_fps, fps):
        raise ValueError("Result did not preserve the source frames and frame rate.")
    limits, _ = common.build_scalar_joint_limits(model, config["robot"].get("joint_limits", {}))
    violations = [v["name"] for col, v in limits.items()
                  if qpos[:, col].min() < v["lower"] - 1e-6 or qpos[:, col].max() > v["upper"] + 1e-6]
    if violations:
        raise ValueError(f"Joint limit violations: {violations}")
    quat_error = float(np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1)))
    if quat_error > 1e-6:
        raise ValueError(f"Root quaternion norm error: {quat_error}")
    joint_steps = {}
    iterations = int(config["solver"]["iters"])
    for name, bound in config["robot"].get("dof_max_dq_box", {}).items():
        joint = model.joint(name)
        col = int(joint.qposadr[0])
        maximum = float(np.abs(np.diff(qpos[:, col])).max()) if frames > 1 else 0.0
        if maximum > iterations * float(bound) + 1e-6:
            raise ValueError(f"Continuity bound violated: {name}, step={maximum}")
        joint_steps[name] = maximum
    solver = SimpleNamespace(**config["solver"])
    cache = common.build_robot_self_penetration_cache(model, solver)
    data = mujoco.MjData(model)
    threshold = float(solver.collision_threshold)
    min_distance = threshold
    worst_frame = None
    penetrating = []
    below_margin = 0
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        _, distances = common.compute_robot_self_penetration_rows(model, data, cache, threshold)
        nearest = min(distances, default=threshold)
        if nearest < min_distance:
            min_distance, worst_frame = nearest, frame
        if nearest < -1e-6:
            penetrating.append(frame)
        below_margin += int(nearest < solver.robot_self_penetration_margin)
    audit = {
        "frames": frames, "fps": saved_fps, "duration_seconds": frames / saved_fps,
        "qpos_shape": list(qpos.shape), "all_finite": True, "all_frame_ids_preserved": True,
        "joint_limit_violations": violations, "max_root_quaternion_norm_error": quat_error,
        "max_shoulder_step_rad": max(joint_steps.values(), default=0.0),
        "bounded_joint_max_steps_rad": joint_steps,
        "temporal_joint_radians_per_frame": common.qpos_temporal_summary(qpos, limits),
        "self_collision": {
            "min_geom_distance_m": min_distance, "worst_frame": worst_frame,
            "penetrating_frame_count": len(penetrating), "penetrating_frames": penetrating,
            "frames_below_requested_margin": below_margin,
            "scope": "MuJoCo collision geometries/pairs used by the solver, not visual triangle meshes",
        },
    }
    with np.load(path, allow_pickle=True) as result:
        if "robot_sole_height" in result:
            import retarget_foot_clearance as foot
            soles = foot.build_robot_soles(model, config["robot"].get("foot_bodies", {}))
            heights, _, speeds = foot.robot_sole_trajectory(model, qpos, soles, saved_fps)
            np.testing.assert_allclose(heights, result["robot_sole_height"], atol=1e-6)
            audit["foot_clearance"] = foot.clearance_summary(
                result["source_sole_height"], result["target_sole_height"], heights,
                result["foot_swing_mask"], result["foot_stance_mask"],
                json.loads(str(result["foot_clearance_events_json"])), speeds)
            if "robot_support_height_max" in result:
                foot.build_support_patches(model, soles)
                maximum, span = foot.support_patch_trajectory(model, qpos, soles)
                np.testing.assert_allclose(maximum, result["robot_support_height_max"], atol=1e-6)
                np.testing.assert_allclose(span, result["robot_support_height_span"], atol=1e-6)
                stance = result["foot_stance_mask"]
                audit["foot_clearance"].update(
                    stance_support_height_p95_m=float(np.percentile(maximum[stance], 95)) if stance.any() else None,
                    stance_support_height_span_p95_m=float(np.percentile(span[stance], 95)) if stance.any() else None)
            tolerance = config["retarget"].get("foot_clearance", {}).get("max_ground_penetration")
            if tolerance is not None:
                tolerance = foot.positive(tolerance, "max_ground_penetration", allow_zero=True)
                if heights.min() < -tolerance:
                    raise ValueError(f"Foot mesh penetrates ground: depth={-heights.min():.6f}m, allowed={tolerance}m")
        if "jump_compensation_json" in result:
            import retarget_jump_compensation as jump
            com, _ = jump.robot_com_trajectory(model, qpos)
            np.testing.assert_allclose(com, result["robot_com"], atol=1e-6)
            spec = json.loads(str(result["jump_compensation_json"]))
            active = result["jump_active_mask"]
            support = result["jump_support_mask"]
            target = result["jump_target_com_z"]
            if not np.isfinite(target).all():
                raise ValueError("Non-finite jump COM targets")
            for event in spec["events"]:
                start, end = int(event["start"]), int(event["end"])
                if not 0 <= start < end < len(qpos) or support[start+1:end].any():
                    raise ValueError(f"Invalid airborne support interval: {start}:{end}")
                if not np.isclose((end-start)/saved_fps, event["duration_s"]):
                    raise ValueError("Jump compensation changed the identified flight duration")
                for feet_key, frames_key in (("takeoff_feet", "takeoff_contact_frames"),
                                              ("landing_feet", "landing_contact_frames")):
                    for side in event[feet_key]:
                        frame = int(event[frames_key][str(side)])
                        if not 0 <= frame < len(qpos) or not support[frame, side]:
                            raise ValueError(f"Missing {feet_key} constraint: frame={frame}, side={side}")
            contact_max = float(np.max(result["robot_sole_height"][support])) if support.any() else None
            if contact_max is not None and float(spec["config"].get("support_slack_cost", 0.0)) == 0:
                allowed = (float(spec["config"].get("support_height_tolerance", .001))
                           + float(spec["config"].get("support_refine_tolerance", .0001)) + .00005)
                if contact_max > allowed:
                    raise ValueError(f"Jump support contact did not converge: gap={contact_max}, allowed={allowed}")
            audit["jump_compensation"] = {
                "compensated_events": len(spec["events"]), "skipped_events": len(spec["rejected"]),
                "com_tracking_rms_m": float(np.sqrt(np.mean((com[active,2]-target[active])**2))) if active.any() else None,
                "com_tracking_max_error_m": float(np.max(np.abs(com[active,2]-target[active]))) if active.any() else None,
                "support_sole_height_max_m": contact_max,
            }
    return audit


def prepare_config(args):
    from retarget_smpl_to_humanoid_surface_vector import absolutize_asset_paths, model_has_freejoint

    config, original = load_batch_config(args)
    robot_xml = resolve_path(config["robot"]["xml"], original)
    args.output_robot_name = robot_xml.stem.removesuffix("_limit")
    config["smplx_model_dir"] = str(resolve_path(config["smplx_model_dir"], original))
    tree = ET.parse(robot_xml)
    if not model_has_freejoint(tree.getroot()):
        raise ValueError("This batch runner requires a robot XML with a free root joint.")
    if tree.getroot().find("include") is not None:
        raise ValueError("Use a self-contained robot MJCF without include tags.")
    absolutize_asset_paths(tree.getroot(), robot_xml)
    xml_text = ET.tostring(tree.getroot(), encoding="unicode")
    # Snapshot in the output tree: parallel clips never write into the asset directory.
    xml_hash = hashlib.sha256(xml_text.encode()).hexdigest()
    snapshot = args.output_root / "robot" / f"{robot_xml.stem}_{xml_hash[:12]}.xml"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        snapshot.write_text(xml_text)
    config["robot"]["xml"] = str(snapshot)
    config["robot"]["xml_policy"] = {"add_freejoint_root": False}
    if config["solver"].get("trajectory_warm_start_mode", "sequential") != "sequential":
        raise ValueError("Sequential warm starts are required for the continuity audit.")
    if config["solver"].get("trajectory_filter_mode") != "off":
        raise ValueError("Post-filtering must be off to retain the solved constraints.")
    with np.load(args.slots, allow_pickle=True) as slots:
        names = slots["names"].astype(str).tolist()
        if "smplx_neutral" not in names or config["robot"]["name"] not in names:
            raise ValueError(f"Correspondence samples do not match the configured robot: {names}")
    config["smpl_template"].update(source="manual", type="smplx", use_betas=False,
                                   gender="neutral", name="smplx_neutral")
    config["correspondence"]["smpl_name"] = "smplx_neutral"
    assets = [Path(item.get("file")) for item in tree.getroot().findall("./asset/*") if item.get("file")]
    model_dir = Path(config["smplx_model_dir"])
    assets.extend(p for p in model_dir.glob("SMPLX_NEUTRAL.*") if p.is_file())
    asset_stats = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in assets]
    recipe = {
        "config": config, "robot_xml_sha256": xml_hash, "slots_sha256": digest(args.slots),
        "asset_stats": asset_stats, "unit_scale": args.unit_scale,
        "output_name_template": args.output_name_template,
        "output_date": args.output_date if "{date" in args.output_name_template else None,
        "output_robot_name": args.output_robot_name, "source_format": args.source_format,
        "scripts": {p.name: digest(p) for p in sorted(SCRIPTS.glob("*.py"))},
    }
    return config, hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def process_clip(item, args, base_config, recipe_hash):
    started = time.monotonic()
    source = Path(item["source"])
    relative = Path(item["relative"])
    paths = {key: args.output_root / key / relative.with_suffix(suffix)
             for key, suffix in [("smplx", ".npz"), ("retarget", ".npz"), ("configs", ".json"),
                                 ("logs", ".log"), ("validation", ".json"), ("videos", ".mp4")]}
    result_relative = result_relative_path(relative, args.output_name_template,
                                          args.output_date, args.output_robot_name)
    paths["retarget"] = args.output_root / "retarget" / result_relative
    paths["videos"] = args.output_root / "videos" / result_relative.with_suffix(".mp4")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256((recipe_hash + digest(source)).encode()).hexdigest()
    entry = dict(item, status="running", out=str(paths["retarget"]),
                 log=str(paths["logs"]), validation=str(paths["validation"]), fingerprint=signature)
    config = json.loads(json.dumps(base_config))
    config["motion"].update(data=str(paths["smplx"]), seq_key=source.stem, seq_index=0,
                            start=0, end=-1, stride=1, max_frames=0)
    config["retarget"]["out"] = str(paths["retarget"])
    config["view"]["enabled"] = False
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.cpu_threads)
    env.update(MUJOCO_GL="egl", EGL_PLATFORM="surfaceless")
    reused = False
    try:
        if not args.force and paths["validation"].exists() and paths["retarget"].exists():
            previous = json.loads(paths["validation"].read_text())
            if (previous.get("status") == "ok" and previous.get("fingerprint") == signature
                    and previous.get("result_sha256") == digest(paths["retarget"])
                    and paths["smplx"].exists()
                    and previous.get("smplx_sha256") == digest(paths["smplx"])):
                entry.update(previous)
                reused = True
        if not reused:
            write_json(paths["configs"], config)
            commands = [
                [sys.executable, str(SCRIPTS / f"convert_{item['source_format']}_bvh_to_smplx.py"), "--input", str(source),
                 "--out", str(paths["smplx"]), "--smplx-model-dir", config["smplx_model_dir"],
                 "--unit-scale", str(args.unit_scale)],
                [sys.executable, "-u", str(SCRIPTS / "retarget_smpl_to_humanoid_surface_vector.py"),
                 "--config", str(paths["configs"]), "--slots", str(args.slots), "--force-no-floating-root"],
            ]
            with paths["logs"].open("w") as log:
                for command in commands:
                    log.write(json.dumps(command, ensure_ascii=False) + "\n")
                    log.flush()
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            audit = validate_result(paths["retarget"], config, item["frames"], item["fps"])
            entry.update(status="ok", audit=audit, result_sha256=digest(paths["retarget"]),
                         smplx_sha256=digest(paths["smplx"]))
            # A new result invalidates any previous preview, even if rendering is deferred.
            entry.pop("video_result_sha256", None)
        if args.render:
            if not (paths["videos"].exists() and entry.get("video_result_sha256") == entry["result_sha256"]):
                command = [sys.executable, str(SCRIPTS / "visualize_robot_retarget_result.py"),
                           "--result", str(paths["retarget"]), "--record-video", str(paths["videos"]),
                           "--record-width", "640", "--record-height", "480", "--stride", "4",
                           "--camera-mode", "root", "--camera-distance", "1.3"]
                with paths["logs"].open("a") as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                entry["video_result_sha256"] = entry["result_sha256"]
            entry["video"] = str(paths["videos"])
        entry.update(reused=reused, elapsed_seconds=time.monotonic() - started)
    except Exception as exc:
        entry.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                     elapsed_seconds=time.monotonic() - started)
    write_json(paths["validation"], entry)
    return entry


def save_summary(args, items, results):
    by_source = {entry["source"]: entry for entry in results}
    entries = [by_source.get(item["source"], dict(item, status="pending")) for item in items]
    summary = {"source_root": str(args.source_root), "config": str(args.config), "slots": str(args.slots),
               "file_list": str(args.file_list) if args.file_list else None,
               "patterns": args.pattern, "preserve_foot_timing": args.preserve_foot_timing,
               "source_format": args.source_format, "output_name_template": args.output_name_template,
               "output_date": args.output_date, "output_robot_name": args.output_robot_name,
               "total": len(items), "ok": sum(e["status"] == "ok" for e in entries),
               "failed": sum(e["status"] == "failed" for e in entries),
               "pending": sum(e["status"] == "pending" for e in entries),
               "total_source_frames": sum(e["frames"] for e in entries), "results": entries}
    write_json(args.output_root / "batch_summary.json", summary)
    fields = ["motion", "status", "frames", "fps", "max_shoulder_step_rad", "penetrating_frames",
              "max_penetration_mm", "min_sole_height_mm", "stance_height_rms_mm", "boosted_swings", "out", "error"]
    with (args.output_root / "batch_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            audit = entry.get("audit", {})
            collision = audit.get("self_collision", {})
            feet = audit.get("foot_clearance", {})
            writer.writerow(dict(motion=entry["relative"], status=entry["status"], frames=entry["frames"],
                                 fps=entry["fps"], max_shoulder_step_rad=audit.get("max_shoulder_step_rad", ""),
                                 penetrating_frames=collision.get("penetrating_frame_count", ""),
                                 max_penetration_mm=max(0.0, -collision["min_geom_distance_m"] * 1000)
                                 if collision else "",
                                 min_sole_height_mm=feet.get("min_robot_sole_height_m", 0) * 1000 if feet else "",
                                 stance_height_rms_mm=feet["stance_height_rms_m"] * 1000
                                 if feet and feet.get("stance_height_rms_m") is not None else "",
                                 boosted_swings=feet.get("boosted_swing_count", ""),
                                 out=entry.get("out", ""), error=entry.get("error", "")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pattern", action="append", default=None,
                        help="Select by case-insensitive filename keyword (e.g. 'jump') or path glob "
                             "(e.g. 'dance*.bvh'); repeat to combine patterns.")
    parser.add_argument("--preserve-foot-timing", action="store_true",
                        help="Preserve original foot XY/orientation timing throughout all selected clips, "
                             "including held legs, jump preparation and flight. Disables transport_timing "
                             "and its near-ground speed limits; keeps configured height/COM compensation.")
    parser.add_argument("--file-list", type=Path,
                        help="Only process exact source-relative BVH paths listed one per line (e.g. jump scan selected_files.txt).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--slots", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-format", choices=("auto", "noitom", "lafan"), default="auto")
    parser.add_argument("--output-name-template", default="{name}",
                        help="Result stem using {name}, {date}, {robot}; e.g. '{name}_{date}_{robot}_umr'.")
    parser.add_argument("--output-date", default=datetime.now().strftime("%y%m%d"),
                        help="Date used in output names (default: today's YYMMDD); pin this when resuming.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--unit-scale", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--render", action="store_true", help="Also render full-duration previews at source FPS / 4.")
    parser.add_argument("--dry-run", action="store_true", help="List selected clips and exit before conversion or robot setup.")
    args = parser.parse_args()
    if args.workers < 1 or args.cpu_threads < 1 or args.limit < 0:
        parser.error("workers and cpu-threads must be positive; limit must be nonnegative")
    for name in ("source_root", "config", "slots", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    selected = None
    if args.file_list:
        args.file_list = args.file_list.expanduser().resolve()
        selected = read_source_file_list(args.file_list, args.source_root)
    items = []
    for source in sorted(args.source_root.rglob("*")):
        if (source.is_file() and source.suffix.lower() == ".bvh"
                and (selected is None or source.relative_to(args.source_root).as_posix() in selected)
                and (not args.pattern or any(matches_source_pattern(source, pattern) for pattern in args.pattern))):
            frames, frame_time = _bvh_metadata(source)
            if frames <= 0 or not np.isfinite(frame_time) or frame_time <= 0:
                raise ValueError(f"Invalid BVH header: {source}")
            source_format = detect_source_format(source)
            if args.source_format != "auto" and source_format != args.source_format:
                raise ValueError(f"Expected {args.source_format}, detected {source_format}: {source}")
            items.append(dict(source=str(source), relative=str(source.relative_to(args.source_root)),
                              frames=frames, fps=1.0 / frame_time, source_format=source_format))
    if not items:
        parser.error("No BVH files matched --source-root, --pattern and --file-list")
    items.sort(key=lambda item: (item["frames"], item["relative"]))
    if args.limit:
        items = items[:args.limit]
    if args.preserve_foot_timing:
        print("[BVHBatch] preserve-foot-timing: original foot XY/orientation timing for every selected clip; "
              "height/COM compensation follows config", flush=True)
    if args.dry_run:
        for item in items:
            print(f"[BVHBatch] {item['relative']} frames={item['frames']} fps={item['fps']:.6f}")
        print(f"[BVHBatch] dry-run clips={len(items)} frames={sum(i['frames'] for i in items)}")
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    config, recipe_hash = prepare_config(args)
    outputs = [result_relative_path(item["relative"], args.output_name_template,
                                    args.output_date, args.output_robot_name) for item in items]
    if len(set(outputs)) != len(outputs):
        parser.error("Output name template produces duplicate result paths")
    write_json(args.output_root / "resolved_base_config.json", config)
    print(f"[BVHBatch] clips={len(items)} frames={sum(i['frames'] for i in items)} workers={args.workers}", flush=True)
    results = []
    save_summary(args, items, results)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_clip, item, args, config, recipe_hash) for item in items]
        for future in as_completed(futures):
            entry = future.result()
            results.append(entry)
            save_summary(args, items, results)
            print(f"[BVHBatch] {len(results)}/{len(items)} {entry['status']} {entry['relative']} "
                  f"reused={entry.get('reused', False)} elapsed={entry['elapsed_seconds']:.1f}s", flush=True)
    failed = sum(e["status"] != "ok" for e in results)
    print(f"[BVHBatch] complete ok={len(results)-failed} failed={failed} summary={args.output_root / 'batch_summary.json'}", flush=True)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())

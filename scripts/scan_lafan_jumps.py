#!/usr/bin/env python3
"""Screen native LAFAN BVH for confident jumps without SMPL-X or robot solves."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np

from humanoid_retarget_config import load_config
from nr_source import _bvh_metadata
from retarget_foot_clearance import bvh_foot_phase, positive
from retarget_jump_compensation import detect_jumps
from retarget_noitom_bvh_batch import detect_source_format, digest, write_json

ROOT = Path(__file__).resolve().parents[1]


def scan_clip(source, source_root, config, unit_scale=.01):
    """Use exactly the source tracks and detector used by COM compensation."""
    started = time.monotonic()
    entry = {"source": str(source), "relative": source.relative_to(source_root).as_posix()}
    try:
        if detect_source_format(source) != "lafan":
            raise ValueError("Expected a LAFAN BVH skeleton")
        frames, frame_time = _bvh_metadata(source)
        if frames < 3:
            raise ValueError("At least three source frames are required")
        positive(frame_time, "BVH frame time")
        fps = 1 / frame_time
        _, _, _, info = bvh_foot_phase(
            source, np.arange(frames), scale=1., unit_scale=unit_scale,
            expected_frames=frames, expected_fps=fps, include_markers=True)
        analysis = detect_jumps(info.pop("tracks"), fps, config)
        events = [dict(event, start_time_s=event["start"] / fps,
                       end_time_s=event["end"] / fps) for event in analysis["events"]]
        labels = dict(sorted(Counter(event["label"] for event in events).items()))
        entry.update(status="ok", selected=bool(events), frames=frames, fps=fps,
                     duration_s=frames / fps, source_sha256=digest(source),
                     event_count=len(events), label_counts=labels,
                     single_takeoff_count=sum(len(e["takeoff_feet"]) == 1 for e in events),
                     double_takeoff_count=sum(len(e["takeoff_feet"]) == 2 for e in events),
                     max_confidence=max((e["confidence"] for e in events), default=None),
                     events=events, rejected=analysis["rejected"],
                     rejection_counts=dict(sorted(Counter(e["reason"] for e in analysis["rejected"]).items())),
                     ground=info)
    except Exception as exc:
        # A failed scan is unknown, never evidence that the motion has no jumps.
        entry.update(status="failed", selected=None, error=f"{type(exc).__name__}: {exc}")
    entry["elapsed_seconds"] = time.monotonic() - started
    return entry


def save_report(output_root, source_root, config_path, config, unit_scale, entries, elapsed):
    selected = [entry for entry in entries if entry.get("selected")]
    failed = sum(entry["status"] == "failed" for entry in entries)
    labels = Counter()
    for entry in selected:
        labels.update(entry["label_counts"])
    summary = {
        "schema_version": 1, "source_root": str(source_root), "config": str(config_path),
        "jump_config": config, "unit_scale": unit_scale,
        "detector_sha256": {name: digest(ROOT / "scripts" / name) for name in
                            ("scan_lafan_jumps.py", "retarget_jump_compensation.py",
                             "retarget_foot_clearance.py", "nr_source.py")},
        "frame_convention": "zero-based original BVH frames; start/end are inclusive contact boundaries",
        "selection_rule": "at least one accepted event from the compensation detector",
        "confidence_kind": "heuristic_score_not_probability",
        "complete": failed == 0, "total": len(entries), "failed": failed,
        "selected": len(selected), "not_selected": len(entries) - failed - len(selected),
        "total_source_frames": sum(e.get("frames", 0) for e in entries),
        "selected_source_frames": sum(e["frames"] for e in selected),
        "event_count": sum(e["event_count"] for e in selected),
        "label_counts": dict(sorted(labels.items())), "elapsed_seconds": elapsed,
        "results": entries,
    }
    write_json(output_root / "jump_scan.json", summary)
    (output_root / "selected_files.txt").write_text(
        "".join(entry["relative"] + "\n" for entry in selected), encoding="utf-8")
    fields = ["relative", "status", "selected", "frames", "fps", "event_count",
              "single_takeoff_count", "double_takeoff_count", "max_confidence",
              "label_counts", "rejection_counts", "error"]
    with (output_root / "jump_scan.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            row = dict(entry)
            for key in ("label_counts", "rejection_counts"):
                row[key] = json.dumps(entry.get(key, {}), sort_keys=True)
            writer.writerow(row)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "demo_configs/umr_lafan_jump_compensation.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output/lafan_jump_scan")
    parser.add_argument("--pattern", action="append", help="Repeat to combine BVH glob patterns.")
    parser.add_argument("--unit-scale", type=float, default=.01)
    args = parser.parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    positive(args.unit_scale, "unit_scale")
    config = load_config(args.config).get("retarget", {}).get("jump_compensation", {})
    if not config.get("enabled", False):
        parser.error("The selected config must enable retarget.jump_compensation")
    sources = [p for p in sorted(args.source_root.rglob("*")) if p.is_file()
               and p.suffix.lower() == ".bvh"
               and (not args.pattern or any(p.match(pattern) for pattern in args.pattern))]
    if not sources:
        parser.error("No BVH files matched --source-root and --pattern")
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    entries = []
    for i, source in enumerate(sources, 1):
        entry = scan_clip(source, args.source_root, config, args.unit_scale)
        entries.append(entry)
        detail = (f"jumps={entry['event_count']} {entry['label_counts']}"
                  if entry["status"] == "ok" else entry["error"])
        print(f"[JumpScan] {i}/{len(sources)} {entry['relative']} {detail}", flush=True)
    summary = save_report(args.output_root, args.source_root, args.config, config,
                          args.unit_scale, entries, time.monotonic() - started)
    print(f"[JumpScan] selected={summary['selected']}/{summary['total']} "
          f"events={summary['event_count']} failed={summary['failed']} "
          f"elapsed={summary['elapsed_seconds']:.1f}s report={args.output_root / 'jump_scan.json'}", flush=True)
    return int(summary["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())

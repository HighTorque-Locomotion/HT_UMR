#!/usr/bin/env python3
"""Exercise native BVH screening and exact batch selection without robot assets."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from convert_lafan_bvh_to_smplx import BODY_PARENTS, LAFAN_BODY_JOINT_NAMES
import retarget_noitom_bvh_batch as batch
import retarget_foot_clearance as foot
from scan_lafan_jumps import scan_clip, save_report


def write_motion(path, takeoff=(0, 1), jumping=True):
    """Centimetre, Y-up skeleton: move the root ballistically and bend one leg."""
    lines, order = ["HIERARCHY"], []

    def node(index):
        name = LAFAN_BODY_JOINT_NAMES[index]
        root = index == 0
        offset = (0, 10, 0)
        if root:
            offset = (0, 0, 0)
        elif name.endswith("UpLeg"):
            offset = (10 if name.startswith("Left") else -10, -10, 0)
        elif name.endswith(("Leg", "Foot")):
            offset = (0, -45, 0)
        elif name.endswith("Toe"):
            offset = (0, 0, 15)
        lines.extend([("ROOT " if root else "JOINT ") + name, "{",
                      "OFFSET " + " ".join(map(str, offset)),
                      "CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
                      if root else "CHANNELS 3 Xrotation Yrotation Zrotation"])
        order.append(index)
        for child in np.flatnonzero(BODY_PARENTS == index):
            node(int(child))
        lines.append("}")

    node(0)
    n, a, b, fps = 120, 50, 68, 30
    values = np.zeros((n, 3 + 3 * len(order)))
    values[:, 1] = 100
    if jumping:
        t = np.arange(b - a + 1) / fps
        values[a:b+1, 1] += 100 * .5 * 9.81 * t * (t[-1] - t)
    for side, name in enumerate(("LeftUpLeg", "RightUpLeg")):
        if side not in takeoff:
            column = 3 + 3 * order.index(LAFAN_BODY_JOINT_NAMES.index(name)) + 2
            values[a-10:b+11, column] = 50
    lines.extend(["MOTION", f"Frames: {n}", f"Frame Time: {1 / fps:.12f}"])
    with path.open("w") as stream:
        stream.write("\n".join(lines) + "\n")
        np.savetxt(stream, values, fmt="%.10f")


class JumpScanChecks(unittest.TestCase):
    def test_keyword_and_glob_selection_with_preserved_timing_dry_run(self):
        with tempfile.TemporaryDirectory(prefix="jump_batch_") as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            for name in ("jumps1_subject2.bvh", "nested/jumps2_subject1.bvh", "dance1_subject1.bvh"):
                write_motion(root / name)
            base = ["batch", "--source-root", str(root), "--source-format", "lafan",
                    "--config", "unused.json", "--slots", "unused.npz",
                    "--output-root", str(root / "output"), "--preserve-foot-timing", "--dry-run"]
            for patterns, expected in [(["jump"], 2), (["JUMP"], 2), (["jumps*.bvh"], 2),
                                       (["nested/*.bvh"], 1), (["jumps1_subject2.bvh"], 1),
                                       (["jump", "dance"], 3)]:
                with self.subTest(patterns=patterns):
                    argv = base + [v for pattern in patterns for v in ("--pattern", pattern)]
                    output = io.StringIO()
                    with patch("sys.argv", argv), contextlib.redirect_stdout(output):
                        self.assertEqual(batch.main(), 0)
                    self.assertIn(f"dry-run clips={expected} ", output.getvalue())
                    self.assertIn("preserve-foot-timing", output.getvalue())
                    if "dance" not in patterns:
                        self.assertNotIn("dance1_subject1.bvh", output.getvalue())
                    self.assertFalse((root / "output").exists())

    def test_preserved_timing_keeps_held_foot_and_flight_synchronized(self):
        args = SimpleNamespace(config=batch.ROOT / "demo_configs/umr_lafan_jump_compensation_5cm_bvh_reference.json",
                               preserve_foot_timing=False)
        default, _ = batch.load_batch_config(args)
        args.preserve_foot_timing = True
        preserved, _ = batch.load_batch_config(args)
        h = np.zeros((240, 2))
        h[20:160, 0] = .02  # Long held leg with motion, followed by a double-foot flight.
        h[190:210] = .10
        speed = np.zeros_like(h)
        speed[h > 0] = .3
        centers = np.zeros((len(h), 2, 3))
        centers[:, :, 0] = np.cumsum(speed, axis=0) / 30
        centers[:, :, 2] = h
        before = foot.plan_swing_clearance(h, 30, default["retarget"]["foot_clearance"], speed, centers)
        after = foot.plan_swing_clearance(h, 30, preserved["retarget"]["foot_clearance"], speed, centers)
        self.assertTrue(before["transport_timing_active"].any())
        self.assertNotIn("transport_sample_times", after)
        self.assertNotIn("transport_timing_active", after)
        groups = [np.array([0]), np.array([1])]
        positions = foot.shift_foot_targets(centers, after, groups)
        np.testing.assert_array_equal(positions[:, :, :2], centers[:, :, :2])
        angles = np.arange(len(h)) * .025
        vectors = np.repeat(np.stack([np.cos(angles), np.sin(angles), np.zeros(len(h))], axis=1)[:, None], 2, axis=1)
        np.testing.assert_array_equal(foot.retime_foot_vectors(vectors, after, groups), vectors)
        # Height compensation still clears low travel; high airborne motion is retained.
        self.assertGreaterEqual(after["target_heights"][40:140, 0].min(), .05)
        np.testing.assert_allclose(after["target_heights"][190:210], .10)
        np.testing.assert_array_equal(after["target_heights"][after["stance"]], 0)
        self.assertEqual(preserved["retarget"]["jump_compensation"], default["retarget"]["jump_compensation"])
        self.assertTrue(preserved["retarget"]["jump_compensation"]["enabled"])
        unchanged, _ = batch.load_batch_config(SimpleNamespace(config=args.config, preserve_foot_timing=False))
        self.assertEqual(unchanged, default)

    def test_bvh_fk_floor_and_single_double_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for feet, label in [((0, 1), "double_to_double"), ((0,), "left_to_left"),
                                ((1,), "right_to_right")]:
                with self.subTest(label=label):
                    path = root / (label + ".bvh")
                    write_motion(path, feet)
                    result = scan_clip(path, root, {})
                    self.assertEqual(result["status"], "ok", result)
                    self.assertTrue(result["selected"])
                    self.assertEqual(result["label_counts"], {label: 1})
                    event = result["events"][0]
                    self.assertEqual((event["start"], event["end"]), (50, 68))
                    self.assertAlmostEqual(event["start_time_s"], 50 / 30)

    def test_foot_lift_without_flight_is_not_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "foot_lift.bvh"
            write_motion(path, (0,), jumping=False)
            result = scan_clip(path, root, {})
            self.assertEqual(result["status"], "ok", result)
            self.assertFalse(result["selected"])

    def test_bad_input_is_unknown_and_makes_report_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "broken.bvh"
            path.write_text("invalid BVH")
            result = scan_clip(path, root, {})
            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["selected"])
            summary = save_report(root, root, root / "config.json", {}, .01, [result], 0)
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["not_selected"], 0)
            self.assertEqual((root / "selected_files.txt").read_text(), "")

    def test_report_list_drives_batch_before_robot_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jump_path, still_path = root / "dance.bvh", root / "jumps_standing.bvh"
            write_motion(jump_path)
            write_motion(still_path, jumping=False)
            entries = [scan_clip(path, root, {}) for path in (jump_path, still_path)]
            summary = save_report(root, root, root / "config.json", {}, .01, entries, 0)
            self.assertEqual(summary["selected"], 1)
            self.assertEqual(summary["event_count"], 1)
            file_list = root / "selected_files.txt"
            self.assertEqual(batch.read_source_file_list(file_list, root), {"dance.bvh"})
            self.assertEqual(json.loads((root / "jump_scan.json").read_text())["total"], 2)
            argv = ["batch", "--source-root", str(root), "--source-format", "lafan",
                    "--file-list", str(file_list), "--config", "unused.json", "--slots", "unused.npz",
                    "--output-root", str(root / "no_robot_output"), "--dry-run"]
            output = io.StringIO()
            with patch("sys.argv", argv), contextlib.redirect_stdout(output):
                self.assertEqual(batch.main(), 0)
            self.assertIn("dance.bvh", output.getvalue())
            self.assertNotIn("jumps_standing.bvh", output.getvalue())
            self.assertIn("dry-run clips=1 frames=120", output.getvalue())
            self.assertFalse((root / "no_robot_output").exists())

    def test_file_list_is_exact_and_rejects_empty_or_missing_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "nested" / "dance [1].bvh").touch()
            file_list = root / "files.txt"
            file_list.write_text("nested/dance [1].bvh\nnested/dance [1].bvh\n\n")
            self.assertEqual(batch.read_source_file_list(file_list, root), {"nested/dance [1].bvh"})
            for content in ("", "missing.bvh", "../outside.bvh", str(root / "absolute.bvh"), "*.bvh"):
                with self.subTest(content=content):
                    file_list.write_text(content)
                    with self.assertRaises(ValueError):
                        batch.read_source_file_list(file_list, root)


if __name__ == "__main__":
    unittest.main()

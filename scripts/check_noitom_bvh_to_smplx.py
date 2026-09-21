#!/usr/bin/env python3
"""Regression checks for Noitom conversion; no body model or motion asset needed."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from convert_noitom_bvh_to_smplx import (
    BODY_JOINT_NAMES, BODY_PARENTS, bvh_body_transforms, transfer_body_rotations,
)
from nr_source import _bvh_local_tracks, _parse_bvh


class ConversionChecks(unittest.TestCase):
    def skeleton(self):
        nodes = []
        name_to_id = {}
        cursor = 0
        for joint, name in enumerate(BODY_JOINT_NAMES):
            parent = BODY_PARENTS[joint]
            parent_name = BODY_JOINT_NAMES[parent] if parent >= 0 else None
            if name == "Spine3":
                nodes.append(dict(name="Spine2", parent=name_to_id["Spine1"], offset=np.array([0., 8., 0.]),
                                  channels=["Yrotation", "Xrotation", "Zrotation"], channel_start=cursor))
                name_to_id["Spine2"] = len(nodes) - 1
                cursor += 3
                parent_name = "Spine2"
            channels = (["Xposition", "Yposition", "Zposition"] if parent < 0 else []) + ["Yrotation", "Xrotation", "Zrotation"]
            name_to_id[name] = len(nodes)
            nodes.append(dict(name=name, parent=name_to_id[parent_name] if parent_name else -1,
                              offset=np.array([0., 92.2 if parent < 0 else 10., 0.]),
                              channels=channels, channel_start=cursor))
            cursor += len(channels)
        values = np.zeros((2, cursor))
        values[:, :3] = [[0., 92.2, 0.], [10., 95., 20.]]
        return nodes, values, name_to_id

    def test_absolute_root_and_units(self):
        nodes, values, _ = self.skeleton()
        _, rest, root = bvh_body_transforms(nodes, values)
        assert_allclose(root, [[0, .922, 0], [.1, .95, .2]])
        assert_allclose(rest[0], [0, .922, 0])

    def test_collapsed_spine_preserves_rotation(self):
        nodes, values, names = self.skeleton()
        values[1, nodes[names["Spine2"]]["channel_start"]] = 30
        values[1, nodes[names["Spine3"]]["channel_start"] + 1] = 20
        world, _, _ = bvh_body_transforms(nodes, values)
        expected = Rotation.from_euler("Y", 30, degrees=True) * Rotation.from_euler("X", 20, degrees=True)
        assert_allclose(world[1, 9], expected.as_matrix(), atol=1e-10)
        assert_allclose(world[1, 16], expected.as_matrix(), atol=1e-10)

    def test_transfer_aligns_bones_and_pelvis(self):
        rng = np.random.default_rng(42)
        source_rest = np.zeros((22, 3))
        for joint, parent in enumerate(BODY_PARENTS):
            source_rest[joint] = rng.normal(size=3) if parent < 0 else source_rest[parent] + rng.normal(size=3)
        target_rest = source_rest @ Rotation.from_euler("Z", 15, degrees=True).as_matrix().T
        world = Rotation.random(44, random_state=rng).as_matrix().reshape(2, 22, 3, 3)
        root = np.array([[1., 2., 3.], [4., 5., 6.]])
        poses, trans, _ = transfer_body_rotations(world, source_rest, target_rest, root)
        local = Rotation.from_rotvec(poses[:, :66].reshape(-1, 3)).as_matrix().reshape(2, 22, 3, 3)
        reconstructed = local.copy()
        for joint, parent in enumerate(BODY_PARENTS):
            if parent >= 0:
                reconstructed[:, joint] = reconstructed[:, parent] @ local[:, joint]
        for child, parent in enumerate(BODY_PARENTS):
            if parent <= 0:
                continue
            src = source_rest[child] - source_rest[parent]
            dst = target_rest[child] - target_rest[parent]
            assert_allclose(reconstructed[:, parent] @ (dst / np.linalg.norm(dst)),
                            world[:, parent] @ (src / np.linalg.norm(src)), atol=1e-6)
        assert_allclose(trans + target_rest[0], root, atol=1e-6)
        assert_allclose(poses[:, 66:], 0)

    def test_rejects_unsupported_skeleton(self):
        nodes, values, _ = self.skeleton()
        nodes[1]["name"] = "UnknownLeg"
        with self.assertRaisesRegex(ValueError, "missing joints"):
            bvh_body_transforms(nodes, values)

    def test_parser_channel_order_and_single_frame(self):
        text = """HIERARCHY
ROOT Hips
{
 OFFSET 0 92.2 0
 CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
 End Site
 {
  OFFSET 0 1 0
 }
}
MOTION
Frames: 1
Frame Time: 0.016667
1 2 3 30 20 10
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "single.bvh"
            path.write_text(text)
            nodes, values, dt = _parse_bvh(path)
            tracks = _bvh_local_tracks(nodes, values, [0])
        assert_allclose(tracks["Hips"][0], [[1, 2, 3]])
        actual = Rotation.from_quat(tracks["Hips"][1]).as_matrix()[0]
        expected = (Rotation.from_euler("Y", 30, degrees=True)
                    * Rotation.from_euler("X", 20, degrees=True)
                    * Rotation.from_euler("Z", 10, degrees=True)).as_matrix()
        assert_allclose(actual, expected, atol=1e-10)
        self.assertEqual(dt, .016667)


if __name__ == "__main__":
    unittest.main()

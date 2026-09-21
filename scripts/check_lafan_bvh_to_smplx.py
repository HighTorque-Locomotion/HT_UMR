#!/usr/bin/env python3
"""Check LAFAN basis conversion against independently constructed BVH motion."""
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from convert_lafan_bvh_to_smplx import (
    BODY_PARENTS, LAFAN_BODY_JOINT_NAMES, lafan_body_transforms, lafan_joint_bases,
    transfer_body_rotations,
)
from retarget_noitom_bvh_batch import result_relative_path


class LAFANConversionChecks(unittest.TestCase):
    def test_tpose_and_animated_global_rotations(self):
        bases = lafan_joint_bases()
        assert_allclose(bases @ bases.transpose(0, 2, 1), np.broadcast_to(np.eye(3), bases.shape))
        assert_allclose(np.linalg.det(bases), 1)
        rng = np.random.default_rng(13)
        # Arbitrary nonzero lengths ensure the test exercises translation and
        # every edge, including both hips, the third spine, and leaf rotations.
        rest = np.zeros((22, 3))
        for joint, parent in enumerate(BODY_PARENTS):
            if parent >= 0:
                rest[joint] = rest[parent] + rng.normal(size=3)
        expected = np.repeat(np.eye(3)[None, None], 22, axis=1)
        expected = np.concatenate([expected, Rotation.random(44, random_state=rng).as_matrix().reshape(2, 22, 3, 3)])
        raw_world = expected @ bases.transpose(0, 2, 1)[None]
        roots = np.array([[0., 0.9, 0.], [1., 2., 3.], [-2., 0.8, 4.]])
        values = np.zeros((3, 69))
        values[:, :3] = roots * 100
        nodes = []
        cursor = 0
        for joint, (name, parent) in enumerate(zip(LAFAN_BODY_JOINT_NAMES, BODY_PARENTS)):
            local = raw_world[:, joint] if parent < 0 else raw_world[:, parent].transpose(0, 2, 1) @ raw_world[:, joint]
            channels = (["Xposition", "Yposition", "Zposition"] if parent < 0 else []) + ["Zrotation", "Yrotation", "Xrotation"]
            offset = np.array([123., 92., -45.]) if parent < 0 else bases[parent] @ (rest[joint] - rest[parent]) * 100
            nodes.append(dict(name=name, parent=int(parent), offset=offset, channels=channels, channel_start=cursor))
            start = cursor + (3 if parent < 0 else 0)
            values[:, start:start + 3] = Rotation.from_matrix(local).as_euler("ZYX", degrees=True)
            cursor += len(channels)
        world, canonical_rest, root = lafan_body_transforms(nodes, values)
        assert_allclose(world, expected, atol=1e-12)
        assert_allclose(canonical_rest, rest, atol=1e-12)
        assert_allclose(root, roots, atol=1e-12)
        poses, trans, _ = transfer_body_rotations(world, canonical_rest, rest, root)
        actual = Rotation.from_rotvec(poses[:, :66].reshape(-1, 3)).as_matrix().reshape(3, 22, 3, 3)
        for joint, parent in enumerate(BODY_PARENTS):
            if parent >= 0:
                actual[:, joint] = actual[:, parent] @ actual[:, joint]
        assert_allclose(actual, expected, atol=4e-7)
        assert_allclose(trans + rest[0], roots, atol=1e-7)
        assert_allclose(poses[:, 66:], 0)

    def test_result_name_and_relative_directory(self):
        path = result_relative_path("dance/push1_subject2.bvh", "{name}_{date}_{robot}_umr",
                                    "260915", "PiPlus_S_12L8A0G2H0W_LSE_ZedMini_40V_260908")
        self.assertEqual(str(path), "dance/push1_subject2_260915_PiPlus_S_12L8A0G2H0W_LSE_ZedMini_40V_260908_umr.npz")
        for template in ("../{name}", "{name}/bad", ""):
            with self.assertRaises(ValueError):
                result_relative_path("push1_subject2.bvh", template, "260915", "robot")


if __name__ == "__main__":
    unittest.main()

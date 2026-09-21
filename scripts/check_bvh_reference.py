"""Asset-independent checks for BVH rest calibration and retained motion."""
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

from convert_noitom_bvh_to_smplx import BODY_PARENTS
from retarget_body_segment_surface import transport_tpose_robot_normals
from retarget_bvh_reference import (reference_body_pose, calibrate_reference_normals,
                                    calibrate_lateral_positions, lafan_reference_alignment)


class BVHReferenceChecks(unittest.TestCase):
    def test_reference_reconstructs_global_corrections(self):
        world = Rotation.random(22, random_state=42).as_matrix()
        local = Rotation.from_rotvec(reference_body_pose(world).reshape(22, 3)).as_matrix()
        result = local.copy()
        for joint, parent in enumerate(BODY_PARENTS):
            if parent >= 0:
                result[joint] = result[parent] @ local[joint]
        assert_allclose(result, world, atol=5e-7)

    def test_invalid_alignment_rejected(self):
        identity = np.tile(np.eye(3), (22, 1, 1))
        for value in (np.zeros((22, 3, 3)), identity * np.nan, identity[:21], -identity):
            with self.assertRaises(ValueError):
                reference_body_pose(value)

    def fixture(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        faces = np.array([[0, 1, 2]])
        binding = {"face_ids": np.zeros(3, dtype=int)}
        normals = np.eye(3, dtype=np.float32)
        bias = Rotation.from_euler("xyz", [8, -17, 4], degrees=True).as_matrix()
        reference = vertices @ bias.T + [1, 2, 3]
        return vertices, faces, binding, normals, reference

    def test_removes_fixed_bias_but_keeps_tilt_and_heading(self):
        vertices, faces, binding, normals, reference = self.fixture()
        corrected = calibrate_reference_normals(normals, binding, vertices, faces, reference, [True]*3)
        # Both a resting foot and a genuinely tilted/turning foot. No contact
        # flag or ground normal is supplied to calibration or transport.
        movement = Rotation.from_euler("xyz", [[0, 0, 0], [25, -40, 80]], degrees=True).as_matrix()
        motion = np.einsum("fij,vj->fvi", movement, reference)
        actual = transport_tpose_robot_normals(corrected, binding, vertices, faces, motion)
        expected = np.einsum("fij,nj->fni", movement, normals)
        assert_allclose(actual, expected, atol=1e-6)
        assert_allclose(np.linalg.norm(actual, axis=-1), 1, atol=1e-6)

    def test_unselected_segments_are_unchanged(self):
        vertices, faces, binding, normals, reference = self.fixture()
        corrected = calibrate_reference_normals(normals, binding, vertices, faces, reference, [True, False, False])
        assert_array_equal(corrected[1:], normals[1:])
        assert_array_equal(normals, np.eye(3))
        unchanged = calibrate_reference_normals(normals, binding, vertices, faces, reference, [False]*3)
        assert_array_equal(unchanged, normals)

    def test_identity_reference_preserves_vectors(self):
        vertices, faces, binding, normals, _ = self.fixture()
        corrected = calibrate_reference_normals(normals, binding, vertices, faces, vertices, [True]*3)
        assert_allclose(corrected, normals, atol=1e-7)

    def test_lateral_reference_matches_width_and_follows_heading(self):
        # The source's rest feet are narrower than the robot's. Match the
        # reference once, while keeping a genuine inward/outward foot motion.
        source_ref = np.array([[.025, 0, 0], [-.025, 0, 0], [0., 0, 1]])
        robot_ref = np.array([[.05, 0, 0], [-.05, 0, 0], [0., 0, 1]])
        slots = np.array([source_ref, source_ref.copy(), source_ref.copy()])
        slots[1, 0, 0] += .08
        groups = {'leftFoot': np.array([0]), 'rightFoot': np.array([1])}
        headings = np.array([[1., 0], [1., 0], [0., 1]])
        result, offsets = calibrate_lateral_positions(slots, source_ref, robot_ref, groups, [0, 1, 2], headings)
        assert_allclose(result[0], robot_ref)
        self.assertAlmostEqual(result[1, 0, 0] - result[0, 0, 0], .08)
        assert_allclose(result[2, :2, :2] - slots[2, :2, :2], [[0, .025], [0, -.025]])
        assert_array_equal(result[:, :, 2], slots[:, :, 2])
        assert_array_equal(result[:, 2], slots[:, 2])
        self.assertAlmostEqual(offsets['leftFoot'], .025)
        retimed, _ = calibrate_lateral_positions(slots, source_ref, robot_ref, groups, [0, 1, 2], headings,
                                                 {'leftFoot': np.array([0., 0., 0.])})
        assert_allclose(retimed[2, 0, :2] - slots[2, 0, :2], [.025, 0])

    def test_lafan_fixed_toe_offset_is_reference_not_motion(self):
        nodes = []
        for side in ['Left', 'Right']:
            parent = len(nodes)
            for name, ancestor in [(side+'Foot', -1), (side+'Toe', parent)]:
                nodes.append(dict(name=name, parent=ancestor, offset=np.zeros(3),
                                  channels=['Zrotation', 'Yrotation', 'Xrotation'], channel_start=3*len(nodes)))
        values = np.zeros((3, 12))
        values[:, 3] = values[:, 9] = 21.45
        # Foot motion is arbitrary and must not affect the inferred reference.
        values[:, 0] = [40, -20, 80]
        identity = np.tile(np.eye(3), (22, 1, 1))
        aligned, audit = lafan_reference_alignment(identity, nodes, values)
        expected = Rotation.from_euler('x', 21.45, degrees=True).as_matrix()
        assert_allclose(aligned[7], expected, atol=1e-7)
        assert_allclose(aligned[8], expected, atol=1e-7)
        assert_array_equal(aligned[10:], identity[10:])
        self.assertLess(audit['LeftToe']['max_variation_rad'], 1e-7)
        values[1, 3] += 5
        with self.assertRaisesRegex(ValueError, 'animated'):
            lafan_reference_alignment(identity, nodes, values)


if __name__ == "__main__":
    unittest.main()

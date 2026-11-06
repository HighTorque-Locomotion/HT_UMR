"""Geometry invariants for the local anatomical clipping adapter."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/local_regions"))
from clipped_surface import clipped_mesh


class ClippedSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.faces = np.array([[0, 1, 2]])

    def test_crossing_triangle_preserves_area_and_ownership(self):
        vertices, faces, owners = clipped_mesh(
            self.vertices,
            self.faces,
            17,
            np.array([0.0, -2.0, -2.0]),
            np.array([2.0, 2.0, 2.0]),
        )
        self.assertTrue(np.all(vertices[:, 0] >= 0))
        triangles = vertices[faces]
        area = (
            np.linalg.norm(
                np.cross(
                    triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
                ),
                axis=1,
            ).sum()
            / 2
        )
        self.assertAlmostEqual(area, 0.5)
        np.testing.assert_array_equal(owners, np.full(len(faces), 17))

    def test_disjoint_region_is_empty(self):
        vertices, faces, owners = clipped_mesh(
            self.vertices, self.faces, 17, np.full(3, 5.0), np.full(3, 6.0)
        )
        self.assertEqual(vertices.shape, (0, 3))
        self.assertEqual(faces.shape, (0, 3))
        self.assertEqual(owners.shape, (0,))

    def test_unbounded_region_preserves_triangle(self):
        vertices, faces, owners = clipped_mesh(
            self.vertices, self.faces, 17, np.full(3, -np.inf), np.full(3, np.inf)
        )
        np.testing.assert_array_equal(vertices[faces], self.vertices[self.faces])


if __name__ == "__main__":
    unittest.main()

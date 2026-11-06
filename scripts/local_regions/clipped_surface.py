"""Clip triangles to anatomical bounds, preserving geometry ownership."""

import numpy as np


def clipped_mesh(vertices, faces, geom_id, lo, hi):
    triangles = vertices[faces]
    keep = np.all(triangles.max(1) >= lo - 1e-9, axis=1) & np.all(
        triangles.min(1) <= hi + 1e-9, axis=1
    )
    triangles = triangles[keep]
    inside = np.all((triangles >= lo - 1e-9) & (triangles <= hi + 1e-9), axis=(1, 2))
    out = list(triangles[inside])
    for tri in triangles[~inside]:
        poly = list(tri)
        for axis in range(3):
            for bound, sign in [(lo[axis], 1), (hi[axis], -1)]:
                if not np.isfinite(bound) or not poly:
                    continue
                new = []
                for a, b in zip(poly, poly[1:] + poly[:1]):
                    da, db = sign * (a[axis] - bound), sign * (b[axis] - bound)
                    if da >= -1e-10:
                        new.append(a)
                    if (da >= 0) != (db >= 0):
                        new.append(a + (b - a) * da / (da - db))
                poly = new
        for i in range(1, len(poly) - 1):
            t = np.array([poly[0], poly[i], poly[i + 1]])
            if np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0])) > 1e-12:
                out.append(t)
    if not out:
        return np.empty((0, 3)), np.empty((0, 3), int), np.empty(0, int)
    v = np.asarray(out).reshape(-1, 3)
    return v, np.arange(len(v)).reshape(-1, 3), np.full(len(out), geom_id, int)

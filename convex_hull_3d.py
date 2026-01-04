from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

from geometry_utils import (
    Point3D,
    centroid3,
    cross3,
    norm3,
    signed_volume,
    sub3,
)


@dataclass(frozen=True)
class Face:
    a: int
    b: int
    c: int


def face_normal(points: Sequence[Point3D], face: Face) -> Point3D:
    a, b, c = points[face.a], points[face.b], points[face.c]
    return cross3(sub3(b, a), sub3(c, a))


def face_distance(points: Sequence[Point3D], face: Face, p: Point3D) -> float:
    a = points[face.a]
    normal = face_normal(points, face)
    denom = norm3(normal)
    if denom == 0:
        return 0.0
    return (
        normal[0] * (p[0] - a[0])
        + normal[1] * (p[1] - a[1])
        + normal[2] * (p[2] - a[2])
    ) / denom


def orient_face(points: Sequence[Point3D], face: Face, interior: Point3D) -> Face:
    a, b, c = face.a, face.b, face.c
    if signed_volume(points[a], points[b], points[c], interior) > 0:
        return Face(a, c, b)
    return face


def initial_tetrahedron(points: Sequence[Point3D]) -> Tuple[List[int], List[Face]]:
    n = len(points)
    if n < 4:
        raise ValueError("Need at least 4 points for 3D convex hull")
    p0 = min(range(n), key=lambda i: points[i][0])
    p1 = max(range(n), key=lambda i: points[i][0])
    if p0 == p1:
        raise ValueError("Degenerate input: all x identical")

    def distance_to_line(i: int) -> float:
        v = sub3(points[i], points[p0])
        w = sub3(points[p1], points[p0])
        cross = cross3(v, w)
        return norm3(cross)

    p2 = max((i for i in range(n) if i not in (p0, p1)), key=distance_to_line)
    if distance_to_line(p2) == 0:
        raise ValueError("Degenerate input: all points collinear")

    def distance_to_plane(i: int) -> float:
        return abs(signed_volume(points[p0], points[p1], points[p2], points[i]))

    p3 = max(
        (i for i in range(n) if i not in (p0, p1, p2)),
        key=distance_to_plane,
    )
    if distance_to_plane(p3) == 0:
        raise ValueError("Degenerate input: all points coplanar")

    base = [p0, p1, p2, p3]
    tetra_faces = [
        Face(p0, p1, p2),
        Face(p0, p2, p3),
        Face(p0, p3, p1),
        Face(p1, p3, p2),
    ]
    interior = centroid3([points[i] for i in base])
    oriented_faces = [orient_face(points, f, interior) for f in tetra_faces]
    return base, oriented_faces


def convex_hull_3d(points: Sequence[Point3D], eps: float = 1e-9) -> Tuple[List[Point3D], Set[Face]]:
    pts = list(points)
    base_indices, faces = initial_tetrahedron(pts)
    face_set: Set[Face] = set(faces)
    interior = centroid3([pts[i] for i in base_indices])
    count = len(base_indices)

    for idx, p in enumerate(pts):
        if idx in base_indices:
            continue
        visible = {f for f in face_set if face_distance(pts, f, p) > eps}
        if not visible:
            count += 1
            interior = (
                (interior[0] * (count - 1) + p[0]) / count,
                (interior[1] * (count - 1) + p[1]) / count,
                (interior[2] * (count - 1) + p[2]) / count,
            )
            continue

        edge_use: dict[Tuple[int, int], int] = {}
        for f in visible:
            edges = [(f.a, f.b), (f.b, f.c), (f.c, f.a)]
            for e in edges:
                key = tuple(sorted(e))
                edge_use[key] = edge_use.get(key, 0) + 1

        horizon: List[Tuple[int, int]] = []
        for f in visible:
            edges = [(f.a, f.b), (f.b, f.c), (f.c, f.a)]
            for e in edges:
                key = tuple(sorted(e))
                if edge_use[key] == 1:
                    horizon.append(e)

        face_set -= visible
        count += 1
        interior = (
            (interior[0] * (count - 1) + p[0]) / count,
            (interior[1] * (count - 1) + p[1]) / count,
            (interior[2] * (count - 1) + p[2]) / count,
        )

        for u, v in horizon:
            new_face = orient_face(pts, Face(u, v, idx), interior)
            face_set.add(new_face)

    return pts, face_set

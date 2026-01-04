from __future__ import annotations

from typing import List, Sequence, Tuple

import matplotlib

from convex_hull_3d import Face, convex_hull_3d, face_normal
from geometry_utils import Point2D, Point3D

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def lift_points(points: Sequence[Point2D]) -> List[Point3D]:
    return [(x, y, x * x + y * y) for x, y in points]


def delaunay_triangulation(points: Sequence[Point2D]) -> List[Tuple[Point2D, Point2D, Point2D]]:
    lifted = lift_points(points)
    lifted_pts, faces = convex_hull_3d(lifted)
    triangles: List[Tuple[Point2D, Point2D, Point2D]] = []
    for f in faces:
        normal = face_normal(lifted_pts, f)
        if normal[2] < 0:  # lower hull
            a, b, c = lifted_pts[f.a], lifted_pts[f.b], lifted_pts[f.c]
            triangles.append(((a[0], a[1]), (b[0], b[1]), (c[0], c[1])))
    return triangles


def plot_delaunay(
    points: Sequence[Point2D],
    triangles: Sequence[Tuple[Point2D, Point2D, Point2D]],
    path: str,
    title: str | None = None,
) -> None:
    fig, ax = plt.subplots()
    for tri in triangles:
        xs = [tri[0][0], tri[1][0], tri[2][0], tri[0][0]]
        ys = [tri[0][1], tri[1][1], tri[2][1], tri[0][1]]
        ax.plot(xs, ys, "b-", linewidth=1)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, "ro", markersize=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title or "Delaunay triangulation via lifting")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_delaunay_steps(points: Sequence[Point2D], path_prefix: str, step_stride: int = 1) -> List[str]:
    if step_stride < 1:
        raise ValueError("step_stride must be >= 1")
    paths: List[str] = []
    if len(points) < 4:
        path = f"{path_prefix}_{len(points):02d}.png"
        plot_delaunay(points, [], path, title=f"Delaunay step {len(points)}")
        return [path]
    start = 4
    indices = list(range(start, len(points) + 1, step_stride))
    if indices[-1] != len(points):
        indices.append(len(points))
    for i in indices:
        subset = points[:i]
        triangles = delaunay_triangulation(subset)
        path = f"{path_prefix}_{i:02d}.png"
        plot_delaunay(subset, triangles, path, title=f"Delaunay step {i}")
        paths.append(path)
    return paths

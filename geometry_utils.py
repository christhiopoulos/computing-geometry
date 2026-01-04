from __future__ import annotations

import math
import random
from typing import Iterable, List, Sequence, Tuple

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


def cross(o: Point2D, a: Point2D, b: Point2D) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def dot3(a: Point3D, b: Point3D) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def sub3(a: Point3D, b: Point3D) -> Point3D:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross3(a: Point3D, b: Point3D) -> Point3D:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm3(a: Point3D) -> float:
    return math.sqrt(dot3(a, a))


def triangle_area2(a: Point2D, b: Point2D, c: Point2D) -> float:
    return cross(a, b, c)


def distance2(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def signed_volume(a: Point3D, b: Point3D, c: Point3D, d: Point3D) -> float:
    return dot3(cross3(sub3(b, a), sub3(c, a)), sub3(d, a)) / 6.0


def centroid3(points: Sequence[Point3D]) -> Point3D:
    sx = sy = sz = 0.0
    for p in points:
        sx += p[0]
        sy += p[1]
        sz += p[2]
    n = float(len(points))
    return (sx / n, sy / n, sz / n)


def random_points2d(
    n: int, seed: int | None = None, lower: float = 0.0, upper: float = 100.0
) -> List[Point2D]:
    rng = random.Random(seed)
    return [(rng.uniform(lower, upper), rng.uniform(lower, upper)) for _ in range(n)]


def random_points3d(
    n: int, seed: int | None = None, lower: float = 0.0, upper: float = 100.0
) -> List[Point3D]:
    rng = random.Random(seed)
    return [
        (rng.uniform(lower, upper), rng.uniform(lower, upper), rng.uniform(lower, upper))
        for _ in range(n)
    ]


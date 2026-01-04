from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from geometry_utils import Point2D, cross, distance2, triangle_area2


def _point_on_segment(a: Point2D, b: Point2D, p: Point2D, eps: float) -> bool:
    if abs(triangle_area2(a, b, p)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _point_in_convex_polygon(hull: Sequence[Point2D], p: Point2D, eps: float = 1e-12) -> bool:
    if len(hull) == 0:
        return False
    if len(hull) == 1:
        return p == hull[0]
    if len(hull) == 2:
        return _point_on_segment(hull[0], hull[1], p, eps)
    for i in range(len(hull)):
        if triangle_area2(hull[i], hull[(i + 1) % len(hull)], p) < -eps:
            return False
    return True


def _prune_collinear(hull: List[Point2D], eps: float = 1e-12) -> List[Point2D]:
    if len(hull) <= 2:
        return hull
    changed = True
    cleaned = hull
    while changed and len(cleaned) > 2:
        changed = False
        updated: List[Point2D] = []
        for i in range(len(cleaned)):
            prev = cleaned[(i - 1) % len(cleaned)]
            curr = cleaned[i]
            nxt = cleaned[(i + 1) % len(cleaned)]
            if abs(triangle_area2(prev, curr, nxt)) <= eps:
                changed = True
                continue
            updated.append(curr)
        cleaned = updated
    return cleaned


def _add_point_to_hull(hull: List[Point2D], p: Point2D, eps: float = 1e-12) -> List[Point2D]:
    if _point_in_convex_polygon(hull, p, eps):
        return hull
    n = len(hull)
    visible = [i for i in range(n) if triangle_area2(hull[i], hull[(i + 1) % n], p) < -eps]
    if not visible:
        return hull
    visible.sort()
    start = visible[0]
    end = visible[-1]
    if len(visible) > 1:
        for k in range(1, len(visible)):
            if visible[k] != visible[k - 1] + 1:
                start = visible[k]
                end = visible[k - 1]
                break
    right = (end + 1) % n
    left = start
    new_hull: List[Point2D] = []
    idx = right
    new_hull.append(hull[idx])
    while idx != left:
        idx = (idx + 1) % n
        new_hull.append(hull[idx])
    new_hull.append(p)
    return _prune_collinear(new_hull, eps)


def incremental_hull(points: Sequence[Point2D]) -> List[Point2D]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    if len(pts) == 2:
        return pts
    a = pts[0]
    b = pts[1]
    c = None
    for p in pts[2:]:
        if triangle_area2(a, b, p) != 0:
            c = p
            break
    if c is None:
        return [pts[0], pts[-1]]
    if triangle_area2(a, b, c) > 0:
        hull = [a, b, c]
    else:
        hull = [a, c, b]
    used = {a, b, c}
    for p in pts:
        if p in used:
            continue
        hull = _add_point_to_hull(hull, p)
    return hull


def gift_wrapping(points: Sequence[Point2D]) -> List[Point2D]:
    pts = list(set(points))
    if len(pts) <= 1:
        return pts
    start = min(pts, key=lambda p: (p[0], p[1]))
    hull = [start]
    current = start
    while True:
        next_point = pts[0] if pts[0] != current else pts[1]
        for candidate in pts:
            if candidate == current:
                continue
            area = triangle_area2(current, next_point, candidate)
            if area < 0 or (area == 0 and distance2(current, candidate) > distance2(current, next_point)):
                next_point = candidate
        if next_point == start:
            break
        hull.append(next_point)
        current = next_point
    return hull


def quickhull(points: Sequence[Point2D]) -> List[Point2D]:
    pts = list(set(points))
    if len(pts) <= 1:
        return pts
    min_x = min(pts, key=lambda p: p[0])
    max_x = max(pts, key=lambda p: p[0])

    def side(a: Point2D, b: Point2D, p: Point2D) -> float:
        return cross(a, b, p)

    def add_hull_segment(a: Point2D, b: Point2D, candidates: List[Point2D], hull: List[Point2D]) -> None:
        if not candidates:
            return
        farthest = max(candidates, key=lambda p: abs(side(a, b, p)))
        left_of_af = [p for p in candidates if side(a, farthest, p) > 0]
        left_of_fb = [p for p in candidates if side(farthest, b, p) > 0]
        add_hull_segment(a, farthest, left_of_af, hull)
        hull.append(farthest)
        add_hull_segment(farthest, b, left_of_fb, hull)

    left_set = [p for p in pts if side(min_x, max_x, p) > 0]
    right_set = [p for p in pts if side(max_x, min_x, p) > 0]
    hull: List[Point2D] = [min_x]
    add_hull_segment(min_x, max_x, left_set, hull)
    hull.append(max_x)
    add_hull_segment(max_x, min_x, right_set, hull)
    return hull


def divide_and_conquer(points: Sequence[Point2D]) -> List[Point2D]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def rightmost_index(hull: Sequence[Point2D]) -> int:
        return max(range(len(hull)), key=lambda i: (hull[i][0], hull[i][1]))

    def leftmost_index(hull: Sequence[Point2D]) -> int:
        return min(range(len(hull)), key=lambda i: (hull[i][0], hull[i][1]))

    def upper_tangent(left: List[Point2D], right: List[Point2D]) -> Tuple[int, int]:
        i = rightmost_index(left)
        j = leftmost_index(right)
        n = len(left)
        m = len(right)
        moved = True
        while moved:
            moved = False
            while triangle_area2(left[i], right[j], left[(i + 1) % n]) >= 0:
                i = (i + 1) % n
            while triangle_area2(left[i], right[j], right[(j - 1) % m]) >= 0:
                j = (j - 1) % m
                moved = True
        return i, j

    def lower_tangent(left: List[Point2D], right: List[Point2D]) -> Tuple[int, int]:
        i = rightmost_index(left)
        j = leftmost_index(right)
        n = len(left)
        m = len(right)
        moved = True
        while moved:
            moved = False
            while triangle_area2(left[i], right[j], left[(i - 1) % n]) <= 0:
                i = (i - 1) % n
            while triangle_area2(left[i], right[j], right[(j + 1) % m]) <= 0:
                j = (j + 1) % m
                moved = True
        return i, j

    def merge(left: List[Point2D], right: List[Point2D]) -> List[Point2D]:
        if not left:
            return right
        if not right:
            return left
        if len(left) < 3 or len(right) < 3:
            return incremental_hull(left + right)

        upper_left, upper_right = upper_tangent(left, right)
        lower_left, lower_right = lower_tangent(left, right)

        merged: List[Point2D] = []
        idx = upper_left
        merged.append(left[idx])
        while idx != lower_left:
            idx = (idx + 1) % len(left)
            merged.append(left[idx])
        idx = lower_right
        merged.append(right[idx])
        while idx != upper_right:
            idx = (idx + 1) % len(right)
            merged.append(right[idx])
        return _prune_collinear(merged)

    def recurse(ps: List[Point2D]) -> List[Point2D]:
        if len(ps) <= 5:
            return incremental_hull(ps)
        mid = len(ps) // 2
        left = recurse(ps[:mid])
        right = recurse(ps[mid:])
        return merge(left, right)

    return recurse(pts)


def timed_hulls(points: Sequence[Point2D]) -> Dict[str, Tuple[List[Point2D], float]]:
    algorithms: Dict[str, Callable[[Sequence[Point2D]], List[Point2D]]] = {
        "incremental": incremental_hull,
        "gift_wrapping": gift_wrapping,
        "divide_conquer": divide_and_conquer,
        "quickhull": quickhull,
    }
    results: Dict[str, Tuple[List[Point2D], float]] = {}
    for name, algo in algorithms.items():
        start = time.perf_counter()
        hull = algo(points)
        elapsed = time.perf_counter() - start
        results[name] = (hull, elapsed)
    return results

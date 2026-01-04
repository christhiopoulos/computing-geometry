from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib

from geometry_utils import Point2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class KDNode:
    point: Point2D
    axis: int
    left: Optional["KDNode"]
    right: Optional["KDNode"]
    bounds: Tuple[float, float, float, float]  # minx, maxx, miny, maxy


def build_kd_tree(points: Sequence[Point2D], depth: int = 0, bounds: Tuple[float, float, float, float] | None = None) -> Optional[KDNode]:
    if not points:
        return None
    axis = depth % 2
    pts = sorted(points, key=lambda p: p[axis])
    mid = len(pts) // 2
    median = pts[mid]
    if bounds is None:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bounds = (min(xs), max(xs), min(ys), max(ys))
    left_bounds = bounds
    right_bounds = bounds
    if axis == 0:
        left_bounds = (bounds[0], median[0], bounds[2], bounds[3])
        right_bounds = (median[0], bounds[1], bounds[2], bounds[3])
    else:
        left_bounds = (bounds[0], bounds[1], bounds[2], median[1])
        right_bounds = (bounds[0], bounds[1], median[1], bounds[3])

    left = build_kd_tree(pts[:mid], depth + 1, left_bounds)
    right = build_kd_tree(pts[mid + 1 :], depth + 1, right_bounds)
    return KDNode(median, axis, left, right, bounds)


def range_search(node: Optional[KDNode], rect: Tuple[float, float, float, float]) -> List[Point2D]:
    if node is None:
        return []
    xmin, xmax, ymin, ymax = rect
    results: List[Point2D] = []
    x, y = node.point
    if xmin <= x <= xmax and ymin <= y <= ymax:
        results.append(node.point)
    axis = node.axis
    if axis == 0:
        if xmin <= x:
            results.extend(range_search(node.left, rect))
        if x <= xmax:
            results.extend(range_search(node.right, rect))
    else:
        if ymin <= y:
            results.extend(range_search(node.left, rect))
        if y <= ymax:
            results.extend(range_search(node.right, rect))
    return results


def plot_kd_tree(node: Optional[KDNode], points: Sequence[Point2D], path: str) -> None:
    fig, ax = plt.subplots()
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, "ko")

    def draw(node: Optional[KDNode]) -> None:
        if node is None:
            return
        x, y = node.point
        if node.axis == 0:
            ax.plot([x, x], [node.bounds[2], node.bounds[3]], "r--", linewidth=0.8)
        else:
            ax.plot([node.bounds[0], node.bounds[1]], [y, y], "b--", linewidth=0.8)
        draw(node.left)
        draw(node.right)

    draw(node)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("kd-tree splits")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


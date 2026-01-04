from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from geometry_utils import Point2D, cross


@dataclass
class Constraint:
    a: float
    b: float
    c: float  # a * x + b * y <= c

    def satisfied(self, p: Point2D, eps: float = 1e-9) -> bool:
        return self.a * p[0] + self.b * p[1] <= self.c + eps


def line_intersection(
    p1: Point2D, p2: Point2D, q1: Point2D, q2: Point2D
) -> Point2D | None:
    d = cross((0.0, 0.0), (p2[0] - p1[0], p2[1] - p1[1]), (q2[0] - q1[0], q2[1] - q1[1]))
    if abs(d) < 1e-12:
        return None
    t = cross((0.0, 0.0), (q1[0] - p1[0], q1[1] - p1[1]), (q2[0] - q1[0], q2[1] - q1[1])) / d
    return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))


def clip_polygon(poly: List[Point2D], cons: Constraint) -> List[Point2D]:
    if not poly:
        return []
    clipped: List[Point2D] = []
    for i in range(len(poly)):
        cur, nxt = poly[i], poly[(i + 1) % len(poly)]
        cur_inside = cons.satisfied(cur)
        nxt_inside = cons.satisfied(nxt)
        if cur_inside and nxt_inside:
            clipped.append(nxt)
        elif cur_inside and not nxt_inside:
            inter = intersect_edge_with_constraint(cur, nxt, cons)
            if inter:
                clipped.append(inter)
        elif not cur_inside and nxt_inside:
            inter = intersect_edge_with_constraint(cur, nxt, cons)
            if inter:
                clipped.append(inter)
            clipped.append(nxt)
    return clipped


def intersect_edge_with_constraint(a: Point2D, b: Point2D, cons: Constraint) -> Point2D | None:
    direction = (b[0] - a[0], b[1] - a[1])
    normal = (cons.a, cons.b)
    denom = normal[0] * direction[0] + normal[1] * direction[1]
    if abs(denom) < 1e-12:
        return None
    t = (cons.c - (normal[0] * a[0] + normal[1] * a[1])) / denom
    return (a[0] + t * direction[0], a[1] + t * direction[1])


def incremental_lp(constraints: Sequence[Constraint], objective: Tuple[float, float]) -> Tuple[Point2D | None, float | None, List[Point2D]]:
    INF = 1e6
    poly: List[Point2D] = [(-INF, -INF), (INF, -INF), (INF, INF), (-INF, INF)]
    for cons in constraints:
        poly = clip_polygon(poly, cons)
        if not poly:
            return None, None, []
    best_point = max(poly, key=lambda p: objective[0] * p[0] + objective[1] * p[1])
    value = objective[0] * best_point[0] + objective[1] * best_point[1]
    return best_point, value, poly


def plot_feasible_region(poly: Sequence[Point2D], constraints: Sequence[Constraint], objective: Tuple[float, float], solution: Point2D | None, path: str) -> None:
    fig, ax = plt.subplots()
    if poly:
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.fill(xs, ys, color="lightblue", alpha=1.0, label="Feasible", zorder=10)
    for cons in constraints:
        # plot line a x + b y = c on a window
        vals = []
        for x in range(-5, 40):
            if abs(cons.b) < 1e-9:
                continue
            y = (cons.c - cons.a * x) / cons.b
            vals.append((x, y))
        if vals:
            xs = [v[0] for v in vals]
            ys = [v[1] for v in vals]
            ax.plot(xs, ys, linewidth=0.5, linestyle="--")
    if solution:
        ax.plot(solution[0], solution[1], "ro", label="Optimum")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.set_title(f"Incremental LP objective ({objective[0]}x1 + {objective[1]}x2)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


from __future__ import annotations

import math
import time
from pprint import pprint

import matplotlib

from convex_hull_2d import timed_hulls
from convex_hull_3d import convex_hull_3d, face_normal
from delaunay import delaunay_triangulation, plot_delaunay, plot_delaunay_steps
from geometry_utils import Point2D, random_points2d, random_points3d
from kd_tree import build_kd_tree, plot_kd_tree, range_search
from lp_incremental import Constraint, incremental_lp, plot_feasible_region

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def compare_convex_hulls() -> None:
    points = random_points2d(200, seed=42, lower=0, upper=100)
    results = timed_hulls(points)

    print("Convex hull sizes and timing (200 points):")
    for name, (hull, t) in results.items():
        print(f"  {name:16s} -> {len(hull):3d} points, {t*1000:.3f} ms")

    hull_sets = {name: set(h) for name, (h, _) in results.items()}
    all_same = len({frozenset(s) for s in hull_sets.values()}) == 1
    print(f"Hulls identical across algorithms: {all_same}")

    sizes = [100, 500, 1000, 2000]
    print("\nTiming table (ms):")
    header = ["n", "incremental", "gift_wrap", "divide_conquer", "quickhull"]
    print(" | ".join(f"{h:>15s}" for h in header))
    for n in sizes:
        pts = random_points2d(n, seed=123 + n, lower=0, upper=100)
        times = timed_hulls(pts)
        row = [
            n,
            times["incremental"][1] * 1000,
            times["gift_wrapping"][1] * 1000,
            times["divide_conquer"][1] * 1000,
            times["quickhull"][1] * 1000,
        ]
        print(" | ".join(f"{v:15.3f}" if isinstance(v, float) else f"{v:15d}" for v in row))


def demo_3d_convex_hull() -> None:
    pts3d = random_points3d(180, seed=7, lower=0, upper=50)
    _, faces = convex_hull_3d(pts3d)
    print(f"\n3D convex hull computed with {len(faces)} faces from 180 points.")


def solve_given_lp() -> None:
    cons = [
        Constraint(-1, 3, 3),
        Constraint(-6, -1, -8),
        Constraint(-2, 7, 25),
        Constraint(-1, 8, 31),
        Constraint(-1, 1, -12),
        Constraint(1, -6, 2),
        Constraint(3, -12, -6),
        Constraint(-1, 0, 0),
        Constraint(0, -1, 0),
    ]
    objective = (-5.0, 12.0)
    best, value, poly = incremental_lp(cons, objective)
    print("\nLP solution using incremental half-plane intersection:")
    if best is None:
        print("  Infeasible problem")
    else:
        print(f"  Optimal point: ({best[0]:.3f}, {best[1]:.3f})")
        print(f"  Objective: {value:.3f}")
        plot_feasible_region(poly, cons, objective, best, "lp_feasible.png")
        print("  Feasible region plot saved to lp_feasible.png")


def demo_delaunay() -> None:
    pts = random_points2d(15, seed=21, lower=0, upper=30)
    tris = delaunay_triangulation(pts)
    plot_delaunay(pts, tris, "delaunay_example.png")
    step_paths = plot_delaunay_steps(pts, "delaunay_step")
    print("\nDelaunay triangulation created for 15 points (delaunay_example.png).")
    print(f"Delaunay construction steps saved to {len(step_paths)} frames (delaunay_step_XX.png).")


def demo_kd_tree() -> None:
    pts = random_points2d(200, seed=9, lower=0, upper=100)
    tree = build_kd_tree(pts)
    plot_kd_tree(tree, pts, "kd_tree.png")
    rect = (25, 75, 10, 60)
    hits = range_search(tree, rect)
    print("\nKD-tree range query within rectangle [25,75] x [10,60]:")
    for p in hits:
        print(f"  {p}")
    fig, ax = plt.subplots()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, "ko")
    ax.add_patch(plt.Rectangle((rect[0], rect[2]), rect[1] - rect[0], rect[3] - rect[2], fill=False, edgecolor="green"))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Range query result")
    ax.plot([p[0] for p in hits], [p[1] for p in hits], "ro")
    fig.tight_layout()
    fig.savefig("kd_range_query.png")
    plt.close(fig)
    print("KD-tree split plot: kd_tree.png, range query plot: kd_range_query.png")


if __name__ == "__main__":
    compare_convex_hulls()
    demo_3d_convex_hull()
    solve_given_lp()
    demo_delaunay()
    demo_kd_tree()

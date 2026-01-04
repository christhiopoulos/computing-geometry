# Computational Geometry Assignments (Python)

This workspace includes self-contained Python implementations for convex hull algorithms, incremental linear programming, Delaunay triangulation (via lifting), and kd-tree range search.

## Files
- `geometry_utils.py`: Shared point helpers and random point generation.
- `convex_hull_2d.py`: Incremental insertion, gift wrapping, divide-and-conquer (tangent merge), and QuickHull implementations with timing helper.
- `convex_hull_3d.py`: Incremental 3D convex hull construction.
- `delaunay.py`: Delaunay triangulation through lifting to a paraboloid plus plotting helper.
- `lp_incremental.py`: Incremental half-plane intersection LP solver for 2D with feasible-region plotting.
- `kd_tree.py`: kd-tree construction, orthogonal range search, and visualization.
- `demo.py`: Runs representative experiments and saves figures.

## Running the demos
Use Python 3.10+; matplotlib is used for plots.

```bash
python3 demo.py
```

Artifacts produced:
- Convex hull timing table (printed to stdout).
- `lp_feasible.png`: Feasible region and optimum for the provided LP.
- `delaunay_example.png`: Delaunay triangulation of a small point set.
- `delaunay_step_XX.png`: Step-by-step Delaunay construction frames.
- `kd_tree.png` and `kd_range_query.png`: kd-tree splits and a sample range query.

All randomness is seeded for reproducibility; adjust seeds or point counts inside `demo.py` for further experiments.

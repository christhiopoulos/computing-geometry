FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir matplotlib numpy

COPY geometry_utils.py convex_hull_2d.py convex_hull_3d.py \
     delaunay.py lp_incremental.py kd_tree.py demo.py ./

CMD ["python", "demo.py"]
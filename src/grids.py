"""Frozen P5-01 half-cell shifts, reused by P5-03."""
from __future__ import annotations
from typing import Any
import geopandas as gpd
from shapely.geometry import box


def shifted_grid_p501(
    original: gpd.GeoDataFrame,
    region: Any,
    scale: int,
    dx: float,
    dy: float,
) -> gpd.GeoDataFrame:
    """逐字复用 results/P5_01/code/spatial.py 的 shifted_grid 实现。

    对于 grid_id = f"{ix}_{iy}"：
    - 原始网格框为 [ix*scale, (ix+1)*scale] x [iy*scale, (iy+1)*scale]
    - 平移后网格框为 [(ix+dx)*scale, (ix+dx+1)*scale] x [(iy+dy)*scale, (iy+dy+1)*scale]
      即向 +X (East) 偏移 dx*scale 米，向 +Y (North) 偏移 dy*scale 米；
      同时当 dx>0 时补充 ix-1 边界邻居，当 dy>0 时补充 iy-1 边界邻居，以完整覆盖 region。
    """
    indices = set()
    for key in original.grid_id:
        ix, iy = map(int, key.split("_"))
        for x in ((ix - 1, ix) if dx else (ix,)):
            for y in ((iy - 1, iy) if dy else (iy,)):
                indices.add((x, y))
    rows = []
    for ix, iy in sorted(indices):
        geom = box(
            (ix + dx) * scale,
            (iy + dy) * scale,
            (ix + dx + 1) * scale,
            (iy + dy + 1) * scale,
        ).intersection(region)
        if geom.area > 1e-6:
            rows.append(
                dict(
                    grid_id=f"{ix}_{iy}",
                    geometry=geom,
                    observed_m2=geom.area,
                    observed_fraction=geom.area / scale**2,
                    partial_cell=int(geom.area / scale**2 < 1 - 1e-6),
                    scale_m=scale,
                )
            )
    grid = gpd.GeoDataFrame(rows, crs=original.crs)
    if not len(grid) or not grid.is_valid.all() or abs(grid.observed_m2.sum() - region.area) >= 0.05:
        raise ValueError("STOP: shifted grid support conservation failed")
    return grid

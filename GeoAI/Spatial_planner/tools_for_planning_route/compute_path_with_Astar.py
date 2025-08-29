#!/usr/bin/env python
# coding: utf-8

# In[57]:


import json
import heapq
from pathlib import Path
import sys
import math

import numpy as np
import rasterio
from rasterio.transform import xy
import fiona
from shapely.geometry import LineString, mapping
from pyproj import Transformer

# In[58]:


try:
    from lib_planning_route.bootstrap_planning_route import add_project_root
except ModuleNotFoundError:
    current_path = Path.cwd().resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / "lib_planning_route").exists():
            sys.path.insert(0, str(parent))
            break
    from lib_planning_route.bootstrap_planning_route import add_project_root

add_project_root()

from lib_planning_route.utils_planning_route import get_project_paths

data_dir, task_info_path = get_project_paths()



def get_task_info(task_json_path: str | None = None) -> dict:
    """
    Load the current task JSON. If a custom path is provided, use it;
    otherwise use the standard path from get_project_paths().
    """
    if task_json_path is None:
        _, default_path = get_project_paths()
        task_json_path = default_path
    with open(task_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# In[59]:


TURN_PENALTY       = 0.01   # small cost when changing direction (helps kill zig-zag)
STICK_TO_ROAD      = 5   # extra cost for being far from road (0..1). 0 disables.
SMOOTH_ITERS       = 1      # Chaikin smoothing iterations for output line (0..2)
NO_CORNER_CUTTING  = True   # disallow diagonals that “cut corners”
ROAD_Q             = 0.80   # infer roads as cheapest 20% if no road raster
NODATA_THRESHOLD   = 9.9e8
EPS                = 1e-3
MAX_SNAP_SAMPLES   = 50000

# ---------- helpers you already have (kept) ----------
def _octile(r0, c0, r1, c1):
    dx = abs(c1 - c0); dy = abs(r1 - r0)
    return (dx + dy) + (math.sqrt(2.0) - 2.0) * min(dx, dy)

def _movement_cost(cost_array, r, c, nr, nc):
    base = 0.5 * (float(cost_array[r, c]) + float(cost_array[nr, nc]))
    return base * (math.sqrt(2.0) if (r != nr and c != nc) else 1.0)

def _reconstruct(parent_r, parent_c, r, c):
    path = []
    while r != -1 and c != -1:
        path.append((r, c))
        r, c = parent_r[r, c], parent_c[r, c]
    path.reverse()
    return path

def _chamfer_dt_bool(road_mask: np.ndarray) -> np.ndarray:
    H, W = road_mask.shape
    inf = 10**9
    dt = np.full((H, W), inf, dtype=np.int32)
    dt[road_mask] = 0
    for r in range(H):
        for c in range(W):
            v = dt[r, c]
            if r > 0:
                v = min(v, dt[r-1, c] + 3)
                if c > 0: v = min(v, dt[r-1, c-1] + 4)
                if c < W-1: v = min(v, dt[r-1, c+1] + 4)
            if c > 0: v = min(v, dt[r, c-1] + 3)
            dt[r, c] = v
    for r in range(H-1, -1, -1):
        for c in range(W-1, -1, -1):
            v = dt[r, c]
            if r < H-1:
                v = min(v, dt[r+1, c] + 3)
                if c > 0: v = min(v, dt[r+1, c-1] + 4)
                if c < W-1: v = min(v, dt[r+1, c+1] + 4)
            if c < W-1: v = min(v, dt[r, c+1] + 3)
            dt[r, c] = v
    return dt.astype(np.float32) / 3.0

def _normalize01(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(arr, dtype=np.float32)
    v = arr[mask]
    lo = float(np.nanpercentile(v, 1))
    hi = float(np.nanpercentile(v, 99))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    out[out < 0] = 0.0
    out[out > 1] = 1.0
    return out

def _load_or_infer_road_mask(cost_array: np.ndarray, road_mask: np.ndarray = None):
    if road_mask is not None:
        return road_mask.astype(bool)
    finite = np.isfinite(cost_array) & (cost_array < NODATA_THRESHOLD)
    if not np.any(finite):
        return np.zeros_like(cost_array, dtype=bool)
    thr = float(np.quantile(cost_array[finite], ROAD_Q))
    mask = np.zeros_like(cost_array, dtype=bool)
    mask[finite] = cost_array[finite] <= thr
    return mask

def _snap_to_road(r, c, road_mask: np.ndarray):
    rr, cc = np.where(road_mask)
    if rr.size == 0:
        return r, c
    if rr.size > MAX_SNAP_SAMPLES:
        idx = np.random.choice(rr.size, size=MAX_SNAP_SAMPLES, replace=False)
        rr, cc = rr[idx], cc[idx]
    d2 = (rr - r)**2 + (cc - c)**2
    k = int(np.argmin(d2))
    return int(rr[k]), int(cc[k])

# ---------- NEW: 8-neighbor generator with corner-cutting guard ----------
def _neighbors8(r, c, impassable: np.ndarray):
    rows, cols = impassable.shape
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]:
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            continue
        if impassable[nr, nc]:
            continue
        # prevent corner cutting on diagonals
        if NO_CORNER_CUTTING and dr != 0 and dc != 0:
            if impassable[r, nc] or impassable[nr, c]:
                continue
        yield nr, nc, dr, dc

# ---------- NEW: light simplifier & smoothing for output ----------
def _remove_colinear_steps(pixels):
    if not pixels: return pixels
    out = [pixels[0]]
    prev_dr = prev_dc = None
    for i in range(1, len(pixels)):
        r0, c0 = out[-1]
        r1, c1 = pixels[i]
        dr = np.sign(r1 - r0); dc = np.sign(c1 - c0)
        if prev_dr is None or (dr, dc) != (prev_dr, prev_dc):
            out.append((r1, c1))
            prev_dr, prev_dc = dr, dc
        else:
            # continue straight; replace last point
            out[-1] = (r1, c1)
    return out

def _chaikin_smooth(coords_xy, iterations=1, w=0.25):
    if iterations <= 0 or len(coords_xy) < 3: 
        return coords_xy
    pts = coords_xy
    for _ in range(iterations):
        new_pts = [pts[0]]
        for i in range(len(pts)-1):
            x0, y0 = pts[i]; x1, y1 = pts[i+1]
            Q = ( (1-w)*x0 + w*x1, (1-w)*y0 + w*y1 )
            R = ( w*x0 + (1-w)*x1, w*y0 + (1-w)*y1 )
            new_pts.extend([Q, R])
        new_pts.append(pts[-1])
        pts = new_pts
    return pts

# --------------- A* (road-aware, smoother) ---------------
def astar(cost_array, start, end, road_mask=None):
    rows, cols = cost_array.shape
    sr, sc = int(start[0]), int(start[1])
    er, ec = int(end[0]), int(end[1])
    if not (0 <= sr < rows and 0 <= sc < cols and 0 <= er < rows and 0 <= ec < cols):
        return None

    # strictly positive traversable; block huge/nodata
    arr = np.array(cost_array, dtype=np.float32, copy=True)
    arr = np.where(np.isfinite(arr), np.maximum(arr, EPS), np.inf)
    impassable = (arr >= NODATA_THRESHOLD) | ~np.isfinite(arr)

    # road guidance: use provided mask or infer from costs
    inferred_used = False
    if road_mask is None:
        road_mask = _load_or_infer_road_mask(arr)
        inferred_used = True
    else:
        road_mask = road_mask.astype(bool)

    # clean mask where impassable
    road_mask = road_mask & (~impassable)
    # snap endpoints if we have any roads
    if road_mask.any():
        sr, sc = _snap_to_road(sr, sc, road_mask)
        er, ec = _snap_to_road(er, ec, road_mask)

    if impassable[sr, sc] or impassable[er, ec]:
        return None

    # heuristic scale uses min traversable step
    finite = (~impassable)
    min_step = float(np.nanmin(arr[finite])) if np.any(finite) else 1.0
    if not np.isfinite(min_step) or min_step <= 0:
        min_step = 1.0

    # road proximity field for gentle "stickiness"
    near_road = None
    if STICK_TO_ROAD > 0 and road_mask.any():
        dt = _chamfer_dt_bool(road_mask)
        # normalize inside the traversable area
        near_road = 1.0 - _normalize01(dt, finite)  # 1 on road → 0 far away

    inf = np.float32(np.inf)
    g = np.full((rows, cols), inf, dtype=np.float32)
    f_closed = np.zeros((rows, cols), dtype=bool)
    parent_r = np.full((rows, cols), -1, dtype=np.int32)
    parent_c = np.full((rows, cols), -1, dtype=np.int32)

    g[sr, sc] = 0.0
    h0 = _octile(sr, sc, er, ec) * min_step
    open_heap = []
    # include tiny tie-break to prefer progress toward goal
    heapq.heappush(open_heap, (h0 + 1e-6 * h0, sr, sc))

    while open_heap:
        f, r, c = heapq.heappop(open_heap)
        if f_closed[r, c]:
            continue
        f_closed[r, c] = True

        if r == er and c == ec:
            path = _reconstruct(parent_r, parent_c, r, c)
            # simplify stair-steps before coordinate smoothing
            return _remove_colinear_steps(path)

        # previous move direction (for turn penalty)
        pr, pc = parent_r[r, c], parent_c[r, c]
        prev_dr = prev_dc = 0
        if pr != -1 and pc != -1:
            prev_dr = int(np.sign(r - pr))
            prev_dc = int(np.sign(c - pc))

        for nr, nc, dr, dc in _neighbors8(r, c, impassable):
            if f_closed[nr, nc]:
                continue

            # movement cost
            step = _movement_cost(arr, r, c, nr, nc)

            # gentle road stickiness
            if near_road is not None:
                # penalize being away from road; smaller penalty when already on road
                step += STICK_TO_ROAD * (1.0 - float(near_road[nr, nc]))

            # turn penalty (applied always; it's tiny)
            if prev_dr != 0 or prev_dc != 0:
                if (dr, dc) != (prev_dr, prev_dc):
                    step += TURN_PENALTY

            tentative_g = g[r, c] + step
            if tentative_g < g[nr, nc]:
                g[nr, nc] = tentative_g
                parent_r[nr, nc] = r
                parent_c[nr, nc] = c
                h = _octile(nr, nc, er, ec) * min_step
                # very small goal-directed tie-break (helps reduce meander)
                heapq.heappush(open_heap, (tentative_g + h + 1e-6 * h, nr, nc))

    return None


# In[60]:


def compute_path_with_Astar(write_shp=False, write_geojson=True):
    info = get_task_info()
    task_id = info["task_id"]

    start_pixel = (int(info["start_pixel"]["row"]), int(info["start_pixel"]["col"]))
    end_pixel   = (int(info["end_pixel"]["row"]),   int(info["end_pixel"]["col"]))

    cost_raster_path = Path(data_dir) / f"total_cost_{task_id}.tif"
    shp_path         = Path(data_dir) / f"route_{task_id}.shp"
    geojson_path     = Path(data_dir) / f"route_{task_id}.geojson"

    with rasterio.open(cost_raster_path) as src:
        cost_array = src.read(1).astype(np.float32, copy=False)
        transform = src.transform
        crs = src.crs  # may be None

    # optional: explicit road raster
    road_mask = None
    road_raster_path = Path(data_dir) / f"roads_{task_id}.tif"
    if road_raster_path.exists():
        with rasterio.open(road_raster_path) as rsrc:
            road_mask = (rsrc.read(1) != 0)

    print(f"Running A* pathfinding for Task ID: {task_id}")
    path_pixels = astar(cost_array, start_pixel, end_pixel, road_mask=road_mask)
    if not path_pixels:
        print("No path found.")
        return None

    # pixel -> source CRS coords
    coords_src = [xy(transform, r, c) for r, c in path_pixels]  # (x,y) in raster CRS

    # Optional smoothing (in source CRS)
    if SMOOTH_ITERS > 0:
        coords_src = _chaikin_smooth(coords_src, iterations=SMOOTH_ITERS, w=0.25)

    # Ensure GeoJSON is lon/lat (EPSG:4326)
    coords_ll = coords_src
    if crs is not None and crs.to_epsg() != 4326:
        try:
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            xs, ys = zip(*coords_src)
            lons, lats = transformer.transform(xs, ys)
            coords_ll = list(zip(lons, lats))
        except Exception as e:
            # fall back: write in source CRS (not ideal for web maps)
            print(f"CRS transform to EPSG:4326 failed: {e}; writing source CRS to GeoJSON.")

    # Build the LineString
    line_geom_ll = LineString(coords_ll)

    # --- Write GeoJSON (preferred for app.py) ---
    if write_geojson:
        feature = {
            "type": "Feature",
            "geometry": mapping(line_geom_ll),
            "properties": {
                "task_id": task_id
            }
        }
        fc = {"type": "FeatureCollection", "features": [feature]}
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False)
        print(f"GeoJSON saved to {geojson_path}")

    # --- (Optional) also write shapefile if you still want ---
    if write_shp:
        schema = {"geometry": "LineString", "properties": {"task_id": "str"}}
        with fiona.open(
            shp_path, "w",
            driver="ESRI Shapefile",
            crs=crs,
            schema=schema
        ) as shp:
            shp.write({
                "geometry": LineString(coords_src).__geo_interface__,
                "properties": {"task_id": str(task_id)}
            })
        print(f"Shapefile saved to {shp_path}")

    return str(geojson_path if write_geojson else shp_path)

if __name__ == "__main__":
    compute_path_with_Astar(write_shp=False, write_geojson=True)



#!/usr/bin/env python
# coding: utf-8
import os
import json
import math
from shapely.geometry import box
import osmnx as ox
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import numpy as np
import sys
from pathlib import Path

# --- Project root bootstrap ---
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


def load_task_info(json_path=None):
    if json_path is None:
        _, json_path = get_project_paths()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    task_id = data.get("task_id", "no_task_id")
    start_place = (data.get("start_point") or {}).get("place", "Unknown start")
    end_place   = (data.get("end_point") or {}).get("place", "Unknown end")

    aoi = data.get("aoi_bbox")
    if not aoi:
        raise ValueError("Missing 'aoi_bbox' in task info. Run the AOI/DEM step first.")
    for k in ("lon_min", "lat_min", "lon_max", "lat_max"):
        if k not in aoi:
            raise ValueError(f"Missing '{k}' in 'aoi_bbox'.")
    bounds = (float(aoi["lon_min"]), float(aoi["lat_min"]),
              float(aoi["lon_max"]), float(aoi["lat_max"]))
    return task_id, start_place, end_place, bounds


def get_bbox_from_json(bounds):
    minx, miny, maxx, maxy = bounds
    return box(minx, miny, maxx, maxy), bounds


def download_osm_buildings(polygon):
    """Download building footprints from OSM within the given polygon."""
    tags = {"building": True}
    gdf = ox.features.features_from_polygon(polygon, tags=tags)
    # Ensure valid geometries only
    if not gdf.empty:
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf = gdf[gdf.is_valid]
    return gdf


def rasterize_vector(gdf, bounds, resolution_m=10, out_path="output.tif"):
    """
    Rasterize geometries to a binary mask (1=building, 0=else) in EPSG:4326 grid,
    using approximate meters-per-degree factors at the bbox center latitude.
    """
    minx, miny, maxx, maxy = bounds
    lat_c = (miny + maxy) / 2.0

    # meters per degree (approx)
    meters_per_deg_lat = 111_132.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_c))

    dx_m = max((maxx - minx) * meters_per_deg_lon, 1e-6)
    dy_m = max((maxy - miny) * meters_per_deg_lat, 1e-6)

    width  = int(math.ceil(dx_m / resolution_m))
    height = int(math.ceil(dy_m / resolution_m))
    width = max(width, 1)
    height = max(height, 1)

    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    shapes = ((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty and geom.is_valid)

    raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=raster.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(raster, 1)


def download_buildings_main(resolution_m=10):
    # Step 1: Load task info + AOI bbox from JSON
    task_id, start_place, end_place, bbox_coords = load_task_info()

    # Step 2: Create bbox polygon from JSON (no geocoding)
    bbox_poly, _ = get_bbox_from_json(bbox_coords)

    # Step 3: Download building footprints
    gdf = download_osm_buildings(bbox_poly)
    if gdf.empty:
        print(f"⚠ No building data found for AOI covering: {start_place} → {end_place}")
        return

    # Step 4: Output path
    output_folder, _ = get_project_paths()
    out_path = os.path.join(output_folder, f"osm_buildings_{task_id}.tif")

    # Step 5: Rasterize (binary mask)
    rasterize_vector(gdf, bbox_coords, resolution_m=resolution_m, out_path=out_path)
    print(f" Building footprint raster saved as: {out_path} | AOI: {start_place} → {end_place}")


if __name__ == "__main__":
    download_buildings_main()

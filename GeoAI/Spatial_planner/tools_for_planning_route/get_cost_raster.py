
import os
import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import sys
from pathlib import Path

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

# ----------------- Paths -----------------
data_dir, task_info_path = get_project_paths()

def get_task_info():
    with open(task_info_path, 'r') as f:
        return json.load(f)

# ----------------- Raster Matching -----------------
def match_raster(reference_path, to_match_path, output_path):
    with rasterio.open(reference_path) as ref:
        ref_profile = ref.profile
        ref_data = ref.read(1)

    with rasterio.open(to_match_path) as src:
        matched = np.empty_like(ref_data)

        reproject(
            source=src.read(1),
            destination=matched,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            resampling=Resampling.nearest 
        )

        ref_profile.update(dtype=matched.dtype)

        with rasterio.open(output_path, "w", **ref_profile) as dst:
            dst.write(matched, 1)

# ----------------- Cost Raster -----------------
def generate_total_cost_raster_main(base_dir=data_dir):
    task_info = get_task_info()
    task_id = task_info["task_id"]
    cost_config = task_info["cost_config"]

    # File paths
    lulc_path = os.path.join(base_dir, f"lulc_cost_{task_id}.tif")
    slope_path = os.path.join(base_dir, f"slope_cost_{task_id}.tif")
    road_path = os.path.join(base_dir, f"osm_roads_{task_id}.tif")
    water_path = os.path.join(base_dir, f"osm_waterbody_{task_id}.tif")
    building_path = os.path.join(base_dir, f"osm_buildings_{task_id}.tif")

    # Matched output paths
    slope_matched = slope_path.replace(".tif", "_matched.tif")
    road_matched = road_path.replace(".tif", "_matched.tif")
    water_matched = water_path.replace(".tif", "_matched.tif")
    building_matched = building_path.replace(".tif", "_matched.tif")

    # Match all rasters to LULC
    match_raster(lulc_path, slope_path, slope_matched)
    match_raster(lulc_path, road_path, road_matched)
    match_raster(lulc_path, water_path, water_matched)
    match_raster(lulc_path, building_path, building_matched)

    # Read rasters
    with rasterio.open(lulc_path) as src:
        lulc = src.read(1)
        profile = src.profile

    slope = rasterio.open(slope_matched).read(1)
    road = rasterio.open(road_matched).read(1)
    water = rasterio.open(water_matched).read(1)
    building = rasterio.open(building_matched).read(1)

    # Extract weights & penalties
    lulc_w = cost_config["lulc"]["weight"]
    slope_w = cost_config["slope"]["weight"]
    road_w = cost_config["road"]["weight"]
    water_penalty_val = cost_config["water"]["penalty"]
    building_penalty_val = cost_config["building"]["penalty"]

    # Weighted base cost
    base_cost = (lulc_w * lulc) + (slope_w * slope)

    # Road bonus
    road_bonus = (road == 1) * road_w

    # Water penalty
    water_penalty = (water == 1) * water_penalty_val

    # Building penalty
    building_penalty = (building == 1) * building_penalty_val

    # Final cost
    total_cost = base_cost + road_bonus + water_penalty + building_penalty
    total_cost = total_cost.astype("float32")

    # Save output
    profile.update(dtype="float32", nodata=None)
    output_path = os.path.join(base_dir, f"total_cost_{task_id}.tif")

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(total_cost, 1)

    print(f" Final cost raster saved to {output_path}")


if __name__ == "__main__":
    generate_total_cost_raster_main()
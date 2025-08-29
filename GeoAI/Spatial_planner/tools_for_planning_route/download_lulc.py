import ee
import time
import json
from pathlib import Path
import os
import sys

try:
    BASE_DIR = Path(__file__).resolve().parent.parent  # tools_for_planning_route → Spatial_planner
except NameError:
    BASE_DIR = Path.cwd()

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from lib_planning_route.bootstrap_planning_route import add_project_root
add_project_root()

from lib_planning_route.utils_planning_route import get_project_paths


def download_lulc_main(
    ee_project_id: str = "ee-an23cem5r02akash",
    worldcover_asset: str = "ESA/WorldCover/v100/2020",  # keep consistent with your stack
    scale_m: int = 10
):
    # --- Utility ---
    def wait_for_task(task, poll_interval=10):
        desc = task.config.get("description", "unnamed")
        print(f" Waiting for task '{desc}' to finish...")
        while task.active():
            print("  Still running...")
            time.sleep(poll_interval)
        print(f" Task '{desc}' completed with state: {task.status().get('state')}")

    def read_aoi_from_json(task_info: dict):
        aoi = task_info.get("aoi_bbox")
        if not aoi:
            raise ValueError(
                "Missing 'aoi_bbox' in task info. Run the DEM/AOI step first to write the bounding box."
            )
        region_ring = aoi.get("gee_region")
        if not region_ring or not isinstance(region_ring, list) or len(region_ring) < 4:
            raise ValueError(
                "Invalid or missing 'aoi_bbox.gee_region' in task info. It must be a closed ring (list of lon/lat pairs)."
            )
        return region_ring

    # --- 1) Init Earth Engine ---
    ee.Initialize(project=ee_project_id)

    # --- 2) Get paths & read task info ---
    data_dir, task_info_path = get_project_paths()
    task_info_path = Path(task_info_path)
    with task_info_path.open("r", encoding="utf-8") as f:
        task_info = json.load(f)

    task_id = task_info.get("task_id", "no_task_id")

    # --- 3) STRICT: use AOI from JSON only ---
    region_ring = read_aoi_from_json(task_info)
    aoi_geom = ee.Geometry.Polygon(region_ring)

    # Optional pretty context for logs (doesn't recompute anything)
    start_place = (task_info.get("start_point") or {}).get("place", "Unknown start")
    end_place   = (task_info.get("end_point") or {}).get("place", "Unknown end")

    # --- 4) Prepare LULC image ---
    lulc = ee.Image(worldcover_asset)

    # --- 5) Export LULC using existing task_id and JSON AOI ---
    lulc_task = ee.batch.Export.image.toDrive(
        image=lulc,
        description=f"lulc_{task_id}",
        folder="EarthEngine",
        fileNamePrefix=f"lulc_{task_id}",
        region=aoi_geom,      # <- from JSON only
        scale=scale_m,
        maxPixels=1e13
    )
    lulc_task.start()

    print(f" Export started for LULC covering: {start_place} → {end_place} | Task ID: {task_id}")
    wait_for_task(lulc_task)

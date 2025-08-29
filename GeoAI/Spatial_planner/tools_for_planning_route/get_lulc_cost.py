#!/usr/bin/env python
# coding: utf-8


import rasterio
import numpy as np
import os
import json
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

LULC_COST_MAPPING = {
    10: 4,     # Tree Cover
    20: 4,     # Shrubland
    30: 3.5,      # Grassland
    40: 3.5,      # Cropland
    50: 5,     # Built-up
    60: 2.5,     # Bare / Sparse vegetation
    70: 2,     # Snow and Ice
    80: 10,   # Water
    90: 2.5,     # Mangroves
    95: 3.5,     # Wetlands
    100: 2.5      # Moss and Lichen
}

HIGH_COST = 9999

data_dir, task_info_path = get_project_paths()
def load_task_info():
    with open(task_info_path, "r") as f:
        return json.load(f)

def adjust_lulc_cost_mapping(task):
    """
    Adjusts the LULC cost mapping based on 'avoid' or 'prefer' instructions in task JSON.
    """
    mapping = LULC_COST_MAPPING.copy()

    # Normalize user instructions
    avoid_classes = [s.lower() for s in task.get("avoid_features", [])]
    prefer_classes = [s.lower() for s in task.get("prefer_features", [])]

    # Mapping from human-readable to LULC codes
    NAME_TO_CODE = {
        "tree cover": 10,
        "shrubland": 20,
        "grassland": 30,
        "cropland": 40,
        "built-up": 50,
        "bare": 60,
        "snow": 70,
        "water": 80,
        "mangroves": 90,
        "wetlands": 95,
        "moss": 100
    }

    for name, code in NAME_TO_CODE.items():
        if name in avoid_classes:
            mapping[code] = HIGH_COST
        elif name in prefer_classes:
            mapping[code] = max(1, mapping[code] * 0.5)  # Half the cost for preference

    return mapping


def reclassify_lulc_to_cost(lulc_array, cost_mapping):
    cost_array = np.full(lulc_array.shape, fill_value=65535, dtype=np.uint16)  # default = invalid
    for class_val, cost_val in cost_mapping.items():
        cost_array[lulc_array == class_val] = cost_val
    return cost_array

def process_lulc_raster(task_id, cost_mapping, base_dir=data_dir):
    input_path = os.path.join(base_dir, f"lulc_{task_id}.tif")
    output_path = os.path.join(base_dir, f"lulc_cost_{task_id}.tif")

    with rasterio.open(input_path) as src:
        lulc = src.read(1)
        profile = src.profile

    cost = reclassify_lulc_to_cost(lulc, cost_mapping)

    profile.update(
        dtype=rasterio.uint16,
        nodata=65535,
        count=1
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(cost.astype(np.uint16), 1)

    print(f"Reclassified cost raster saved to: {output_path}")

def get_lulc_cost_main():
    task = load_task_info()
    task_id = task["task_id"]
    cost_mapping = adjust_lulc_cost_mapping(task)
    process_lulc_raster(task_id, cost_mapping)

if __name__ == "__main__":
    get_lulc_cost_main()

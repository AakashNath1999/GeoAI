#!/usr/bin/env python
# coding: utf-8

# In[25]:


import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import numpy as np
import os
import json
import sys
from pathlib import Path


# In[26]:


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


# In[27]:


data_dir, task_info_path = get_project_paths()

def load_task_info():
    with open(task_info_path, "r") as f:
        return json.load(f)


# In[28]:


def resample_raster_10m(input_path, output_path, target_res_deg=0.00008983):
    with rasterio.open(input_path) as src:
        left, bottom, right, top = src.bounds
        new_width = int((right - left) / target_res_deg)
        new_height = int((top - bottom) / target_res_deg)

        if new_width <= 0 or new_height <= 0:
            raise ValueError(f"Invalid resampled dimensions: {new_width} x {new_height}")

        transform = from_origin(left, top, target_res_deg, target_res_deg)

        kwargs = src.meta.copy()
        kwargs.update({
            'transform': transform,
            'width': new_width,
            'height': new_height,
            'dtype': 'float32'
        })

        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear
        )

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            dst.write(data)

    print(f"Resampled raster written to {output_path}")


# In[29]:


def slope_to_cost(slope_array):
    cost = np.full_like(slope_array, fill_value=65535, dtype=np.uint16)  # default: steep
    cost[(slope_array >= 0) & (slope_array < 5)] = 1
    cost[(slope_array >= 5) & (slope_array < 10)] = 2
    cost[(slope_array >= 10) & (slope_array < 20)] = 5
    cost[(slope_array >= 20) & (slope_array < 30)] = 10
    cost[(slope_array >= 30) & (slope_array < 45)] = 20
    return cost


# In[30]:


def generate_slope_cost(task_id, base_dir=data_dir):
    resampled_path = os.path.join(base_dir, f"slope_10m_{task_id}.tif")
    cost_path = os.path.join(base_dir, f"slope_cost_{task_id}.tif")
    input_path = os.path.join(base_dir, f"slope_{task_id}.tif")

    resample_raster_10m(input_path, resampled_path)

    with rasterio.open(resampled_path) as src:
        slope = src.read(1)
        profile = src.profile

    slope_cost = slope_to_cost(slope)

    profile.update(dtype='uint16', nodata=65535)

    with rasterio.open(cost_path, "w", **profile) as dst:
        dst.write(slope_cost, 1)

    print(f" Slope cost raster saved to {cost_path}")


# In[31]:

# 🔹 Main callable for controller
def get_slope_cost_main(task_id=None):
    """Main function to be called from controller."""
    if task_id is None:
        task = load_task_info()
        task_id = task["task_id"]
    generate_slope_cost(task_id)

if __name__ == "__main__":
    get_slope_cost_main()





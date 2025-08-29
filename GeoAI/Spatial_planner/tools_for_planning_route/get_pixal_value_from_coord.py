
import json
import rasterio
from rasterio.transform import rowcol
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


def geocoords_to_pixel(transform, lon, lat):
    row, col = rowcol(transform, lon, lat)
    return int(row), int(col)


# In[28]:


def append_pixel_coords_to_task():

    data_dir, task_info_path = get_project_paths()
    # Load existing task info
    with open(task_info_path, "r") as f:
        task_info = json.load(f)

    task_id = task_info["task_id"]
    start_lat = task_info["start_point"]["lat"]
    start_lon = task_info["start_point"]["lon"]
    end_lat = task_info["end_point"]["lat"]
    end_lon = task_info["end_point"]["lon"]

    # Open cost raster to get transform
    cost_raster_path = data_dir/f"total_cost_{task_id}.tif"
    with rasterio.open(cost_raster_path) as src:
        transform = src.transform

    # Convert geographic coordinates to pixel indices
    start_row, start_col = geocoords_to_pixel(transform, start_lon, start_lat)
    end_row, end_col = geocoords_to_pixel(transform, end_lon, end_lat)

    # Append pixel indices as new keys
    task_info["start_pixel"] = {"row": start_row, "col": start_col}
    task_info["end_pixel"] = {"row": end_row, "col": end_col}

    # Write back to the same JSON file
    with open(task_info_path, "w") as f:
        json.dump(task_info, f, indent=4)

    return task_info 

if __name__ == "__main__":
    updated_data = append_pixel_coords_to_task()
    print("Updated task file with pixel values:", updated_data)






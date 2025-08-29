#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
from geopy.geocoders import Nominatim
import sys
from pathlib import Path


# In[2]:


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


# In[3]:


def get_coordinates(place_name):
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(place_name)
    if not location:
        raise ValueError(f"Location not found: {place_name}")
    return {"lat": location.latitude, "lon": location.longitude}


# In[4]:


def update_task_file_with_coordinates():
    """
    Reads place names from task JSON and adds lat/lon coordinates.
    Returns updated task_info dict.
    """
    _, task_file_path = get_project_paths()

    if not os.path.exists(task_file_path):
        raise FileNotFoundError(f"Task file not found: {task_file_path}")

    with open(task_file_path, "r") as f:
        data = json.load(f)

    start_place = data.get("start_point", {}).get("place")
    end_place = data.get("end_point", {}).get("place")

    if not start_place or not end_place:
        raise ValueError("Start or end place is missing from the JSON file.")

    data["start_point"].update(get_coordinates(start_place))
    data["end_point"].update(get_coordinates(end_place))

    with open(task_file_path, "w") as f:
        json.dump(data, f, indent=2)

    return data  


if __name__ == "__main__":
    updated_data = update_task_file_with_coordinates()
    print("Updated task file with coordinates:", updated_data)


# In[ ]:





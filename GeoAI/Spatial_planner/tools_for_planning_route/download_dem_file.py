import ee
import time
import json
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import math
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


def download_dem_main(
    ee_project_id: str = "ee-an23cem5r02akash",
    default_pad_km: float = 3.0,   # padding around the bbox that encloses start & end
    fallback_bbox_km: float = 20.0  # if only one place available, use this square bbox
):
    # --- Helpers ---
    def wait_for_task(task, poll_interval=10):
        desc = task.config.get("description", "unnamed")
        print(f" Waiting for task '{desc}' to finish...")
        while task.active():
            print("  Still running...")
            time.sleep(poll_interval)
        status = task.status()
        print(f" Task '{desc}' completed with state: {status.get('state')}")

    def km_to_deg_lat(km):
        return km / 111.0  # ~111 km per degree latitude

    def km_to_deg_lon(km, lat_deg):
        return km / (111.0 * math.cos(math.radians(lat_deg)))

    def geocode_place(geocoder, place: str):
        if not place or not str(place).strip():
            return None
        try:
            loc = geocoder(place)
            if loc:
                return {"lat": float(loc.latitude), "lon": float(loc.longitude)}
        except Exception as e:
            print(f"  Geocoding error for '{place}': {e}")
        return None

    def ensure_point_dict(d):
        return d if isinstance(d, dict) else {}

    def update_json_atomic(path: Path, obj: dict):
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        tmp.replace(path)

    # --- 1) Init Earth Engine ---
    ee.Initialize(project=ee_project_id)

    # --- 2) Paths & load current task info ---
    data_dir, task_info_path = get_project_paths()
    task_info_path = Path(task_info_path)
    with task_info_path.open("r", encoding="utf-8") as f:
        task_info = json.load(f)

    task_id = task_info.get("task_id", "no_task_id")

    # Unified shape:
    #   "start_point": {"place": "...", "lat": ..., "lon": ...}
    #   "end_point":   {"place": "...", "lat": ..., "lon": ...}
    start_point = ensure_point_dict(task_info.get("start_point"))
    end_point   = ensure_point_dict(task_info.get("end_point"))

    # Back-compat names if present
    start_place = start_point.get("place") or task_info.get("start_point_place")
    end_place   = end_point.get("place")   or task_info.get("end_point_place")

    # --- 3) Geocode missing lat/lon and write back ---
    geolocator = Nominatim(user_agent="geoai_route_planner")
    safe_geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, swallow_exceptions=True)

    changed = False

    if start_place:
        if "lat" not in start_point or "lon" not in start_point:
            res = geocode_place(safe_geocode, start_place)
            if res:
                start_point["place"] = start_place
                start_point["lat"] = res["lat"]
                start_point["lon"] = res["lon"]
                task_info["start_point"] = start_point
                changed = True
            else:
                raise ValueError(f"Could not geocode start point place: '{start_place}'")
    else:
        print(" Warning: start_point.place missing")

    if end_place:
        if "lat" not in end_point or "lon" not in end_point:
            res = geocode_place(safe_geocode, end_place)
            if res:
                end_point["place"] = end_place
                end_point["lat"] = res["lat"]
                end_point["lon"] = res["lon"]
                task_info["end_point"] = end_point
                changed = True
            else:
                raise ValueError(f"Could not geocode end point place: '{end_place}'")
    else:
        print(" Warning: end_point.place missing")

    # Persist lat/lon updates early
    if changed:
        update_json_atomic(task_info_path, task_info)
        print(" Updated start/end lat/lon in task info JSON.")

    # --- 4) Build AOI covering both points (or fallback) ---
    have_start = "lat" in start_point and "lon" in start_point
    have_end   = "lat" in end_point and "lon" in end_point

    bbox_meta = {"crs": "EPSG:4326"}

    if have_start and have_end:
        lat_min = min(start_point["lat"], end_point["lat"])
        lat_max = max(start_point["lat"], end_point["lat"])
        lon_min = min(start_point["lon"], end_point["lon"])
        lon_max = max(start_point["lon"], end_point["lon"])

        center_lat = (lat_min + lat_max) / 2.0
        dlat = km_to_deg_lat(default_pad_km)
        dlon = km_to_deg_lon(default_pad_km, center_lat)

        lon_min_p, lon_max_p = lon_min - dlon, lon_max + dlon
        lat_min_p, lat_max_p = lat_min - dlat, lat_max + dlat

        method = "start_end_with_padding"
        pad_km = default_pad_km

    elif have_start or have_end:
        p = start_point if have_start else end_point
        lat = p["lat"]
        lon = p["lon"]
        dlat = km_to_deg_lat(fallback_bbox_km / 2.0)
        dlon = km_to_deg_lon(fallback_bbox_km / 2.0, lat)

        lon_min_p, lon_max_p = lon - dlon, lon + dlon
        lat_min_p, lat_max_p = lat - dlat, lat + dlat

        method = "single_point_fallback"
        pad_km = fallback_bbox_km / 2.0

    else:
        place_name = task_info.get("place_name")
        if not place_name:
            raise ValueError("No geocoded points or place_name available to build AOI.")
        res = geolocator.geocode(place_name)
        if not res:
            raise ValueError(f"Could not geocode place_name: '{place_name}'")
        lat = float(res.latitude)
        lon = float(res.longitude)
        dlat = km_to_deg_lat(fallback_bbox_km / 2.0)
        dlon = km_to_deg_lon(fallback_bbox_km / 2.0, lat)

        lon_min_p, lon_max_p = lon - dlon, lon + dlon
        lat_min_p, lat_max_p = lat - dlat, lat + dlat

        method = "place_name_fallback"
        pad_km = fallback_bbox_km / 2.0

    # Region ring (closed) for EE & GeoJSON polygon
    region_ring = [
        [lon_min_p, lat_min_p],
        [lon_min_p, lat_max_p],
        [lon_max_p, lat_max_p],
        [lon_max_p, lat_min_p],
        [lon_min_p, lat_min_p],
    ]

    # Save AOI info into JSON so later tools can reuse it directly
    aoi_bbox = {
        "type": "bbox",
        "crs": "EPSG:4326",
        "method": method,
        "padding_km": pad_km,
        "lon_min": lon_min_p,
        "lat_min": lat_min_p,
        "lon_max": lon_max_p,
        "lat_max": lat_max_p,
        # Ready-to-use in Earth Engine exports:
        "gee_region": region_ring,
        # Ready-to-use in typical Geo tooling:
        "geojson": {
            "type": "Polygon",
            "coordinates": [region_ring]
        }
    }
    task_info["aoi_bbox"] = aoi_bbox
    update_json_atomic(task_info_path, task_info)
    print(" Wrote aoi_bbox to task info JSON.")

    # --- 5) Prepare DEM (SRTM) and export using the saved region ---
    ee_region = aoi_bbox["gee_region"]
    dem = ee.Image("CGIAR/SRTM90_V4")

    dem_task = ee.batch.Export.image.toDrive(
        image=dem,
        description=f"dem_{task_id}",
        folder="EarthEngine",
        fileNamePrefix=f"dem_{task_id}",
        region=ee_region,  # reuse the exact ring we saved
        scale=30,
        maxPixels=1e13
    )
    dem_task.start()

    pretty_from = start_point.get("place") or "Unknown start"
    pretty_to   = end_point.get("place") or "Unknown end"
    print(f" Export started for DEM covering: {pretty_from} → {pretty_to}  | Task ID: {task_id}")

    wait_for_task(dem_task)

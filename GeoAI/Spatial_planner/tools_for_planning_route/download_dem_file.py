import ee
import time
import json
import math
from pathlib import Path
import sys
from math import radians, sin, cos, asin, sqrt

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# --------------------------------------------------------------------
# Project bootstrap
# --------------------------------------------------------------------
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
    fallback_bbox_km: float = 20.0 # if only one place available, use this square bbox
):
    # ---------------- Helpers ----------------
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

    def ensure_point_dict(d):
        return d if isinstance(d, dict) else {}

    def update_json_atomic(path: Path, obj: dict):
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        tmp.replace(path)

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat_r = radians(lat2 - lat1)
        dlon_r = radians(lon2 - lon1)
        a = sin(dlat_r/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon_r/2)**2
        return 2 * R * asin(sqrt(a))

    # ---------------- 1) Init Earth Engine ----------------
    ee.Initialize(project=ee_project_id)

    # ---------------- 2) Load current task info ----------------
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

    # ---------------- 3) Geocode start/end using LIVE UI bounds ----------------
    geolocator = Nominatim(user_agent="geoai_route_planner")
    safe_geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, swallow_exceptions=True)

    ui_ctx = task_info.get("ui_context") or {}
    bounds = ui_ctx.get("bounds") or {}
    center = ui_ctx.get("center") or {}

    if not (bounds and all(k in bounds for k in ("south", "west", "north", "east")) and center and "lat" in center and "lon" in center):
        raise ValueError("ui_context.bounds/center missing; cannot do map-biased geocoding.")

    south = float(bounds["south"]); west  = float(bounds["west"])
    north = float(bounds["north"]); east  = float(bounds["east"])
    map_lat = float(center["lat"]); map_lon = float(center["lon"])

    # Build viewboxes in all formats geopy/Nominatim accept (for cross-version compatibility)
    # Pair-of-pairs must be ((lat1, lon1), (lat2, lon2)) = ((south, west), (north, east))
    viewbox_pairs_latlon = ((south, west), (north, east))
    # String must be "left,top,right,bottom" = west,north,east,south
    viewbox_string_lt_rb = f"{west},{north},{east},{south}"

    print(f"  UI bounds received: S={south:.6f} W={west:.6f} N={north:.6f} E={east:.6f}")
    print(f"  Viewbox (pairs, latlon): {viewbox_pairs_latlon}")
    print(f"  Viewbox (string l,t,r,b): {viewbox_string_lt_rb}")
    print(f"  UI center: {map_lat:.6f}, {map_lon:.6f}")

    def geocode_near_view(name: str, limit: int = 10, country_codes: str | None = "in"):
        """
        Try in this order:
          1) bounded=True with pair-of-pairs viewbox
          2) bounded=True with string viewbox
          3) bounded=False (bias only) with pair-of-pairs, then string; filter to ≤20 km of UI center
          4) no viewbox, country-only (exactly_one=False)  ➜ pick nearest to center
          5) global (no viewbox, no country filter)       ➜ pick nearest to center
        Returns {lat, lon, raw} or None.
        """
        if not name or not str(name).strip():
            return None

        common = dict(exactly_one=False, limit=limit, addressdetails=True, extratags=True)
        if country_codes:
            common["country_codes"] = country_codes

        # A) strict inside current map view — pairs
        results = []
        try:
            results = safe_geocode(name, viewbox=viewbox_pairs_latlon, bounded=True, **common) or []
            if results:
                print(f"   geocode(bounded=True, pairs) OK: '{name}' -> {len(results)} candidate(s)")
        except Exception as e:
            print(f"   geocode(bounded=True, pairs) '{name}' -> {e}")

        # A2) strict inside — string
        if not results:
            try:
                results = safe_geocode(name, viewbox=viewbox_string_lt_rb, bounded=True, **common) or []
                if results:
                    print(f"   geocode(bounded=True, string) OK: '{name}' -> {len(results)} candidate(s)")
            except Exception as e:
                print(f"   geocode(bounded=True, string) '{name}' -> {e}")

        # B) bias-only + distance filter (≤20 km)
        if not results:
            r = []
            try:
                r = safe_geocode(name, viewbox=viewbox_pairs_latlon, bounded=False, **common) or []
            except Exception as e:
                print(f"   geocode(bounded=False, pairs) '{name}' -> {e}")
            if not r:
                try:
                    r = safe_geocode(name, viewbox=viewbox_string_lt_rb, bounded=False, **common) or []
                except Exception as e:
                    print(f"   geocode(bounded=False, string) '{name}' -> {e}")
            results = [
                x for x in r
                if hasattr(x, "latitude") and hasattr(x, "longitude")
                and haversine_km(map_lat, map_lon, float(x.latitude), float(x.longitude)) <= 20.0
            ]
            if results:
                print(f"   geocode(bounded=False, filtered≤20km) OK: '{name}' -> {len(results)} candidate(s)")

        # C) country-only, no viewbox
        if not results:
            try:
                c_common = dict(common)
                if country_codes:
                    c_common["country_codes"] = country_codes
                results = safe_geocode(name, **c_common) or []
                if results:
                    print(f"   geocode(country-only) OK: '{name}' -> {len(results)} candidate(s)")
            except Exception as e:
                print(f"   geocode(country-only) '{name}' -> {e}")
                results = []

        # D) global, no viewbox, no country filter
        if not results:
            try:
                g_common = dict(common)
                g_common.pop("country_codes", None)
                results = safe_geocode(name, **g_common) or []
                if results:
                    print(f"   geocode(global) OK: '{name}' -> {len(results)} candidate(s)")
            except Exception as e:
                print(f"   geocode(global) '{name}' -> {e}")
                results = []

        if not results:
            print(f"  No results for '{name}' even after fallbacks; skipping.")
            return None

        # Pick the nearest candidate to UI center if possible
        try:
            best = min(
                results,
                key=lambda rr: haversine_km(map_lat, map_lon, float(rr.latitude), float(rr.longitude))
            )
        except Exception:
            # If a result lacks lat/lon, just take the first
            best = results[0]

        return {"lat": float(best.latitude), "lon": float(best.longitude), "raw": getattr(best, "raw", {})}

    changed = False

    # Only fill coords if missing (do NOT overwrite existing lat/lon)
    if start_place and ("lat" not in start_point or "lon" not in start_point):
        res = geocode_near_view(start_place) or (geocode_near_view(start_place.replace(" ", "")) if " " in start_place else None)
        if not res:
            raise ValueError(f"No '{start_place}' in/near current map view.")
        start_point["place"] = start_place
        start_point["lat"] = res["lat"]
        start_point["lon"] = res["lon"]
        task_info["start_point"] = start_point
        changed = True
        print(f"  Start geocoded: {start_place} → {res['lat']:.6f},{res['lon']:.6f}")

    if end_place and ("lat" not in end_point or "lon" not in end_point):
        res = geocode_near_view(end_place) or (geocode_near_view(end_place.replace(" ", "")) if " " in end_place else None)
        if not res:
            raise ValueError(f"No '{end_place}' in/near current map view.")
        end_point["place"] = end_place
        end_point["lat"] = res["lat"]
        end_point["lon"] = res["lon"]
        task_info["end_point"] = end_point
        changed = True
        print(f"  End geocoded: {end_place} → {res['lat']:.6f},{res['lon']:.6f}")

    if changed:
        update_json_atomic(task_info_path, task_info)
        print(" Updated start/end lat/lon in task info JSON.")

    # ---------------- 4) Build AOI ----------------
    have_start = "lat" in start_point and "lon" in start_point
    have_end   = "lat" in end_point and "lon" in end_point

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
        lat = p["lat"]; lon = p["lon"]
        dlat = km_to_deg_lat(fallback_bbox_km / 2.0)
        dlon = km_to_deg_lon(fallback_bbox_km / 2.0, lat)
        lon_min_p, lon_max_p = lon - dlon, lon + dlon
        lat_min_p, lat_max_p = lat - dlat, lat + dlat
        method = "single_point_fallback"
        pad_km = fallback_bbox_km / 2.0
    else:
        # LAST resort: if nothing specified, geocode the broader place_name (not center-biased)
        place_name = task_info.get("place_name")
        if not place_name:
            raise ValueError("No geocoded points or place_name available to build AOI.")
        res = geolocator.geocode(place_name)
        if not res:
            raise ValueError(f"Could not geocode place_name: '{place_name}'")
        lat = float(res.latitude); lon = float(res.longitude)
        dlat = km_to_deg_lat(fallback_bbox_km / 2.0)
        dlon = km_to_deg_lon(fallback_bbox_km / 2.0, lat)
        lon_min_p, lon_max_p = lon - dlon, lon + dlon
        lat_min_p, lat_max_p = lat - dlat, lat + dlat
        method = "place_name_fallback"
        pad_km = fallback_bbox_km / 2.0

    region_ring = [
        [lon_min_p, lat_min_p],
        [lon_min_p, lat_max_p],
        [lon_max_p, lat_max_p],
        [lon_max_p, lat_min_p],
        [lon_min_p, lat_min_p],
    ]

    aoi_bbox = {
        "type": "bbox",
        "crs": "EPSG:4326",
        "method": method,
        "padding_km": pad_km,
        "lon_min": lon_min_p,
        "lat_min": lat_min_p,
        "lon_max": lon_max_p,
        "lat_max": lat_max_p,
        "gee_region": region_ring,
        "geojson": {"type": "Polygon", "coordinates": [region_ring]},
    }
    task_info["aoi_bbox"] = aoi_bbox
    update_json_atomic(task_info_path, task_info)
    print(" Wrote aoi_bbox to task info JSON.")

    # ---------------- 5) Export DEM ----------------
    ee_region = aoi_bbox["gee_region"]
    dem = ee.Image("CGIAR/SRTM90_V4")

    dem_task = ee.batch.Export.image.toDrive(
        image=dem,
        description=f"dem_{task_id}",
        folder="EarthEngine",
        fileNamePrefix=f"dem_{task_id}",
        region=ee_region,
        scale=30,
        maxPixels=1e13
    )
    dem_task.start()

    pretty_from = start_point.get("place") or "Unknown start"
    pretty_to   = end_point.get("place") or "Unknown end"
    print(f" Export started for DEM covering: {pretty_from} → {pretty_to}  | Task ID: {task_id}")

    wait_for_task(dem_task)

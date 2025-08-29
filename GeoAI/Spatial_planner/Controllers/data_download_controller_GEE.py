#!/usr/bin/env python
# coding: utf-8
import os
import sys
import time
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# PATH/BOOTSTRAP SECTION
# Ensures the package imports work regardless of how this script is executed
# (as a module, from a notebook, or directly via python script.py).
try:
    # If running as a file inside the package, resolve two levels up-
    #   <...>/Spatial_planner/lib_planning_route/this_file.py
    # → BASE_DIR = <...>/Spatial_planner
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback for environments (e.g., notebooks) where __file__ is undefined
    BASE_DIR = Path.cwd().parent

if str(BASE_DIR) not in sys.path:
    # Make sure BASE_DIR is on sys.path so internal imports work
    sys.path.insert(0, str(BASE_DIR))

# Project bootstrap (adds project root to sys.path, sets env, etc.)
from lib_planning_route.bootstrap_planning_route import add_project_root
add_project_root()

# Small utility that returns canonical project paths (download dir, task_info path)
from lib_planning_route.utils_planning_route import get_project_paths

# TOOL IMPORTS
# Keep these thin wrapper functions in their own modules so this controller remains orchestration-only and easy to test.
from tools_for_planning_route.download_lulc import download_lulc_main
from tools_for_planning_route.download_slope import download_slope_main
from tools_for_planning_route.download_dem_file import download_dem_main

from tools_for_planning_route.retrieve_dem_from_drive_file import retrieve_dem_from_drive_main
from tools_for_planning_route.retrieve_lulc_from_drive import retrieve_lulc_from_drive_main
from tools_for_planning_route.retrieve_slope_from_drive import retrieve_slope_from_drive_main


class DataDownloadControllerGEE:
    """
    Orchestrates the end-to-end flow of:
      1) Triggering Earth Engine (GEE) exports for DEM, LULC, Slope
      2) Retrieving exported files from Google Drive to local storage
      3) Waiting robustly until files are fully present and stable
      4) Returning standardized paths/metadata for downstream steps

    Notes:
    - Export time is driven by EE servers; we poll local filesystem as the
      Drive → local copy completes.
    - Customize MAX WAIT via env var: GEE_MAX_WAIT_SECS (default 1800 = 30 min).
    """

    def __init__(self):
        # Resolve project-wide download dir and task info JSON file.
        self.download_dir, self.task_info_path = get_project_paths()
        Path(self.download_dir).mkdir(parents=True, exist_ok=True)
        self._init_task()

        # Standardize expected filenames: these must match your retrieve_* tools.
        self.dem_file = Path(self.download_dir) / f"dem_{self.task_id}.tif"
        self.lulc_file = Path(self.download_dir) / f"lulc_{self.task_id}.tif"
        self.slope_file = Path(self.download_dir) / f"slope_{self.task_id}.tif"

    def _init_task(self):
        """
        Create or load a persistent task record in JSON so all tools share the same task_id.
        If there's no task_id, generate a new UUID and persist it.
        """
        import uuid

        if not self.task_info_path.exists():
            task_info = {
                "task_id": str(uuid.uuid4()),
                "place_name": "Hyderabad",  # default
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(self.task_info_path, "w", encoding="utf-8") as f:
                json.dump(task_info, f, indent=2, ensure_ascii=False)
        else:
            # Load existing task record (created earlier in the pipeline).
            with open(self.task_info_path, "r", encoding="utf-8") as f:
                task_info = json.load(f)
            # Backfill task_id if missing (robustness for partially written files).
            if not task_info.get("task_id"):
                task_info["task_id"] = str(uuid.uuid4())
                with open(self.task_info_path, "w", encoding="utf-8") as f:
                    json.dump(task_info, f, indent=2, ensure_ascii=False)

        # Cache fields for easy access throughout the controller.
        self.task_id = task_info["task_id"]
        self.place_name = task_info["place_name"]

    def _wait_for_file(self, filepath: Path, timeout=None, quiet=False):
        """
        Block until `filepath` exists and its size remains unchanged for 10 seconds.

        Args:
            filepath: Path to the expected output file (e.g., dem_<task_id>.tif).
            timeout: Max seconds to wait. Default: 1800 (30 min) or env GEE_MAX_WAIT_SECS.
            quiet: If True, suppress heartbeat logs.

        Raises:
            TimeoutError: If the file doesn't appear or stabilize within the timeout.

        Why "stability"?
            Some copy/export flows create the file early and then stream bytes into it.
            Waiting until size stops changing helps avoid reading partial/corrupt data.
        """
        if timeout is None:
            timeout = int(os.environ.get("GEE_MAX_WAIT_SECS", "1800"))  # 30 min default

        start = time.time()
        last_size = -1
        stable_since = None

        while True:
            if filepath.exists():
                try:
                    size = filepath.stat().st_size
                except FileNotFoundError:
                    size = -1

                if size == last_size and size > 0:
                    if stable_since is None:
                        stable_since = time.time()
                    if time.time() - stable_since >= 10:
                        if not quiet:
                            print(f"File ready and stable: {filepath} ({size} bytes)")
                        return
                else:
                    stable_since = None
                    last_size = size

            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(f"Timeout after {timeout}s: File not ready - {filepath}")

            # heartbeat every ~15s
            if int(elapsed) % 15 == 0 and not quiet:
                eta = max(0, timeout - int(elapsed))
                print(f"… waiting for {filepath.name} (elapsed {int(elapsed)}s, ETA {eta}s)")

            time.sleep(2)

    def download_dem(self):
        """Start DEM export from GEE (asynchronous on the server side)."""
        print("  Starting DEM export from GEE…")
        download_dem_main()

    def download_lulc(self):
        print("  Starting LULC export from GEE…")
        download_lulc_main()

    def download_slope(self):
        print("  Starting SLOPE export from GEE…")
        download_slope_main()

    def start_all_exports(self):
        """
        Fire off all three exports immediately. GEE will run them concurrently
        up to your account's concurrency quota; excess tasks will queue.
        """
        print("  Kicking off concurrent GEE exports: DEM, LULC, Slope…")
        self.download_dem()
        self.download_lulc()
        self.download_slope()

    # Retrieve Methods (from Drive → local)
    def retrieve_dem(self):
        """
        Retrieve DEM from Drive to local storage, then wait until it's fully written.
        Assumes retrieve_dem_from_drive_main() writes to self.dem_file.
        """
        print("  Retrieving DEM from Drive…")
        retrieve_dem_from_drive_main()
        self._wait_for_file(self.dem_file)

    def retrieve_lulc(self):
        print("  Retrieving LULC from Drive…")
        retrieve_lulc_from_drive_main()
        self._wait_for_file(self.lulc_file)

    def retrieve_slope(self):
        print("  Retrieving SLOPE from Drive…")
        retrieve_slope_from_drive_main()
        self._wait_for_file(self.slope_file)

    def retrieve_all(self, parallel: bool = False):
        """
        Retrieve all three artifacts from Drive → local.

        Args:
            parallel: If True, run the three retrieve+wait steps in parallel threads.
                      Default False (safer if your retrieve_* functions query Drive
                      in a way that prefers sequential access).
        """
        if not parallel:
            # Safe, simple sequential retrieval
            self.retrieve_dem()
            self.retrieve_lulc()
            self.retrieve_slope()
            return

        # Optional: parallelize retrieval + wait
        print("  Retrieving all in parallel…")
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(self.retrieve_dem): "DEM",
                pool.submit(self.retrieve_lulc): "LULC",
                pool.submit(self.retrieve_slope): "SLOPE",
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    fut.result()
                    print(f"  {name} retrieval complete.")
                except Exception as e:
                    # Surface which layer failed; let the exception bubble up.
                    raise RuntimeError(f"{name} retrieval failed") from e

    def download_and_retrieve_all(self, parallel_retrieval: bool = False):
        """
        New concurrent pattern:
          1) Start all three GEE exports immediately (concurrent on server side)
          2) Retrieve each output and wait until stable (optionally in parallel)
        """
        self.start_all_exports()
        self.retrieve_all(parallel=parallel_retrieval)

    def run(self, **kwargs):
        """
        Main task execution method (agent entrypoint).
        Downloads (exports concurrently) and retrieves DEM, LULC, and Slope files.
        Returns dict with file paths & metadata.
        """
        print("  DataDownloadControllerGEE.run()")
        self.download_and_retrieve_all(parallel_retrieval=False)

        result = {
            "task_id": self.task_id,
            "place_name": self.place_name,
            "dem_raster": str(self.dem_file),
            "lulc_raster": str(self.lulc_file),
            "slope_raster": str(self.slope_file),
        }

        print("  DataDownloadControllerGEE finished:", result)
        return result


if __name__ == "__main__":
    controller = DataDownloadControllerGEE()
    output = controller.run()
    print("Final Output:", output)

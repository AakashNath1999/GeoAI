#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
from pathlib import Path
import json

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from lib_planning_route.bootstrap_planning_route import add_project_root
add_project_root()

from lib_planning_route.utils_planning_route import get_project_paths

# Tool entry points
from tools_for_planning_route.get_lulc_cost import get_lulc_cost_main
from tools_for_planning_route.get_slope_cost import get_slope_cost_main
from tools_for_planning_route.get_cost_raster import generate_total_cost_raster_main


class CostSurfaceController:
    def __init__(self, base_dir=None):
        """
        Reads existing task_id, sets expected output paths,
        and prepares to run the cost-surface pipeline.
        """
        # Resolve project paths
        self.download_dir, self.task_info_path = get_project_paths()
        self.download_dir = Path(self.download_dir)

        # Read task info without modifying it
        if not self.task_info_path.exists():
            raise FileNotFoundError(
                f"Task info not found: {self.task_info_path}. "
                "Run the GEE controller first so it creates task metadata."
            )
        with open(self.task_info_path, "r", encoding="utf-8") as f:
            task_info = json.load(f)

        self.task_id = task_info.get("task_id")
        if not self.task_id:
            raise ValueError(
                f"No task_id present in {self.task_info_path}. "
                "Ensure the GEE step populated it before building cost surfaces."
            )

        # Define expected output file paths
        self.lulc_cost_file  = self.download_dir / f"lulc_cost_{self.task_id}.tif"
        self.slope_cost_file = self.download_dir / f"slope_cost_{self.task_id}.tif"
        self.total_cost_file = self.download_dir / f"total_cost_{self.task_id}.tif"

    # helper to wait for files to be written & stable
    def _wait_for_file(self, filepath: Path, timeout=300, quiet=False):
        start = time.time()
        last_size, stable_since = -1, None
        while True:
            if filepath.exists():
                try:
                    size = filepath.stat().st_size
                except FileNotFoundError:
                    size = -1
                if size == last_size and size > 0:
                    if stable_since is None:
                        stable_since = time.time()
                    if time.time() - stable_since >= 4:
                        if not quiet:
                            print(f"File ready: {filepath} ({size} bytes)")
                        return
                else:
                    stable_since = None
                    last_size = size
            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout after {timeout}s waiting for {filepath}")
            time.sleep(0.5)

    # Steps
    def prepare_lulc_cost(self):
        print("\n Generating LULC cost raster...")
        # If get_lulc_cost_main returns a path, capture it; else use expected path
        out = get_lulc_cost_main()
        if isinstance(out, (str, Path)):
            self.lulc_cost_file = Path(out)
        # Ensure file exists (use expected path if function didn't return)
        self._wait_for_file(Path(self.lulc_cost_file), quiet=True)
        print("LULC cost raster:", self.lulc_cost_file)

    def prepare_slope_cost(self):
        print("\n Generating slope cost raster...")
        out = get_slope_cost_main()
        if isinstance(out, (str, Path)):
            self.slope_cost_file = Path(out)
        self._wait_for_file(Path(self.slope_cost_file), quiet=True)
        print("Slope cost raster:", self.slope_cost_file)

    def combine_costs(self):
        print("\n Combining LULC, slope, roads, buildings and waterbodies into final cost raster...")
        out = generate_total_cost_raster_main()
        if isinstance(out, (str, Path)):
            self.total_cost_file = Path(out)
        self._wait_for_file(Path(self.total_cost_file), quiet=True)
        print("Final cost (total) raster:", self.total_cost_file)

    def run_full_pipeline(self):
        print("\n Starting Cost Surface Preparation Pipeline...")
        self.prepare_lulc_cost()
        self.prepare_slope_cost()
        self.combine_costs()
        print("\n Cost surface preparation complete.")

    def run(self, **kwargs):
        """
        Entry point. Runs the full pipeline and returns paths.
        Also returns 'cost_surface' alias so the dispatcher can consume it.
        """
        print("Running CostSurfaceController...")
        self.run_full_pipeline()

        result = {
            "lulc_cost_raster":  str(self.lulc_cost_file),
            "slope_cost_raster": str(self.slope_cost_file),
            "total_cost_raster": str(self.total_cost_file),
            # alias for the dispatcher node that expects 'cost_surface'
            "cost_surface":      str(self.total_cost_file),
        }

        print("CostSurfaceController finished:", result)
        return result


if __name__ == "__main__":
    controller = CostSurfaceController()
    output = controller.run()
    print("Final Output:", output)

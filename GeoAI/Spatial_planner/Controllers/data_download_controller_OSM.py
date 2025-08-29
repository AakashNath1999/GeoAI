import sys
import time
import json
from pathlib import Path

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from lib_planning_route.bootstrap_planning_route import add_project_root
add_project_root()

from lib_planning_route.utils_planning_route import get_project_paths

# Import OSM download main functions
from tools_for_planning_route.download_OSM_raster_for_highways import download_highways_main as download_highways_main
from tools_for_planning_route.download_OSM_raster_for_waterbody import download_waterbody_main as download_waterbody_main
from tools_for_planning_route.download_OSM_raster_for_building import download_buildings_main as download_buildings_main


class DataDownloadControllerOSM:
    """
    Downloads & rasterizes OSM layers (roads, waterbodies, buildings).
    - DOES NOT create/modify task_id. It only reads the existing task info.
    - Exposes .run() returning dispatcher-compatible keys:
        road_raster, waterbody_raster, building_raster
    """
    def __init__(self):
        self.download_dir, self.task_info_path = get_project_paths()
        Path(self.download_dir).mkdir(parents=True, exist_ok=True)

        self._read_task_info()

        self.highways_file   = Path(self.download_dir) / f"highways_{self.task_id}.tif"
        self.waterbody_file  = Path(self.download_dir) / f"waterbody_{self.task_id}.tif"
        self.buildings_file  = Path(self.download_dir) / f"buildings_{self.task_id}.tif"

    def _read_task_info(self):
        if not self.task_info_path.exists():
            raise FileNotFoundError(
                f"Task info file not found: {self.task_info_path}. "
                "GEE controller should have created this. Run GEE step first."
            )
        with open(self.task_info_path, "r", encoding="utf-8") as f:
            task_info = json.load(f)

        task_id = task_info.get("task_id")
        if not task_id:
            raise ValueError(
                f"No task_id present in {self.task_info_path}. "
                "Please ensure GEE step populated it before OSM download."
            )

        self.task_id = task_id
        self.place_name = task_info.get("place_name", "Unknown")

    # small wait to ensure files are written
    def _wait_for_file(self, filepath: Path, timeout=300, quiet=False):
        """
        Wait until `filepath` exists and is stable (size unchanged for ~4s).
        """
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
                    if time.time() - stable_since >= 4:
                        if not quiet:
                            print(f" File ready and stable: {filepath} ({size} bytes)")
                        return
                else:
                    stable_since = None
                    last_size = size

            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout after {timeout}s: File not ready - {filepath}")
            time.sleep(0.5)

    # Individual layer methods
    def download_highways(self):
        print("\n  Downloading OSM Highways Raster...")
        download_highways_main()
        # ensure the file exists:
        if not self.highways_file.exists():
            pass
        else:
            self._wait_for_file(self.highways_file, quiet=True)

    def download_waterbodies(self):
        print("\n  Downloading OSM Waterbodies Raster...")
        download_waterbody_main()
        if not self.waterbody_file.exists():
            pass
        else:
            self._wait_for_file(self.waterbody_file, quiet=True)

    def download_buildings(self):
        print("\n  Downloading OSM Buildings Raster...")
        download_buildings_main()
        if not self.buildings_file.exists():
            pass
        else:
            self._wait_for_file(self.buildings_file, quiet=True)

    def download_all(self):
        self.download_waterbodies()
        self.download_highways()
        self.download_buildings()

    def run(self, **kwargs):
        """
        Main execution entry point.
        Downloads all OSM rasters and returns structured dict
        matching what the dispatcher expects.
        """
        print("  DataDownloadControllerOSM.run()")
        self.download_all()

        result = {
            "road_raster":      str(self.highways_file),
            "waterbody_raster": str(self.waterbody_file),
            "building_raster":  str(self.buildings_file),
        }

        print(" DataDownloadControllerOSM finished:", result)
        return result


if __name__ == "__main__":
    controller = DataDownloadControllerOSM()
    output = controller.run()
    print("Final Output:", output)
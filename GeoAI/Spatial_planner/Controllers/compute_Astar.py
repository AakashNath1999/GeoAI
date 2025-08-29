#!/usr/bin/env python
# coding: utf-8

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

#from tools_for_planning_route.get_route_start_end import update_task_file_with_coordinates
from tools_for_planning_route.get_pixal_value_from_coord import append_pixel_coords_to_task
from tools_for_planning_route.compute_path_with_Astar import compute_path_with_Astar


class RouteComputationController:
    def run_pipeline(self):
        """
        1) Resolve start/end coordinates from names into task file.
        2) Convert coordinates to pixel indices and write back to task file.
        3) Run A* and produce a shapefile path.
        Returns: (task_info_dict, shapefile_path_str)
        """
        #print("Step 1: Getting coordinates from place names...")
        #task_info = update_task_file_with_coordinates()

        print("Step 2: Converting coordinates to pixel values...")
        task_info = append_pixel_coords_to_task()

        print("Step 3: Running A* path computation...")
        shapefile_path = compute_path_with_Astar()

        print(f"Route computation complete! Shapefile saved at: {shapefile_path}")
        return task_info, shapefile_path

    def run(self, **kwargs):
        """
        Entry point for the dispatcher.
        Runs the full route computation and returns structured output.
        IMPORTANT: includes 'route' so the dispatcher can pick it up.
        """
        print("Running RouteComputationController...")
        task_info, shapefile_path = self.run_pipeline()

        result = {
            # raw info
            "start_coordinates": task_info.get("start_coords") if isinstance(task_info, dict) else None,
            "end_coordinates": task_info.get("end_coords") if isinstance(task_info, dict) else None,
            "start_pixel": task_info.get("start_pixel") if isinstance(task_info, dict) else None,
            "end_pixel": task_info.get("end_pixel") if isinstance(task_info, dict) else None,

            # artifact paths
            "route_shapefile": str(shapefile_path) if shapefile_path is not None else None,

            # alias used by LangGraph node: s["route"] = out.get("route")
            "route": str(shapefile_path) if shapefile_path is not None else None,
        }

        print("RouteComputationController finished:", result)
        return result


if __name__ == "__main__":
    controller = RouteComputationController()
    output = controller.run()
    print("Final Output:", output)

# lib_planning/utils_planning.py

from pathlib import Path

def get_project_paths():

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    data_dir = project_root / "Downloads_for_planning_route"
    task_info_dir = project_root / "task_info_planning_route"
    
    # Allow dynamic switching between route/site/etc.
    task_info_file = task_info_dir / f"current_task_info_route.json"

    data_dir.mkdir(exist_ok=True)
    task_info_dir.mkdir(exist_ok=True)

    return data_dir, task_info_file

import sys
from pathlib import Path

def add_project_root():
    """
    Adds the project root to sys.path so relative imports like 'from lib.utils import ...' work.
    It finds the nearest parent directory containing 'lib/'.
    """
    # This works both for Jupyter and .py files
    try:
        current_path = Path(__file__).resolve()
    except NameError:
        current_path = Path.cwd().resolve()

    for parent in [current_path] + list(current_path.parents):
        if (parent / "lib_planning_route").exists():
            sys.path.insert(0, str(parent))
            break

# retrieve_dem_from_drive.py

import os
import sys
import json
from pathlib import Path
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# Get the folder where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent

# Paths to credentials
CLIENT_SECRET_PATH = SCRIPT_DIR / "client_secret.json"
MYCREDS_PATH = SCRIPT_DIR / "mycreds.txt"

# Add project root to sys.path
BASE_DIR = SCRIPT_DIR.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from lib_planning_route.bootstrap_planning_route import add_project_root
add_project_root()
from lib_planning_route.utils_planning_route import get_project_paths


def retrieve_dem_from_drive_main():
    """
    Retrieves the DEM file for the current task_id from Google Drive
    and saves it into the download directory.
    """
    download_dir, task_info_path = get_project_paths()

    # Step 1: Load current task_id
    with open(task_info_path, "r") as f:
        task_id = json.load(f)["task_id"]

    dem_name = f"dem_{task_id}.tif"

    # Ensure download folder exists
    os.makedirs(download_dir, exist_ok=True)


# Authenticate Google Drive
    gauth = GoogleAuth(settings={
        "client_config_file": str(CLIENT_SECRET_PATH)  # Explicit path
    })
    gauth.settings['get_refresh_token'] = True
    gauth.settings['oauth_scope'] = ['https://www.googleapis.com/auth/drive']
    gauth.settings['oauth_flow_params'] = {'access_type': 'offline'}

    # Load or refresh credentials
    gauth.LoadCredentialsFile(str(MYCREDS_PATH))
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        print(" Refreshing expired token...")
        gauth.Refresh()
    else:
        gauth.Authorize()

    # Save credentials for reuse
    gauth.SaveCredentialsFile(str(MYCREDS_PATH))

    # Step 3: Search and download DEM
    drive = GoogleDrive(gauth)
    file_list = drive.ListFile({
        'q': f"title='{dem_name}' and trashed=false",
        'corpora': 'allDrives',
        'includeItemsFromAllDrives': True,
        'supportsAllDrives': True
    }).GetList()

    if file_list:
        file = file_list[0]
        output_path = os.path.join(download_dir, dem_name)
        file.GetContentFile(output_path)
        print(f" Downloaded {dem_name} to {output_path}")
    else:
        print(f" {dem_name} not found in Drive.")


# Allow running directly
if __name__ == "__main__":
    retrieve_dem_from_drive_main()

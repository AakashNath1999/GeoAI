# GeoAI — Agentic Route Planning from Natural Language

> Type: “Give me the optimal route from Gachibowli to Charminar, avoiding water.”  
> The system fetches DEM/LULC/Slope from Google Earth Engine (GEE), pulls OSM water/buildings/roads, rasterizes, builds a **cost raster**, then computes an **A*** shortest path, and returns a shapefile/GeoJSON.

---

## ✨ What this repo contains
- **Agentic orchestration** with LangChain/LangGraph controlling tools that:
  - Download/Fetch **DEM, LULC, Slope** from GEE (Python API)
  - Download OSM layers (**roads, waterbodies, buildings**), convert to rasters (rasterio)
  - Compute **cost raster** from weighted layers
  - Run **A*** path over the cost surface
- Cross‑platform **setup scripts** for a one‑click environment:
  - `scripts/setup_geoai.sh` (macOS/Linux)
  - `scripts/setup_geoai.bat` (Windows CMD)
  - `scripts/setup_geoai.ps1` (Windows PowerShell)
- Environment specs:
  - `environment.yml` (Conda — recommended)
  - `requirements.txt` (pip fallback; assumes GDAL/GEOS/PROJ present)

---

## 🗂 Project structure (key parts)
```
GeoAI/
├─ main_LLM.py     
├─ crew_dispatcher.py 
├─ app.py            # Entrypoint/orchestration (LangChain/LangGraph)
├─ Spatial_planner/
│  ├─ Downloads_for_planning_route/   # Generated data (ignored)
│  ├─ lib_planning_route/
│  │  ├─ __init__.py
│  │  ├─ bootstrap_planning_route.py
│  │  └─ utils_planning_route.py
│  ├─ task_info_planning_route/
│  │  └─ current_task_info_route.json
│  └─ tools_for_planning_route/
│     ├─ download_dem/
│     ├─ download_lulc/
│     ├─ download_slope/
│     ├─ retrieve_dem_from_drive/
│     ├─ retrieve_lulc_from_drive/
│     ├─ retrieve_slope_from_drive/
│     ├─ download_osm_raster_for_highways/
│     ├─ download_osm_raster_for_waterbody/
│     ├─ check_osm_name/
│     ├─ extract_place_name/
│     ├─ get_lulc_cost/
│     ├─ get_slope_cost/
│     ├─ get_cost_raster/
│     ├─ get_route_start_end/
│     └─ compute_path_with_A*/
└─ scripts/
   ├─ setup_geoai.sh
   ├─ setup_geoai.bat
   └─ setup_geoai.ps1
```

> **Note:** Large rasters/outputs live in `Downloads_for_planning_route/` and should be **.gitignored**.

---

## ✅ System requirements
- **Operating system:** Windows / macOS / Linux
- **Conda (Miniconda/Anaconda)** — strongly recommended for GDAL/Rasterio/GeoPandas
- **Python:** 3.10 (via Conda env)
- **Ollama:** installed & running locally (for LLM inference)
- **Google Account with Earth Engine access** (first run will prompt auth)
- **Git** (optional; for cloning)

---

## 🚀 Quickstart (recommended path)

### 0) Clone or download the repo
- GitHub web → **Code** → **Download ZIP** (or `git clone ...`)

### 1) Use the one‑click setup script (per OS)
> The script will create/activate the `geoai` conda env, install deps, run GEE auth, ensure an Ollama model is present, and sanity‑check imports.

**macOS / Linux**
```bash
cd GeoAI
chmod +x scripts/setup_geoai.sh
./scripts/setup_geoai.sh           # default model: llama3
# or specify:
./scripts/setup_geoai.sh llama3:8b
```

**Windows (CMD)**
```bat
cd GeoAI
scripts\setup_geoai.bat            REM default model: llama3
scripts\setup_geoai.bat llama3:8b
```

**Windows (PowerShell)**
```powershell
cd GeoAI
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\scripts\setup_geoai.ps1                  # default model: llama3
.\scripts\setup_geoai.ps1 -ModelName llama3:8b
```

### 2) Run the app
```bash
conda activate geoai
python app.py         # or your chosen entrypoint
```

---

## 🧠 LLM & Ollama
- Install **Ollama** from the official site for your OS.
- Start the Ollama daemon (usually automatic after install).
- Models can be pulled automatically by the setup script, or manually:
  ```bash
  ollama pull llama3
  # or: ollama pull llama3:8b
  ```
- Default API URL is `http://localhost:11434`. If you change it, configure in code or via env var.

---

## 🌐 Google Earth Engine (GEE) authentication
On first run, you’ll be prompted to authenticate:
```python
import ee
ee.Authenticate()
ee.Initialize()
```
The setup scripts attempt this automatically. Complete the browser flow once; subsequent runs won’t prompt.

---

# Generated geodata
Spatial_planner/Downloads_for_planning_route/
*.tif *.tiff *.img *.vrt
*.gpkg *.geojson *.shp *.shx *.dbf *.prj *.cpg *.zip
.ipynb_checkpoints/
.DS_Store
```
Keep OAuth tokens/credentials outside version control. Use `.env` for config if needed.

---

## 🧩 Typical workflow
1. **Parse user prompt** → LangChain agent extracts start/end, AOI, layer weights.
2. **Fetch data**
   - GEE: DEM/LULC/Slope
   - OSM: roads, waterbodies, buildings → rasterize
3. **Build cost raster** (weights/penalties per layer)
4. **Compute path** with A* over the cost surface
5. **Export** route (Shapefile/GeoJSON) and any diagnostic rasters

---

## ⚙️ Configuration tips
- **Layer weights/penalties**: stored in JSON (e.g., `task_info_planning_route/current_task_info_route.json`) or passed via agent prompt.
- **AOI & resolution**: choose a reasonable grid size to keep memory under control.
- **CRS/Projection**: ensure consistent CRS before raster math (use `pyproj`/`rasterio.warp`).

---

## 🧯 Troubleshooting
- `rasterio/geopandas/fiona` install issues → use **Conda** (`environment.yml`), not pure pip.
- `ee.Authenticate()` loops → ensure the browser completes & same environment is active.
- `Ollama not found` → install Ollama and ensure it’s running; then re-run setup.
- Memory errors when rasterizing → downsample resolution, clip AOI tighter, or tile processing.
- `ModuleNotFoundError` for `langgraph` / `langchain_ollama` → verify active env and reinstall.

---

## 📦 Alternate installs (manual)
**Conda (preferred)**
```bash
conda env create -f environment.yml
conda activate geoai
python -c "import ee; ee.Authenticate(); ee.Initialize()"
```

**pip (fallback; only if GDAL/GEOS/PROJ preinstalled)**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts ctivate
pip install -r requirements.txt
python -c "import ee; ee.Authenticate(); ee.Initialize()"
```

---

## How to start
- In the terminal or Anaconda prompt, go to the file path where the app.py is saved, change the 'cd' to that filepath and the type - python app.py
- remember to install python beforehands.

## 🤝 Contributing
- Open an issue for bugs/ideas.
- PRs welcome. Keep large data out of the repo; add new tools under `Spatial_planner/tools_for_planning_route/`.

---

## 📜 License
Choose a license (MIT recommended) and add a `LICENSE` file.

---

## 🙌 Credits
Built by Akash. Uses Google Earth Engine, OSM, Rasterio, GeoPandas, LangChain/LangGraph, and Ollama LLMs.

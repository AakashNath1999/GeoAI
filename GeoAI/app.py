from __future__ import annotations
import json
import time
import threading
import logging
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any

import sys, io, re, glob, os, uuid
from contextlib import redirect_stdout, redirect_stderr

from flask import Flask, request, jsonify, render_template

# NOTE: We no longer import fiona/shapely/pyproj because we don't convert SHP → GeoJSON anymore.

# -----------------------------------------------------------------------------
# Project bootstrapping
# -----------------------------------------------------------------------------
try:
    from Spatial_planner.lib_planning_route.bootstrap_planning_route import add_project_root
    add_project_root()
except Exception:
    pass

from lib_planning_route.utils_planning_route import get_project_paths
from main_LLM import run_main_llm
from crew_dispatcher import Controller_dispatch

# -----------------------------------------------------------------------------
# Flask + logging
# -----------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
app.logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# In-memory state
# -----------------------------------------------------------------------------
progress: Dict[str, Any] = {
    "status": "idle",        # idle | running | done | error
    "step": "Idle",
    "percent": 0,
    "started_at": None,
    "finished_at": None,
    "error": None,
    "meta": {}
}
_progress_lock = threading.Lock()

_logs = deque(maxlen=4000)
_logs_lock = threading.Lock()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _set_progress(
    *,
    status: Optional[str] = None,
    step: Optional[str] = None,
    percent: Optional[int] = None,
    error: Optional[str] = None,
    finished: bool = False,
    meta: Optional[Dict[str, Any]] = None,
):
    with _progress_lock:
        if status is not None:
            progress["status"] = status
        if step is not None:
            progress["step"] = step
        if percent is not None:
            progress["percent"] = max(0, min(100, int(percent)))
        if error is not None:
            progress["error"] = error
        if meta:
            progress["meta"].update(meta)
        if progress["started_at"] is None and status == "running":
            progress["started_at"] = time.time()
        if finished:
            progress["finished_at"] = time.time()

def _log(msg: str, level=logging.INFO):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with _logs_lock:
        _logs.append(line)
    app.logger.log(level, msg)

def _read_task(json_path: Path) -> dict:
    return json.loads(Path(json_path).read_text(encoding="utf-8"))

def _ensure_task_id(task: dict, task_json_path: Path) -> str:
    """Guarantee a task_id in the JSON; if missing, create one and persist it."""
    tid = task.get("task_id")
    if not tid:
        tid = str(uuid.uuid4())
        task["task_id"] = tid
        try:
            task_json_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
            _log(f"[fix] Injected missing task_id into task JSON: {tid}")
        except Exception as e:
            _log(f"[warn] Could not persist injected task_id: {e}")
    return tid

def _route_geojson_from_task(task: dict) -> Path:
    data_dir, _ = get_project_paths()
    return Path(data_dir) / f"route_{task['task_id']}.geojson"

def _glob_latest_geojson(download_dir: Path) -> Optional[Path]:
    # Prefer route_* first, then any *.geojson
    candidates = glob.glob(str(download_dir / "route_*.geojson"))
    if not candidates:
        candidates = glob.glob(str(download_dir / "*.geojson"))
    if not candidates:
        return None
    return Path(max(candidates, key=lambda p: os.path.getmtime(p)))

def _safe_under(base: Path, candidate: Path) -> bool:
    try:
        return candidate.resolve().is_relative_to(base.resolve())
    except AttributeError:
        # py<3.9 fallback
        base_resolved = str(base.resolve())
        cand_resolved = str(candidate.resolve())
        return cand_resolved.startswith(base_resolved)

def _list_json_like(download_dir: Path):
    files = []
    for pattern in ("*.geojson", "*.json"):
        for p in sorted(Path(download_dir).glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                files.append({
                    "name": p.name,
                    "size": p.stat().st_size,
                    "mtime": p.stat().st_mtime
                })
            except FileNotFoundError:
                continue
    return files

# -----------------------------------------------------------------------------
# UI Log plumbing: intercept print() and logging from controllers
# -----------------------------------------------------------------------------
class _UILogWriter(io.TextIOBase):
    """Line-buffered writer that forwards anything written to _log(...)"""
    def __init__(self, log_cb):
        self.log_cb = log_cb
        self._buf = ""

    def writable(self): return True

    def write(self, s):
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if line.strip():
                self.log_cb(line)
        return len(s)

    def flush(self):
        if self._buf.strip():
            self.log_cb(self._buf.strip())
        self._buf = ""

class _UILogHandler(logging.Handler):
    """Logging handler that forwards log records to _log(...)"""
    def __init__(self, log_cb):
        super().__init__()
        self.log_cb = log_cb

    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        level = record.levelno
        self.log_cb(msg, level)

# Map specific backend log lines to progress bumps
_PROGRESS_MAP = [
    (r"Starting DEM export|Downloading DEM", 12, "Downloading DEM from GEE…"),
    (r"Retrieving DEM|Downloaded dem",       18, "Retrieving DEM from Drive…"),
    (r"Starting LULC export|Downloading LULC", 22, "Downloading LULC from GEE…"),
    (r"Retrieving LULC|Downloaded lulc",     28, "Retrieving LULC from Drive…"),
    (r"Starting SLOPE export|Computing slope", 32, "Computing slope…"),
    (r"Retrieving SLOPE|Downloaded slope",   38, "Retrieving slope from Drive…"),
    (r"OSM Waterbodies|waterbod(y|ies)",     45, "Fetching OSM waterbodies…"),
    (r"OSM Highways|roads",                  50, "Fetching OSM roads…"),
    (r"OSM Buildings|building",              55, "Fetching OSM buildings…"),
    (r"LULC cost",                           62, "Generating LULC cost raster…"),
    (r"Slope cost|Resampled raster",         68, "Generating slope cost raster…"),
    (r"Combining .* into final cost|Final cost raster", 75, "Building total cost raster…"),
    (r"Running A\*|A\* pathfinding",         85, "Running A* pathfinding…"),
    (r"Path saved|Route computation complete|Exported GeoJSON", 98, "Exporting route…"),
]
def _maybe_bump_progress_from_line(line: str):
    for pattern, pct, step in _PROGRESS_MAP:
        if re.search(pattern, line, re.IGNORECASE):
            with _progress_lock:
                current = progress.get("percent", 0) or 0
            if pct > current:
                _set_progress(step=step, percent=pct)
            break

# -----------------------------------------------------------------------------
# Pipeline runner (GeoJSON-first; no SHP dependency)
# -----------------------------------------------------------------------------
def _run_pipeline(prompt: str):
    """Runs your full pipeline in a background thread and updates progress/logs."""
    try:
        _set_progress(status="running", step="Starting…", percent=1, meta={"prompt": prompt})
        _log(f"▶ Task started: {prompt}")

        # 1) LLM builds/updates current_task_info_route.json
        _set_progress(step="Parsing prompt via local LLM", percent=10)
        _log("Running main_LLM to build task JSON …")
        task_json_path_str = run_main_llm(prompt)
        task_json_path = Path(task_json_path_str)
        task = _read_task(task_json_path)
        tid = _ensure_task_id(task, task_json_path)
        _log(f"Task JSON ready: {task_json_path} (task_id={tid})")
        _set_progress(meta={"task_id": tid})

        # 2) Dispatch controllers to compute the route
        _set_progress(step="Dispatching controllers (downloads, rasters, A*)", percent=35)
        _log("Dispatching controllers …")

        # Intercept ALL prints/logging during the controller run
        ui_writer = _UILogWriter(lambda m: (_log(m), _maybe_bump_progress_from_line(m)))
        ui_handler = _UILogHandler(lambda m, lvl=logging.INFO: (_log(m, lvl), _maybe_bump_progress_from_line(m)))
        ui_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(ui_handler)
        prev_level = root_logger.level
        root_logger.setLevel(logging.INFO)

        try:
            with redirect_stdout(ui_writer), redirect_stderr(ui_writer):
                # If you can, pass tid/log_cb/progress_cb into Controller_dispatch
                # Controller_dispatch(task_id=tid, log_cb=_log, progress_cb=_set_progress)
                Controller_dispatch()
                ui_writer.flush()
        finally:
            root_logger.removeHandler(ui_handler)
            root_logger.setLevel(prev_level)

        # 3) GeoJSON output (preferred). Do NOT require/expect SHP.
        _set_progress(step="Preparing GeoJSON output", percent=98)
        data_dir, _ = get_project_paths()

        # Prefer the exact file for the current task id
        gj_path = _route_geojson_from_task(task)
        if gj_path.exists():
            fc = json.loads(gj_path.read_text(encoding="utf-8"))
            _log(f"Found GeoJSON: {gj_path}")
        else:
            # Fallback: the latest route_*.geojson in downloads
            latest = _glob_latest_geojson(Path(data_dir))
            if not latest:
                raise FileNotFoundError(
                    "No route GeoJSON found. Expected "
                    f"{gj_path.name} or any route_*.geojson in {data_dir}"
                )
            _log(f"[fallback] Using latest GeoJSON: {latest.name}")
            fc = json.loads(latest.read_text(encoding="utf-8"))
            gj_path = latest  # point meta to what we actually used

        _set_progress(
            status="done", step="Completed", percent=100, finished=True,
            meta={"route_geojson": str(gj_path)}
        )
        _log("✅ Task completed.")

    except Exception as e:
        _set_progress(status="error", step="Failed", error=str(e), finished=True)
        _log(f"❗ Error: {e}", logging.ERROR)

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def home():
    # Put your Leaflet HTML at templates/index.html
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({"ok": True})

@app.get("/api/status")
def api_status():
    with _progress_lock:
        return jsonify(progress)

@app.get("/api/logs")
def api_logs():
    with _logs_lock:
        return jsonify({"logs": list(_logs)})

@app.get("/api/route")
def api_route():
    """
    Returns the last produced route as GeoJSON.
    Preference order:
      1) route_{task_id}.geojson (exact match)
      2) latest route_*.geojson in downloads
    """
    try:
        data_dir, task_json_path = get_project_paths()
        task = _read_task(Path(task_json_path))
        _ensure_task_id(task, Path(task_json_path))

        # 1) exact geojson from task id
        gj_path = _route_geojson_from_task(task)
        if gj_path.exists():
            fc = json.loads(gj_path.read_text(encoding="utf-8"))
            return jsonify({"ok": True, "route": fc, "source": gj_path.name})

        # 2) latest geojson fallback
        latest_gj = _glob_latest_geojson(Path(data_dir))
        if latest_gj and latest_gj.exists():
            fc = json.loads(latest_gj.read_text(encoding="utf-8"))
            _log(f"[fallback] /api/route using latest geojson: {latest_gj.name}")
            return jsonify({"ok": True, "route": fc, "source": latest_gj.name})

        return jsonify({"ok": False, "error": "No route GeoJSON found."}), 404

    except Exception as e:
        _log(f"/api/route error: {e}", logging.ERROR)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/api/jsons")
def api_list_jsons():
    """List JSON/GeoJSON files available in the downloads folder."""
    try:
        data_dir, _ = get_project_paths()
        files = _list_json_like(Path(data_dir))
        return jsonify({"ok": True, "files": files})
    except Exception as e:
        _log(f"/api/jsons error: {e}", logging.ERROR)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/api/jsons/<path:filename>")
def api_get_json_file(filename: str):
    """
    Return the contents of a JSON/GeoJSON file from the downloads folder.
    Security: ensures the path stays under downloads directory.
    """
    try:
        data_dir, _ = get_project_paths()
        target = Path(data_dir) / filename
        if not _safe_under(Path(data_dir), target):
            return jsonify({"ok": False, "error": "Invalid path"}), 400
        if not target.exists():
            return jsonify({"ok": False, "error": f"{filename} not found"}), 404
        if target.suffix.lower() not in {".json", ".geojson"}:
            return jsonify({"ok": False, "error": "Only .json/.geojson allowed"}), 400

        text = target.read_text(encoding="utf-8")
        return jsonify({"ok": True, "name": target.name, "content": json.loads(text)})
    except json.JSONDecodeError:
        # If it's not valid JSON, still serve as text for debugging
        return jsonify({"ok": True, "name": filename, "content": target.read_text(encoding="utf-8")})
    except Exception as e:
        _log(f"/api/jsons/{filename} error: {e}", logging.ERROR)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/api/run")
def api_run():
    """
    Body: { "prompt": "Give me the optimal road from ..." }
    Kicks off the long-running pipeline in a background thread.
    Frontend should poll /api/status and /api/logs,
    and call /api/route when progress.status === 'done'.
    """
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"ok": False, "error": "prompt is required"}), 400

    # Reset state
    with _progress_lock:
        progress.update({
            "status": "running",
            "step": "Starting…",
            "percent": 1,
            "started_at": time.time(),
            "finished_at": None,
            "error": None,
            "meta": {}
        })
    with _logs_lock:
        _logs.clear()

    # Start background worker
    t = threading.Thread(target=_run_pipeline, args=(prompt,), daemon=True)
    t.start()

    return jsonify({"ok": True, "message": "Task started"})

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Dev server
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
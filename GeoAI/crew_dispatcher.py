#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import annotations
import os, json, sys
from pathlib import Path
from typing import TypedDict, Optional, List

# ---- Project paths ----
try:
    GEOAI_ROOT = Path(__file__).resolve().parent
except NameError:
    GEOAI_ROOT = Path().resolve()

LIB_PATH = GEOAI_ROOT / "Spatial_planner" / "lib_planning_route"
for p in {str(GEOAI_ROOT), str(LIB_PATH)}:
    if p not in sys.path:
        sys.path.insert(0, p)

TASK_JSON = GEOAI_ROOT / "Spatial_planner" / "task_info_planning_route" / "current_task_info_route.json"

# ---- Controllers ----
from Spatial_planner.Controllers.data_download_controller_GEE import DataDownloadControllerGEE
from Spatial_planner.Controllers.data_download_controller_OSM import DataDownloadControllerOSM
from Spatial_planner.Controllers.cost_surface_controller import CostSurfaceController
from Spatial_planner.Controllers.compute_Astar import RouteComputationController

# ---- LangChain / LangGraph ----
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END


# ## route

# In[2]:


class RouteState(TypedDict, total=False):
    # inputs
    task_type: str
    # artifacts/paths
    dem_raster: Optional[str]
    slope_raster: Optional[str]
    lulc_raster: Optional[str]
    road_raster: Optional[str]
    waterbody_raster: Optional[str]
    building_raster: Optional[str]
    cost_surface: Optional[str]
    route: Optional[dict]  # geojson/coords
    # logs
    log: List[str]

# ---------------- LLM (optional; keep for summaries/reasoning if you need it) ----------------
llm = ChatOllama(
    model="llama3.2:3b",       # ensure:  ollama pull llama3.2:3b
    base_url="http://localhost:11434",
    temperature=0.0,
    num_ctx=4096,
    timeout=600,
)


# In[3]:


def _ensure_log(s: dict) -> None:
    s["log"] = list(s.get("log", []))

def node_load_task(state: RouteState) -> RouteState:
    with open(TASK_JSON, "r", encoding="utf-8") as f:
        task_info = json.load(f)
    s = dict(state)
    _ensure_log(s)
    s["task_type"] = task_info.get("task_type", "")
    s["log"].append(f"Detected task_type: {s['task_type']}")
    return s

def node_download_gee(state: RouteState) -> RouteState:
    s = dict(state)
    _ensure_log(s)
    ctl = DataDownloadControllerGEE()
    out = ctl.run()  # expected: {"dem_raster": "...", "slope_raster": "...", "lulc_raster": "..."}
    s["dem_raster"]   = out.get("dem_raster")
    s["slope_raster"] = out.get("slope_raster")
    s["lulc_raster"]  = out.get("lulc_raster")
    s["log"].append("[GEE] Downloaded DEM/Slope/LULC")
    return s

def node_download_osm(state: RouteState) -> RouteState:
    s = dict(state)
    _ensure_log(s)
    ctl = DataDownloadControllerOSM()
    out = ctl.run()  # expected: {"road_raster": "...", "waterbody_raster": "...", "building_raster": "..."}
    s["road_raster"]      = out.get("road_raster")
    s["waterbody_raster"] = out.get("waterbody_raster")
    s["building_raster"]  = out.get("building_raster")
    s["log"].append("[OSM] Downloaded & rasterized roads/water/buildings")
    return s

def node_build_cost(state: RouteState) -> RouteState:
    s = dict(state)
    _ensure_log(s)
    ctl = CostSurfaceController()
    out = ctl.run()  # expected: {"cost_surface": "..."}
    s["cost_surface"] = out.get("cost_surface")
    s["log"].append("[COST] Built cost surface")
    return s

def node_run_astar(state: RouteState) -> RouteState:
    s = dict(state)
    _ensure_log(s)
    ctl = RouteComputationController()
    out = ctl.run()  # expected: {"route": <geojson/coords dict>}
    s["route"] = out.get("route")
    s["log"].append("[ROUTE] Computed A* path")
    return s


# In[4]:


def need_gee(state: RouteState) -> bool:
    return not all([state.get("dem_raster"), state.get("slope_raster"), state.get("lulc_raster")])

def need_osm(state: RouteState) -> bool:
    return not all([state.get("road_raster"), state.get("waterbody_raster"), state.get("building_raster")])

def need_cost(state: RouteState) -> bool:
    return not bool(state.get("cost_surface"))

def need_route(state: RouteState) -> bool:
    return not bool(state.get("route"))

def add_conditional_with_end(graph: StateGraph, node_name: str, chooser, mapping: dict):
    """
    Wrapper for LangGraph >=0.2 where add_conditional_edges has no `fallback`.
    Ensures chooser always maps to a known key; unknowns go to __end__ -> END.
    """
    local = dict(mapping)
    if "__end__" not in local:
        local["__end__"] = END

    def _choose(s):
        key = chooser(s)
        return key if key in local else "__end__"

    graph.add_conditional_edges(node_name, _choose, local)


# In[5]:


def build_graph():
    graph = StateGraph(RouteState)
    graph.add_node("load_task", node_load_task)
    graph.add_node("download_gee", node_download_gee)
    graph.add_node("download_osm", node_download_osm)
    graph.add_node("build_cost", node_build_cost)
    graph.add_node("run_astar", node_run_astar)

    # entry
    graph.set_entry_point("load_task")

    # After load_task, branch by needs (download_data / route_planning -> GEE; else END)
    add_conditional_with_end(
        graph, "load_task",
        lambda s: "download_gee" if s.get("task_type") in {"download_data", "route_planning"} else "__end__",
        {"download_gee": "download_gee"},
    )

    add_conditional_with_end(
        graph, "download_gee",
        lambda s: "download_osm" if need_osm(s) else ("build_cost" if s.get("task_type") == "route_planning" else "__end__"),
        {"download_osm": "download_osm", "build_cost": "build_cost"},
    )

    # After download_osm, either END (download_data) or build cost (route_planning)
    add_conditional_with_end(
        graph, "download_osm",
        lambda s: "build_cost" if s.get("task_type") == "route_planning" else "__end__",
        {"build_cost": "build_cost"},
    )

    # After build_cost, maybe run A*
    add_conditional_with_end(
        graph, "build_cost",
        lambda s: "run_astar" if need_route(s) else "__end__",
        {"run_astar": "run_astar"},
    )

    # After run_astar -> END
    graph.add_edge("run_astar", END)

    return graph.compile()


# In[6]:


def Controller_dispatch():
    os.environ.setdefault("GEE_MAX_WAIT_SECS", "600")
    app = build_graph()
    # initial empty state; graph will read TASK_JSON
    out = app.invoke({})
    print("---- LOG ----")
    for line in out.get("log", []):
        print(line)
    print("---- RESULT ----")
    print(json.dumps({
        "dem_raster": out.get("dem_raster"),
        "slope_raster": out.get("slope_raster"),
        "lulc_raster": out.get("lulc_raster"),
        "road_raster": out.get("road_raster"),
        "waterbody_raster": out.get("waterbody_raster"),
        "building_raster": out.get("building_raster"),
        "cost_surface": out.get("cost_surface"),
        "route": out.get("route"),
    }, indent=2))

if __name__ == "__main__":
    Controller_dispatch()


# In[ ]:





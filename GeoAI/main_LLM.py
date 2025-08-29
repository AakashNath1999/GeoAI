#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations
import json
import re
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
import math
from typing import Tuple

from langchain_ollama import ChatOllama
from Spatial_planner.Crew_memory.crew_memory_LC import CrewMemoryLC 


OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:3b"

MEMORY_JSON = "Spatial_planner/Crew_memory/crew_memory.json"
MEMORY_PT   = "Spatial_planner/Crew_memory/crew_memory_embedding.pt"

OUT_PATH = Path("Spatial_planner/task_info_planning_route/current_task_info_route.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def _log(msg: str):
    print(msg)

def ensure_ollama_up():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/version", timeout=5)
        r.raise_for_status()
        return True
    except Exception as e:
        raise RuntimeError(
            "Ollama not reachable at http://localhost:11434. "
            "Start it with: ollama serve"
        ) from e

# JSON helpers
_CODEFENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

def strip_code_fences(text: str) -> str:
    return _CODEFENCE.sub("", text.strip())

def extract_first_json(text: str) -> Dict[str, Any]:
    t = strip_code_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    start = t.find("{")
    if start == -1:
        raise ValueError("No JSON object found in output")
    brace = 0
    for i, ch in enumerate(t[start:], start=start):
        if ch == "{": brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0:
                snippet = t[start:i+1]
                try: return json.loads(snippet)
                except Exception: break
    for m in re.finditer(r"\{.*\}", t, flags=re.DOTALL):
        try: return json.loads(m.group(0))
        except Exception: continue
    raise ValueError(f"Could not parse JSON from output:\n{text}")


# In[80]:


_RE_FROM_TO = re.compile(
    r"\bfrom\s+(.+?)\s+to\s+(.+?)"
    r"(?=$|\s(?:in|into|within|near|around|avoiding|avoid|while|maximizing|maximising|"
    r"prefer|preferring|prioritize|prioritise|prioritizing|prioritising|but|with|and|&|"
    r"via|through|using|by)\b|[.,;])",
    flags=re.IGNORECASE,
)

_RE_BETWEEN_AND = re.compile(
    r"\bbetween\s+(.+?)\s+and\s+(.+?)"
    r"(?=$|\s(?:in|into|within|near|around|avoiding|avoid|while|maximizing|maximising|"
    r"prefer|preferring|prioritize|prioritise|prioritizing|prioritising|but|with|and|&|"
    r"via|through|using|by)\b|[.,;])",
    flags=re.IGNORECASE,
)

# optional tidy-up for edges
_STOP_EDGE = re.compile(r"^(?:the|a|an)\s+|(?:\s+(?:area|city|town|village|district))$", re.IGNORECASE)

def _clean_place(s: str) -> str:
    s = s.strip(' "\'')
    s = _STOP_EDGE.sub("", s).strip()
    return s

def extract_start_end(instruction: str) -> Tuple[Optional[str], Optional[str]]:
    text = instruction.strip()
    m = _RE_FROM_TO.search(text)
    if m:
        return _clean_place(m.group(1)) or None, _clean_place(m.group(2)) or None
    m = _RE_BETWEEN_AND.search(text)
    if m:
        return _clean_place(m.group(1)) or None, _clean_place(m.group(2)) or None
    return (None, None)


llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_URL,
    temperature=0.0,
    timeout=600,
)

# Few-shot Memory
crew_mem = CrewMemoryLC(
    memory_file=MEMORY_JSON,
    embeddings_file=MEMORY_PT,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    instruction_key="instruction",
)

# NOT using it... but as a demo. 
TASK_DEFS: Dict[str, Dict[str, Any]] = {
    "route_planning": {
        "skeleton": {
            "task_type": "route_planning",
            "place_name": "Unknown",
            "cost_config": {
                "lulc":     {"weight": None, "penalty": None},
                "slope":    {"weight": None, "penalty": None},
                "road":     {"weight": None, "penalty": None},
                "water":    {"weight": None, "penalty": None},
                "building": {"weight": None, "penalty": None}
            },
            "start_point": {"place": None},
            "end_point": {"place": None}
        },
        "explain": (
          "Produce ONLY a JSON object for route_planning.\n"
          "- Copy caller-provided start/end/place exactly.\n"
          "- DO NOT give zero value in weight for LULC raster ever, give weight according to the examples.\n"
          "- DO NOT give negative value in weight anywhere except Road layer.\n"
          "- You MUST compute numeric weights and penalties for ALL layers in cost_config "
          "(lulc, slope, road, water, building). Use floats/ints only.\n"
          "- Only if you see any layers which is to be avoided in the prompt, give high penalty like [100,1000] range.\n"
          "- If no information is mentioned for any layer then give the weight according to the shot examples but DO NOT leave it empty.\n"
          "- Do NOT copy numbers from the examples or any defaults; set values that reflect the instruction."
        ),
        "keywords": [r"route", r"road", r"path", r"\bfrom\b.+\bto\b", r"\bbetween\b.+\band\b", r"avoid", r"avoiding"],
    },
    "download_data": {
        "skeleton": {
            "task_type": "download_data",
            "place_name": "Unknown",
            "data_requirements": {
                "dem": False, "slope": False, "lulc": False, "road": False, "water": False, "building": False
            },
            "output_paths": {"dem_raster": None, "slope_raster": None, "lulc_raster": None}
        },
        "explain": (
            "Produce ONLY a JSON object for download_data. "
            "Set data_requirements booleans according to the instruction."
        ),
        "keywords": [r"\bdownload\b", r"\braster\b", r"\bdem\b", r"\bslope\b", r"\blulc\b"],
    },
    # Adding other tasks here ---
    "site_selection": {
        "skeleton": {
            "task_type": "site_selection",
            "place_name": "Unknown",
            "criteria": {
                "slope_max": None, "distance_to_road_max": None, "avoid_water": True
            },
            "aoi": None
        },
        "explain": "Produce ONLY a JSON object for site_selection. Fill criteria based on the instruction.",
        "keywords": [r"site selection", r"suitable site", r"select site", r"suitability"],
    },
    "prediction": {
        "skeleton": {
            "task_type": "prediction",
            "place_name": "Unknown",
            "target": None,
            "features": [],
            "horizon": None
        },
        "explain": "Produce ONLY a JSON object for prediction tasks.",
        "keywords": [r"predict", r"forecast", r"estimate", r"classification", r"regression"],
    },
}

# Intent classifier
def classify_task_type(instruction: str) -> str:
    text = instruction.lower()
    for tname, tdef in TASK_DEFS.items():
        for pat in tdef["keywords"]:
            if re.search(pat, text, re.I):
                return tname
    # default: route_planning (most common)
    return "route_planning"

# Prompt builder
SYSTEM_BLOCK = (
    "You are an assistant that emits ONLY one valid JSON object for the requested task type. "
    "Do not add explanations or code fences."
)

def build_prompt_for(task_type: str, instruction: str) -> str:
    # entities from caller
    start, end = extract_start_end(instruction)
    place_name = start if start else "Unknown"

    # filter top 5 memory examples to this task_type
    ex_raw = crew_mem.top_k(query=instruction, k=5)
    ex_filtered = []
    for ex in ex_raw:
        payload = ex.get("json_output", ex)
        if isinstance(payload, dict) and payload.get("task_type") == task_type:
            ex_filtered.append(payload)
    # fallback: take any examples if none match (keeps the format)
    if not ex_filtered and ex_raw:
        ex_filtered = [ex.get("json_output", ex) for ex in ex_raw]

    examples_text = ""
    for p in ex_filtered[:3]:
        examples_text += json.dumps(p, ensure_ascii=False, indent=2) + "\n"

    skeleton = TASK_DEFS[task_type]["skeleton"].copy()
    # inject caller-provided fields into the skeleton preview to “pin” them
    if task_type == "route_planning":
        skeleton["place_name"] = place_name
        skeleton["start_point"]["place"] = start
        skeleton["end_point"]["place"] = end
    else:
        skeleton["place_name"] = place_name

    constraints = (
        "CONTEXT (caller-provided fields — copy EXACTLY):\n"
        f"- place_name: {json.dumps(place_name)}\n"
    )
    if task_type == "route_planning":
        constraints += (
            f"- start_point.place: {json.dumps(start)}\n"
            f"- end_point.place: {json.dumps(end)}\n"
        )
    constraints += (
        "\nHARD RULES:\n"
        "- Output ONLY a single JSON object.\n"
        "- Copy the caller-provided fields exactly; do not invent locations.\n"
        "- Follow the schema of the task-specific skeleton below.\n"
    )

    prompt = (
        SYSTEM_BLOCK + "\n\n"
        f"TASK TYPE: {task_type}\n"
        + TASK_DEFS[task_type]["explain"] + "\n\n"
        + constraints + "\n"
        "EXAMPLES (representative):\n" + (examples_text or "(none)\n") + "\n"
        "SKELETON (fill/adjust but keep structure):\n"
        + json.dumps(skeleton, ensure_ascii=False, indent=2) + "\n\n"
        f'INSTRUCTION: "{instruction}"\n'
        "OUTPUT (JSON only):\n"
    )
    return prompt, {"start": start, "end": end, "place_name": place_name}

# Validators / normalizers
REQUIRED_CC_KEYS = {"lulc", "slope", "road", "water", "building"}

def _clamp_float(x, lo, hi):
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return max(lo, min(hi, v))

def validate_cost_config_strict(cfg: dict) -> dict:
    """
    Strict validation:
    - Must contain ALL required layers.
    - Each layer must have numeric (finite) weight and penalty.
    - Clamp into sane ranges.
    - No merging with defaults. Raises on issues.
    """
    if not isinstance(cfg, dict):
        raise ValueError("cost_config must be a dict")

    missing = REQUIRED_CC_KEYS - set(cfg.keys())
    if missing:
        raise ValueError(f"Missing layers in cost_config: {sorted(missing)}")

    cleaned = {}
    for k in REQUIRED_CC_KEYS:
        sub = cfg.get(k)
        if not isinstance(sub, dict):
            raise ValueError(f"{k} must be an object with 'weight' and 'penalty'")
        w = _clamp_float(sub.get("weight"), -100, 100)
        p = _clamp_float(sub.get("penalty"), 0, 100000)
        if w is None or p is None:
            raise ValueError(f"{k} has invalid weight/penalty; got {sub!r}")
        cleaned[k] = {"weight": w, "penalty": p}

    if all(cleaned[k]["weight"] == 0 for k in cleaned):
        raise ValueError("All weights are zero; degenerate configuration")

    return cleaned

def normalize_route_planning(task: Dict[str, Any], pins: Dict[str, Any]) -> Dict[str, Any]:
    task["task_type"] = "route_planning"
    task["place_name"] = pins["place_name"]
    task["start_point"] = {"place": pins["start"]}
    task["end_point"]   = {"place": pins["end"]}

    # STRICT: require LLM to provide full numeric cost_config
    if "cost_config" not in task:
        raise ValueError("LLM output missing 'cost_config'")
    task["cost_config"] = validate_cost_config_strict(task["cost_config"])
    return task

def normalize_download_data(task: Dict[str, Any], pins: Dict[str, Any], instruction: str) -> Dict[str, Any]:
    task["task_type"] = "download_data"
    task["place_name"] = pins["place_name"]
    dr = task.get("data_requirements") or {}
    text = instruction.lower()
    task["data_requirements"] = {
        "dem":   dr.get("dem",   "dem" in text or "elevation" in text),
        "slope": dr.get("slope", "slope" in text),
        "lulc":  dr.get("lulc",  "lulc" in text or "land use" in text),
        "road":  bool(dr.get("road", False)),
        "water": bool(dr.get("water", False)),
        "building": bool(dr.get("building", False)),
    }
    task["output_paths"] = task.get("output_paths") or {
        "dem_raster": None, "slope_raster": None, "lulc_raster": None
    }
    return task

def normalize_site_selection(task: Dict[str, Any], pins: Dict[str, Any]) -> Dict[str, Any]:
    task["task_type"] = "site_selection"
    task.setdefault("place_name", pins["place_name"])
    task.setdefault("criteria", {"slope_max": None, "distance_to_road_max": None, "avoid_water": True})
    task.setdefault("aoi", None)
    return task

def normalize_prediction(task: Dict[str, Any], pins: Dict[str, Any]) -> Dict[str, Any]:
    task["task_type"] = "prediction"
    task.setdefault("place_name", pins["place_name"])
    task.setdefault("target", None)
    task.setdefault("features", [])
    task.setdefault("horizon", None)
    return task

NORMALIZERS = {
    "route_planning": normalize_route_planning,
    "download_data":  normalize_download_data,
    "site_selection": normalize_site_selection,
    "prediction":     normalize_prediction,
}


def get_task_object(instruction: str) -> Dict[str, Any]:
    ensure_ollama_up()

    # 1) classify task
    task_type = classify_task_type(instruction)
    _log(f"[llm] classified task_type = {task_type}")

    # 2) build prompt locked to that task, with pins
    prompt, pins = build_prompt_for(task_type, instruction)

    # 3) call LLM
    resp = llm.invoke(prompt)
    raw_text = resp.content if hasattr(resp, "content") else str(resp)

    # 4) parse JSON
    obj = extract_first_json(raw_text)
    task = obj.get("json_output", obj) or {}

    # 5) normalize/validate
    normalizer = NORMALIZERS[task_type]
    if task_type == "download_data":
        task = normalizer(task, pins, instruction)  # needs instruction to infer booleans
    else:
        task = normalizer(task, pins)

    return task

def run_main_llm(instruction: str) -> str:
    """
    Build task JSON from a user instruction, save it, return path.
    """
    task_obj = get_task_object(instruction)
    OUT_PATH.write_text(json.dumps(task_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f" Task JSON written to {OUT_PATH}")
    return str(OUT_PATH)

# Demo if using notebook,
if __name__ == "__main__":
    demo = "Get me a road from Atila Gaon to Choladhara avoiding water and buildings and prefer roads"
    path = run_main_llm(demo)
    print("\n----- JSON Content -----")
    print(Path(path).read_text(encoding="utf-8"))
    print("------------------------")






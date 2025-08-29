#!/usr/bin/env bash
set -e

PROJECT_NAME="geoai"
MODEL_NAME="${1:-llama3}"

echo "==> Checking for conda..."
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: Conda not found. Install Miniconda/Anaconda and re-run."
  exit 1
fi

# Try to make 'conda activate' available in this script
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [ -f "environment.yml" ]; then
  echo "==> Creating env from environment.yml"
  conda env create -f environment.yml || echo "Env may already exist; proceeding..."
else
  echo "WARNING: environment.yml not found in current directory."
  echo "         Using a minimal env as fallback (geo libs may be tricky via pip)."
  conda create -y -n "$PROJECT_NAME" python=3.10
fi

echo "==> Activating env: $PROJECT_NAME"
conda activate "$PROJECT_NAME" || conda activate $(head -n1 environment.yml | awk '{print $2}')

# If environment.yml wasn't present, install requirements.txt if available
if [ ! -f "environment.yml" ] && [ -f "requirements.txt" ]; then
  echo "==> Installing pip requirements (fallback)"
  pip install -r requirements.txt
fi

echo "==> Earth Engine first-time auth (ok to skip if already authenticated)"
python - <<'PY'
try:
    import ee
    ee.Initialize()
    print("Earth Engine: already authenticated & initialized.")
except Exception as e:
    print("Earth Engine not initialized. Starting authentication flow...")
    import ee
    ee.Authenticate()
    ee.Initialize()
    print("Earth Engine: authenticated.")
PY

echo "==> Checking Ollama..."
if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: Ollama is not installed. Install it from https://ollama.com and re-run."
  exit 1
fi

echo "==> Ensuring model '${MODEL_NAME}' is available"
ollama pull "${MODEL_NAME}" || true

echo "==> Sanity check: LangChain + LangGraph + Rasterio"
python - <<'PY'
import importlib
for m in ["langchain", "langgraph", "rasterio", "geopandas", "osmnx"]:
    try:
        importlib.import_module(m)
        print(f"OK: {m}")
    except Exception as e:
        print(f"CHECK: {m} -> {e}")
PY

echo "==> All set!"
echo "To start working:"
echo "  conda activate $PROJECT_NAME"
echo "  python main_phi2.py   # or your entrypoint"

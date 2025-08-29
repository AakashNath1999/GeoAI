Param(
  [string]$ProjectName = "geoai",
  [string]$ModelName = "llama3"
)

Write-Host "==> Checking for conda..."
$conda = Get-Command conda -ErrorAction SilentlyContinue
if (-not $conda) {
  Write-Error "Conda not found. Install Miniconda/Anaconda and re-run."
  exit 1
}

$condaBase = (& conda info --base).Trim()
& "$condaBase\shell\condabin\conda-hook.ps1" | Out-Null
conda activate base | Out-Null

if (Test-Path "environment.yml") {
  Write-Host "==> Creating env from environment.yml"
  conda env create -f environment.yml | Out-Null
} else {
  Write-Warning "environment.yml not found. Creating minimal env (geo libs via pip may be tricky)."
  conda create -y -n $ProjectName python=3.10 | Out-Null
}

Write-Host "==> Activating env: $ProjectName"
conda activate $ProjectName | Out-Null

if ((-not (Test-Path "environment.yml")) -and (Test-Path "requirements.txt")) {
  Write-Host "==> Installing pip requirements (fallback)"
  pip install -r requirements.txt
}

Write-Host "==> Earth Engine first-time auth (ok to skip if already authenticated)"
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

Write-Host "==> Checking Ollama..."
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
  Write-Error "Ollama is not installed. Install it from https://ollama.com and re-run."
  exit 1
}

Write-Host "==> Ensuring model '$ModelName' is available"
& ollama pull $ModelName

Write-Host "==> Sanity check: LangChain + LangGraph + Rasterio"
python - <<'PY'
import importlib
for m in ["langchain", "langgraph", "rasterio", "geopandas", "osmnx"]:
    try:
        importlib.import_module(m)
        print(f"OK: {m}")
    except Exception as e:
        print(f"CHECK: {m} -> {e}")
PY

Write-Host "==> All set!"
Write-Host "  conda activate $ProjectName"
Write-Host "  python main_phi2.py"

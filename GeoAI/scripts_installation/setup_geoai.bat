\
@echo off
setlocal enabledelayedexpansion

set PROJECT_NAME=geoai
set MODEL_NAME=%1
if "%MODEL_NAME%"=="" set MODEL_NAME=llama3

echo ==^> Checking for conda...
where conda >nul 2>nul
if errorlevel 1 (
  echo ERROR: Conda not found. Install Miniconda/Anaconda and re-run.
  exit /b 1
)

for /f "delims=" %%i in ('conda info --base') do set CONDA_BASE=%%i
call "%CONDA_BASE%\Scripts\activate.bat"

if exist environment.yml (
  echo ==^> Creating env from environment.yml
  conda env create -f environment.yml || echo Env may already exist; proceeding...
) else (
  echo WARNING: environment.yml not found. Creating minimal env (geo libs via pip may be tricky).
  conda create -y -n %PROJECT_NAME% python=3.10
)

echo ==^> Activating env: %PROJECT_NAME%
call conda activate %PROJECT_NAME% || (for /f "tokens=2 delims= " %%n in ('findstr /B /C:"name:" environment.yml') do call conda activate %%n)

if not exist environment.yml if exist requirements.txt (
  echo ==^> Installing pip requirements (fallback)
  pip install -r requirements.txt
)

echo ==^> Earth Engine first-time auth (ok to skip if already authenticated)
python - <<PY
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

echo ==^> Checking Ollama...
where ollama >nul 2>nul
if errorlevel 1 (
  echo ERROR: Ollama is not installed. Install it from https://ollama.com and re-run.
  exit /b 1
)

echo ==^> Ensuring model '%MODEL_NAME%' is available
ollama pull %MODEL_NAME%

echo ==^> Sanity check: LangChain + LangGraph + Rasterio
python - <<PY
import importlib
for m in ["langchain", "langgraph", "rasterio", "geopandas", "osmnx"]:
    try:
        importlib.import_module(m)
        print(f"OK: {m}")
    except Exception as e:
        print(f"CHECK: {m} -> {e}")
PY

echo ==^> All set!
echo   conda activate %PROJECT_NAME%
echo   python main_phi2.py   ^>^> start the app

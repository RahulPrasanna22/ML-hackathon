@echo off
REM PathSnap Laptop 1 API - listens on all interfaces so Laptop 2 can reach it over Wi-Fi.
REM Optional: set PATHSNAP_EVAL_DIR=C:\path\to\evaluator_test   (folder with the 10 evaluator trips)
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo [ERROR] .venv not found. Run setup_windows.bat first.
  exit /b 1
)
echo Laptop 1 IPv4 addresses (give one of these to Laptop 2):
ipconfig | findstr /R /C:"IPv4"
.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info

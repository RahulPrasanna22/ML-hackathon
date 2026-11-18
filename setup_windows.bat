@echo off
REM One-time setup on Laptop 1 (Python 3.11 on PATH required).
cd /d "%~dp0"
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\pip.exe install -r requirements.txt
echo.
echo Allow inbound connections on port 8000 (run this window as Administrator once):
echo   netsh advfirewall firewall add rule name="PathSnap API 8000" dir=in action=allow protocol=TCP localport=8000

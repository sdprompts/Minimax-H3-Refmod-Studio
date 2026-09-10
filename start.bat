@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
  set "BOOTSTRAP=py -3"
) else (
  set "BOOTSTRAP=python"
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %BOOTSTRAP% -m venv .venv
  if errorlevel 1 (
    echo Failed to create .venv. Install Python 3 and try again.
    pause
    exit /b 1
  )
)

echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo pip install failed.
  pause
  exit /b 1
)

echo Starting H3 RefMod Studio...
echo Close this window or press Ctrl+C to stop.
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause

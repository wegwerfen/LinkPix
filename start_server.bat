@echo off
setlocal ENABLEEXTENSIONS

REM Change to directory of this script
cd /d "%~dp0"

echo.
echo Starting Image Generation Server (Flask + Gradio)
echo.

REM Activate virtual environment if available
if exist "venv\Scripts\activate.bat" (
  echo Activating virtual environment...
  call "venv\Scripts\activate.bat"
) else (
  echo [Info] No virtual environment found at venv\Scripts\activate.bat
  echo        Continuing with system Python.
)

REM Run the Python launcher (starts Flask API and Gradio UI)
python start_server.py

endlocal

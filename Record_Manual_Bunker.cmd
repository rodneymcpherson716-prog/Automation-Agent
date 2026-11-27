@echo off
setlocal
cd /d "%~dp0"
title Automation Read-Only Manual Terrain Recorder
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo ERROR: The Python environment is missing.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -B -u "%~dp0manual_terrain_recorder.py" %*
set "RECORDER_EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %RECORDER_EXIT_CODE%

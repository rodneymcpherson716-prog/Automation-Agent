@echo off
setlocal
cd /d "%~dp0"
title Automation Execution Engine - F9 Stop

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo ERROR: The Python environment is missing.
    echo Expected: %~dp0.venv\Scripts\python.exe
    pause
    exit /b 1
)

"%~dp0.venv\Scripts\python.exe" -B -u "%~dp0execution_engine.py" %*
set "AGENT_EXIT_CODE=%ERRORLEVEL%"
if not "%AGENT_EXIT_CODE%"=="0" (
    echo.
    echo Automation stopped. Read the reason above. No automatic retry will be made.
)
if /i not "%~1"=="--check" if /i not "%~1"=="--observe" pause
exit /b %AGENT_EXIT_CODE%

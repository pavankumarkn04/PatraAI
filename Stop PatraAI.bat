@echo off
title Stop PatraAI
setlocal
cd /d "%~dp0"
echo Stopping PatraAI...
where py >nul 2>nul
if errorlevel 1 (
    python "%~dp0run_app.py" --stop
) else (
    py -3.11 "%~dp0run_app.py" --stop
)
echo To start it again, run "Start PatraAI.bat".
rem A short pause so the message can be read. ping works in any window;
rem timeout fails when VS Code runs this file.
ping -n 4 127.0.0.1 >nul 2>nul
exit /b 0

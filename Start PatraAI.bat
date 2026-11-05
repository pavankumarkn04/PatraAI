@echo off
title PatraAI
setlocal
cd /d "%~dp0"
echo Starting PatraAI... (any older PatraAI server is stopped first)
echo Keep this window open while you use the app. Close it, or run "Stop PatraAI.bat", to stop.
echo.

where py >nul 2>nul
if errorlevel 1 (
    python "%~dp0run_app.py"
) else (
    py -3.11 "%~dp0run_app.py"
)

echo.
echo PatraAI has stopped.
rem Never wait forever: tools such as VS Code Code Runner cannot press a key.
timeout /t 15 >nul 2>nul

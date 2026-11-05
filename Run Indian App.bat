@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    python "%~dp0run_app.py"
) else (
    py -3.11 "%~dp0run_app.py"
)

if errorlevel 1 timeout /t 15 >nul 2>nul

@echo off
title PatraAI evaluation dashboard
setlocal
cd /d "%~dp0"
echo Starting the PatraAI evaluation dashboard (developer tool, separate from the website)...
where py >nul 2>nul
if errorlevel 1 (
    python -m streamlit run "%~dp0evaluation\dashboard.py" --server.port 8502 --server.address 127.0.0.1 --browser.gatherUsageStats false
) else (
    py -3.11 -m streamlit run "%~dp0evaluation\dashboard.py" --server.port 8502 --server.address 127.0.0.1 --browser.gatherUsageStats false
)
pause

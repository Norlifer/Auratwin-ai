@echo off
title AuraTwin AI - Digital Twin & Virtual HVAC Engine
echo ================================================================
echo    Starting AuraTwin AI Building Digital Twin & Virtual HVAC
echo ================================================================
echo.

IF EXIST "C:\Users\HP\anaconda3\python.exe" (
    "C:\Users\HP\anaconda3\python.exe" run_server.py
) ELSE (
    python run_server.py
)

pause

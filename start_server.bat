@echo off
title AI Stock Predictor Backend Server
echo ========================================================
echo   Starting AI Stock Predictor Server (FastAPI + XGBoost)
echo   Dashboard URL: http://localhost:8000/
echo   API Docs URL:  http://localhost:8000/docs
echo ========================================================
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
) else (
    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
)
pause

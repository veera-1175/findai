@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat 2>nul
if errorlevel 1 (
    echo Run setup.bat first.
    pause
    exit /b 1
)
echo Starting FindAI at http://localhost:8000
python run.py

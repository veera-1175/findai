@echo off
cd /d "%~dp0"
echo === FindAI Setup ===

python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
echo Installing dependencies...
pip install -r requirements.txt -q

if not exist .env copy .env.example .env

echo.
echo Setup complete! Run start.bat to launch FindAI.
pause

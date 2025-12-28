@echo off
REM Setup script for CL-CoT project (Windows)

echo ==========================================
echo CL-CoT Project Setup
echo ==========================================

REM Check Python
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Create directories
echo.
echo Creating directories...
if not exist "data\processed" mkdir data\processed
if not exist "data\cache" mkdir data\cache
if not exist "experiments" mkdir experiments
if not exist "logs" mkdir logs
if not exist "results" mkdir results

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo To activate the environment:
echo   venv\Scripts\activate.bat
echo.
echo To start training:
echo   python scripts\train_stage1.py --dataset wtq --output_dir experiments\stage1\wtq
echo.
pause

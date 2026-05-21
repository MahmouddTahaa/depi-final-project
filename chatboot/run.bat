@echo off
REM ChatBoot launcher - Windows.
REM This starts the app only. Use setup_once.bat if dependencies are missing.
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
  echo.
  echo No local virtual environment was found.
  echo Run setup_once.bat one time, then run this file again.
  echo.
  pause
  exit /b 1
)

call "venv\Scripts\activate.bat"

echo.
echo Launching ChatBoot at http://localhost:8501
echo.
python -m streamlit run streamlit_app.py

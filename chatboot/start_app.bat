@echo off
REM Start ChatBoot without reinstalling dependencies every time.
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
  echo.
  echo No virtual environment was found.
  echo Run setup_once.bat first, then run this file again.
  echo.
  pause
  exit /b 1
)

call "venv\Scripts\activate.bat"

python -c "import streamlit" >nul 2>nul
if errorlevel 1 (
  echo.
  echo Streamlit is not installed in this venv.
  echo Run setup_once.bat one time, then use start_app.bat after that.
  echo.
  pause
  exit /b 1
)

echo.
echo Starting ChatBoot...
echo.
python -m streamlit run streamlit_app.py

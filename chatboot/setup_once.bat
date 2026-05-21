@echo off
REM Install ChatBoot dependencies one time only.
cd /d "%~dp0"

if not exist "venv" (
  echo Creating virtual environment...
  python -m venv venv
)

call "venv\Scripts\activate.bat"

echo.
echo Installing dependencies. This can take several minutes, but you only need to do it once.
echo.
python -m pip install --upgrade pip --no-cache-dir
python -m pip install -r requirements.txt --no-cache-dir
python -m spacy download en_core_web_sm

echo.
echo Setup complete. From now on, run start_app.bat instead of installing again.
echo.
pause

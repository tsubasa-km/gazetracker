@echo off
rem First-time setup: create .venv, install requirements, download the MediaPipe model.
cd /d "%~dp0"
if not exist .venv (
  py -3.11 -m venv .venv || py -3 -m venv .venv || goto :err
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || goto :err
python download_model.py || goto :err
echo.
echo Setup finished. Next: run app.bat (or double-click GazeTracker.pyw).
pause
exit /b 0
:err
echo Setup failed.
pause
exit /b 1

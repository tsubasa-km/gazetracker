@echo off
rem Start the GUI with a console window (useful to see errors).
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python app.py
if errorlevel 1 pause

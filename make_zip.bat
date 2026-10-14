@echo off
rem Create the distribution zip in release\ (run build_exe.bat first).
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python make_zip.py
pause

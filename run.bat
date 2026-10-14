@echo off
rem Usage: run.bat [profile]
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set PROFILE=%1
if "%PROFILE%"=="" set PROFILE=default
python run.py --profile %PROFILE% --preview
pause

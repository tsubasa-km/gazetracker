@echo off
rem Usage: calibrate.bat [profile] [monitor]
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set PROFILE=%1
if "%PROFILE%"=="" set PROFILE=default
set MON=%2
if "%MON%"=="" set MON=0
python calibrate.py --list-monitors
python calibrate.py --profile %PROFILE% --monitor %MON% --passes 5
pause

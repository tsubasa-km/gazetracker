@echo off
rem Sign dist\GazeTracker\GazeTracker.exe with the self-signed code-signing certificate
rem (created in the current user's Personal store if missing). Exports cert\GazeTracker.cer.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0cert\sign.ps1" -Exe "%~dp0dist\GazeTracker\GazeTracker.exe"
if errorlevel 1 (
  echo Signing failed.
  pause
  exit /b 1
)

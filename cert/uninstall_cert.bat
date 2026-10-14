@echo off
rem Remove the certificate added by install_cert.bat (run as administrator).
net session >nul 2>&1
if errorlevel 1 (
  echo Please run this file as administrator.
  pause
  exit /b 1
)
certutil -delstore Root "GazeTracker Self-Signed"
certutil -delstore TrustedPublisher "GazeTracker Self-Signed"
pause

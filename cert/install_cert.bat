@echo off
rem Trust the GazeTracker signature on another PC. Right-click and "Run as administrator".
rem Adds GazeTracker.cer to "Trusted Root Certification Authorities" and "Trusted Publishers".
cd /d "%~dp0"
net session >nul 2>&1
if errorlevel 1 (
  echo Please run this file as administrator: right-click, "Run as administrator".
  pause
  exit /b 1
)
certutil -addstore -f Root "%~dp0GazeTracker.cer" || goto :err
certutil -addstore -f TrustedPublisher "%~dp0GazeTracker.cer" || goto :err
echo.
echo Installed. GazeTracker.exe - Properties - Digital Signatures now shows "GazeTracker Self-Signed" as valid.
echo If SmartScreen appears on first launch, click "More info" then "Run anyway" once.
pause
exit /b 0
:err
echo Failed to install the certificate.
pause
exit /b 1

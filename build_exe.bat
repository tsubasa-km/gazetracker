@echo off
rem Build dist\GazeTracker\GazeTracker.exe with PyInstaller (run setup.bat first), then sign it.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m pip install -q pyinstaller || goto :err
if not exist models\face_landmarker.task python download_model.py
pyinstaller --noconfirm --clean --windowed --name GazeTracker ^
  --collect-all customtkinter ^
  --collect-data mediapipe ^
  --collect-submodules pygrabber ^
  --collect-submodules comtypes ^
  --collect-submodules keyboard ^
  --add-data "%~dp0models;models" ^
  --add-data "%~dp0overlay;overlay" ^
  "%~dp0GazeTracker.py" || goto :err
call "%~dp0sign_exe.bat"
echo.
echo Done: dist\GazeTracker\GazeTracker.exe  (copy the whole folder to distribute)
echo On another PC, run cert\install_cert.bat as administrator to trust the signature.
pause
exit /b 0
:err
echo Build failed.
pause
exit /b 1

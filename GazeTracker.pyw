"""ダブルクリックでコンソールなしに GUI を起動する。

システムの Python で開かれた場合は、.venv の pythonw.exe で自分を起動し直す
(依存ライブラリは .venv に入っているため)。
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV_PYW = HERE / ".venv" / "Scripts" / "pythonw.exe"

if VENV_PYW.exists() and Path(sys.executable).resolve() != VENV_PYW.resolve():
    subprocess.Popen([str(VENV_PYW), str(HERE / "GazeTracker.py")], cwd=str(HERE),
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    sys.exit(0)

if not VENV_PYW.exists():
    import ctypes
    ctypes.windll.user32.MessageBoxW(0, "先に setup.bat を実行してください。", "Gaze Tracker", 0x10)
    sys.exit(1)

os.chdir(HERE)
sys.path.insert(0, str(HERE))
import GazeTracker  # noqa: E402

GazeTracker.main()

"""ファイル配置。通常実行と PyInstaller で固めた exe の両方で動くようにする。

- BUNDLE: 同梱リソース (モデル、オーバーレイ HTML)。exe では展開先
- APP_DIR: ユーザーデータ (プロファイル、ログ、設定)。exe では exe と同じフォルダ
"""
from __future__ import annotations

import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))
_SRC_ROOT = Path(__file__).resolve().parent.parent
BUNDLE = Path(getattr(sys, "_MEIPASS", _SRC_ROOT))
APP_DIR = Path(sys.executable).resolve().parent if FROZEN else _SRC_ROOT

MODEL_PATH = BUNDLE / "models" / "face_landmarker.task"
OVERLAY_DIR = BUNDLE / "overlay"
PROFILE_DIR = APP_DIR / "profiles"
LOG_DIR = APP_DIR / "logs"
SETTINGS_PATH = APP_DIR / "settings.json"


def subcommand(name: str, *args: str) -> list[str]:
    """calibrate / heatmap を子プロセスで起動するためのコマンド列。"""
    if FROZEN:
        return [sys.executable, name, *args]
    return [sys.executable, str(_SRC_ROOT / f"{name}.py"), *args]

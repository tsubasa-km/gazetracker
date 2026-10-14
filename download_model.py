"""MediaPipe Face Landmarker のモデルファイルを models/ にダウンロードする。"""
import urllib.request
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracker.paths import MODEL_PATH  # noqa: E402

URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
       "face_landmarker/float16/1/face_landmarker.task")
DEST = MODEL_PATH

if __name__ == "__main__":
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists():
        print(f"既に存在します: {DEST}")
    else:
        print(f"ダウンロード中: {URL}")
        urllib.request.urlretrieve(URL, DEST)
        print(f"保存しました: {DEST} ({DEST.stat().st_size // 1024} KB)")

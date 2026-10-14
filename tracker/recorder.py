"""検証用の録画: カメラ映像 (MJPG の AVI) と、各フレームの時刻・表示中の点の位置 (JSONL) を保存する。

あとで analyze_recording.py で映像を処理し直し、特徴量・フィルター・モデルの違いを
同じデータで比べるために使う。録画は別スレッドで書き出すので、キャリブレーションの表示は遅くならない。

保存先: recordings/<日時>/
  video.avi     カメラ映像 (撮ったままの解像度)
  frames.jsonl  1 行 1 フレーム: {"i": 番号, "t": 撮影時刻, "phase": 段階, "tx": 点の x, "ty": 点の y, ...}
  meta.json     画面サイズ・カメラ・開始時刻など
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import cv2

from .paths import APP_DIR

RECORD_DIR = APP_DIR / "recordings"


class Recorder:
    def __init__(self, screen_w: int, screen_h: int, cam_desc: str, frame_size: tuple[int, int],
                 fps: float = 30.0, extra: dict | None = None):
        self.dir = RECORD_DIR / time.strftime("%Y%m%d_%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        w, h = frame_size
        # FFmpeg を同梱しないので OpenCV 内蔵の MJPEG 書き出しを使う
        self._vw = cv2.VideoWriter(str(self.dir / "video.avi"), cv2.CAP_OPENCV_MJPEG,
                                   cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        if not self._vw.isOpened():
            raise RuntimeError("録画ファイルを作れません")
        self._log = open(self.dir / "frames.jsonl", "w", encoding="utf-8")
        meta = {"screen_w": screen_w, "screen_h": screen_h, "camera": cam_desc, "frame_w": w, "frame_h": h,
                "start_unix": time.time(), "format": 1}
        meta.update(extra or {})
        (self.dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        self._q: queue.Queue = queue.Queue(maxsize=240)
        self.n = 0
        self.dropped = 0
        self._last_t = None
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def add(self, frame, t: float, phase: str, target=None, **info):
        """新しいフレームだけを記録する (同じ時刻のフレームは無視)。target は正規化座標 (x, y) か None。"""
        if frame is None or t == self._last_t:
            return
        self._last_t = t
        rec = {"i": self.n, "t": round(t, 4), "phase": phase}
        if target is not None:
            rec["tx"], rec["ty"] = round(float(target[0]), 5), round(float(target[1]), 5)
        rec.update(info)
        try:
            self._q.put_nowait((frame, rec))
            self.n += 1
        except queue.Full:
            self.dropped += 1

    def _loop(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            frame, rec = item
            self._vw.write(frame)
            self._log.write(json.dumps(rec) + "\n")

    def close(self):
        self._q.put(None)
        self._th.join(timeout=30)
        self._vw.release()
        self._log.close()
        info = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        info.update(frames=self.n, dropped=self.dropped, end_unix=time.time())
        (self.dir / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        return self.dir


class NullRecorder:
    """録画しないときの代わり。"""
    dir = None

    def add(self, *a, **k):
        pass

    def close(self):
        return None

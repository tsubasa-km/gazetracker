"""Web カメラを別スレッドで読み続け、常に最新フレームを返す。"""
from __future__ import annotations

import threading
import time

import cv2


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        # 解像度が高いほど虹彩の精度は上がるが、安いカメラは 720p で 10fps しか出ないことがある
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"カメラ {index} を開けません")
        self._frame = None
        self._ts = 0.0
        self._lock = threading.Lock()
        self._stop = False
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        t0 = time.time()
        while self._frame is None and time.time() - t0 < 5:
            time.sleep(0.01)
        if self._frame is None:
            raise RuntimeError("カメラからフレームが来ません")

    def _loop(self):
        while not self._stop:
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = f
                    self._ts = time.time()
            else:
                time.sleep(0.005)

    def read(self):
        """(frame, unix_time_sec) を返す。"""
        with self._lock:
            return self._frame, self._ts

    def close(self):
        self._stop = True
        self._th.join(timeout=1)
        self.cap.release()

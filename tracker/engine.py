"""リアルタイム視線推定エンジン (CLI と GUI の両方から使う)。

- カメラ → 特徴量 → 回帰 → 平滑化 を別スレッドで回す
- WebSocket で座標を配信、HTTP でオーバーレイ HTML を配信
- JSONL に記録
"""
from __future__ import annotations

import asyncio
import functools
import http.server
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import websockets

from .camera import Camera
from .features import FeatureExtractor, HeadSmoother, draw_face
from .filters import OneEuro
from .model import Profile
from .paths import LOG_DIR, OVERLAY_DIR


class Broadcaster:
    def __init__(self):
        self.clients: set = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self._server = None
        self._stop_evt: asyncio.Event | None = None

    async def handler(self, ws):
        self.clients.add(ws)
        try:
            async for _ in ws:
                pass
        finally:
            self.clients.discard(ws)

    async def _send_all(self, msg: str):
        if self.clients:
            await asyncio.gather(*(c.send(msg) for c in list(self.clients)), return_exceptions=True)

    def send(self, msg: str):
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._send_all(msg), self.loop)

    async def serve(self, port: int):
        self.loop = asyncio.get_running_loop()
        self._stop_evt = asyncio.Event()
        async with websockets.serve(self.handler, "127.0.0.1", port):
            await self._stop_evt.wait()

    def stop(self):
        if self.loop and self._stop_evt:
            self.loop.call_soon_threadsafe(self._stop_evt.set)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a, **k):
        pass


def start_http(port: int):
    handler = functools.partial(_QuietHandler, directory=str(OVERLAY_DIR))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@dataclass
class Status:
    running: bool = False
    fps: float = 0.0
    face: bool = False
    blink: bool = False
    valid: bool = False
    x: float = 0.0
    y: float = 0.0
    clients: int = 0
    log_path: str = ""
    cam_desc: str = ""          # 実際に取れたカメラの解像度・fps・形式
    started: bool = False      # カメラとモデルの初期化が終わったか
    error: str = ""
    profile: str = ""
    cv_error_px: float = 0.0
    offset_x: float = 0.0       # ワンキー補正のずれ量 (正規化座標)
    offset_y: float = 0.0
    recentering: bool = False   # 補正のためのサンプル収集中


class GazeEngine:
    def __init__(self, profile: str, camera: int = 0, ws_port: int = 8765, http_port: int = 8766,
                 log: bool = True, min_cutoff: float = 1.0, beta: float = 0.05,
                 cam_w: int = 0, cam_h: int = 0, hotkey: str = "f8", reset_hotkey: str = "f9",
                 beep: bool = True):
        self.profile_name = profile
        self.hotkey, self.reset_hotkey, self.beep = hotkey, reset_hotkey, beep
        self._offset = (0.0, 0.0)
        self._recenter_until = 0.0
        self._recenter_buf: list[tuple[float, float]] = []
        self._hotkey_handles = []
        self.camera_index = camera
        self.cam_w, self.cam_h = cam_w, cam_h
        self.ws_port, self.http_port = ws_port, http_port
        self.log_enabled = log
        self.min_cutoff, self.beta = min_cutoff, beta
        self.status = Status()
        self._frame_lock = threading.Lock()
        self._latest_vis: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._bc: Broadcaster | None = None
        self._http = None
        self._fx = None
        self._fy = None

    # ---- 公開 API ----
    @property
    def overlay_url(self) -> str:
        return f"http://localhost:{self.http_port}/?ws={self.ws_port}"

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.status = Status(running=True, profile=self.profile_name)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None

    # --- ワンキー補正 ---
    def request_recenter(self, window: float = 0.5):
        """呼ばれてから window 秒間の生の推定の中央値を画面中央とみなし、ずれを差し引く。
        Splatoon は照準が常に画面中央なので、照準を見ながら押せばよい。"""
        if not self.status.started:
            return
        self._recenter_buf = []
        self._recenter_until = time.time() + window
        self.status.recentering = True

    def clear_offset(self):
        self._offset = (0.0, 0.0)
        self.status.offset_x = self.status.offset_y = 0.0
        self._beep(400, 120)

    def _finish_recenter(self):
        self.status.recentering = False
        if len(self._recenter_buf) < 5:
            self._beep(300, 300)          # 顔が取れておらず補正できなかった
            return
        arr = np.array(self._recenter_buf)
        mx, my = np.median(arr, axis=0)
        self._offset = (0.5 - float(mx), 0.5 - float(my))
        self.status.offset_x, self.status.offset_y = self._offset
        for f in (self._fx, self._fy):
            if f:
                f.reset()
        self._beep(900, 80)
        self._beep(1200, 80)

    def _beep(self, freq: int, ms: int):
        if not self.beep:
            return
        try:
            import winsound
            winsound.Beep(freq, ms)
        except Exception:
            pass

    def _install_hotkeys(self):
        try:
            import keyboard
        except Exception:
            return
        for key, fn in ((self.hotkey, self.request_recenter), (self.reset_hotkey, self.clear_offset)):
            if key:
                try:
                    self._hotkey_handles.append(keyboard.add_hotkey(key, fn))
                except Exception as e:
                    self.status.error = f"ホットキー '{key}' を登録できません: {e}"

    def _remove_hotkeys(self):
        try:
            import keyboard
            for h in self._hotkey_handles:
                keyboard.remove_hotkey(h)
        except Exception:
            pass
        self._hotkey_handles = []

    def set_smoothing(self, min_cutoff: float, beta: float):
        self.min_cutoff, self.beta = min_cutoff, beta
        for f in (self._fx, self._fy):
            if f:
                f.min_cutoff, f.beta = min_cutoff, beta

    def latest_frame(self) -> np.ndarray | None:
        """推定点を描き込んだ最新のカメラ映像 (BGR)。"""
        with self._frame_lock:
            return None if self._latest_vis is None else self._latest_vis.copy()

    # ---- 内部 ----
    def _run(self):
        st = self.status
        log = cam = fx = None
        try:
            prof = Profile.load(self.profile_name)
            st.cv_error_px = prof.cv_error_px

            self._bc = Broadcaster()
            threading.Thread(target=lambda: asyncio.run(self._bc.serve(self.ws_port)), daemon=True).start()
            self._http = start_http(self.http_port)

            if self.log_enabled:
                LOG_DIR.mkdir(exist_ok=True)
                path = LOG_DIR / time.strftime("gaze_%Y%m%d_%H%M%S.jsonl")
                log = open(path, "w", encoding="utf-8", buffering=1)
                log.write(json.dumps({"type": "header", "profile": prof.name, "screen_w": prof.screen_w,
                                      "screen_h": prof.screen_h, "start_unix": time.time()}) + "\n")
                st.log_path = str(path)

            cam = Camera(self.camera_index, self.cam_w, self.cam_h)
            st.cam_desc = cam.description
            fx = FeatureExtractor()
            st.started = True
            self._fx = OneEuro(self.min_cutoff, self.beta)
            self._fy = OneEuro(self.min_cutoff, self.beta)
            head_smooth = HeadSmoother()
            self._install_hotkeys()
            last_ts = 0.0
            n, t_fps = 0, time.time()

            while not self._stop.is_set():
                frame, ts = cam.read()
                if ts == last_ts:
                    time.sleep(0.002)
                    continue
                last_ts = ts
                s = fx.process(frame, int(ts * 1000))
                valid = s.ok and not s.blink
                msg = {"t": int(ts * 1000), "valid": valid, "blink": bool(s.blink),
                       "face": bool(s.ok), "fps": round(st.fps, 1)}
                if valid:
                    px, py = prof.predict(head_smooth(s.feat))
                    if st.recentering:
                        self._recenter_buf.append((px, py))
                    rx, ry = px + self._offset[0], py + self._offset[1]
                    x, y = self._fx(rx, ts), self._fy(ry, ts)
                    msg.update(x=round(x, 4), y=round(y, 4), rx=round(rx, 4), ry=round(ry, 4))
                    st.x, st.y = x, y
                if st.recentering and time.time() >= self._recenter_until:
                    self._finish_recenter()
                st.face, st.blink, st.valid = bool(s.ok), bool(s.blink), valid
                line = json.dumps(msg)
                self._bc.send(line)
                if log:
                    log.write(line + "\n")
                st.clients = len(self._bc.clients)

                n += 1
                if time.time() - t_fps >= 1.0:
                    st.fps = n / (time.time() - t_fps)
                    n, t_fps = 0, time.time()

                vw = 640
                vh = int(vw * frame.shape[0] / frame.shape[1])   # アスペクト比を保つ
                vis = cv2.resize(frame, (vw, vh))
                draw_face(vis, s, vw / frame.shape[1])
                if valid:
                    cv2.circle(vis, (int(st.x * vw), int(st.y * vh)), 10, (0, 200, 255), 2, cv2.LINE_AA)
                with self._frame_lock:
                    self._latest_vis = vis
        except Exception as e:  # GUI に見せるため握りつぶさず記録
            st.error = f"{type(e).__name__}: {e}"
        finally:
            st.running = False
            self._remove_hotkeys()
            if log:
                log.close()
            if self._bc:
                self._bc.stop()
            if self._http:
                self._http.shutdown()
                self._http.server_close()
            if fx:
                fx.close()
            if cam:
                cam.close()

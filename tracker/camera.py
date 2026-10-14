"""Web カメラを別スレッドで読み続け、常に最新フレームを返す。

解像度と形式の決め方:
  1. ドライバ (DirectShow) に対応モードの一覧 (形式・解像度・fps) を問い合わせ、
     MIN_FPS 以上出る中で最も解像度の高いモードを選び、その形式を指定して開く。
     例: 多くの USB カメラは 720p を MJPG なら 30fps、非圧縮 (YUY2) なら 10fps で出す。
     形式を指定しないと YUY2 が選ばれて遅くなるので、形式まで指定するのが重要。
  2. 一覧を返さないカメラ (配信ソフトの仮想カメラなど) は、720p → 480p の順に
     実際に開いて fps を測り、MIN_FPS 以上出る最大の解像度を使う。
"""
from __future__ import annotations

import threading
import time

import cv2

AUTO = (0, 0)            # 解像度の自動選択を表す
AUTO_LADDER = [(1280, 720), (640, 480)]   # 一覧を返さないカメラで、自動のとき上から試す
MIN_FPS = 20             # これ未満の fps は「出ていない」とみなして次を試す


def _open(index: int, width: int, height: int, fps: int, fourcc):
    """形式・解像度・fps を開くときにまとめて渡す。
    開いた後に set() で変えると、DirectShow が形式を非圧縮に戻して 720p が 10fps になるカメラがある。"""
    params = []
    if fourcc:
        params += [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4].ljust(4))]
    params += [cv2.CAP_PROP_FRAME_WIDTH, width, cv2.CAP_PROP_FRAME_HEIGHT, height, cv2.CAP_PROP_FPS, fps]
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW, params)
    except TypeError:              # params を受け付けない古い OpenCV
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4].ljust(4)))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def com_init():
    """このスレッドで COM を初期化する (DirectShow の問い合わせに必要)。
    カメラは画面とは別のスレッドで調べるため、スレッドごとに呼ぶ必要がある。初期化済みなら何もしない。"""
    try:
        import comtypes
        comtypes.CoInitializeEx(comtypes.COINIT_APARTMENTTHREADED)
    except OSError:
        pass          # 別の方式で初期化済み (それでも問い合わせはできる)
    except Exception:
        pass


def list_modes(index: int):
    """ドライバが申告する対応モード [(形式, 幅, 高さ, 最大fps), ...]。問い合わせできなければ None。"""
    try:
        com_init()
        from pygrabber.dshow_graph import FilterGraph
        g = FilterGraph()
        g.add_video_input_device(index)
        fmts = g.get_input_device().get_formats()
    except Exception:
        return None
    modes = set()
    for f in fmts:
        try:
            fps = max(float(f.get("max_framerate") or 0), float(f.get("min_framerate") or 0))
            modes.add((str(f["media_type_str"]), int(f["width"]), int(f["height"]), round(fps, 1)))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(modes, key=lambda m: (-m[1] * m[2], -m[3])) or None


MAX_PIXELS = 1280 * 720        # これより大きいと処理が重いだけで精度は上がらない (MediaPipe は顔を 256px 程度に縮める。
                               # 実録画で 1080p と 640x480 相当を比べても誤差は同等だった)
_PREFER = {"NV12": 3, "YUY2": 2, "MJPG": 1}   # 同じ解像度・fps なら展開の軽い非圧縮を優先


def best_mode(modes):
    """MIN_FPS 以上出る中で最も解像度の高いモード。無ければ最も速いモード。"""
    usable = [m for m in modes if m[1] * m[2] <= MAX_PIXELS and m[0] in _PREFER]
    fast = [m for m in usable if m[3] >= MIN_FPS]
    if fast:
        return max(fast, key=lambda m: (m[1] * m[2], m[3], _PREFER[m[0]]))
    if usable:
        return max(usable, key=lambda m: (m[3], m[1] * m[2], _PREFER[m[0]]))
    return None


def _fourcc_str(cap, requested=None) -> str:
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    b = bytes((v >> (8 * i)) & 0xFF for i in range(4))
    if all(32 <= c < 127 for c in b) and b.strip():
        return b.decode("ascii")
    return requested or "?"      # 読み返しが壊れて返るドライバがある


def _measure(cap, seconds: float = 0.5):
    """数フレーム読んで (幅, 高さ, fps) を返す。立ち上がりの数枚は捨てる。"""
    f = None
    for _ in range(5):
        ok, g = cap.read()
        if ok:
            f = g
    if f is None:
        return None
    t0, n = time.time(), 0
    while time.time() - t0 < seconds:
        ok, g = cap.read()
        if ok:
            n += 1
            f = g
    return f.shape[1], f.shape[0], n / max(time.time() - t0, 1e-6)


def _score(m):
    """良さの比較キー: MIN_FPS 以上出ること → 画素数 → fps"""
    return (m[2] >= MIN_FPS, m[0] * m[1], m[2])


def _try(index: int, width: int, height: int, fps: int):
    """MJPG → 形式指定なし の順で実際に開いて測り、良い方を開いたまま返す: (cap, (w, h, fps), 形式) か None。
    同じカメラを同時に 2 回は開かない。"""
    cap = _open(index, width, height, fps, "MJPG")
    if cap is None:
        return None
    m = _measure(cap)
    if m is not None and m[2] >= MIN_FPS:
        return cap, m, "MJPG"
    cap.release()
    cap2 = _open(index, width, height, fps, None)
    m2 = _measure(cap2) if cap2 is not None else None
    if m2 is not None and (m is None or _score(m2) >= _score(m)):
        return cap2, m2, None
    if cap2 is not None:
        cap2.release()
    if m is None:
        return None
    cap = _open(index, width, height, fps, "MJPG")
    m = _measure(cap) if cap is not None else None
    return (cap, m, "MJPG") if m is not None else None


def _try_mode(index: int, mode):
    """ドライバの申告どおりのモードで開き、実際にその解像度で十分な fps が出るか確かめる。"""
    fmt, w, h, fps = mode
    cap = _open(index, w, h, int(round(fps)), fmt)
    if cap is None:
        return None
    m = _measure(cap)
    if m is None or (m[0], m[1]) != (w, h) or m[2] < min(MIN_FPS, fps) * 0.75:
        cap.release()
        return None
    return cap, m, fmt


def probe(index: int):
    """カメラが自動で選ぶ解像度を調べる。(幅, 高さ, fps, 形式) か None。
    ドライバが一覧を返せばカメラを開かずに答える (速い)。返さなければ実際に開いて測る。"""
    modes = list_modes(index)
    if modes:
        m = best_mode(modes)
        if m is not None:
            return m[1], m[2], m[3], m[0]
    cam = None
    try:
        cam = Camera(index)
        return cam.width, cam.height, cam.fps, cam.fourcc
    except Exception:
        return None
    finally:
        if cam is not None:
            cam.close()


class Camera:
    def __init__(self, index: int = 0, width: int = 0, height: int = 0, fps: int = 30):
        """width/height が 0 (自動) なら 720p → 480p の順に試し、MIN_FPS 以上出る最大の解像度を使う。
        指定があればその解像度を要求する (出せなければ近いものになる)。
        実際の値は self.width / self.height / self.fps / self.fourcc で分かる。"""
        self.auto = not (width and height)
        self.source = "測定"
        best = None
        if self.auto:
            modes = list_modes(index)
            mode = best_mode(modes) if modes else None
            if mode is not None:
                best = _try_mode(index, mode)
                if best is not None:
                    self.source = "ドライバ"
        if best is None:
            ladder = AUTO_LADDER if self.auto else [(width, height)]
            for w, h in ladder:
                if best is not None and w * h >= best[1][0] * best[1][1]:
                    continue          # すでにこれ以下の解像度しか出ないと分かっている
                r = _try(index, w, h, fps)
                if r is None:
                    if best is None and (w, h) == ladder[0]:
                        raise RuntimeError(f"カメラ {index} を開けません")
                    continue
                if best is None or _score(r[1]) > _score(best[1]):
                    if best is not None:
                        best[0].release()
                    best = r
                else:
                    r[0].release()
                if best[1][2] >= MIN_FPS:
                    break
        if best is None:
            raise RuntimeError("カメラからフレームが来ません")
        cap, m, fmt = best
        self.cap = cap
        self.requested = (width, height)
        self.width, self.height, self.fps = m[0], m[1], m[2]
        self.fourcc = _fourcc_str(cap, fmt)

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

    @property
    def description(self) -> str:
        s = f"{self.width}x{self.height} {self.fps:.0f}fps {self.fourcc}"
        if self.auto:
            s += f" (自動・{self.source})"
        elif (self.width, self.height) != tuple(self.requested):
            s += f" (要求 {self.requested[0]}x{self.requested[1]})"
        return s

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


def parse_res(text: str) -> tuple[int, int]:
    """'1280x720' → (1280, 720)。'auto' / '自動' などは AUTO (自動選択)。"""
    t = (text or "").lower()
    if "x" in t:
        try:
            w, h = t.split("x", 1)
            return int(w), int(h.split()[0])
        except ValueError:
            pass
    return AUTO

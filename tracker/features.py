"""MediaPipe Face Landmarker から視線推定用の特徴量を取り出す。

特徴量の中心は「眼球モデルで求めた視線の 3D 方向」:
  MediaPipe の頭部姿勢行列で標準顔モデルの眼球中心をカメラ座標に置き、
  虹彩の画像位置から伸ばした視線 (レイ) と眼球 (球) の交点を虹彩の 3D 位置とみなし、
  眼球中心 → 虹彩 の向きを視線方向とする。頭の回転・位置はここで物理的に扱われる。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2

from . import _mp_compat  # noqa: F401  (mediapipe より先に)
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .paths import MODEL_PATH

# ランドマーク番号 (MediaPipe Face Mesh 478点)
# 画像上で左にある目 (被写体の右目): 目尻 33, 目頭 133 / 画像上で右にある目: 目頭 362, 目尻 263
LEFT_EYE = dict(a=33, b=133, up=159, down=145, iris=list(range(468, 473)), center=468)
RIGHT_EYE = dict(a=362, b=263, up=386, down=374, iris=list(range(473, 478)), center=473)
NOSE = 1

# MediaPipe 標準顔モデル (cm, +z が顔の前方) の目尻・目頭の中点から、眼球中心を少し奥に置く
_CANON_L_CORNERS = np.array([[-4.445859, 2.663991, 3.173422], [-1.856432, 2.585245, 3.757904]])
_CANON_R_CORNERS = np.array([[4.445859, 2.663991, 3.173422], [1.856432, 2.585245, 3.757904]])
EYE_BACK_CM = 0.9          # 目尻・目頭の中点から眼球中心までの奥行き
EYE_RADIUS_CM = 1.2        # 眼球半径
CANON_EYE_L = _CANON_L_CORNERS.mean(0) - [0, 0, EYE_BACK_CM]
CANON_EYE_R = _CANON_R_CORNERS.mean(0) - [0, 0, EYE_BACK_CM]
FOVY_DEG = 63.0            # MediaPipe face geometry が仮定する縦画角

FEATURE_NAMES = [
    "l_h", "l_v", "r_h", "r_v",          # 虹彩の眼窩内での相対位置 (右が正, 下が正)
    "l_open", "r_open",                  # 目の開き具合 (まばたき検出に使う)
    "yaw", "pitch", "roll",              # 頭部回転 (rad)
    "tx", "ty", "tz",                    # 頭部位置 (cm 相当)
    "face_x", "face_y", "face_scale",    # 画像内の顔位置と大きさ
    "g_ta", "g_te",                      # 視線方向 (カメラ座標): tan(方位角), tan(仰角)  右・上が正
    "o_x", "o_y", "o_z",                 # 両目の中点の位置 (cm 相当, o_z はカメラからの距離)
    "gl_ta", "gl_te", "gr_ta", "gr_te",  # 左右それぞれの視線方向
]
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
HEAD_IDX = [IDX[n] for n in ("yaw", "pitch", "roll", "tx", "ty", "tz", "o_x", "o_y", "o_z")]


COMPACT_NAMES = ["l_h", "l_v", "r_h", "r_v", "l_open", "r_open", "yaw", "pitch", "tx", "ty", "tz"]
COMPACT_IDX = [IDX[n] for n in COMPACT_NAMES]


def compact_features(X: np.ndarray) -> np.ndarray:
    """回帰に使う 11 次元: 左右それぞれの虹彩位置 (h, v) とまぶたの開き + 頭の向き・位置。

    虹彩位置は同じフレームの目尻・目頭を基準にした 2D の相対値なので、頭部姿勢の推定誤差に
    引きずられにくい。まぶたの開きは縦方向の視線と強く相関する (下を見ると閉じ気味になる)。
    (3D 眼球モデルの g_ta/g_te は眼球半径 1.2cm に対して頭部位置の誤差が大きすぎて
    実用にならなかったため、解析用の列としてのみ残している)"""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None]
    return X[:, COMPACT_IDX]


@dataclass
class Sample:
    feat: np.ndarray            # len(FEATURE_NAMES) 次元
    blink: bool
    ok: bool
    eye_pts: np.ndarray | None = None   # デバッグ描画用 (画素座標)
    landmarks: np.ndarray | None = None  # 478 点すべて (画素座標)
    head_R: np.ndarray | None = None     # 頭部の回転行列 3x3
    raw_L: np.ndarray | None = None      # 正規化ランドマーク (478, 3)  再計算・解析用
    raw_M: np.ndarray | None = None      # 頭部姿勢行列 (4, 4)
    patches: np.ndarray | None = None    # 目の周りの画像 (2, PATCH_H, PATCH_W) uint8  見た目の特徴用
    t: float = 0.0                       # 撮影時刻 (秒)
    openness_ratio: float = 1.0          # 目の開き / 普段の開き


class BlinkDetector:
    """まばたきの判定。閉じかけ・開きかけの途中も含めて除く。

    途中のフレームでは目の開きが小さく、虹彩もまぶたに隠れるので、推定が「下を見た」側に大きくぶれる。
    - 開き具合が普段 (直近の開いているときの中央値) の ENTER 倍未満、または
      1 フレームで普段の DROP 倍以上急に閉じて DROP_BELOW 倍未満になったら、まばたき開始
    - EXIT 倍以上に戻った状態が EXIT_FRAMES 続いたら終了。その後さらに HOLD 秒は除く
    下を見たときの自然な開きの減少 (普段の 0.85〜0.9 倍程度) では反応しないよう、
    どの閾値もそれより下に置いている。"""
    ENTER, DROP, DROP_BELOW, EXIT, EXIT_FRAMES, HOLD = 0.72, 0.2, 0.8, 0.8, 2, 0.12

    def __init__(self):
        self.hist: deque = deque(maxlen=90)
        self.in_blink = False
        self.exit_count = 0
        self.hold_until = -1.0
        self.prev_ratio = 1.0
        self.events: deque = deque(maxlen=200)   # (開始時刻, 終了時刻)
        self._start = None

    def baseline(self) -> float | None:
        return float(np.median(self.hist)) if len(self.hist) >= 15 else None

    def __call__(self, openness: float, t: float) -> tuple[bool, float]:
        base = self.baseline()
        if base is None:
            blink = openness < 0.05
            if not blink:
                self.hist.append(openness)
            return blink, 1.0
        r = openness / max(base, 1e-6)
        if not self.in_blink:
            if r < self.ENTER or ((self.prev_ratio - r) > self.DROP and r < self.DROP_BELOW):
                self.in_blink, self.exit_count, self._start = True, 0, t
        else:
            self.exit_count = self.exit_count + 1 if r >= self.EXIT else 0
            if self.exit_count >= self.EXIT_FRAMES:
                self.in_blink = False
                self.hold_until = t + self.HOLD
                self.events.append((self._start, t + self.HOLD))
        self.prev_ratio = r
        blink = self.in_blink or t < self.hold_until
        if not blink:
            self.hist.append(openness)
        return blink, r

    def near_blink(self, t: float, before: float = 0.1) -> bool:
        """t がまばたきの直前 before 秒〜終了 (保留を含む) に入るか。閉じ始めの検出漏れを後から除くのに使う。"""
        if self.in_blink and self._start is not None and t >= self._start - before:
            return True
        return any(a - before <= t <= b for a, b in self.events)


# 目の周りの画像 (パッチ) の大きさ。目尻と目頭を結ぶ線を水平にそろえて切り出す
PATCH_W, PATCH_H = 40, 24
PATCH_SPAN_X = 1.6     # 目の幅 (目尻〜目頭) に対する切り出し幅
PATCH_SPAN_Y = 0.96    # 目の幅に対する切り出し高さ (まぶた・眉の下まで入る)


def eye_patches(gray: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """グレースケール画像と画素座標のランドマークから、左右の目のパッチ (2, PATCH_H, PATCH_W) を作る。
    目の傾きと大きさをそろえるので、頭の回転や距離の影響が小さくなる。"""
    out = np.zeros((2, PATCH_H, PATCH_W), np.uint8)
    for k, eye in enumerate((LEFT_EYE, RIGHT_EYE)):
        p1, p2 = pts[eye["a"]], pts[eye["b"]]
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        c = (p1 + p2) / 2
        d = p2 - p1
        width = float(np.hypot(*d)) + 1e-6
        ux = d / width                       # 目の横方向 (単位)
        uy = np.array([-ux[1], ux[0]])       # 下方向
        sx, sy = PATCH_SPAN_X * width, PATCH_SPAN_Y * width
        # パッチの左上・右上・左下 に対応する元画像の点
        src = np.float32([c - ux * sx / 2 - uy * sy / 2, c + ux * sx / 2 - uy * sy / 2, c - ux * sx / 2 + uy * sy / 2])
        dst = np.float32([[0, 0], [PATCH_W, 0], [0, PATCH_H]])
        A = cv2.getAffineTransform(src, dst)
        out[k] = cv2.warpAffine(gray, A, (PATCH_W, PATCH_H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    return out


# 描画用の輪郭 (MediaPipe Face Mesh の番号)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
NOSE_LINE = [6, 197, 195, 5, 4]
LEFT_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_RING = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]


def draw_face(img: np.ndarray, s: "Sample", scale: float = 1.0, axis_len: int = 45) -> None:
    """検出結果を映像に描く: 顔の輪郭・眉・鼻筋・口 (灰)、目の輪郭 (緑)、虹彩 (シアン)、頭の向きの軸、視線。"""
    if not s.ok or s.landmarks is None:
        return
    P = (s.landmarks * scale).astype(np.int32)

    def poly(idx, color, closed, thick=1):
        cv2.polylines(img, [P[idx].reshape(-1, 1, 2)], closed, color, thick, cv2.LINE_AA)

    poly(FACE_OVAL, (150, 150, 150), True)
    poly(LIPS_OUTER, (150, 150, 150), True)
    poly(NOSE_LINE, (150, 150, 150), False)
    poly(LEFT_BROW, (150, 150, 150), False)
    poly(RIGHT_BROW, (150, 150, 150), False)
    poly(LEFT_EYE_RING, (0, 255, 0), True)
    poly(RIGHT_EYE_RING, (0, 255, 0), True)
    for c in (468, 473):
        r = max(2, int(np.linalg.norm(P[c + 1] - P[c])))
        cv2.circle(img, tuple(P[c]), r, (255, 220, 0), 1, cv2.LINE_AA)
        cv2.circle(img, tuple(P[c]), 2, (255, 220, 0), -1, cv2.LINE_AA)
    if s.head_R is not None:
        o = P[NOSE]
        R = s.head_R
        # 頭の X (赤) Y (緑) Z (青) 軸を鼻先から描く。画像は y 下向きなので符号を合わせる
        for axis, color in ((0, (0, 0, 255)), (1, (0, 255, 0)), (2, (255, 128, 0))):
            d = R[:, axis]
            end = (int(o[0] + d[0] * axis_len), int(o[1] - d[1] * axis_len))
            cv2.line(img, tuple(o), end, color, 2, cv2.LINE_AA)
    # 視線方向 (黄) を虹彩から描く
    f = s.feat
    for c, ta, te in ((468, f[IDX["gl_ta"]], f[IDX["gl_te"]]), (473, f[IDX["gr_ta"]], f[IDX["gr_te"]])):
        o = P[c]
        end = (int(o[0] + ta * axis_len * 1.5), int(o[1] - te * axis_len * 1.5))
        cv2.line(img, tuple(o), end, (0, 230, 255), 2, cv2.LINE_AA)


class HeadSmoother:
    """頭部姿勢・位置の特徴は MediaPipe の推定がぶれるので指数移動平均で均す。
    頭はゆっくりしか動かないので視線の応答性には影響しない。"""

    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self._s: np.ndarray | None = None

    def __call__(self, feat: np.ndarray) -> np.ndarray:
        f = feat.copy()
        head = f[HEAD_IDX]
        if self._s is None:
            self._s = head
        else:
            self._s = self.alpha * head + (1 - self.alpha) * self._s
        f[HEAD_IDX] = self._s
        return f

    def reset(self):
        self._s = None


# ---------------------------------------------------------------------------
# 幾何: 眼球モデルによる視線方向
# ---------------------------------------------------------------------------
def pixel_ray(u: float, v: float, aspect: float, fovy_deg: float = FOVY_DEG) -> np.ndarray:
    """正規化画像座標 (u 右, v 下, 0..1) から、カメラ座標の視線レイ方向 (x 右, y 上, z 手前, カメラは -z を向く)。"""
    t = math.tan(math.radians(fovy_deg) / 2)
    return np.array([(2 * u - 1) * aspect * t, (1 - 2 * v) * t, -1.0])


def project(p: np.ndarray, aspect: float, fovy_deg: float = FOVY_DEG) -> tuple[float, float]:
    """カメラ座標の点を正規化画像座標 (u, v) に投影する (pixel_ray の逆)。"""
    t = math.tan(math.radians(fovy_deg) / 2)
    z = -p[2]
    return 0.5 * (p[0] / (z * aspect * t) + 1), 0.5 * (1 - p[1] / (z * t))


def eye_gaze(iris_uv, canon_center, M, aspect, radius=EYE_RADIUS_CM):
    """片目の視線方向 (単位ベクトル, カメラ座標) と眼球中心を返す。"""
    C = (M @ np.append(canon_center, 1.0))[:3]
    d = pixel_ray(iris_uv[0], iris_uv[1], aspect)
    a = d @ d
    b = -2 * d @ C
    c = C @ C - radius ** 2
    disc = b * b - 4 * a * c
    if disc >= 0:
        t = (-b - math.sqrt(disc)) / (2 * a)       # 手前側の交点 (見えている虹彩)
        p = t * d
    else:
        t = (d @ C) / a                              # 交わらない: レイ上の最近点で代用
        p = t * d
    g = p - C
    n = np.linalg.norm(g)
    return (g / n if n > 0 else np.array([0, 0, -1.0])), C


def features_from_raw(L: np.ndarray, M: np.ndarray, w: int, h: int) -> np.ndarray:
    """正規化ランドマーク L (478,3) と頭部姿勢行列 M (4,4) から特徴ベクトルを作る。
    保存した生データから再計算できるよう、process() とは独立させている。"""
    pts = L[:, :2] * [w, h]
    l_h, l_v, l_open = _eye_feats(pts, LEFT_EYE)
    r_h, r_v, r_open = _eye_feats(pts, RIGHT_EYE)
    yaw, pitch, roll = _euler_from_matrix(M[:3, :3])
    tx, ty, tz = M[:3, 3]
    inter = np.linalg.norm(pts[LEFT_EYE["a"]] - pts[RIGHT_EYE["b"]])
    face_x, face_y, face_scale = pts[NOSE][0] / w, pts[NOSE][1] / h, inter / w

    aspect = w / h
    gl, Cl = eye_gaze(L[LEFT_EYE["center"], :2], CANON_EYE_L, M, aspect)
    gr, Cr = eye_gaze(L[RIGHT_EYE["center"], :2], CANON_EYE_R, M, aspect)
    g = gl + gr
    g /= max(np.linalg.norm(g), 1e-9)
    O = (Cl + Cr) / 2

    def tan_ae(v):
        # 視線は被写体 (z<0) から画面・カメラ (z≈0) へ向かうので前方は +z
        z = max(v[2], 1e-6)
        return v[0] / z, v[1] / z

    g_ta, g_te = tan_ae(g)
    gl_ta, gl_te = tan_ae(gl)
    gr_ta, gr_te = tan_ae(gr)
    return np.array([l_h, l_v, r_h, r_v, l_open, r_open,
                     yaw, pitch, roll, tx, ty, tz,
                     face_x, face_y, face_scale,
                     g_ta, g_te, O[0], O[1], -O[2],
                     gl_ta, gl_te, gr_ta, gr_te])


class FeatureExtractor:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} が見つかりません。download_model.py を実行してください。")
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1
        self.blink = BlinkDetector()

    def process(self, frame_bgr: np.ndarray, ts_ms: int) -> Sample:
        # MediaPipe は単調増加のタイムスタンプを要求する
        if ts_ms <= self._last_ts:
            ts_ms = self._last_ts + 1
        self._last_ts = ts_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._lm.detect_for_video(img, ts_ms)
        if not res.face_landmarks:
            return Sample(np.zeros(len(FEATURE_NAMES)), False, False)

        h, w = frame_bgr.shape[:2]
        L = np.array([(p.x, p.y, p.z) for p in res.face_landmarks[0]], dtype=np.float64)
        M = np.array(res.facial_transformation_matrixes[0], dtype=np.float64)
        feat = features_from_raw(L, M, w, h)
        pts = L[:, :2] * [w, h]
        t = ts_ms / 1000.0
        blink, ratio = self.blink(min(feat[IDX["l_open"]], feat[IDX["r_open"]]), t)
        eye_pts = pts[[LEFT_EYE["a"], LEFT_EYE["b"], RIGHT_EYE["a"], RIGHT_EYE["b"], 468, 473]]
        patches = eye_patches(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), pts)
        return Sample(feat, blink, True, eye_pts, pts, M[:3, :3], L.astype(np.float32), M.astype(np.float32), patches,
                      t, ratio)

    def close(self):
        self._lm.close()


def _eye_feats(pts, eye):
    """虹彩中心を目の両端・上下まぶたで正規化した相対座標を返す。
    軸は常に画像の左→右方向にそろえ、左右の目で符号が一致するようにする。"""
    p1, p2 = pts[eye["a"]], pts[eye["b"]]
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    p_up, p_down = pts[eye["up"]], pts[eye["down"]]
    iris = pts[eye["iris"]].mean(axis=0)
    axis = p2 - p1
    width = np.linalg.norm(axis) + 1e-6
    axis_u = axis / width
    normal_u = np.array([-axis_u[1], axis_u[0]])     # 画像の下向き
    center = (p1 + p2) / 2
    d = iris - center
    h = float(d @ axis_u) / width          # 右を向くと正
    v = float(d @ normal_u) / width        # 下を向くと正
    openness = float(np.linalg.norm(p_up - p_down)) / width
    return h, v, openness


def _euler_from_matrix(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0
    return yaw, pitch, roll

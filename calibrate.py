"""キャリブレーション。

Display1 に Windows 画面を出した状態で実行し、注視点を順に見つめる。
頭の位置を変えた複数の姿勢 (パス) で集めると、プレイ中に頭が動いても精度が保てる。
終了後にプロファイル (profiles/<name>.json) と生データ (profiles/<name>_calib.npz) を保存する。

例:
  python calibrate.py --list-monitors
  python calibrate.py --profile home --monitor 0 --passes 5
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from screeninfo import get_monitors

from tracker.camera import Camera
from tracker.features import IDX, FeatureExtractor, HeadSmoother, draw_face
from tracker.filters import OneEuro
from tracker.model import KIND_MOTION, KIND_PURSUIT, KIND_STATIC, GazeModel, align_pursuit, fit_profile
from tracker.recorder import NullRecorder, Recorder
from tracker.paths import PROFILE_DIR

WIN = "gaze-calibration"
REC = NullRecorder()          # --record のときに Recorder に差し替える

STATIC_GRID = 5               # 1 周目の静止点: 5x5。縦の段数を増やして点ごとの偏りを平均しやすくする
STATIC_SETTLE, STATIC_COLLECT = 0.8, 0.6   # 1 点あたり: 視線が落ち着くまで待つ秒数 / 集める秒数
PURSUIT_SEGMENT = 2.0         # 動く点のデータを交差検証用に区切る秒数
BG = (20, 20, 20)
WHITE, GRAY, GREEN, RED, ORANGE = (230, 230, 230), (140, 140, 140), (80, 220, 0), (255, 60, 60), (255, 179, 71)

# 姿勢ごとの指示。1 周目は普段の姿勢、以降は頭の位置を変える。
# 目標は 1 周目の姿勢からのずれ (左へ cm, 上へ cm, 画面に近づく cm)
PASS_GUIDE = [
    ("いつもの姿勢", "普段プレイする姿勢で座ってください", (0, 0, 0)),
    ("頭を左へ", "上体ごと頭を左にずらしてください", (10, 0, 0)),
    ("頭を右へ", "上体ごと頭を右にずらしてください", (-10, 0, 0)),
    ("頭を前へ", "画面に近づいてください", (0, 0, 10)),
    ("頭を後ろへ", "画面から離れてください", (0, 0, -10)),
    ("頭を上へ", "背筋を伸ばして頭を高くしてください", (0, 5, 0)),
    ("頭を下へ", "少し姿勢を低くしてください", (0, -5, 0)),
]
POSE_TOL_CM = 3.0     # 目標との許容差
POSE_HOLD_SEC = 1.2   # 目標内に入ってから自動で進むまでの秒数


def head_offset(feat, ref):
    """基準姿勢 ref からの頭のずれ (左へ, 上へ, 画面に近づく) を cm 相当で返す。
    向きは画像内の顔位置から決め、大きさは MediaPipe の頭部位置 (cm 相当) から取る。"""
    left = np.sign(feat[IDX["face_x"]] - ref[IDX["face_x"]]) * abs(feat[IDX["tx"]] - ref[IDX["tx"]])
    up = np.sign(ref[IDX["face_y"]] - feat[IDX["face_y"]]) * abs(feat[IDX["ty"]] - ref[IDX["ty"]])
    closer = abs(ref[IDX["tz"]]) - abs(feat[IDX["tz"]])
    return float(left), float(up), float(closer)


def draw_head_panel(canvas, cur, target, cx, cy, scale=11.0, tol=POSE_TOL_CM):
    """鏡のように頭の位置を表示する。cur / target = (左へ, 上へ, 近づく) cm。
    正面図: 丸が今の頭、灰色の輪が目標。右の縦バーが奥行き。"""
    left, up, closer = cur
    tl, tu, tc = target
    # 正面図の枠
    cv2.rectangle(canvas, (cx - 170, cy - 130), (cx + 170, cy + 130), (60, 60, 60), 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - 130), (cx, cy + 130), (45, 45, 45), 1)
    cv2.line(canvas, (cx - 170, cy), (cx + 170, cy), (45, 45, 45), 1)
    # 目標 (鏡なので左へ = 画面の左)
    tx, ty = int(cx - tl * scale), int(cy - tu * scale)
    cv2.circle(canvas, (tx, ty), int(tol * scale), (120, 120, 120), 2, cv2.LINE_AA)
    # 今の頭。近づくほど大きく描く
    hx, hy = int(cx - left * scale), int(cy - up * scale)
    r = int(max(10, 26 + closer * 1.2))
    ok_xy = abs(left - tl) <= tol and abs(up - tu) <= tol
    ok_z = abs(closer - tc) <= tol
    cv2.circle(canvas, (hx, hy), r, (80, 220, 0) if ok_xy else (255, 179, 71)[::-1], -1, cv2.LINE_AA)
    # 奥行きバー (上が近い)
    bx, by0, by1 = cx + 215, cy - 130, cy + 130
    cv2.rectangle(canvas, (bx - 14, by0), (bx + 14, by1), (60, 60, 60), 1, cv2.LINE_AA)
    zt = int(cy - tc * scale)
    cv2.rectangle(canvas, (bx - 20, zt - int(tol * scale)), (bx + 20, zt + int(tol * scale)), (120, 120, 120), 2, cv2.LINE_AA)
    zc = int(np.clip(cy - closer * scale, by0, by1))
    cv2.rectangle(canvas, (bx - 12, zc - 5), (bx + 12, zc + 5), (80, 220, 0) if ok_z else (71, 179, 255), -1, cv2.LINE_AA)
    return ok_xy and ok_z


def offset_instruction(cur, target, tol=POSE_TOL_CM):
    """「あと 4cm 左へ」のような文を作る。目標内なら None。"""
    parts = []
    for d, pos, neg in ((target[0] - cur[0], "左へ", "右へ"), (target[1] - cur[1], "上へ", "下へ"),
                        (target[2] - cur[2], "前へ", "後ろへ")):
        if abs(d) > tol:
            parts.append(f"あと {abs(d):.0f}cm {pos if d > 0 else neg}")
    return "、".join(parts) if parts else None

_FONTS: dict[int, ImageFont.FreeTypeFont] = {}


def _font(size: int):
    if size not in _FONTS:
        for name in ("meiryo.ttc", "YuGothM.ttc", "msgothic.ttc"):
            p = Path("C:/Windows/Fonts") / name
            if p.exists():
                _FONTS[size] = ImageFont.truetype(str(p), size)
                break
        else:
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def draw_texts(canvas: np.ndarray, items):
    """items: [(text, y, size, rgb), ...] を中央揃えで描く (日本語対応)。"""
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    W = canvas.shape[1]
    for text, y, size, rgb in items:
        f = _font(size)
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=rgb)
    canvas[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def grid_points(n: int, margin: float = 0.06):
    xs = np.linspace(margin, 1 - margin, n)
    ys = np.linspace(margin, 1 - margin, n)
    pts = [(float(x), float(y)) for y in ys for x in xs]
    if n % 2 == 0:
        pts.append((0.5, 0.5))
    return pts


def open_fullscreen(mon):
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WIN, mon.x, mon.y)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def wait_key(ms=1):
    return cv2.waitKey(ms) & 0xFF


def phase_preview(cam, fx, W, H, passes, motion_n=0, motion_sec=12.0, pursuit_sec=0.0):
    """カメラ映像を見せて顔検出を確認する。SPACE で開始, ESC で中止。"""
    while True:
        frame, ts = cam.read()
        s = fx.process(frame, int(ts * 1000))
        canvas = np.full((H, W, 3), BG, np.uint8)
        ph = H // 2
        pw = int(frame.shape[1] * ph / frame.shape[0])
        small = cv2.resize(frame, (pw, ph))
        draw_face(small, s, pw / frame.shape[1])
        x0, y0 = (W - pw) // 2, (H - ph) // 2 - 40
        canvas[y0:y0 + ph, x0:x0 + pw] = small
        n_pts = len(grid_points(STATIC_GRID)) + (passes - 1) * len(grid_points(3))
        secs = n_pts * (STATIC_SETTLE + STATIC_COLLECT + 0.1) + pursuit_sec + 5 + motion_n * (motion_sec + 2)
        plan = f"{n_pts} 点" + (f" + 動く点 {pursuit_sec:.0f} 秒" if pursuit_sec else "") + \
            (f" + 頭を動かしながら {motion_n} 点" if motion_n else "") + (f" ({passes} 姿勢)" if passes > 1 else "")
        draw_texts(canvas, [
            ("顔・目・頭の向きを検出しています" if s.ok else "顔が検出できません。カメラの向きや明るさを確認してください",
             y0 - 56, 34, GREEN if s.ok else RED),
            (f"{plan} (約 {int(secs // 60) + 1} 分)", y0 + ph + 30, 28, WHITE),
            ("出てくる点を、頭を動かさず目だけで見てください", y0 + ph + 75, 28, WHITE),
            ("SPACE: 開始    ESC: 中止", y0 + ph + 135, 30, GRAY),
        ])
        cv2.imshow(WIN, canvas)
        k = wait_key(1)
        if k == 27:
            return False
        if k == 32:
            return True


def show_pass_guide(cam, fx, W, H, p, passes, ref=None):
    """姿勢の指示を出す。2 周目以降は基準姿勢からのずれを鏡のように表示し、
    目標の位置に入って POSE_HOLD_SEC 秒たつと自動で進む (SPACE でも進める)。"""
    title, detail, target = PASS_GUIDE[min(p, len(PASS_GUIDE) - 1)]
    hs = HeadSmoother(0.3)
    in_target_since = None
    while True:
        frame, ts = cam.read()
        s = fx.process(frame, int(ts * 1000))
        canvas = np.full((H, W, 3), BG, np.uint8)
        ph = H // 3
        pw = int(frame.shape[1] * ph / frame.shape[0])
        small = cv2.resize(frame, (pw, ph))
        draw_face(small, s, pw / frame.shape[1])
        small = cv2.flip(small, 1)     # 鏡表示
        lines = [(f"姿勢 {p + 1} / {passes}: {title}", 100, 56, ORANGE), (detail, 185, 34, WHITE)]
        auto = False
        if ref is None or p == 0:
            x0, y0 = (W - pw) // 2, H // 2 - ph // 2 + 60
            canvas[y0:y0 + ph, x0:x0 + pw] = small
            lines += [("この姿勢のまま、次の点を目だけで追ってください", 235, 28, GRAY),
                      ("顔を検出しています" if s.ok else "顔が検出できません", y0 + ph + 20, 28, GREEN if s.ok else RED),
                      ("SPACE: この姿勢で開始    ESC: 中止", H - 90, 30, GRAY)]
        else:
            # 左にカメラ映像 (鏡)、右に位置パネル
            x0, y0 = W // 2 - pw - 40, max(H // 2 - ph // 2 + 40, 300)
            canvas[y0:y0 + ph, x0:x0 + pw] = small
            pcx, pcy = W // 2 + 230, y0 + ph // 2
            if s.ok:
                cur = head_offset(hs(s.feat), ref)
                ok = draw_head_panel(canvas, cur, target, pcx, pcy)
                msg = offset_instruction(cur, target)
                if ok:
                    in_target_since = in_target_since or time.time()
                    remain = POSE_HOLD_SEC - (time.time() - in_target_since)
                    lines.append((f"OK！そのまま動かないでください ({max(remain, 0):.1f})", 240, 34, GREEN))
                    auto = remain <= 0
                else:
                    in_target_since = None
                    lines.append((msg or "OK", 240, 34, ORANGE))
            else:
                in_target_since = None
                draw_head_panel(canvas, (0, 0, 0), target, pcx, pcy)
                lines.append(("顔が検出できません", 240, 34, RED))
            lines += [("緑の丸が今の頭、灰色の輪が目標。右のバーは奥行き (上ほど画面に近い)", y0 + ph + 30, 24, GRAY),
                      ("目標に入ると自動で進みます    SPACE: この位置で開始    ESC: 中止", H - 90, 28, GRAY)]
        draw_texts(canvas, lines)
        cv2.imshow(WIN, canvas)
        k = wait_key(1)
        if k == 27:
            return False
        if (k == 32 and s.ok) or auto:
            return True


def phase_collect(cam, fx, W, H, passes, settle=STATIC_SETTLE, collect=STATIC_COLLECT):
    """静止点を順に見つめてもらう。返り値: (サンプルの list, ラベルの dict of list, 基準姿勢)"""
    S, Y, G, P = [], [], [], []
    gid = 0
    ref = None
    for p in range(passes):
        if p == 1 and S:
            ref = np.median(np.array([x.feat for x in S]), axis=0)      # 1 周目の頭の位置を基準にする
        if not show_pass_guide(cam, fx, W, H, p, passes, ref):
            return None
        order = grid_points(STATIC_GRID) if p == 0 else grid_points(3)
        random.shuffle(order)
        order.sort(key=lambda q: 0 if q == (0.5, 0.5) else 1)    # 中心から始めると目が慣れやすい
        for i, (nx, ny) in enumerate(order):
            cx, cy = int(nx * W), int(ny * H)
            t0 = time.time()
            got = []
            last = None
            while True:
                el = time.time() - t0
                canvas = np.full((H, W, 3), BG, np.uint8)
                if el < settle:
                    r = int(40 - 28 * (el / settle))
                    col = (200, 200, 200)
                else:
                    r = 12
                    col = (0, 220, 80)
                cv2.circle(canvas, (cx, cy), r, col, 2, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 3, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(canvas, f"{p + 1}/{passes}  {i + 1}/{len(order)}   ESC: cancel", (20, H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
                cv2.imshow(WIN, canvas)
                frame, ts = cam.read()
                if ts != last:          # 同じフレームを 2 回数えない
                    last = ts
                    s = fx.process(frame, int(ts * 1000))
                    collecting = el >= settle
                    REC.add(frame, ts, "static", (nx, ny), gid=gid, collecting=collecting, face=bool(s.ok))
                    if collecting and s.ok and not s.blink:
                        got.append(s)
                if el >= settle + collect and len(got) >= 6:
                    break
                if el > settle + collect + 3:   # 顔が取れない場合の打ち切り
                    break
                if wait_key(1) == 27:
                    return None
            got = drop_near_blinks(fx, got)
            if len(got) < 4:
                print(f"  pass {p + 1} point {i + 1}: サンプル不足 ({len(got)}) スキップ")
                continue
            for x in got:
                S.append(x)
                Y.append((nx, ny))
                G.append(gid)
                P.append(p)
            gid += 1
    if ref is None and S:
        ref = np.median(np.array([x.feat for x in S]), axis=0)
    return S, {"Y": Y, "G": G, "P": P}, ref


def pursuit_path(t: float):
    """動く点の軌跡 (正規化座標)。周期の違う 2 つの正弦波 (リサージュ曲線) で画面全体をなめらかになぞる。
    周期は「画面を 5x5 に区切った全マスを約 40 秒で通る」組み合わせを探して決めた (13 秒 / 9 秒だと 80 秒かかった)。
    速さは最大で横が画面幅の 0.28 倍/秒、縦が画面高さの 0.19 倍/秒 (目で無理なく追える速さ)。"""
    return (0.5 + 0.42 * np.sin(2 * np.pi * t / 9.5),
            0.5 + 0.40 * np.sin(2 * np.pi * t / 13.5 + np.pi / 2))


def show_pursuit_guide(cam, fx, W, H, duration):
    while True:
        frame, ts = cam.read()
        s = fx.process(frame, int(ts * 1000))
        canvas = np.full((H, W, 3), BG, np.uint8)
        draw_texts(canvas, [
            ("次は、動く点を目で追います", 120, 56, ORANGE),
            (f"{duration:.0f} 秒ほど、点がゆっくり画面全体を動きます (瞬きで取りこぼした場所があれば少し延長)", 215, 30, WHITE),
            ("頭は動かさず、目だけで点を追い続けてください", 265, 30, WHITE),
            ("上下の位置をたくさん集めて、縦方向のずれを減らします", 315, 26, GRAY),
            ("顔を検出しています" if s.ok else "顔が検出できません", H // 2 + 60, 28, GREEN if s.ok else RED),
            ("SPACE: 開始    ESC: 中止", H - 90, 30, GRAY),
        ])
        cv2.imshow(WIN, canvas)
        k = wait_key(1)
        if k == 27:
            return False
        if k == 32 and s.ok:
            return True


PURSUIT_CELLS = 5            # 網羅の判定: 画面を 5x5 に区切る
PURSUIT_PER_CELL = 10        # 各マスに必要な (瞬き以外の) サンプル数
PURSUIT_MAX_FACTOR = 2.25    # 最低秒数の何倍まで延長するか (既定 40 秒 → 最大 90 秒)
PURSUIT_ASSUMED_LAG = 0.23   # 網羅の判定に使う目の遅れの目安 (秒)。学習では改めて推定する


def pursuit_cell(x, y):
    """正規化座標 → 5x5 のマス番号 (点が動く範囲 0.08〜0.92 を等分)"""
    cx = int(np.clip((x - 0.08) / 0.84 * PURSUIT_CELLS, 0, PURSUIT_CELLS - 1))
    cy = int(np.clip((y - 0.10) / 0.80 * PURSUIT_CELLS, 0, PURSUIT_CELLS - 1))
    return cy * PURSUIT_CELLS + cx


def phase_pursuit(cam, fx, W, H, duration=40.0, lead_in=1.5):
    """動く点を目で追ってもらう。点は最初 lead_in 秒止まってから動き出す。
    duration 秒たっても、画面を 5x5 に区切った全マスで瞬き以外のサンプルが PURSUIT_PER_CELL 個そろうまでは続ける
    (瞬きで取りこぼした場所を自動で取り直す)。最長は duration の PURSUIT_MAX_FACTOR 倍。
    返り値: (サンプル, 撮影時刻, 軌跡の時刻, 軌跡の座標) か None (中止)。"""
    samples, times, track_t, track_xy = [], [], [], []
    counts = np.zeros(PURSUIT_CELLS * PURSUIT_CELLS, int)
    t_start = time.time() + lead_in
    t_max = duration * PURSUIT_MAX_FACTOR
    last = None
    while True:
        now = time.time()
        el = now - t_start
        covered = float(np.mean(np.minimum(counts, PURSUIT_PER_CELL)) / PURSUIT_PER_CELL)
        if el >= t_max or (el >= duration and covered >= 1.0):
            break
        canvas = np.full((H, W, 3), BG, np.uint8)
        nx, ny = pursuit_path(max(el, 0.0))
        cx, cy = int(nx * W), int(ny * H)
        cv2.circle(canvas, (cx, cy), 16, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)
        # 進み具合: 時間 (最低秒数まで) と網羅率の低い方
        cv2.rectangle(canvas, (0, H - 6), (int(W * min(max(el, 0) / duration, 1) * (covered if el >= duration else 1)), H),
                      (0, 200, 255), -1)
        if el < 0:
            draw_texts(canvas, [("この点を目で追ってください", max(cy - 90, 20), 32, ORANGE)])
        elif el >= duration:
            cv2.putText(canvas, f"collecting missed areas {int(covered * 100)}%", (20, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.imshow(WIN, canvas)
        # 表示した時刻と位置を軌跡として残す (撮影時刻との対応は後で遅れ込みで合わせる)
        track_t.append(now)
        track_xy.append((nx, ny))
        frame, ts = cam.read()
        if ts != last:
            last = ts
            s = fx.process(frame, int(ts * 1000))
            tgt = pursuit_path(max(ts - t_start, 0.0))
            REC.add(frame, ts, "pursuit", tgt, face=bool(s.ok), moving=bool(ts >= t_start))
            if ts >= t_start + 0.5 and s.ok and not s.blink:      # 動き出してすぐは目が追いついていない
                samples.append(s)
                times.append(ts)
                lx, ly = pursuit_path(max(ts - t_start - PURSUIT_ASSUMED_LAG, 0.0))
                counts[pursuit_cell(lx, ly)] += 1
        if wait_key(1) == 27:
            return None
    keep = set(id(x) for x in drop_near_blinks(fx, samples))
    times = [t for x, t in zip(samples, times) if id(x) in keep]
    samples = [x for x in samples if id(x) in keep]
    print(f"動く点: {max(time.time() - t_start, 0):.0f} 秒、網羅率 {int(covered * 100)}%")
    return samples, np.array(times), np.array(track_t), np.array(track_xy)


MOTION_STEPS = [   # (開始秒, 指示)
    (0, "頭をゆっくり左右に回す"),
    (4, "ゆっくり上下にうなずく"),
    (8, "画面に近づいたり離れたりする"),
]


def motion_points(n: int, margin: float = 0.12):
    if n <= 0:
        return []
    if n <= 5:
        return [(0.5, 0.5), (margin, margin), (1 - margin, margin), (margin, 1 - margin), (1 - margin, 1 - margin)][:n]
    return grid_points(3, margin)


def show_motion_guide(cam, fx, W, H, n_points, duration):
    while True:
        frame, ts = cam.read()
        s = fx.process(frame, int(ts * 1000))
        canvas = np.full((H, W, 3), BG, np.uint8)
        draw_texts(canvas, [
            ("次は、点を見つめたまま頭を動かします", 120, 56, ORANGE),
            (f"{n_points} 点それぞれで {duration:.0f} 秒間、画面の指示どおりに頭をゆっくり動かしてください", 215, 32, WHITE),
            ("目は点から離さないでください。動きは大きめでも構いません", 265, 28, GRAY),
            ("これで、頭の向きや奥行きが変わったときのずれを学習します", 310, 28, GRAY),
            ("顔を検出しています" if s.ok else "顔が検出できません", H // 2 + 60, 28, GREEN if s.ok else RED),
            ("SPACE: 開始    ESC: 中止", H - 90, 30, GRAY),
        ])
        cv2.imshow(WIN, canvas)
        k = wait_key(1)
        if k == 27:
            return False
        if k == 32 and s.ok:
            return True


MOTION_TARGET = (20.0, 15.0, 6.0)   # 目標の振れ幅: yaw ±deg, pitch ±deg, 奥行き ±cm


def draw_gauge(canvas, x, y, length, cur, lo, hi, target, label, active, vertical=False):
    """今の値 (cur) と、これまでに動かした範囲 (lo..hi) を、目標範囲 (±target) と一緒に描く。"""
    col_frame = (255, 179, 71)[::-1] if active else (70, 70, 70)
    half = length // 2
    def pos(v):
        f = float(np.clip(v / (target * 1.5), -1, 1))
        return (x, int(y - f * half)) if vertical else (int(x + f * half), y)
    a, b = pos(-target * 1.5), pos(target * 1.5)
    cv2.line(canvas, a, b, (55, 55, 55), 6, cv2.LINE_AA)
    ta, tb = pos(-target), pos(target)
    cv2.line(canvas, ta, tb, col_frame, 2, cv2.LINE_AA)
    if hi > lo:
        ca, cb = pos(lo), pos(hi)
        cv2.line(canvas, ca, cb, (80, 220, 0), 6, cv2.LINE_AA)
    cv2.circle(canvas, pos(cur), 9, (255, 255, 255), -1, cv2.LINE_AA)
    cov = min(1.0, (hi - lo) / (2 * target)) if hi > lo else 0.0
    if abs(cur) > target * 1.25:     # 動かしすぎ: この範囲のデータは学習に使われない
        cv2.circle(canvas, pos(cur), 12, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{label} {int(cov * 100)}%", (x - half if not vertical else x + 20, y + half + 22 if not vertical else y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return cov


def phase_motion(cam, fx, W, H, points, duration=12.0, gid0=0, max_per_point=16, ref=None):
    """各点を見つめたまま頭を動かしてもらい、連続的にサンプルを集める。"""
    S, Y, G = [], [], []
    hs = HeadSmoother(0.4)
    for i, (nx, ny) in enumerate(points):
        cx, cy = int(nx * W), int(ny * H)
        got = []
        t0 = None
        last = None
        rng = {"yaw": [0.0, 0.0], "pitch": [0.0, 0.0], "z": [0.0, 0.0]}
        cur = {"yaw": 0.0, "pitch": 0.0, "z": 0.0}
        base = None
        s_prev = None
        while True:
            frame, ts = cam.read()
            new = ts != last
            last = ts
            s = fx.process(frame, int(ts * 1000)) if new or s_prev is None else s_prev
            s_prev = s
            canvas = np.full((H, W, 3), BG, np.uint8)
            if t0 is None:
                # 顔を検出して 1 秒待ってから開始
                if s.ok:
                    t0 = time.time() + 1.0
                el = -1.0
            else:
                el = time.time() - t0
            cv2.circle(canvas, (cx, cy), 14, (0, 220, 80) if el >= 0 else (200, 200, 200), 2, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)
            step_i = 0
            if el >= 0:
                cv2.ellipse(canvas, (cx, cy), (34, 34), -90, 0, int(360 * min(el / duration, 1)), (0, 200, 255), 3, cv2.LINE_AA)
                step_i = max(k for k, (t0s, _) in enumerate(MOTION_STEPS) if el >= t0s)
                step = MOTION_STEPS[step_i][1]
                if new and s.ok and not s.blink:
                    got.append(s)
                if new:
                    REC.add(frame, ts, "motion", (nx, ny), gid=gid0 + i, face=bool(s.ok))
            else:
                step = "顔を検出したら始まります" if not s.ok else "まもなく開始"
            if s.ok:
                f = hs(s.feat)
                if base is None:
                    base = ref if ref is not None else f.copy()
                cur = {"yaw": float(np.degrees(f[IDX["yaw"]] - base[IDX["yaw"]])),
                       "pitch": float(np.degrees(f[IDX["pitch"]] - base[IDX["pitch"]])),
                       "z": float(abs(base[IDX["tz"]]) - abs(f[IDX["tz"]]))}
                if el >= 0:
                    for k, v in cur.items():
                        rng[k][0] = min(rng[k][0], v)
                        rng[k][1] = max(rng[k][1], v)
            # 指示とゲージは点の邪魔にならない位置 (点が上半分なら下、下半分なら上) に出す
            ty = H - 210 if ny < 0.5 else 80
            gy = ty + 150 if ny < 0.5 else ty + 135
            gx = W // 2
            draw_gauge(canvas, gx - 260, gy, 220, cur["yaw"], *rng["yaw"], MOTION_TARGET[0], "left-right", step_i == 0)
            draw_gauge(canvas, gx, gy, 220, cur["pitch"], *rng["pitch"], MOTION_TARGET[1], "up-down", step_i == 1)
            draw_gauge(canvas, gx + 260, gy, 220, cur["z"], *rng["z"], MOTION_TARGET[2], "near-far", step_i == 2)
            draw_texts(canvas, [
                (f"点を見つめたまま: {step}", ty, 40, ORANGE),
                (f"{i + 1} / {len(points)}    サンプル {len(got)}    緑が動かした範囲、枠が目標。枠を大きく超えた分は使われません    ESC: 中止", ty + 58, 24, GRAY),
            ])
            cv2.imshow(WIN, canvas)
            if wait_key(1) == 27:
                return None
            if el >= duration:
                break
        got = drop_near_blinks(fx, got)
        if len(got) < 20:
            print(f"  motion point {i + 1}: サンプル不足 ({len(got)}) スキップ")
            continue
        # 時間方向に均等に間引く。頭を動かしたデータは全体の 15% 程度が最もよかった (実録画での交差検証):
        # 多すぎると普段の姿勢での縦の精度が落ち、無いと頭を動かしたときに大きくずれる
        idx = np.linspace(0, len(got) - 1, min(max_per_point, len(got))).astype(int)
        for j in idx:
            S.append(got[j])
            Y.append((nx, ny))
            G.append(gid0 + i)
    return S, Y, G


def phase_verify(cam, fx, W, H, prof):
    """推定結果をリアルタイム表示して確認する。SPACE 保存 / R やり直し / ESC 破棄。"""
    fxf, fyf, hs = OneEuro(), OneEuro(), HeadSmoother()
    pts = grid_points(4)
    pct = prof.cv_error_px / W * 100
    grade = "良好" if pct < 5 else ("許容範囲" if pct < 10 else "やり直しを推奨")
    while True:
        frame, ts = cam.read()
        s = fx.process(frame, int(ts * 1000))
        canvas = np.full((H, W, 3), BG, np.uint8)
        for nx, ny in pts:
            cv2.circle(canvas, (int(nx * W), int(ny * H)), 6, (90, 90, 90), 1, cv2.LINE_AA)
        if s.ok and not s.blink:
            x, y = prof.predict(hs(s.feat), s.patches)
            x, y = fxf(x, ts), fyf(y, ts)
            cv2.circle(canvas, (int(x * W), int(y * H)), 18, (0, 200, 255), 3, cv2.LINE_AA)
        lines = [(f"推定誤差の目安: 約 {prof.cv_error_px:.0f} px (画面幅の {pct:.1f}%)  {grade}", 40, 32, WHITE)]
        sub = f"横 約 {prof.error_x_px:.0f} px  縦 約 {prof.error_y_px:.0f} px   |   静止点 約 {prof.same_pose_error_px:.0f} px"
        if not np.isnan(prof.pursuit_error_px):
            sub += f"  動く点 約 {prof.pursuit_error_px:.0f} px"
        if prof.n_motion_points > 0 and not np.isnan(prof.motion_error_px):
            sub += f"  頭を動かしたとき 約 {prof.motion_error_px:.0f} px"
        lines.append((sub, 90, 26, GRAY))
        lines.append((("目の画像の特徴: 使用" if prof.uses_appearance else "目の画像の特徴: 不使用")
                      + f" (効果 {prof.appearance_gain_px:+.0f} px)"
                      + (f"   キャリブレーション {prof.n_sessions} 回分のデータで学習" if getattr(prof, "n_sessions", 1) > 1 else ""),
                      130, 22, GRAY))
        lines.append(("画面のあちこちを見て、頭も少し動かして確かめてください", H - 130, 28, GRAY))
        lines.append(("SPACE: 保存    R: やり直し    ESC: 破棄", H - 80, 30, WHITE))
        draw_texts(canvas, lines)
        cv2.imshow(WIN, canvas)
        k = wait_key(1)
        if k == 27:
            return "discard"
        if k == 32:
            return "save"
        if k in (ord("r"), ord("R")):
            return "redo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="default", help="プロファイル名 (環境ごとに分ける)")
    ap.add_argument("--monitor", type=int, default=0, help="キャリブレーションを表示するモニター番号")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--passes", type=int, default=1, help="座り直して繰り返す姿勢の数 (1, 3, 5, 7)")
    ap.add_argument("--motion-points", type=int, default=5, help="点を見つめたまま頭を動かす点の数 (0, 5, 9)")
    ap.add_argument("--motion-seconds", type=float, default=12.0, help="1 点あたり頭を動かす秒数")
    ap.add_argument("--pursuit-seconds", type=float, default=40.0, help="動く点を目で追う秒数 (0 で省略)")
    ap.add_argument("--record", action="store_true", help="検証用にカメラ映像と点の位置を recordings/ に録画する")
    ap.add_argument("--accumulate", type=int, default=3,
                    help="同じプロファイルの前回までのデータと合わせて学習する回数 (1 で今回だけ。既定 3 = 前々回まで)")
    ap.add_argument("--cam-res", default="auto", help="カメラ解像度 (auto=出せる最大, 1920x1080, 1280x720, 640x480)")
    ap.add_argument("--list-monitors", action="store_true")
    a = ap.parse_args()

    mons = get_monitors()
    if a.list_monitors:
        for i, m in enumerate(mons):
            print(f"[{i}] {m.name}  {m.width}x{m.height}  at ({m.x},{m.y})  primary={m.is_primary}")
        return
    mon = mons[a.monitor]
    W, H = mon.width, mon.height
    passes = max(1, min(a.passes, len(PASS_GUIDE)))
    print(f"monitor [{a.monitor}] {W}x{H}, profile '{a.profile}', passes={passes}")

    from tracker.camera import parse_res
    cam = Camera(a.camera, *parse_res(a.cam_res))
    cw, ch = cam.width, cam.height
    print(f"camera [{a.camera}] {cam.description}")
    fx = FeatureExtractor()
    global REC
    if a.record:
        REC = Recorder(W, H, cam.description, (cam.width, cam.height), fps=cam.fps,
                       extra={"profile": a.profile, "pursuit": "0.5+0.42sin(2πt/9.5), 0.5+0.40sin(2πt/13.5+π/2)"})
        print(f"録画: {REC.dir}")
    open_fullscreen(mon)
    try:
        while True:
            r = run_once(cam, fx, W, H, cw, ch, passes, a)
            if r in ("save", "discard", "cancel"):
                print({"save": f"保存しました: {a.profile}", "discard": "破棄しました", "cancel": "中止"}[r])
                return
    finally:
        cv2.destroyAllWindows()
        fx.close()
        cam.close()
        d = REC.close()
        if d is not None:
            print(f"録画を保存しました: {d}")


def drop_near_blinks(fx, samples):
    """まばたきの閉じ始め (検出より少し前) のサンプルを後から除く。"""
    bd = getattr(fx, "blink", None)
    if bd is None or not hasattr(bd, "near_blink"):
        return samples
    return [x for x in samples if not bd.near_blink(x.t)]


def stack(samples):
    """Sample の list → 特徴量・生ランドマーク・頭部姿勢・目のパッチの配列"""
    return (np.array([x.feat for x in samples]), np.array([x.raw_L for x in samples]),
            np.array([x.raw_M for x in samples]), np.array([x.patches for x in samples]))


def run_once(cam, fx, W, H, cw, ch, passes, a):
    if not phase_preview(cam, fx, W, H, passes, a.motion_points, a.motion_seconds, a.pursuit_seconds):
        return "cancel"
    data = phase_collect(cam, fx, W, H, passes)
    if data is None:
        return "cancel"
    S, lab, ref = data
    Y, G, P = list(lab["Y"]), list(lab["G"]), list(lab["P"])
    K = [KIND_STATIC] * len(S)
    T = [float("nan")] * len(S)
    lag = float("nan")

    # 動く点: 静止点だけのモデルで遅れを推定し、遅れ分ずらした点の位置を正解にする
    if a.pursuit_seconds > 0:
        if not show_pursuit_guide(cam, fx, W, H, a.pursuit_seconds):
            return "cancel"
        pr = phase_pursuit(cam, fx, W, H, a.pursuit_seconds)
        if pr is None:
            return "cancel"
        Sp, tp, track_t, track_xy = pr
        if len(Sp) >= 30 and len(S) >= 10:
            Xs = np.array([x.feat for x in S])
            base = GazeModel.fit(Xs, np.array(Y, float), 1.0, *_z0o0(Xs))
            pred = base.predict(np.array([x.feat for x in Sp]))
            tgt, keep, lag = align_pursuit(pred, tp, track_t, track_xy)
            g0 = max(G) + 1 if G else 0
            seg = ((tp - tp[0]) // PURSUIT_SEGMENT).astype(int)
            for x, yv, k, sg, t in zip(Sp, tgt, keep, seg, tp):
                if k:
                    S.append(x); Y.append(tuple(yv)); G.append(g0 + int(sg)); P.append(-1)
                    K.append(KIND_PURSUIT); T.append(float(t))
            print(f"動く点: {len(Sp)} サンプル中 {int(keep.sum())} を使用、遅れ {lag * 1000:.0f} ms")

    mpts = motion_points(a.motion_points)
    if mpts:
        if not show_motion_guide(cam, fx, W, H, len(mpts), a.motion_seconds):
            return "cancel"
        m = phase_motion(cam, fx, W, H, mpts, a.motion_seconds, gid0=max(G) + 1, ref=ref)
        if m is None:
            return "cancel"
        Sm, Ym, Gm = m
        S += Sm; Y += Ym; G += Gm; P += [-1] * len(Sm); K += [KIND_MOTION] * len(Sm); T += [float("nan")] * len(Sm)

    X, RL, RM, PA = stack(S)
    Y, G, P, K, T = np.array(Y, float), np.array(G), np.array(P), np.array(K), np.array(T)
    SES = np.zeros(len(X), int)            # 何回前のキャリブレーションのデータか (0 = 今回)
    prev = load_previous(a.profile, W, H, X.shape[1], PA.shape[1:], a.accumulate - 1)
    if prev is not None:
        pX, pY, pG, pP, pK, pT, pPA, pS, pRL, pRM = prev
        G0 = int(G.max()) + 1
        X = np.vstack([X, pX]); Y = np.vstack([Y, pY]); G = np.concatenate([G, pG + G0 + 100000 * pS])
        P = np.concatenate([P, np.where(pP >= 0, pP + 100 * pS, pP)]); K = np.concatenate([K, pK]); T = np.concatenate([T, pT])
        PA = np.concatenate([PA, pPA]); SES = np.concatenate([SES, pS]); RL = np.concatenate([RL, pRL]); RM = np.concatenate([RM, pRM])
        print(f"前回までのデータを追加: {len(pX)} サンプル ({len(np.unique(pS))} 回分)")
    print(f"samples={len(X)} (今回 {int((SES == 0).sum())}), points={len(np.unique(G))}, static={int((K == KIND_STATIC).sum())}, "
          f"pursuit={int((K == KIND_PURSUIT).sum())}, motion={int((K == KIND_MOTION).sum())}")
    prof = fit_profile(a.profile, X, Y, G, W, H, P, K, patches=PA, pursuit_lag_s=lag)
    print(f"交差検証誤差: {prof.cv_error_px:.1f} px (横 {prof.error_x_px:.1f} / 縦 {prof.error_y_px:.1f}, "
          f"静止 {prof.same_pose_error_px:.1f}, 動く点 {prof.pursuit_error_px:.1f}, 頭を動かしたとき {prof.motion_error_px:.1f}, "
          f"目の画像 {'使用' if prof.uses_appearance else '不使用'} 効果 {prof.appearance_gain_px:+.1f} px, "
          f"除外 {getattr(prof, 'n_dropped', 0)} サンプル)")
    prof.n_sessions = int(len(np.unique(SES)))
    r = phase_verify(cam, fx, W, H, prof)
    if r == "save":
        prof.save()
        # 生データも残す (モデルの調整や再学習、次回の追加学習に使う)。SES は何回前のデータか
        np.savez_compressed(PROFILE_DIR / f"{a.profile}_calib.npz", X=X, Y=Y, G=G, P=P, K=K, T=T, L=RL, M=RM, PA=PA,
                            SES=SES, screen=np.array([W, H]), cam=np.array([cw, ch]), lag=np.array(lag),
                            ref=ref if ref is not None else np.zeros(0))
    return r


def load_previous(profile, W, H, n_feat, patch_shape, max_sessions):
    """前回までのキャリブレーションの生データを読む。画面サイズや特徴の形が違う、古い形式などなら None。
    返すデータの SES は 1 = 前回, 2 = 前々回 ...。max_sessions 回分まで。"""
    if max_sessions <= 0:
        return None
    path = PROFILE_DIR / f"{profile}_calib.npz"
    if not path.exists():
        return None
    try:
        d = np.load(path)
        if tuple(d["screen"]) != (W, H) or d["X"].shape[1] != n_feat or "PA" not in d.files \
                or tuple(d["PA"].shape[1:]) != tuple(patch_shape) or len(d["PA"]) != len(d["X"]):
            print("前回のデータは画面や形式が違うので使いません")
            return None
        ses = (d["SES"] if "SES" in d.files else np.zeros(len(d["X"]), int)) + 1
        m = ses <= max_sessions
        T = d["T"] if "T" in d.files else np.full(len(d["X"]), np.nan)
        RL = d["L"] if "L" in d.files and len(d["L"]) == len(d["X"]) else np.zeros((len(d["X"]), 478, 3), np.float32)
        RM = d["M"] if "M" in d.files and len(d["M"]) == len(d["X"]) else np.zeros((len(d["X"]), 4, 4), np.float32)
        return (d["X"][m], d["Y"][m], d["G"][m], d["P"][m], d["K"][m], T[m], d["PA"][m], ses[m], RL[m], RM[m])
    except Exception as e:
        print("前回のデータを読めません:", e)
        return None


def _z0o0(X):
    from tracker.model import _scale_params
    return _scale_params(X)


if __name__ == "__main__":
    sys.exit(main())

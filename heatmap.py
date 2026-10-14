"""ログ (logs/gaze_*.jsonl) からヒートマップ PNG を作る。

例:
  python heatmap.py logs/gaze_20260921_203000.jsonl
  python heatmap.py logs/gaze_20260921_203000.jsonl --start 120 --end 300  # 開始からの秒で範囲指定
  python heatmap.py logs/gaze_20260921_203000.jsonl --bg capture.png       # ゲーム画面に重ねる
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--start", type=float, default=0, help="開始からの秒")
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--bg", help="背景画像 (任意)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sigma", type=float, default=40, help="ぼかし半径 px")
    a = ap.parse_args()

    lines = Path(a.log).read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    W, H = header["screen_w"], header["screen_h"]
    t0 = header["start_unix"] * 1000
    acc = np.zeros((H, W), np.float32)
    n = 0
    for ln in lines[1:]:
        m = json.loads(ln)
        if not m.get("valid"):
            continue
        rel = (m["t"] - t0) / 1000
        if rel < a.start or (a.end is not None and rel > a.end):
            continue
        x = int(np.clip(m["x"], 0, 0.999) * W)
        y = int(np.clip(m["y"], 0, 0.999) * H)
        acc[y, x] += 1
        n += 1
    if n == 0:
        print("該当データがありません")
        return
    k = int(a.sigma * 3) | 1
    acc = cv2.GaussianBlur(acc, (k, k), a.sigma)
    acc = (acc / acc.max() * 255).astype(np.uint8)
    heat = cv2.applyColorMap(acc, cv2.COLORMAP_JET)
    if a.bg:
        bg = cv2.resize(cv2.imread(a.bg), (W, H))
        alpha = (acc.astype(np.float32) / 255 * 0.7)[..., None]
        out = (bg * (1 - alpha) + heat * alpha).astype(np.uint8)
    else:
        out = heat
    dest = a.out or str(Path(a.log).with_suffix(".heatmap.png"))
    cv2.imwrite(dest, out)
    print(f"{n} samples -> {dest}")


if __name__ == "__main__":
    main()

"""キャリブレーションの生データ (profiles/<name>_calib.npz) を解析する。

- 生のランドマークと頭部姿勢から特徴量を再計算し、いくつかの特徴量の組で
  「注視点をまたいだ交差検証誤差」を比べる (静止 / 頭を動かしたとき を別々に)
- 頭の動きの範囲や、各点での推定のばらつきも出す

例:
  python analyze_calib.py default
  python analyze_calib.py "dist/GazeTracker/profiles/default_calib.npz"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tracker import features as F
from tracker.model import KIND_MOTION, KIND_STATIC, cv_predict
from tracker.paths import PROFILE_DIR

I = F.IDX


def fs_2d(X):
    X = np.asarray(X)
    h = (X[:, I["l_h"]] + X[:, I["r_h"]]) / 2
    v = (X[:, I["l_v"]] + X[:, I["r_v"]]) / 2
    yaw, pitch, tx, ty, tz = (X[:, I[k]] for k in ("yaw", "pitch", "tx", "ty", "tz"))
    return np.column_stack([h, v, yaw, pitch, tx, ty, tz, h * tz, v * tz, yaw * tz, pitch * tz,
                            h * yaw, v * pitch, h * tx, v * ty, yaw * tx, pitch * ty])


def fs_2d_linear(X):
    X = np.asarray(X)
    h = (X[:, I["l_h"]] + X[:, I["r_h"]]) / 2
    v = (X[:, I["l_v"]] + X[:, I["r_v"]]) / 2
    return np.column_stack([h, v] + [X[:, I[k]] for k in ("yaw", "pitch", "tx", "ty", "tz")])


def fs_2d_pereye(X):
    X = np.asarray(X)
    cols = [X[:, I[k]] for k in ("l_h", "l_v", "r_h", "r_v", "yaw", "pitch", "tx", "ty", "tz")]
    h = (X[:, I["l_h"]] + X[:, I["r_h"]]) / 2
    v = (X[:, I["l_v"]] + X[:, I["r_v"]]) / 2
    tz, yaw, pitch = X[:, I["tz"]], X[:, I["yaw"]], X[:, I["pitch"]]
    return np.column_stack(cols + [h * tz, v * tz, yaw * tz, pitch * tz, h * yaw, v * pitch])


def fs_2d_noHead(X):
    X = np.asarray(X)
    h = (X[:, I["l_h"]] + X[:, I["r_h"]]) / 2
    v = (X[:, I["l_v"]] + X[:, I["r_v"]]) / 2
    return np.column_stack([h, v])


def fs_3d(X):
    X = np.asarray(X)
    ta, te, ox, oy, oz, yaw, pitch = (X[:, I[k]] for k in ("g_ta", "g_te", "o_x", "o_y", "o_z", "yaw", "pitch"))
    return np.column_stack([ta, te, ox, oy, oz, ta * oz, te * oz, yaw, pitch])


def fs_both(X):
    return np.column_stack([fs_2d(X), fs_3d(X)])


FEATURE_SETS = {
    "2D 虹彩+頭姿勢+積 (現行)": fs_2d,
    "2D 虹彩+頭姿勢 (線形のみ)": fs_2d_linear,
    "2D 左右別+頭姿勢+積": fs_2d_pereye,
    "2D 虹彩のみ (頭無視)": fs_2d_noHead,
    "3D 眼球モデル": fs_3d,
    "2D + 3D": fs_both,
}


def cv_errors(fs, X, Y, G, K, W, H):
    A = fs(X)
    sd = A.std(axis=0)
    A = (A - A.mean(axis=0)) / np.where(sd > 0, sd, 1)      # 比較用なので標準化する
    best = None
    for alpha in (0.03, 0.1, 0.3, 1, 3, 10, 100):
        pred = cv_predict(A, Y, G, alpha)
        e = np.linalg.norm((pred - Y) * [W, H], axis=1)
        tot = e.mean()
        if best is None or tot < best[0]:
            best = (tot, alpha, e)
    tot, alpha, e = best
    st = e[K == KIND_STATIC].mean() if np.any(K == KIND_STATIC) else float("nan")
    mo = e[K == KIND_MOTION].mean() if np.any(K == KIND_MOTION) else float("nan")
    return tot, st, mo, alpha


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "default"
    path = Path(arg) if arg.endswith(".npz") else PROFILE_DIR / f"{arg}_calib.npz"
    d = np.load(path)
    W, H = map(int, d["screen"])
    Y, G, K = d["Y"], d["G"], d["K"]
    print(f"{path}\n  画面 {W}x{H}  サンプル {len(Y)}  点 {len(np.unique(G))}  "
          f"静止 {(K == KIND_STATIC).sum()}  動き {(K == KIND_MOTION).sum()}")
    if "L" in d and len(d["L"]):
        cw, ch = map(int, d["cam"])
        X = np.array([F.features_from_raw(L.astype(np.float64), M.astype(np.float64), cw, ch)
                      for L, M in zip(d["L"], d["M"])])
        print(f"  生データから特徴量を再計算 (カメラ {cw}x{ch})")
    else:
        X = d["X"]
        if X.shape[1] != len(F.FEATURE_NAMES):
            print("  生データが無く、特徴量の次元が今の定義と違うため再計算できません")
            return

    # 頭の動きの範囲
    for k in ("yaw", "pitch", "tx", "ty", "tz"):
        col = X[:, I[k]]
        unit = "deg" if k in ("yaw", "pitch") else "cm"
        conv = np.degrees if k in ("yaw", "pitch") else (lambda a: a)
        s = conv(col[K == KIND_STATIC]); m = conv(col[K == KIND_MOTION]) if np.any(K == KIND_MOTION) else s
        print(f"  {k:5s} 静止 {s.min():7.1f}..{s.max():6.1f} {unit}   動き {m.min():7.1f}..{m.max():6.1f} {unit}")
    # 静止点ごとのばらつき (虹彩位置の安定度)
    h = (X[:, I["l_h"]] + X[:, I["r_h"]]) / 2
    sd = [h[(G == g) & (K == KIND_STATIC)].std() for g in np.unique(G[K == KIND_STATIC])]
    rng_h = h[K == KIND_STATIC].max() - h[K == KIND_STATIC].min()
    print(f"  虹彩 h: 点内ばらつき 中央値 {np.median(sd):.4f}  画面全体の振れ幅 {rng_h:.3f}  (比 {np.median(sd) / max(rng_h, 1e-9) * 100:.1f}%)")

    print("\n  特徴量の組                  全体    静止    動き   alpha")
    for name, fs in FEATURE_SETS.items():
        tot, st, mo, alpha = cv_errors(fs, X, Y, G, K, W, H)
        print(f"  {name:26s} {tot:6.0f} {st:7.0f} {mo:7.0f}   {alpha}")


if __name__ == "__main__":
    main()

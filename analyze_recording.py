"""検証用の録画 (recordings/<日時>/) を処理し直して、手法ごとの精度を比べる。

キャリブレーションで「検証用に録画する」をオンにすると録画が残る。それを使って、
同じデータに対して次の組み合わせを試し、横・縦それぞれの誤差を表にする。

  画像の前処理:  なし / CLAHE (局所的なコントラスト強調) / ガンマ補正 (暗部を持ち上げる)
  解像度:       そのまま / 640x480 相当に縮小 (--downscale)
  学習データ:   静止点のみ / 静止点 + 動く点
  特徴:         ランドマークのみ / + 目の画像 (PCA)

評価は「注視点 (動く点は 2 秒ごとの区切り) を丸ごと除いて学習し、除いた分を予測する」交差検証。

例:
  python analyze_recording.py recordings/20261014_221933
  python analyze_recording.py recordings/20261014_221933 --pre none clahe --downscale
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from tracker.features import FeatureExtractor
from tracker.model import (KIND_MOTION, KIND_PURSUIT, KIND_STATIC, AppearancePCA, GazeModel, _scale_params,
                           align_pursuit, compact_features, cv_predict, geometric_features)

PURSUIT_SEGMENT = 2.0


def preprocess(name: str):
    if name == "none":
        return None
    if name == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        def f(img):
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return f
    if name == "gamma":
        lut = np.array([((i / 255.0) ** 0.7) * 255 for i in range(256)], np.uint8)
        return lambda img: cv2.LUT(img, lut)
    raise ValueError(name)


def load(rec_dir: Path):
    meta = json.loads((rec_dir / "meta.json").read_text(encoding="utf-8"))
    frames = [json.loads(l) for l in (rec_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    return meta, frames


def extract(rec_dir: Path, frames, pre: str, downscale: bool):
    """映像を読み直して各フレームのサンプルを作る。"""
    fn = preprocess(pre)
    fx = FeatureExtractor()
    cap = cv2.VideoCapture(str(rec_dir / "video.avi"))
    out = []
    for rec in frames:
        ok, img = cap.read()
        if not ok:
            break
        if downscale:
            img = cv2.resize(cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA),
                             (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        if fn is not None:
            img = fn(img)
        s = fx.process(img, int(rec["t"] * 1000))
        out.append(s)
    cap.release()
    fx.close()
    return out


def build(frames, samples):
    """静止点・動く点・頭を動かした点のデータセットを作る。動く点はここで遅れを推定して正解をずらす。"""
    X, Y, G, K, PA = [], [], [], [], []
    for rec, s in zip(frames, samples):
        if rec["phase"] == "static" and rec.get("collecting") and s.ok and not s.blink:
            X.append(s.feat); Y.append((rec["tx"], rec["ty"])); G.append(rec["gid"]); K.append(KIND_STATIC); PA.append(s.patches)
    if not X:
        raise SystemExit("静止点のデータがありません")
    X, Y = np.array(X), np.array(Y, float)
    z0, o0 = _scale_params(X)
    base = GazeModel.fit(X, Y, 1.0, z0, o0)

    pur = [(rec, s) for rec, s in zip(frames, samples)
           if rec["phase"] == "pursuit" and rec.get("moving") and s.ok and not s.blink]
    lag = float("nan")
    Xp = Yp = Gp = PAp = None
    if len(pur) >= 30:
        t = np.array([r["t"] for r, _ in pur])
        t0 = min(r["t"] for r in frames if r["phase"] == "pursuit" and r.get("moving"))
        pur = [(r, s) for r, s in pur if r["t"] >= t0 + 0.5]
        t = np.array([r["t"] for r, _ in pur])
        track_t = np.array([r["t"] for r in frames if r["phase"] == "pursuit"])
        track_xy = np.array([(r["tx"], r["ty"]) for r in frames if r["phase"] == "pursuit"])
        feats = np.array([s.feat for _, s in pur])
        tgt, keep, lag = align_pursuit(base.predict(feats), t, track_t, track_xy)
        g0 = max(G) + 1
        Xp, Yp = feats[keep], tgt[keep]
        Gp = (g0 + ((t - t[0]) // PURSUIT_SEGMENT).astype(int))[keep]
        PAp = np.array([s.patches for _, s in pur])[keep]
    mot = [(rec, s) for rec, s in zip(frames, samples) if rec["phase"] == "motion" and s.ok and not s.blink]
    data = {"static": (X, Y, np.array(G), np.array(PA))}
    if Xp is not None:
        data["pursuit"] = (Xp, Yp, Gp, PAp)
    if mot:
        data["motion"] = (np.array([s.feat for _, s in mot]), np.array([(r["tx"], r["ty"]) for r, _ in mot], float),
                          np.array([10000 + r["gid"] for r, _ in mot]), np.array([s.patches for _, s in mot]))
    return data, lag, (z0, o0)


def design(X, PA, z0, o0, pca):
    A = geometric_features(compact_features(X), z0, o0)
    return np.hstack([A, pca.transform(PA)]) if pca is not None else A


def evaluate(data, parts, use_app, z0o0, W, H):
    """parts を学習・評価に使い、交差検証の誤差 (横, 縦) をデータの種類ごとに返す。"""
    X = np.vstack([data[p][0] for p in parts]); Y = np.vstack([data[p][1] for p in parts])
    G = np.concatenate([data[p][2] for p in parts]); PA = np.concatenate([data[p][3] for p in parts])
    kind = np.concatenate([[p] * len(data[p][0]) for p in parts])
    pca = AppearancePCA.fit(PA) if use_app else None
    A = design(X, PA, *z0o0, pca)
    best = None
    for alpha in (0.1, 0.3, 1, 3, 10, 30, 100):
        pred = cv_predict(A, Y, G, alpha)
        e = np.linalg.norm((pred - Y) * [W, H], axis=1).mean()
        if best is None or e < best[0]:
            best = (e, alpha, pred)
    _, alpha, pred = best
    res = {}
    for p in parts:
        m = kind == p
        res[p] = (np.abs(pred[m, 0] - Y[m, 0]).mean() * W, np.abs(pred[m, 1] - Y[m, 1]).mean() * H)
    # 静止点だけで学習したときの、動く点への汎化 (学習に入れていない動く点を予測)
    if parts == ["static"] and "pursuit" in data:
        mdl_pca = pca
        A_tr = A
        from tracker.model import ridge_fit
        coef, b = ridge_fit(A_tr, Y, alpha)
        Xp, Yp, _, PAp = data["pursuit"]
        pp = design(Xp, PAp, *z0o0, mdl_pca) @ coef + b
        res["pursuit (未学習)"] = (np.abs(pp[:, 0] - Yp[:, 0]).mean() * W, np.abs(pp[:, 1] - Yp[:, 1]).mean() * H)
    return res, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", help="recordings/<日時> のフォルダ")
    ap.add_argument("--pre", nargs="+", default=["none", "clahe", "gamma"], help="前処理 (none clahe gamma)")
    ap.add_argument("--downscale", action="store_true", help="640x480 相当に落とした場合も比べる")
    ap.add_argument("--out", help="結果を JSON で保存するファイル")
    a = ap.parse_args()
    rec_dir = Path(a.recording)
    meta, frames = load(rec_dir)
    W, H = meta["screen_w"], meta["screen_h"]
    print(f"{rec_dir}  カメラ {meta.get('camera')}  画面 {W}x{H}  フレーム {len(frames)}")
    rows = []
    for ds in ([False, True] if a.downscale else [False]):
        for pre in a.pre:
            t0 = time.time()
            samples = extract(rec_dir, frames, pre, ds)
            face = np.mean([s.ok for s in samples])
            data, lag, z0o0 = build(frames, samples)
            label = f"{pre}{' / 640x480相当' if ds else ''}"
            print(f"\n== 前処理 {label}  (顔検出 {face * 100:.0f}%, 動く点の遅れ {lag * 1000:.0f} ms, {time.time() - t0:.0f} 秒)")
            print(f"   {'学習データ':16s} {'特徴':14s} {'評価':18s} {'横 px':>7s} {'縦 px':>7s}")
            combos = [["static"]]
            if "pursuit" in data:
                combos.append(["static", "pursuit"])
            if "motion" in data:
                combos.append(list(data.keys()))
            for parts in combos:
                for use_app in (False, True):
                    res, alpha = evaluate(data, parts, use_app, z0o0, W, H)
                    for k, (ex, ey) in res.items():
                        name = "+".join(parts)
                        feat = "ランドマーク+目画像" if use_app else "ランドマーク"
                        print(f"   {name:16s} {feat:14s} {k:18s} {ex:7.0f} {ey:7.0f}")
                        rows.append({"pre": pre, "downscale": ds, "train": name, "appearance": use_app,
                                     "eval": k, "err_x": ex, "err_y": ey, "alpha": alpha, "lag": lag})
    if a.out:
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n結果: {a.out}")


if __name__ == "__main__":
    main()

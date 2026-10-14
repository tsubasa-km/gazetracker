"""特徴量 -> 画面座標 (0..1 正規化) の回帰モデルとプロファイル入出力。

回帰は Ridge 回帰と、注視点ごとに分けた交差検証だけなので numpy で実装している
(scikit-learn と SciPy を同梱すると exe が 80MB ほど大きくなるため)。

入力は 2 種類:
  - ランドマーク特徴: 虹彩位置・まぶたの開き・頭部姿勢 (geometric_features)
  - 見た目の特徴 (任意): 目の周りの画像を主成分分析 (PCA) で圧縮したもの。
    ランドマークでは取りにくい、まぶたの形や白目の見え方といった縦方向の手がかりを拾う。
    キャリブレーションの交差検証で誤差が下がるときだけ使う。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from .features import COMPACT_NAMES, FEATURE_NAMES, compact_features
from .paths import PROFILE_DIR

PROFILE_VERSION = 4   # 4: モデルを JSON で保存 (3 までは scikit-learn の joblib)

KIND_STATIC = 0    # 通常姿勢で点を見つめたサンプル
KIND_MOTION = 1    # 点を見つめたまま頭を動かしたサンプル
KIND_PURSUIT = 2   # 動く点を目で追ったサンプル

APPEARANCE_DIMS = (16, 32)  # 見た目の特徴の主成分の数の候補 (交差検証で選ぶ)
APPEARANCE_DIM = 16
APPEARANCE_MIN_SAMPLES = 200


def geometric_features(C: np.ndarray, z0: float = 40.0, o0: float = 0.35) -> np.ndarray:
    """compact_features の 11 次元から回帰の入力を作る (実データでの比較で最良だった構成)。

    - 左右の目それぞれの虹彩位置の 2 次多項式 (h, v, h², v², h·v): 古典的な 9 点キャリブレーションと同じ形。
      虹彩は球面上を動き、目尻・目頭の線も水平でないので、縦位置が横位置に依存する歪みを吸収する
    - まぶたの開き: 縦方向の手がかり
    - 頭の位置・向きと、距離 tz との積: 頭が動いたときの補正
    標準化はしない。キャリブレーション中にほとんど変化しない列 (頭の位置など) を
    標準化すると係数が暴れ、プレイ中に頭が動いたときに大きく外れるため、物理的な単位で固定スケールにする。
    z0, o0 はキャリブレーション時の距離とまぶたの開きの中央値。"""
    C = np.asarray(C, dtype=np.float64)
    lh, lv, rh, rv, lo, ro, yaw, pitch, tx, ty, tz = C.T
    tz = np.abs(tz)
    h, v = (lh + rh) / 2, (lv + rv) / 2

    def poly(a, b):
        return [a / 0.1, b / 0.05, (a * a) / 0.01, (b * b) / 0.0025, (a * b) / 0.005]

    cols = poly(lh, lv) + poly(rh, rv) + [
        (lo - o0) / 0.03, (ro - o0) / 0.03,
        tx / 5, ty / 5, (tz - z0) / 5,
        (tz * h) / 4, (tz * v) / 2, (tz * yaw) / 8, (tz * pitch) / 8,
        yaw / 0.2, pitch / 0.2,
    ]
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# 見た目の特徴 (目のパッチの PCA)
# ---------------------------------------------------------------------------
def patch_vectors(patches: np.ndarray) -> np.ndarray:
    """(N, 2, h, w) uint8 → (N, 2*h*w)。目ごとに明るさとコントラストをそろえる (照明の変化に強くする)。"""
    P = np.asarray(patches, dtype=np.float32).reshape(len(patches), 2, -1)
    P = (P - P.mean(axis=2, keepdims=True)) / (P.std(axis=2, keepdims=True) + 1e-3)
    return P.reshape(len(patches), -1)


@dataclass
class AppearancePCA:
    mean: np.ndarray        # (D,)
    comp: np.ndarray        # (k, D)
    scale: np.ndarray       # (k,)  各主成分の標準偏差 (スコアを 1 程度にそろえる)

    @classmethod
    def fit(cls, patches: np.ndarray, k: int = APPEARANCE_DIM) -> "AppearancePCA":
        V = patch_vectors(patches)
        mean = V.mean(axis=0)
        _, S, Vt = np.linalg.svd(V - mean, full_matrices=False)
        k = min(k, len(S))
        scale = S[:k] / np.sqrt(max(len(V) - 1, 1))
        return cls(mean.astype(np.float32), Vt[:k].astype(np.float32), np.maximum(scale, 1e-6).astype(np.float32))

    def transform(self, patches: np.ndarray) -> np.ndarray:
        return ((patch_vectors(patches) - self.mean) @ self.comp.T) / self.scale

    def to_dict(self):
        return {"mean": self.mean.round(4).tolist(), "comp": self.comp.round(5).tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["mean"], np.float32), np.array(d["comp"], np.float32), np.array(d["scale"], np.float32))


# ---------------------------------------------------------------------------
# Ridge 回帰と交差検証 (scikit-learn の Ridge / GroupKFold と同じ計算)
# ---------------------------------------------------------------------------
def ridge_fit(A: np.ndarray, Y: np.ndarray, alpha: float):
    """切片には正則化をかけない Ridge 回帰。(係数, 切片) を返す。"""
    A = np.asarray(A, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    am, ym = A.mean(axis=0), Y.mean(axis=0)
    Ac, Yc = A - am, Y - ym
    coef = np.linalg.solve(Ac.T @ Ac + alpha * np.eye(A.shape[1]), Ac.T @ Yc)
    return coef, ym - am @ coef


def group_kfold(groups: np.ndarray, n_splits: int):
    """グループ (注視点) を分けずに n_splits 個に分ける。大きいグループから、いちばん小さい組に入れていく。
    返り値: 各サンプルの組番号。"""
    groups = np.asarray(groups)
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    fold_of_group = np.zeros(len(uniq), dtype=int)
    sizes = np.zeros(n_splits, dtype=int)
    for g in np.argsort(-counts, kind="stable"):
        f = int(np.argmin(sizes))
        fold_of_group[g] = f
        sizes[f] += counts[g]
    return fold_of_group[inv]


def cv_predict(A: np.ndarray, Y: np.ndarray, groups: np.ndarray, alpha: float, n_splits: int = 5):
    """グループ単位の交差検証で、各サンプルを「そのグループを除いて学習したモデル」で予測する。"""
    n = len(np.unique(groups))
    if n < 2:
        return None
    folds = group_kfold(groups, min(n_splits, n))
    pred = np.zeros_like(np.asarray(Y, dtype=np.float64))
    for f in np.unique(folds):
        te = folds == f
        coef, b = ridge_fit(A[~te], Y[~te], alpha)
        pred[te] = A[te] @ coef + b
    return pred


@dataclass
class GazeModel:
    """特徴ベクトル (FEATURE_NAMES) [+ 目のパッチ] → 画面の正規化座標。"""
    alpha: float
    z0: float
    o0: float
    coef: np.ndarray        # (入力次元, 2)
    intercept: np.ndarray   # (2,)
    appearance: AppearancePCA | None = None

    def design(self, X, patches=None) -> np.ndarray:
        A = geometric_features(compact_features(X), self.z0, self.o0)
        if self.appearance is not None:
            if patches is None:
                raise ValueError("このモデルは目の画像 (patches) が必要です")
            A = np.hstack([A, self.appearance.transform(np.asarray(patches).reshape(len(A), 2, *np.shape(patches)[-2:]))])
        return A

    def predict(self, X, patches=None) -> np.ndarray:
        return self.design(X, patches) @ self.coef + self.intercept

    @classmethod
    def fit(cls, X, Y, alpha: float, z0: float, o0: float, patches=None, use_appearance=False,
            k: int = APPEARANCE_DIM) -> "GazeModel":
        app = AppearancePCA.fit(patches, k) if use_appearance else None
        m = cls(alpha, z0, o0, np.zeros((1, 2)), np.zeros(2), app)
        coef, b = ridge_fit(m.design(X, patches), Y, alpha)
        m.coef, m.intercept = coef, b
        return m

    def to_dict(self) -> dict:
        d = {"alpha": self.alpha, "z0": self.z0, "o0": self.o0,
             "coef": self.coef.tolist(), "intercept": self.intercept.tolist()}
        if self.appearance is not None:
            d["appearance"] = self.appearance.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GazeModel":
        app = AppearancePCA.from_dict(d["appearance"]) if d.get("appearance") else None
        return cls(float(d["alpha"]), float(d["z0"]), float(d["o0"]),
                   np.array(d["coef"], dtype=np.float64), np.array(d["intercept"], dtype=np.float64), app)


# 頭を動かしたサンプルのうち、通常姿勢からこれ以上離れたものは学習に使わない
# (プレイ中に起きない極端な姿勢は虹彩の検出も崩れ、モデルを歪める)
MOTION_LIMIT = dict(yaw_deg=25.0, pitch_deg=20.0, dz_cm=12.0)


@dataclass
class Profile:
    name: str
    screen_w: int
    screen_h: int
    model: GazeModel
    created: float = field(default_factory=time.time)
    cv_error_px: float = 0.0          # 点をまたいだ交差検証誤差 (全サンプル)
    same_pose_error_px: float = 0.0   # 通常姿勢のサンプルでの誤差
    motion_error_px: float = 0.0      # 頭を動かしたサンプルでの誤差 (無ければ nan)
    pursuit_error_px: float = float("nan")   # 動く点を追ったサンプルでの誤差
    error_x_px: float = float("nan")  # 横方向だけの誤差
    error_y_px: float = float("nan")  # 縦方向だけの誤差
    uses_appearance: bool = False
    appearance_gain_px: float = 0.0   # 見た目の特徴で減った誤差 (正なら改善)
    pursuit_lag_s: float = float("nan")
    n_sessions: int = 1               # 何回分のキャリブレーションのデータで学習したか
    n_passes: int = 1
    n_motion_points: int = 0
    version: int = PROFILE_VERSION

    META_KEYS = ("name", "screen_w", "screen_h", "created", "cv_error_px", "same_pose_error_px",
                 "motion_error_px", "pursuit_error_px", "error_x_px", "error_y_px", "uses_appearance",
                 "appearance_gain_px", "pursuit_lag_s", "n_sessions", "n_passes", "n_motion_points", "version")

    def predict(self, feat: np.ndarray, patches: np.ndarray | None = None) -> tuple[float, float]:
        p = None if patches is None else np.asarray(patches)[None]
        xy = self.model.predict(np.asarray(feat).reshape(1, -1), p)[0]
        return float(np.clip(xy[0], -0.1, 1.1)), float(np.clip(xy[1], -0.1, 1.1))

    def save(self):
        PROFILE_DIR.mkdir(exist_ok=True)
        meta = {k: getattr(self, k) for k in self.META_KEYS}
        meta = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in meta.items()}
        meta["model"] = self.model.to_dict()
        (PROFILE_DIR / f"{self.name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        old = PROFILE_DIR / f"{self.name}.joblib"     # 旧形式のモデルファイルは不要
        if old.exists():
            old.unlink()

    @classmethod
    def load(cls, name: str) -> "Profile":
        meta = json.loads((PROFILE_DIR / f"{name}.json").read_text(encoding="utf-8"))
        if meta.get("version", 1) != PROFILE_VERSION or "model" not in meta:
            prof = _migrate(name)
            if prof is None:
                raise ValueError(f"プロファイル '{name}' は古い形式です。キャリブレーションをやり直してください。")
            return prof
        model = GazeModel.from_dict(meta.pop("model"))
        meta = {k: (float("nan") if v is None else v) for k, v in meta.items() if k in cls.META_KEYS}
        return cls(model=model, **meta)


def _migrate(name: str) -> "Profile | None":
    """古い形式のプロファイルを、保存してあるキャリブレーションの生データから学習し直す。"""
    raw = PROFILE_DIR / f"{name}_calib.npz"
    if not raw.exists():
        return None
    d = np.load(raw)
    X = np.zeros((len(d["X"]), len(FEATURE_NAMES)))
    X[:, :d["X"].shape[1]] = d["X"]           # 特徴の列が少ない古いデータは 0 で埋める (回帰が使う列は先頭にある)
    W, H = (int(v) for v in d["screen"])
    PA = d["PA"] if "PA" in d.files and len(d["PA"]) == len(X) else None
    prof = fit_profile(name, X, d["Y"], d["G"], W, H, d["P"], d["K"], patches=PA)
    prof.save()
    return prof


def _scale_params(X_static):
    C = compact_features(X_static)
    ci = {n: i for i, n in enumerate(COMPACT_NAMES)}
    z0 = float(np.median(np.abs(C[:, ci["tz"]])))
    o0 = float(np.median((C[:, ci["l_open"]] + C[:, ci["r_open"]]) / 2))
    return z0, o0


def _moderate_mask(X, static):
    """通常姿勢の中央値から MOTION_LIMIT 以内のサンプルだけ True。"""
    C = compact_features(X)
    ci = {n: i for i, n in enumerate(COMPACT_NAMES)}
    yaw, pitch, tz = np.degrees(C[:, ci["yaw"]]), np.degrees(C[:, ci["pitch"]]), np.abs(C[:, ci["tz"]])
    ref = static if np.any(static) else np.ones(len(X), bool)
    y0, p0, z0 = np.median(yaw[ref]), np.median(pitch[ref]), np.median(tz[ref])
    return ((np.abs(yaw - y0) <= MOTION_LIMIT["yaw_deg"]) & (np.abs(pitch - p0) <= MOTION_LIMIT["pitch_deg"])
            & (np.abs(tz - z0) <= MOTION_LIMIT["dz_cm"]))


def _px_err(pred, Y, mask, screen_w, screen_h) -> float:
    if pred is None or not np.any(mask):
        return float("nan")
    return float(np.linalg.norm((pred[mask] - Y[mask]) * [screen_w, screen_h], axis=1).mean())


def _axis_err(pred, Y, mask, scale) -> float:
    if pred is None or not np.any(mask):
        return float("nan")
    return float(np.abs(pred[mask] - Y[mask]).mean() * scale)


# ---------------------------------------------------------------------------
# 動く点のデータの遅れ補正
# ---------------------------------------------------------------------------
def align_pursuit(pred_xy: np.ndarray, t: np.ndarray, track_t: np.ndarray, track_xy: np.ndarray,
                  max_lag: float = 0.6, step: float = 0.01, outlier: float = 2.5):
    """目は動く点から少し遅れて追う (+ 表示と処理の遅れ)。
    静止点だけで作ったモデルの予測 pred_xy と、点の軌跡 (track_t, track_xy) を比べて遅れを探す。
    遅れは「予測と点の位置の相関が最も高くなるずれ」で決める。縦は手がかりが弱く相関の山がぼやけるので、
    横と縦のうち相関の高い (信頼できる) 軸で決める。誤差の中央値で決めるより、モデルの偏りに引きずられない。
    その遅れ分だけ前の点の位置を正解とし、追いつくための素早い目の動き (サッカード) やまばたき直後など、
    大きく外れたサンプルは除く。
    返り値: (正解の座標 (N, 2), 使うサンプルの mask, 遅れ秒)"""
    lags = np.arange(0.0, max_lag + 1e-9, step)
    corr = np.zeros((len(lags), 2))
    for i, lag in enumerate(lags):
        for k in (0, 1):
            tgt = np.interp(t - lag, track_t, track_xy[:, k])
            c = np.corrcoef(pred_xy[:, k], tgt)[0, 1] if np.std(tgt) > 0 and np.std(pred_xy[:, k]) > 0 else 0.0
            corr[i, k] = 0.0 if np.isnan(c) else c
    axis = int(np.argmax(corr.max(axis=0)))
    lag = float(lags[int(np.argmax(corr[:, axis]))])
    tgt = np.column_stack([np.interp(t - lag, track_t, track_xy[:, k]) for k in (0, 1)])
    err = np.linalg.norm(pred_xy - tgt, axis=1)
    keep = err <= outlier * np.median(err)
    return tgt, keep, lag


def fit_profile(name: str, X: np.ndarray, Y: np.ndarray, groups: np.ndarray,
                screen_w: int, screen_h: int, passes: np.ndarray | None = None,
                kinds: np.ndarray | None = None, patches: np.ndarray | None = None,
                pursuit_lag_s: float = float("nan")) -> Profile:
    """X: 特徴量, Y: 正規化座標(0..1), groups: 注視点 ID (動く点は数秒ごとの区切り),
    passes: 座り直しパス ID, kinds: KIND_STATIC / KIND_MOTION / KIND_PURSUIT, patches: 目のパッチ (任意)。

    正則化の強さと、見た目の特徴を使うかどうかは「ある注視点を除いて学習し、その点で評価」する
    交差検証で選ぶ。"""
    X, Y, groups = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64), np.asarray(groups)
    passes = np.zeros(len(X), dtype=int) if passes is None else np.asarray(passes)
    kinds = np.zeros(len(X), dtype=int) if kinds is None else np.asarray(kinds)
    static = kinds == KIND_STATIC

    # 極端な姿勢の動きサンプルを除外
    keep = ~(kinds == KIND_MOTION) | _moderate_mask(X, static)
    n_dropped = int((~keep).sum())
    X, Y, groups, passes, kinds = X[keep], Y[keep], groups[keep], passes[keep], kinds[keep]
    if patches is not None:
        patches = np.asarray(patches)[keep]
    static, motion, pursuit = kinds == KIND_STATIC, kinds == KIND_MOTION, kinds == KIND_PURSUIT
    n_passes = len(np.unique(passes[static])) if np.any(static) else 1
    n_motion = len(np.unique(groups[motion])) if np.any(motion) else 0
    z0, o0 = _scale_params(X[static] if np.any(static) else X)

    # 姿勢が 3 つ以上あり他のデータが無いときは、姿勢ごとに除く CV の方が実態に近い
    sel_groups = passes if (n_passes >= 3 and not np.any(~static)) else groups

    A_geo = geometric_features(compact_features(X), z0, o0)
    candidates = [(0, A_geo)]
    if patches is not None and len(patches) >= APPEARANCE_MIN_SAMPLES:
        # 交差検証の中で PCA も学習し直すと遅いので、ここでは全データの PCA を使う
        # (PCA は正解座標を見ないので、評価が甘くなる影響は小さい)
        pca = AppearancePCA.fit(patches, max(APPEARANCE_DIMS))
        scores = pca.transform(patches)
        for k in APPEARANCE_DIMS:
            candidates.append((k, np.hstack([A_geo, scores[:, :k]])))

    results = {}
    for use_app, A in candidates:
        best = None
        for alpha in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
            pred = cv_predict(A, Y, sel_groups, alpha)
            err = _px_err(pred, Y, np.ones(len(X), bool), screen_w, screen_h)
            if best is None or err < best[0] or np.isnan(best[0]):
                best = (err, alpha, pred)
        results[use_app] = best
    k_best = min(results, key=lambda k: results[k][0])
    use_app = k_best > 0
    err, alpha, pred = results[k_best]
    gain = results[0][0] - min(results[k][0] for k in results if k > 0) if len(results) > 1 else 0.0

    model = GazeModel.fit(X, Y, alpha, z0, o0, patches, use_appearance=use_app, k=k_best or APPEARANCE_DIM)
    allm = np.ones(len(X), bool)
    prof = Profile(name=name, screen_w=screen_w, screen_h=screen_h, model=model,
                   cv_error_px=err,
                   same_pose_error_px=_px_err(pred, Y, static, screen_w, screen_h),
                   motion_error_px=_px_err(pred, Y, motion, screen_w, screen_h),
                   pursuit_error_px=_px_err(pred, Y, pursuit, screen_w, screen_h),
                   error_x_px=_axis_err(pred[:, :1], Y[:, :1], allm, screen_w) if pred is not None else float("nan"),
                   error_y_px=_axis_err(pred[:, 1:], Y[:, 1:], allm, screen_h) if pred is not None else float("nan"),
                   uses_appearance=use_app, appearance_gain_px=float(gain), pursuit_lag_s=float(pursuit_lag_s),
                   n_passes=n_passes, n_motion_points=n_motion)
    prof.n_dropped = n_dropped
    return prof

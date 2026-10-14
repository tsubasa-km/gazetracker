"""特徴量 -> 画面座標 (0..1 正規化) の回帰モデルとプロファイル入出力。

回帰は Ridge 回帰と、注視点ごとに分けた交差検証だけなので numpy で実装している
(scikit-learn と SciPy を同梱すると exe が 80MB ほど大きくなるため)。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from .features import COMPACT_NAMES, FEATURE_NAMES, compact_features
from .paths import PROFILE_DIR

PROFILE_VERSION = 4   # 4: モデルを JSON で保存 (3 までは scikit-learn の joblib)

KIND_STATIC = 0   # 通常姿勢で点を見つめたサンプル
KIND_MOTION = 1   # 点を見つめたまま頭を動かしたサンプル


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
    """特徴ベクトル (FEATURE_NAMES) → 画面の正規化座標。"""
    alpha: float
    z0: float
    o0: float
    coef: np.ndarray        # (入力次元, 2)
    intercept: np.ndarray   # (2,)

    def design(self, X) -> np.ndarray:
        return geometric_features(compact_features(X), self.z0, self.o0)

    def predict(self, X) -> np.ndarray:
        return self.design(X) @ self.coef + self.intercept

    @classmethod
    def fit(cls, X, Y, alpha: float, z0: float, o0: float) -> "GazeModel":
        A = geometric_features(compact_features(X), z0, o0)
        coef, b = ridge_fit(A, Y, alpha)
        return cls(alpha, z0, o0, coef, b)

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "z0": self.z0, "o0": self.o0,
                "coef": self.coef.tolist(), "intercept": self.intercept.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "GazeModel":
        return cls(float(d["alpha"]), float(d["z0"]), float(d["o0"]),
                   np.array(d["coef"], dtype=np.float64), np.array(d["intercept"], dtype=np.float64))


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
    n_passes: int = 1
    n_motion_points: int = 0
    version: int = PROFILE_VERSION

    META_KEYS = ("name", "screen_w", "screen_h", "created", "cv_error_px", "same_pose_error_px",
                 "motion_error_px", "n_passes", "n_motion_points", "version")

    def predict(self, feat: np.ndarray) -> tuple[float, float]:
        xy = self.model.predict(np.asarray(feat).reshape(1, -1))[0]
        return float(np.clip(xy[0], -0.1, 1.1)), float(np.clip(xy[1], -0.1, 1.1))

    def save(self):
        PROFILE_DIR.mkdir(exist_ok=True)
        meta = {k: getattr(self, k) for k in self.META_KEYS}
        meta = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in meta.items()}
        meta["model"] = self.model.to_dict()
        (PROFILE_DIR / f"{self.name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
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
    prof = fit_profile(name, X, d["Y"], d["G"], W, H, d["P"], d["K"])
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


def fit_profile(name: str, X: np.ndarray, Y: np.ndarray, groups: np.ndarray,
                screen_w: int, screen_h: int, passes: np.ndarray | None = None,
                kinds: np.ndarray | None = None) -> Profile:
    """X: 特徴量, Y: 正規化座標(0..1), groups: 注視点 ID (姿勢ごとに別 ID),
    passes: 座り直しパス ID, kinds: KIND_STATIC / KIND_MOTION。

    正則化は「ある注視点を除いて学習し、その点で評価」する交差検証で選ぶ。
    頭を動かしたサンプルがあれば、それを除いたときの誤差も別に出す。"""
    X, Y, groups = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64), np.asarray(groups)
    passes = np.zeros(len(X), dtype=int) if passes is None else np.asarray(passes)
    kinds = np.zeros(len(X), dtype=int) if kinds is None else np.asarray(kinds)
    static = kinds == KIND_STATIC
    motion = kinds == KIND_MOTION

    # 極端な姿勢の動きサンプルを除外
    keep = static | (motion & _moderate_mask(X, static))
    n_dropped = int((~keep).sum())
    X, Y, groups, passes, kinds = X[keep], Y[keep], groups[keep], passes[keep], kinds[keep]
    static, motion = kinds == KIND_STATIC, kinds == KIND_MOTION
    n_passes = len(np.unique(passes[static])) if np.any(static) else 1
    n_motion = len(np.unique(groups[motion])) if np.any(motion) else 0
    z0, o0 = _scale_params(X[static] if np.any(static) else X)

    # 姿勢が 3 つ以上あり頭の動きデータが無いときは、姿勢ごとに除く CV の方が実態に近い
    sel_groups = passes if (n_passes >= 3 and n_motion == 0) else groups

    A = geometric_features(compact_features(X), z0, o0)
    best = None
    for alpha in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
        pred = cv_predict(A, Y, sel_groups, alpha)
        err = _px_err(pred, Y, np.ones(len(X), bool), screen_w, screen_h)
        if best is None or err < best[0] or np.isnan(best[0]):
            best = (err, alpha, pred)
    err, alpha, pred = best
    model = GazeModel.fit(X, Y, alpha, z0, o0)
    prof = Profile(name=name, screen_w=screen_w, screen_h=screen_h, model=model,
                   cv_error_px=err,
                   same_pose_error_px=_px_err(pred, Y, static, screen_w, screen_h),
                   motion_error_px=_px_err(pred, Y, motion, screen_w, screen_h),
                   n_passes=n_passes, n_motion_points=n_motion)
    prof.n_dropped = n_dropped
    return prof

"""exe / pyw 用のエントリポイント。

  GazeTracker.exe                    GUI を起動
  GazeTracker.exe calibrate ...      キャリブレーション (GUI から子プロセスとして呼ばれる)
  GazeTracker.exe heatmap ...        ヒートマップ生成
  GazeTracker.exe selftest           依存関係の読み込み確認
"""
from __future__ import annotations

import sys


def main():
    args = sys.argv[1:]
    cmd = args[0] if args else ""
    if cmd == "calibrate":
        import calibrate
        sys.argv = [sys.argv[0], *args[1:]]
        return calibrate.main()
    if cmd == "heatmap":
        import heatmap
        sys.argv = [sys.argv[0], *args[1:]]
        return heatmap.main()
    if cmd == "selftest":
        import time
        import numpy as np
        import customtkinter  # noqa: F401
        from tracker.engine import GazeEngine
        from tracker.model import Profile, fit_profile
        from tracker.paths import APP_DIR, MODEL_PATH
        rng = np.random.default_rng(0)
        from tracker.features import FEATURE_NAMES
        X = rng.normal(size=(100, len(FEATURE_NAMES)))
        prof = fit_profile("__selftest", X, X[:, :2] * 0.1 + 0.5, np.repeat(np.arange(10), 10), 1920, 1080)
        prof.save()
        Profile.load("__selftest").predict(X[0])
        eng = GazeEngine("__selftest", log=False)
        eng.start()
        time.sleep(6)
        st = eng.status
        eng.stop()
        for p in (APP_DIR / "profiles").glob("__selftest.*"):
            p.unlink()
        from app import camera_names
        # カメラ名とモード一覧は別スレッドからも取れること (GUI はスレッドで調べる)
        import threading
        from tracker.camera import list_modes
        res = {}
        th = threading.Thread(target=lambda: res.update(names=len(camera_names()), modes=list_modes(0) is not None))
        th.start()
        th.join()
        print(f"selftest ok model={MODEL_PATH.exists()} engine_started={st.started} fps={st.fps:.1f} "
              f"error={st.error!r} cameras={len(camera_names())} cameras_thread={res.get('names')} "
              f"modes={res.get('modes')} cam={st.cam_desc!r} app_dir={APP_DIR}")
        return 0 if st.started and not st.error else 1
    import app
    app.App().mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

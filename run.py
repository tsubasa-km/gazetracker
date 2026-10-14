"""リアルタイム視線推定サーバー (コマンドライン版)。GUI は app.py。

例:
  python run.py --profile home
  python run.py --profile home --preview     # カメラ映像と推定点を別窓で確認
"""
from __future__ import annotations

import argparse
import time

import cv2

from tracker.engine import GazeEngine
from tracker.model import PROFILE_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="default")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--http-port", type=int, default=8766)
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--preview", action="store_true", help="カメラ映像と推定点を表示")
    ap.add_argument("--min-cutoff", type=float, default=1.0, help="平滑化: 小さいほど滑らか (遅延増)")
    ap.add_argument("--beta", type=float, default=0.05, help="平滑化: 大きいほど速い動きに追従")
    ap.add_argument("--cam-res", default="auto", help="カメラ解像度 (auto=出せる最大, 1920x1080, 1280x720, 640x480)")
    ap.add_argument("--hotkey", default="f8", help="ワンキー補正のキー (画面中央を見ながら押す)")
    ap.add_argument("--reset-hotkey", default="f9", help="補正解除のキー")
    a = ap.parse_args()
    from tracker.camera import parse_res
    cw, ch = parse_res(a.cam_res)

    if not (PROFILE_DIR / f"{a.profile}.json").exists():
        print(f"プロファイル '{a.profile}' がありません。先に calibrate.py --profile {a.profile} を実行してください。")
        return

    eng = GazeEngine(a.profile, a.camera, a.ws_port, a.http_port, not a.no_log, a.min_cutoff, a.beta, cw, ch,
                     hotkey=a.hotkey, reset_hotkey=a.reset_hotkey)
    eng.start()
    print(f"OBS ブラウザソース URL:  {eng.overlay_url}")
    print("Ctrl+C で終了")
    try:
        last = 0.0
        while eng.status.running:
            st = eng.status
            if time.time() - last >= 1.0:
                last = time.time()
                g = f"gaze=({st.x:.2f}, {st.y:.2f})" if st.valid else "no gaze"
                print(f"\r{st.fps:5.1f} fps  clients={st.clients}  {g}      ", end="")
            if a.preview:
                f = eng.latest_frame()
                if f is not None:
                    cv2.imshow("gaze preview (q: quit)", f)
                if (cv2.waitKey(30) & 0xFF) == ord("q"):
                    break
            else:
                time.sleep(0.05)
        if eng.status.error:
            print("\nエラー:", eng.status.error)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n終了")
        eng.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

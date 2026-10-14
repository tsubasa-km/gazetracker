"""操作パネル GUI。

  python app.py

キャリブレーション、トラッキング開始/停止、カメラプレビュー、OBS 用 URL、
平滑化の調整、ヒートマップ生成をここから行う。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import cv2
from PIL import Image
from screeninfo import get_monitors

from tracker.engine import GazeEngine
from tracker.paths import APP_DIR, LOG_DIR, PROFILE_DIR, SETTINGS_PATH as SETTINGS, subcommand

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
ctk.ThemeManager.theme["CTkFont"]["family"] = "Yu Gothic UI"   # 日本語を読みやすく


def camera_names() -> list[str]:
    """DirectShow のデバイス名 (順番は OpenCV の CAP_DSHOW 番号と一致)。"""
    try:
        from tracker.camera import com_init
        com_init()
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    except Exception:
        return []


def probe_cameras(max_index: int = 6) -> list[str]:
    """各カメラが出せる最大解像度 (1080p まで) と fps を調べて一覧にする。"""
    from tracker.camera import probe
    names = camera_names()
    found = []
    for i in range(max(max_index, len(names))):
        name = names[i] if i < len(names) else None
        r = probe(i)
        if r is not None:
            w, h, fps, fmt = r
            found.append(f"{i}: {name or 'カメラ'} (最大 {w}x{h} {fps:.0f}fps {fmt})")
        elif name:
            found.append(f"{i}: {name} (映像なし)")
    return found or ["0: (カメラが見つかりません)"]


VIRTUAL_CAMERA_WORDS = ("obs", "virtual camera", "castcraft", "screen", "ndi", "xsplit",
                        "snap camera", "manycam", "mmhmm", "streamlabs")


def is_virtual_camera(label: str) -> bool:
    """配信ソフトの仮想カメラなど、顔を映さないカメラ。スマホ連携 (Windows 仮想カメラ) は実カメラ扱い。"""
    t = label.lower()
    if "windows" in t:
        return False
    return any(w in t for w in VIRTUAL_CAMERA_WORDS)


def pick_camera(labels: list[str]) -> str:
    """仮想カメラと映像の出ないカメラを除き、解像度が最も高いもの (同じなら番号の小さいもの)。"""
    import re
    best, best_px = None, -1
    for lab in labels:
        m = re.search(r"最大 (\d+)x(\d+)", lab)
        if not m or is_virtual_camera(lab):
            continue
        px = int(m.group(1)) * int(m.group(2))
        if px > best_px:
            best, best_px = lab, px
    return best or labels[0]


def monitor_labels() -> list[str]:
    mons = get_monitors()
    main = next((q for q in mons if q.is_primary), mons[0])
    out = []
    for i, m in enumerate(mons):
        pos = []
        if m.is_primary:
            pos.append("メイン")
        elif m.x >= main.x + main.width:
            pos.append("メインの右")
        elif m.x + m.width <= main.x:
            pos.append("メインの左")
        elif m.y + m.height <= main.y:
            pos.append("メインの上")
        elif m.y >= main.y + main.height:
            pos.append("メインの下")
        if m.height > m.width:
            pos.append("縦置き")
        out.append(f"{i}: {m.width}x{m.height} " + "・".join(pos))
    return out


def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Gaze Tracker for OBS")
        self.geometry("1040x640")
        self.minsize(900, 560)
        self.engine: GazeEngine | None = None
        self._img: ctk.CTkImage | None = None
        self._cam_preview = None
        self._cam_opening = False
        self.calib_proc: subprocess.Popen | None = None
        self.settings = self._load_settings()

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_left()
        self._build_right()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._tick)
        threading.Thread(target=self._probe_cameras_async, daemon=True).start()

    # ---------- 左: 設定と操作 ----------
    def _build_left(self):
        f = ctk.CTkScrollableFrame(self, width=360, corner_radius=0)
        f.grid(row=0, column=0, sticky="nsw")
        f.grid_columnconfigure(1, weight=1)
        r = 0

        ctk.CTkLabel(f, text="設定", font=ctk.CTkFont(size=18, weight="bold")).grid(row=r, column=0, columnspan=2, sticky="w", padx=12, pady=(12, 6)); r += 1

        ctk.CTkLabel(f, text="プロファイル").grid(row=r, column=0, sticky="w", padx=12, pady=4)
        profs = list_profiles() or ["default"]
        self.profile_var = ctk.StringVar(value=self.settings.get("profile", profs[0]))
        self.profile_box = ctk.CTkComboBox(f, values=profs, variable=self.profile_var, width=200)
        self.profile_box.grid(row=r, column=1, sticky="ew", padx=12, pady=4); r += 1
        ctk.CTkLabel(f, text="環境ごとに名前を分ける(home/officeなど)。新しい名前を入力すると新規作成", font=ctk.CTkFont(size=11), text_color="gray", wraplength=330, justify="left").grid(row=r, column=0, columnspan=2, sticky="w", padx=12); r += 1

        ctk.CTkLabel(f, text="カメラ").grid(row=r, column=0, sticky="w", padx=12, pady=4)
        self.camera_var = ctk.StringVar(value=self.settings.get("camera", "検出中..."))
        self.camera_box = ctk.CTkComboBox(f, values=["検出中..."], variable=self.camera_var, width=200, state="readonly",
                                          command=lambda _v: self._stop_cam_preview())
        self.camera_box.grid(row=r, column=1, sticky="ew", padx=12, pady=4); r += 1
        self.cam_prev_btn = ctk.CTkButton(f, text="このカメラの映像を確認", fg_color="gray30", hover_color="gray20", command=self._toggle_cam_preview)
        self.cam_prev_btn.grid(row=r, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 4)); r += 1
        ctk.CTkLabel(f, text="起動時に自動で選ばれる (配信ソフトの仮想カメラを除き、解像度が最も高いもの)。解像度もカメラのドライバに問い合わせて、20fps 以上出る最大を自動で使う。違うカメラなら選び直し、右側の映像で顔が正面から映るか確認する", font=ctk.CTkFont(size=11), text_color="gray", wraplength=330, justify="left").grid(row=r, column=0, columnspan=2, sticky="w", padx=12); r += 1

        ctk.CTkLabel(f, text="キャリブ表示モニター").grid(row=r, column=0, sticky="w", padx=12, pady=4)
        mons = monitor_labels()
        self.monitor_var = ctk.StringVar(value=self.settings.get("monitor", mons[0]))
        if self.monitor_var.get() not in mons:
            self.monitor_var.set(mons[0])
        ctk.CTkComboBox(f, values=mons, variable=self.monitor_var, width=200, state="readonly").grid(row=r, column=1, sticky="ew", padx=12, pady=4); r += 1
        ctk.CTkButton(f, text="モニターに番号を表示して識別", fg_color="gray30", hover_color="gray20", command=self._identify_monitors).grid(row=r, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 4)); r += 1
        ctk.CTkLabel(f, text="Switchの映像を映すモニター(Display1)の番号を選ぶ", font=ctk.CTkFont(size=11), text_color="gray", wraplength=330, justify="left").grid(row=r, column=0, columnspan=2, sticky="w", padx=12); r += 1

        ctk.CTkLabel(f, text="頭を動かす点の数").grid(row=r, column=0, sticky="w", padx=12, pady=4)
        self.motion_var = ctk.StringVar(value=self.settings.get("motion_points", "5"))
        if self.motion_var.get() not in ("0", "5", "9"):
            self.motion_var.set("5")
        ctk.CTkSegmentedButton(f, values=["0", "5", "9"], variable=self.motion_var).grid(row=r, column=1, sticky="w", padx=12, pady=4); r += 1
        ctk.CTkLabel(f, text="点を見つめたまま頭を回したり近づいたりして、向きと奥行きの補正を学習する。1点12秒。5推奨", font=ctk.CTkFont(size=11), text_color="gray", wraplength=330, justify="left").grid(row=r, column=0, columnspan=2, sticky="w", padx=12); r += 1

        ctk.CTkLabel(f, text="座り直す姿勢数").grid(row=r, column=0, sticky="w", padx=12, pady=4)
        self.passes_var = ctk.StringVar(value=self.settings.get("passes", "1"))
        if self.passes_var.get() not in ("1", "3", "5", "7"):
            self.passes_var.set("1")
        ctk.CTkSegmentedButton(f, values=["1", "3", "5", "7"], variable=self.passes_var).grid(row=r, column=1, sticky="w", padx=12, pady=4); r += 1
        ctk.CTkLabel(f, text="座る位置を変えて静止点を繰り返す回数。頭を動かす点を使うなら1で十分。使わないなら5", font=ctk.CTkFont(size=11), text_color="gray", wraplength=330, justify="left").grid(row=r, column=0, columnspan=2, sticky="w", padx=12); r += 1

        self.log_var = ctk.BooleanVar(value=self.settings.get("log", True))
        ctk.CTkCheckBox(f, text="視線ログを logs/ に記録する", variable=self.log_var).grid(row=r, column=0, columnspan=2, sticky="w", padx=12, pady=8); r += 1

        ctk.CTkLabel(f, text="操作", font=ctk.CTkFont(size=18, weight="bold")).grid(row=r, column=0, columnspan=2, sticky="w", padx=12, pady=(14, 6)); r += 1
        self.calib_btn = ctk.CTkButton(f, text="1. キャリブレーション開始", height=40, command=self._start_calibration)
        self.calib_btn.grid(row=r, column=0, columnspan=2, sticky="ew", padx=12, pady=4); r += 1
        ctk.CTkLabel(f, text="Display1をWindowsにして実行。プレイ姿勢で座り、カメラと椅子は以後動かさない", font=ctk.CTkFont(size=11), text_color="gray", wraplength=330, justify="left").grid(row=r, column=0, columnspan=2, sticky="w", padx=12); r += 1
        self.track_btn = ctk.CTkButton(f, text="2. トラッキング開始", height=40, fg_color="#2e8b57", hover_color="#247247", command=self._toggle_tracking)
        self.track_btn.grid(row=r, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 4)); r += 1
        ctk.CTkLabel(f, text="Display1をSwitchに切り替えてから開始", font=ctk.CTkFont(size=11), text_color="gray").grid(row=r, column=0, columnspan=2, sticky="w", padx=12); r += 1

        ctk.CTkLabel(f, text="平滑化", font=ctk.CTkFont(size=18, weight="bold")).grid(row=r, column=0, columnspan=2, sticky="w", padx=12, pady=(14, 6)); r += 1
        self.cutoff_var = ctk.DoubleVar(value=float(self.settings.get("min_cutoff", 1.0)))
        self.beta_var = ctk.DoubleVar(value=float(self.settings.get("beta", 0.05)))
        self.cutoff_lbl = ctk.CTkLabel(f, text="")
        self.cutoff_lbl.grid(row=r, column=0, sticky="w", padx=12)
        ctk.CTkSlider(f, from_=0.1, to=5.0, variable=self.cutoff_var, command=self._on_smooth).grid(row=r, column=1, sticky="ew", padx=12); r += 1
        self.beta_lbl = ctk.CTkLabel(f, text="")
        self.beta_lbl.grid(row=r, column=0, sticky="w", padx=12)
        ctk.CTkSlider(f, from_=0.0, to=0.5, variable=self.beta_var, command=self._on_smooth).grid(row=r, column=1, sticky="ew", padx=12); r += 1
        ctk.CTkLabel(f, text="安定度: 小さいほど滑らか (遅れる)。追従: 大きいほど速い動きに追従", font=ctk.CTkFont(size=11), text_color="gray", wraplength=330, justify="left").grid(row=r, column=0, columnspan=2, sticky="w", padx=12); r += 1
        self._on_smooth()

        ctk.CTkLabel(f, text="ワンキー補正", font=ctk.CTkFont(size=18, weight="bold")).grid(row=r, column=0, columnspan=2, sticky="w", padx=12, pady=(14, 6)); r += 1
        ctk.CTkLabel(f, text="補正キー").grid(row=r, column=0, sticky="w", padx=12, pady=4)
        self.hotkey_var = ctk.StringVar(value=self.settings.get("hotkey", "f8"))
        ctk.CTkEntry(f, textvariable=self.hotkey_var, width=100).grid(row=r, column=1, sticky="w", padx=12, pady=4); r += 1
        ctk.CTkLabel(f, text="解除キー").grid(row=r, column=0, sticky="w", padx=12, pady=4)
        self.reset_hotkey_var = ctk.StringVar(value=self.settings.get("reset_hotkey", "f9"))
        ctk.CTkEntry(f, textvariable=self.reset_hotkey_var, width=100).grid(row=r, column=1, sticky="w", padx=12, pady=4); r += 1
        ctk.CTkLabel(f, text="プレイ中に画面中央の照準を見ながら補正キーを押すと、その瞬間のずれを差し引く(ピピッと鳴る)。座り直したら押し直す。ctrl+f8 のような組み合わせも可。キーの変更は次回のトラッキング開始から", font=ctk.CTkFont(size=11), text_color="gray", wraplength=330, justify="left").grid(row=r, column=0, columnspan=2, sticky="w", padx=12); r += 1
        rowf = ctk.CTkFrame(f, fg_color="transparent")
        rowf.grid(row=r, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 4)); r += 1
        rowf.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(rowf, text="今すぐ補正", fg_color="gray30", hover_color="gray20", command=lambda: self.engine and self.engine.request_recenter()).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(rowf, text="補正を解除", fg_color="gray30", hover_color="gray20", command=lambda: self.engine and self.engine.clear_offset()).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ctk.CTkLabel(f, text="解析", font=ctk.CTkFont(size=18, weight="bold")).grid(row=r, column=0, columnspan=2, sticky="w", padx=12, pady=(14, 6)); r += 1
        ctk.CTkButton(f, text="ログからヒートマップを作成", fg_color="gray30", hover_color="gray20", command=self._make_heatmap).grid(row=r, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 16)); r += 1

    # ---------- 右: プレビューと状態 ----------
    def _build_right(self):
        f = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        f.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(f)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(top, text="OBS ブラウザソース URL").grid(row=0, column=0, padx=10, pady=8)
        self.url_var = ctk.StringVar(value="http://localhost:8766/")
        ctk.CTkEntry(top, textvariable=self.url_var).grid(row=0, column=1, sticky="ew", padx=4, pady=8)
        ctk.CTkButton(top, text="コピー", width=70, command=self._copy_url).grid(row=0, column=2, padx=4, pady=8)
        ctk.CTkButton(top, text="ブラウザで確認", width=110, fg_color="gray30", hover_color="gray20", command=lambda: webbrowser.open(self.url_var.get() + "&debug=1")).grid(row=0, column=3, padx=(4, 10), pady=8)

        ctk.CTkLabel(top, text="マーカー").grid(row=1, column=0, padx=10, pady=(0, 8))
        opt = ctk.CTkFrame(top, fg_color="transparent")
        opt.grid(row=1, column=1, columnspan=3, sticky="w", padx=4, pady=(0, 8))
        self.color_var = ctk.StringVar(value=self.settings.get("color", "00e5ff"))
        self.size_var = ctk.StringVar(value=self.settings.get("size", "36"))
        self.trail_var = ctk.StringVar(value=self.settings.get("trail", "600"))
        for label, var, w in (("色 #", self.color_var, 80), ("半径px", self.size_var, 50), ("軌跡ms", self.trail_var, 60)):
            ctk.CTkLabel(opt, text=label).pack(side="left", padx=(6, 2))
            e = ctk.CTkEntry(opt, textvariable=var, width=w)
            e.pack(side="left")
            e.bind("<KeyRelease>", lambda _e: self._update_url())
        self._update_url()

        self.preview = ctk.CTkLabel(f, text="トラッキングを開始するとカメラ映像がここに出ます", fg_color="gray12", corner_radius=8)
        self.preview.grid(row=1, column=0, sticky="nsew", pady=10)

        st = ctk.CTkFrame(f)
        st.grid(row=2, column=0, sticky="ew")
        self.status_lbl = ctk.CTkLabel(st, text="停止中", anchor="w", justify="left", font=ctk.CTkFont(family="Consolas", size=13))
        self.status_lbl.pack(fill="x", padx=10, pady=8)
        self.msg_lbl = ctk.CTkLabel(f, text="", anchor="w", text_color="#ffb347", wraplength=600, justify="left")
        self.msg_lbl.grid(row=3, column=0, sticky="ew", pady=(6, 0))

    # ---------- 動作 ----------
    def _probe_cameras_async(self):
        cams = probe_cameras()
        def apply():
            self.camera_box.configure(values=cams)
            cur = str(self.settings.get("camera", ""))
            idx = cur.split(":")[0]
            match = [c for c in cams if cur and c.split(":")[0] == idx and "映像なし" not in c]
            if match:
                self.camera_var.set(match[0])
            else:
                self.camera_var.set(pick_camera(cams))
                self._msg(f"カメラを自動で選びました: {self.camera_var.get()}")
        self.after(0, apply)

    # --- カメラ映像の確認 (トラッキングなし) ---
    def _toggle_cam_preview(self):
        if self._cam_preview is not None or self._cam_opening:
            self._stop_cam_preview()
            return
        if self.engine:
            self._msg("トラッキング中は右側にそのカメラの映像が出ています")
            return
        idx = self._camera_index()
        from tracker.camera import parse_res
        cw, ch = parse_res("auto")
        self._cam_opening = True
        self.cam_prev_btn.configure(text="カメラを開いています...")
        self.preview.configure(image="", text="カメラを開いています... (数秒かかります)")

        def work():
            try:
                from tracker.camera import Camera
                cam = Camera(idx, cw, ch)
            except Exception as e:
                self.after(0, lambda: (self._msg(f"カメラ {idx} を開けません: {e}"), self._stop_cam_preview()))
                return

            def done():
                if not self._cam_opening:      # 待っている間に中止された
                    cam.close()
                    return
                self._cam_opening = False
                self._cam_preview = cam
                self.cam_prev_btn.configure(text="映像の確認をやめる")
                self._msg(f"カメラ {idx}: 実際の映像 {cam.description}" + self._res_warning(cam))
            self.after(0, done)
        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _res_warning(cam) -> str:
        if cam.width * cam.height < 1280 * 720:
            return "。高解像度で取れていません。カメラ側の設定や、他のアプリ (OBS など) が同じカメラを使っていないか確認してください"
        if cam.fps < 20:
            return "。fps が低いので、解像度を下げるか照明を明るくしてください"
        return ""

    def _stop_cam_preview(self):
        self._cam_opening = False
        if self._cam_preview is not None:
            self._cam_preview.close()
            self._cam_preview = None
        self.cam_prev_btn.configure(text="このカメラの映像を確認")
        if not self.engine:
            self.preview.configure(image="", text="トラッキングを開始するとカメラ映像がここに出ます")
            self._img = None

    def _show_frame(self, fr):
        ar = fr.shape[1] / fr.shape[0]
        pw = max(self.preview.winfo_width() - 8, 320)
        ph = int(pw / ar)
        if ph > self.preview.winfo_height() - 8 > 100:
            ph = self.preview.winfo_height() - 8
            pw = int(ph * ar)
        img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        try:
            if self._img is None:
                self._img = ctk.CTkImage(light_image=img, dark_image=img, size=(pw, ph))
                self.preview.configure(image=self._img, text="")
            else:
                self._img.configure(light_image=img, dark_image=img, size=(pw, ph))
        except Exception:   # 停止直後など Tk 側の画像が破棄されたタイミングは次回にやり直す
            self._img = None

    # --- モニター識別 (各画面に番号を大きく出す) ---
    def _identify_monitors(self, seconds: float = 3.0):
        wins = []
        for i, m in enumerate(get_monitors()):
            w = ctk.CTkToplevel(self)
            w.overrideredirect(True)
            w.attributes("-topmost", True)
            size = 260
            w.geometry(f"{size}x{size}+{m.x + (m.width - size) // 2}+{m.y + (m.height - size) // 2}")
            w.configure(fg_color="#1f6aa5")
            ctk.CTkLabel(w, text=str(i), font=ctk.CTkFont(size=150, weight="bold"), text_color="white").pack(expand=True, fill="both")
            wins.append(w)
        self._msg("各モニターに番号を表示しています。Switch を映すモニターの番号を「キャリブ表示モニター」で選んでください")
        self.after(int(seconds * 1000), lambda: [w.destroy() for w in wins])

    def _camera_index(self) -> int:
        try:
            return int(self.camera_var.get().split(":")[0])
        except ValueError:
            return 0

    def _monitor_index(self) -> int:
        return int(self.monitor_var.get().split(":")[0])

    def _update_url(self):
        p = []
        if self.color_var.get().strip("#") != "00e5ff":
            p.append("color=" + self.color_var.get().strip("#"))
        if self.size_var.get() != "36":
            p.append("size=" + self.size_var.get())
        if self.trail_var.get() != "600":
            p.append("trail=" + self.trail_var.get())
        self.url_var.set("http://localhost:8766/?ws=8765" + ("&" + "&".join(p) if p else ""))

    def _copy_url(self):
        self.clipboard_clear()
        self.clipboard_append(self.url_var.get())
        self._msg("URL をコピーしました。OBS でブラウザソースを追加して貼り付けてください")

    def _on_smooth(self, _v=None):
        self.cutoff_lbl.configure(text=f"安定度 {self.cutoff_var.get():.2f}")
        self.beta_lbl.configure(text=f"追従 {self.beta_var.get():.3f}")
        if self.engine:
            self.engine.set_smoothing(self.cutoff_var.get(), self.beta_var.get())

    def _msg(self, text: str):
        self.msg_lbl.configure(text=text)

    def _start_calibration(self):
        if self.engine:
            self._msg("トラッキング中はキャリブレーションできません。先に停止してください")
            return
        if self.calib_proc and self.calib_proc.poll() is None:
            return
        self._stop_cam_preview()
        prof = self.profile_var.get().strip() or "default"
        cmd = subcommand("calibrate", "--profile", prof, "--monitor", str(self._monitor_index()),
                         "--camera", str(self._camera_index()), "--passes", self.passes_var.get(), "--motion-points", self.motion_var.get(), "--cam-res", "auto")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        self.calib_proc = subprocess.Popen(cmd, cwd=str(APP_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace", env=env,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.calib_btn.configure(state="disabled", text="キャリブレーション中...")
        self._msg(f"モニター {self._monitor_index()} に全画面表示します。画面の指示に従って、姿勢ごとに点を目だけで追ってください")
        threading.Thread(target=self._watch_calibration, args=(prof,), daemon=True).start()

    def _watch_calibration(self, prof: str):
        out = []
        for line in self.calib_proc.stdout:
            line = line.strip()
            if line and not any(k in line for k in ("WARNING", "W0000", "INFO:", "I0000")):
                out.append(line)
        code = self.calib_proc.wait()
        def done():
            self.calib_btn.configure(state="normal", text="1. キャリブレーション開始")
            self.profile_box.configure(values=list_profiles() or ["default"])
            meta = PROFILE_DIR / f"{prof}.json"
            saved = any("保存しました" in l for l in out) and meta.exists()
            if saved:
                info = json.loads(meta.read_text(encoding="utf-8"))
                err = info["cv_error_px"]; pct = err / info["screen_w"] * 100
                grade = "良好" if pct < 5 else ("許容範囲" if pct < 10 else "やり直し推奨")
                parts = [f"通常姿勢 約 {info.get('same_pose_error_px') or err:.0f}px"]
                if info.get("motion_error_px"):
                    parts.append(f"頭を動かしたとき 約 {info['motion_error_px']:.0f}px")
                self._msg(f"プロファイル '{prof}' を保存しました。推定誤差: " + " / ".join(parts) + f" (幅の {pct:.1f}%): {grade}")
                self.profile_var.set(prof)
            elif code != 0:
                self._msg("キャリブレーションが異常終了しました:\n" + "\n".join(out[-5:]))
            else:
                self._msg("キャリブレーションは保存されませんでした")
        self.after(0, done)

    def _toggle_tracking(self):
        if self.engine:
            self._stop_tracking()
            return
        prof = self.profile_var.get().strip()
        if not (PROFILE_DIR / f"{prof}.json").exists():
            self._msg(f"プロファイル '{prof}' がありません。先にキャリブレーションしてください")
            return
        self._stop_cam_preview()
        from tracker.camera import parse_res
        cw, ch = parse_res("auto")
        self.engine = GazeEngine(prof, self._camera_index(), log=self.log_var.get(),
                                 hotkey=self.hotkey_var.get().strip().lower(), reset_hotkey=self.reset_hotkey_var.get().strip().lower(),
                                 min_cutoff=self.cutoff_var.get(), beta=self.beta_var.get(), cam_w=cw, cam_h=ch)
        self.engine.start()
        self.track_btn.configure(text="トラッキング停止", fg_color="#b23b3b", hover_color="#8f2f2f")
        self.calib_btn.configure(state="disabled")
        self._msg("OBS のブラウザソースに上の URL を設定してください (幅・高さはキャプチャ映像と同じに)")
        self._save_settings()

    def _stop_tracking(self):
        if self.engine:
            self.engine.stop()
            self.engine = None
        self.track_btn.configure(text="2. トラッキング開始", fg_color="#2e8b57", hover_color="#247247")
        self.calib_btn.configure(state="normal")
        self.preview.configure(image="", text="停止中")
        self._img = None
        self.status_lbl.configure(text="停止中")

    def _tick(self):
        if self.engine:
            st = self.engine.status
            if not st.running:
                err = st.error
                self._stop_tracking()
                if err:
                    self._msg("エラーで停止しました: " + err)
            elif not st.started:
                self.status_lbl.configure(text="起動中... (カメラとモデルの初期化に数秒かかります)")
            else:
                g = f"x={st.x:.3f} y={st.y:.3f}" if st.valid else ("まばたき" if st.blink else "顔が見つかりません")
                self.status_lbl.configure(text=(
                    f"実行中  {st.fps:5.1f} fps   顔: {'OK' if st.face else 'NG'}   {g}\n"
                    f"OBS 接続数: {st.clients}   プロファイル: {st.profile} (誤差 {st.cv_error_px:.0f}px)\n"
                    f"補正: {'収集中...' if st.recentering else (f'x {st.offset_x * 100:+.1f}%  y {st.offset_y * 100:+.1f}%' if (st.offset_x or st.offset_y) else 'なし')}"
                    f"   カメラ: {st.cam_desc or '-'}\n"
                    f"ログ: {st.log_path or 'なし'}"))
                fr = self.engine.latest_frame()
                if fr is not None:
                    self._show_frame(fr)
        elif self._cam_preview is not None:
            fr, _ = self._cam_preview.read()
            if fr is not None:
                fr = cv2.resize(fr, (640, int(640 * fr.shape[0] / fr.shape[1])))
                cv2.putText(fr, self.camera_var.get()[:40], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2, cv2.LINE_AA)
                self._show_frame(fr)
        self.after(33, self._tick)

    def _make_heatmap(self):
        path = filedialog.askopenfilename(initialdir=str(LOG_DIR), filetypes=[("gaze log", "*.jsonl")])
        if not path:
            return
        out = str(Path(path).with_suffix(".heatmap.png"))
        r = subprocess.run(subcommand("heatmap", path, "--out", out), cwd=str(APP_DIR),
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode == 0 and Path(out).exists():
            self._msg(f"ヒートマップを保存しました: {out}")
            os.startfile(out)
        else:
            self._msg("ヒートマップ作成に失敗: " + (r.stdout + r.stderr)[-300:])

    # ---------- 設定の保存 ----------
    def _load_settings(self) -> dict:
        try:
            return json.loads(SETTINGS.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self):
        d = {"profile": self.profile_var.get(), "camera": self.camera_var.get(), "monitor": self.monitor_var.get(),
             "passes": self.passes_var.get(), "motion_points": self.motion_var.get(), "log": self.log_var.get(), "min_cutoff": self.cutoff_var.get(),
             "beta": self.beta_var.get(), "color": self.color_var.get(), "size": self.size_var.get(),
             "trail": self.trail_var.get(), "hotkey": self.hotkey_var.get(), "reset_hotkey": self.reset_hotkey_var.get()}
        try:
            SETTINGS.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_close(self):
        self._save_settings()
        self._stop_cam_preview()
        if self.engine:
            self.engine.stop()
        if self.calib_proc and self.calib_proc.poll() is None:
            self.calib_proc.terminate()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()

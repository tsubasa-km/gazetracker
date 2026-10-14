"""配布用 zip を release/ に作る (build_exe.bat で exe を作って署名した後に実行)。

中身: 署名済み exe フォルダ、cert/ (公開証明書と登録 bat)、README.txt
プロファイル・ログ・設定などこの PC の個人データは入れない。
"""
import codecs
import pathlib
import time
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent
EXE_DIR = ROOT / "dist" / "GazeTracker"
EXCLUDE_TOP = {"profiles", "logs", "settings.json"}


def main():
    if not (EXE_DIR / "GazeTracker.exe").exists():
        raise SystemExit("dist/GazeTracker/GazeTracker.exe がありません。先に build_exe.bat を実行してください。")
    out = ROOT / "release"
    out.mkdir(exist_ok=True)
    zpath = out / f"GazeTracker_{time.strftime('%Y%m%d')}.zip"
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(EXE_DIR.rglob("*")):
            rel = p.relative_to(EXE_DIR)
            if rel.parts[0] in EXCLUDE_TOP or "__selftest" in p.name:
                continue
            if p.is_file():
                z.write(p, f"GazeTracker/{rel.as_posix()}")
                n += 1
        z.writestr("GazeTracker/profiles/.keep", "")
        z.writestr("GazeTracker/logs/.keep", "")
        for f in ("GazeTracker.cer", "install_cert.bat", "uninstall_cert.bat"):
            z.write(ROOT / "cert" / f, f"cert/{f}")
            n += 1
        txt = (ROOT / "dist_readme.txt").read_text(encoding="utf-8")
        z.writestr("README.txt", codecs.BOM_UTF8 + txt.replace("\n", "\r\n").encode("utf-8"))   # メモ帳で読める
        n += 1
    print(f"{zpath}  ({n} files, {zpath.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()

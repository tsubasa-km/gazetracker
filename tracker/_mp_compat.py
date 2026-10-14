"""MediaPipe を matplotlib 無しで読み込めるようにする。

mediapipe.tasks.python.vision は、描画補助 (drawing_utils) のために読み込み時に
matplotlib.pyplot を import する。視線推定では使わないので、exe では matplotlib を
同梱せず (約 15MB 削減)、ここで空のモジュールを代わりに置く。
matplotlib が入っている環境ではそちらがそのまま使われる。
"""
import sys
import types

try:
    import matplotlib.pyplot  # noqa: F401
except ImportError:
    _mpl = types.ModuleType("matplotlib")
    _mpl.__path__ = []
    _plt = types.ModuleType("matplotlib.pyplot")
    _mpl.pyplot = _plt
    sys.modules["matplotlib"] = _mpl
    sys.modules["matplotlib.pyplot"] = _plt

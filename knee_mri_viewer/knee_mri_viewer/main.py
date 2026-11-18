"""Knee MRI + cartilage segmentation viewer.

    python main.py
    python main.py --mri train_001_V00.im --seg train_001_V00.seg
    python main.py --mri case.im --seg case.seg --config orientation_config.json
"""
import argparse
import os
import sys
from pathlib import Path

# Must be set BEFORE any Qt-using library is imported (pyvistaqt -> qtpy, pyqtgraph).
os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402


def _apply_dark_style(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    p = QtGui.QPalette()
    R = QtGui.QPalette.ColorRole
    p.setColor(R.Window, QtGui.QColor(40, 40, 42))
    p.setColor(R.WindowText, QtGui.QColor(225, 225, 225))
    p.setColor(R.Base, QtGui.QColor(28, 28, 30))
    p.setColor(R.AlternateBase, QtGui.QColor(40, 40, 42))
    p.setColor(R.Text, QtGui.QColor(225, 225, 225))
    p.setColor(R.Button, QtGui.QColor(56, 56, 60))
    p.setColor(R.ButtonText, QtGui.QColor(225, 225, 225))
    p.setColor(R.ToolTipBase, QtGui.QColor(50, 50, 52))
    p.setColor(R.ToolTipText, QtGui.QColor(230, 230, 230))
    p.setColor(R.Highlight, QtGui.QColor(0, 120, 215))
    p.setColor(R.HighlightedText, QtGui.QColor(255, 255, 255))
    app.setPalette(p)


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive knee MRI / cartilage segmentation viewer")
    ap.add_argument("--mri", help="MRI file (.im)")
    ap.add_argument("--seg", help="segmentation file (.seg)")
    ap.add_argument("--config", help="orientation / spacing JSON (default: ./orientation_config.json if present)")
    ap.add_argument("--threshold", type=float, default=0.0, help="binarise segmentation with value > threshold")
    args = ap.parse_args()

    config = args.config
    if config is None:
        for cand in (Path.cwd() / "orientation_config.json", Path(__file__).with_name("orientation_config.json")):
            if cand.is_file():
                config = str(cand)
                break

    app = QtWidgets.QApplication(sys.argv)
    _apply_dark_style(app)

    from kneeviewer.main_window import MainWindow  # imported late: needs the env vars above

    win = MainWindow(config_path=config)
    win.show()
    if args.mri:
        seg = args.seg
        if seg is None:
            from kneeviewer.data_io import guess_seg_path
            g = guess_seg_path(args.mri)
            seg = str(g) if g else None
        QtCore.QTimer.singleShot(0, lambda: win.load_case(args.mri, seg, args.threshold, True))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

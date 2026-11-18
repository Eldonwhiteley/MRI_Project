"""One 2D anatomical view (sagittal, coronal or transverse).

Mouse controls inside a view
----------------------------
* left click            : move the crosshair (updates the other views + 3D planes)
* Shift + left drag     : move the crosshair continuously
* left / middle drag    : pan
* Ctrl + wheel, right drag : zoom
* wheel                 : scroll through slices

Performance: the ImageItems are created once and only ``setImage`` is called
afterwards; slices are NumPy views (no copy) until pyqtgraph converts them to a
QImage. A slice is only re-extracted when the slice index of THIS view changes -
moving the crosshair inside the plane just moves two lines.
"""
from __future__ import annotations

from typing import Optional, Tuple

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .config import PLANE_NAMES
from .overlays import overlay_rgba
from .state import PLANE_COLORS, ViewerSettings
from .volume_model import VolumeModel

# Arrays are indexed [row, column]; row 0 is drawn at the top (see invertY below).
pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

_LEFT = QtCore.Qt.MouseButton.LeftButton
_SHIFT = QtCore.Qt.KeyboardModifier.ShiftModifier
_CTRL = QtCore.Qt.KeyboardModifier.ControlModifier


class SliceViewBox(pg.ViewBox):
    """ViewBox with click-to-pick and wheel-to-scroll behaviour."""

    sigPicked = QtCore.Signal(float, float)   # position in view (mm) coordinates
    sigScrolled = QtCore.Signal(int)          # +n / -n slices

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wheel_acc = 0.0

    def mouseClickEvent(self, ev):
        if ev.button() == _LEFT:
            p = self.mapToView(ev.pos())
            self.sigPicked.emit(float(p.x()), float(p.y()))
            ev.accept()
        else:
            super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == _LEFT and bool(ev.modifiers() & _SHIFT):
            p = self.mapToView(ev.pos())
            self.sigPicked.emit(float(p.x()), float(p.y()))
            ev.accept()
        else:
            super().mouseDragEvent(ev, axis)

    def wheelEvent(self, ev, axis=None):
        if bool(ev.modifiers() & _CTRL):
            super().wheelEvent(ev, axis)
            return
        self._wheel_acc += float(ev.delta())
        steps = int(self._wheel_acc / 120.0)
        if steps:
            self._wheel_acc -= steps * 120.0
            self.sigScrolled.emit(steps)
        ev.accept()


class SliceView(QtWidgets.QFrame):
    """A titled frame: image + crosshair + slice slider + overlay checkbox."""

    cursorRequested = QtCore.Signal(int, int, int)   # canonical (i, j, k) the user asked for

    def __init__(self, plane: int, settings: ViewerSettings, parent=None):
        super().__init__(parent)
        self.plane = plane
        self.settings = settings
        self.model: Optional[VolumeModel] = None
        self._cursor: Tuple[int, int, int] = (0, 0, 0)
        self._shown_idx: Optional[int] = None
        self._geom = (1, 1, 1.0, 1.0)  # n_rows, n_cols, row spacing... see set_model
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        color = PLANE_COLORS[plane]
        self.title = QtWidgets.QLabel(f"{PLANE_NAMES[plane]} plane")
        self.title.setStyleSheet(f"font-weight: 600; color: rgb{color};")
        self.info = QtWidgets.QLabel("no data")
        self.info.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        # --- graphics -----------------------------------------------------
        self.gv = pg.GraphicsView(background=(12, 12, 12))
        self.vb = SliceViewBox(lockAspect=True, enableMenu=False)
        self.vb.invertY(True)          # row 0 at the top
        self.gv.setCentralItem(self.vb)
        self.img = pg.ImageItem(axisOrder="row-major")
        self.ovl = pg.ImageItem(axisOrder="row-major")
        self.img.setZValue(0)
        self.ovl.setZValue(1)
        self.vb.addItem(self.img)
        self.vb.addItem(self.ovl)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color, width=1))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(color, width=1))
        for ln in (self.vline, self.hline):
            ln.setZValue(10)
            self.vb.addItem(ln, ignoreBounds=True)
        self.vb.sigPicked.connect(self._on_picked)
        self.vb.sigScrolled.connect(self._on_scrolled)

        # --- controls under the image ------------------------------------------
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.spin = QtWidgets.QSpinBox()
        self.spin.setRange(0, 0)
        self.spin.setFixedWidth(64)
        self.chk_overlay = QtWidgets.QCheckBox("Segmentation overlay")
        self.chk_overlay.setChecked(bool(settings["overlay_2d"]))
        self.btn_reset = QtWidgets.QToolButton()
        self.btn_reset.setText("Reset view")
        self.btn_reset.setToolTip("Fit the whole slice into the window (undo zoom/pan)")

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.title)
        top.addStretch(1)
        top.addWidget(self.info)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Slice"))
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(self.chk_overlay)
        row2.addStretch(1)
        row2.addWidget(self.btn_reset)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)
        lay.addLayout(top)
        lay.addWidget(self.gv, 1)
        lay.addLayout(row)
        lay.addLayout(row2)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_slider)
        self.chk_overlay.toggled.connect(lambda _=None: self.show_slice(force=True))
        self.btn_reset.clicked.connect(self.reset_view)
        self.setEnabled(False)

    # ------------------------------------------------------------------ model
    def set_model(self, model: VolumeModel, cursor: Tuple[int, int, int]) -> None:
        self.model = model
        tr = model.display[self.plane]
        n_rows, n_cols = model.shape[tr.row_axis], model.shape[tr.col_axis]
        sy, sx = model.spacing[tr.row_axis], model.spacing[tr.col_axis]
        self._geom = (n_rows, n_cols, sx, sy)
        # pixel (col, row) -> view (mm): x = col*sx - sx/2, y = row*sy - sy/2.
        # With this, a voxel CENTRE sits at index*spacing in every view (same as 3D).
        t = QtGui.QTransform(sx, 0.0, 0.0, sy, -0.5 * sx, -0.5 * sy)
        self.img.setTransform(t)
        self.ovl.setTransform(t)

        # crosshair colours: a vertical line marks the plane that runs along col_axis, etc.
        self.vline.setPen(pg.mkPen(PLANE_COLORS[tr.col_axis], width=1))
        self.hline.setPen(pg.mkPen(PLANE_COLORS[tr.row_axis], width=1))

        n = model.shape[self.plane]
        for w in (self.slider, self.spin):
            w.blockSignals(True)
            w.setRange(0, n - 1)
            w.setValue(int(cursor[self.plane]))
            w.blockSignals(False)
        self.chk_overlay.setEnabled(model.seg is not None)
        self.setEnabled(True)
        self._cursor = tuple(cursor)  # type: ignore[assignment]
        self._shown_idx = None
        self.show_slice(force=True)
        self.update_lines()
        self.reset_view()

    def release(self) -> None:
        """Drop the reference to the volume (frees memory before a new case is loaded)."""
        self.model = None
        self._shown_idx = None
        self.img.clear()
        self.ovl.clear()
        self.info.setText("no data")
        self.setEnabled(False)

    def reset_view(self) -> None:
        if self.model is None:
            return
        n_rows, n_cols, sx, sy = self._geom
        self.vb.setAspectLocked(True, 1.0)
        self.vb.setRange(xRange=(-0.5 * sx, (n_cols - 0.5) * sx),
                         yRange=(-0.5 * sy, (n_rows - 0.5) * sy), padding=0.02)

    # ------------------------------------------------------------------ cursor
    def set_cursor(self, idx: Tuple[int, int, int]) -> None:
        self._cursor = tuple(idx)  # type: ignore[assignment]
        v = self._cursor[self.plane]
        for w in (self.slider, self.spin):
            if w.value() != v:
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)
        self.show_slice()          # no-op unless this view's slice index changed
        self.update_lines()

    def update_lines(self) -> None:
        if self.model is None:
            return
        tr = self.model.display[self.plane]
        r, c = tr.to_display(self._cursor, self.model.shape)
        _, _, sx, sy = self._geom
        self.vline.setPos(c * sx)
        self.hline.setPos(r * sy)

    # ------------------------------------------------------------------ drawing
    def _levels(self) -> Tuple[float, float]:
        return self.model.window_levels(self.settings["win2d_lo"], self.settings["win2d_hi"])

    def show_slice(self, force: bool = False) -> None:
        m = self.model
        if m is None:
            return
        idx = self._cursor[self.plane]
        if not force and idx == self._shown_idx:
            return
        self._shown_idx = idx
        tr = m.display[self.plane]
        mode = self.settings["layer_mode"]

        show_base = mode != "seg"
        self.img.setVisible(show_base)
        if show_base:
            lo, hi = self._levels()
            self.img.setImage(tr.apply(m.mri_slice(self.plane, idx)), levels=(lo, hi), autoLevels=False)

        show_ovl = m.seg is not None and self.chk_overlay.isChecked() and mode in ("both", "seg")
        self.ovl.setVisible(show_ovl)
        if show_ovl:
            rgba = overlay_rgba(m, self.plane, idx, self.settings["seg_mode"],
                                self.settings["combined_color"], self.settings["channel_colors"])
            self.ovl.setImage(tr.apply(rgba), autoLevels=False)
            self.ovl.setOpacity(1.0 if mode == "seg" else float(self.settings["overlay_opacity"]))
        self.info.setText(f"slice {idx}  (0-{m.shape[self.plane] - 1})")

    # ------------------------------------------------------------------ settings
    def on_setting(self, key: str) -> None:
        if self.model is None:
            return
        if key == "overlay_2d":
            self.chk_overlay.setChecked(bool(self.settings["overlay_2d"]))  # -> show_slice via toggled
        elif key == "overlay_opacity":
            if self.settings["layer_mode"] != "seg":
                self.ovl.setOpacity(float(self.settings["overlay_opacity"]))
        elif key in ("layer_mode", "seg_mode", "combined_color", "channel_colors"):
            self.show_slice(force=True)
        elif key in ("win2d_lo", "win2d_hi"):
            if self.img.isVisible() and self.img.image is not None:
                self.img.setLevels(self._levels())

    # ------------------------------------------------------------------ input
    def _on_slider(self, value: int) -> None:
        if self.model is None:
            return
        idx = list(self._cursor)
        idx[self.plane] = int(value)
        self.cursorRequested.emit(*idx)

    def _on_scrolled(self, steps: int) -> None:
        if self.model is None:
            return
        # wheel up (positive delta) -> next slice
        self._on_slider(min(max(self._cursor[self.plane] + steps, 0), self.model.shape[self.plane] - 1))

    def _on_picked(self, x: float, y: float) -> None:
        if self.model is None:
            return
        tr = self.model.display[self.plane]
        n_rows, n_cols, sx, sy = self._geom
        col = int(min(max(round(x / sx), 0), n_cols - 1))
        row = int(min(max(round(y / sy), 0), n_rows - 1))
        i_row, i_col = tr.from_display(row, col, self.model.shape)
        idx = list(self._cursor)
        idx[tr.row_axis] = i_row
        idx[tr.col_axis] = i_col
        self.cursorRequested.emit(*idx)

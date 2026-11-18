"""The control panel docked at the right of the main window.

It only writes to ``ViewerSettings`` (and emits ``voxelEdited``); the views react
to the settings, so the panel does not need to know about them.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from .state import ViewerSettings
from .volume_model import VolumeModel

H = QtCore.Qt.Orientation.Horizontal


class ColorButton(QtWidgets.QPushButton):
    colorChanged = QtCore.Signal(tuple)

    def __init__(self, rgb: Tuple[int, int, int], parent=None):
        super().__init__(parent)
        self.setFixedSize(46, 22)
        self._rgb = tuple(rgb)
        self._paint()
        self.clicked.connect(self._pick)

    def _paint(self) -> None:
        self.setStyleSheet(f"background-color: rgb{self._rgb}; border: 1px solid #666;")

    def set_color(self, rgb: Tuple[int, int, int]) -> None:
        self._rgb = tuple(rgb)
        self._paint()

    def _pick(self) -> None:
        col = QtWidgets.QColorDialog.getColor(QtGui.QColor(*self._rgb), self, "Choose colour")
        if col.isValid():
            self._rgb = (col.red(), col.green(), col.blue())
            self._paint()
            self.colorChanged.emit(self._rgb)


class ControlPanel(QtWidgets.QWidget):
    voxelEdited = QtCore.Signal(int, int, int)   # file-array indices typed by the user

    def __init__(self, settings: ViewerSettings, parent=None):
        super().__init__(parent)
        self.s = settings
        self.model: Optional[VolumeModel] = None
        self._color_buttons: List[ColorButton] = []

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        outer.addWidget(scroll)
        body = QtWidgets.QWidget()
        scroll.setWidget(body)
        self.lay = QtWidgets.QVBoxLayout(body)

        self._build_voxel_group()
        self._build_layers_group()
        self._build_seg_group()
        self._build_win2d_group()
        self._build_vol_group()
        self._build_surf_group()
        self._build_res_group()
        self._build_planes_group()
        self.lay.addStretch(1)
        self.setMinimumWidth(310)

    # --------------------------------------------------------------- helpers
    def _group(self, title: str) -> QtWidgets.QVBoxLayout:
        g = QtWidgets.QGroupBox(title)
        v = QtWidgets.QVBoxLayout(g)
        self.lay.addWidget(g)
        return v

    def _slider(self, parent: QtWidgets.QVBoxLayout, label: str, key: str, lo: int, hi: int,
                scale: float, fmt: Callable[[float], str], tip: str = "") -> QtWidgets.QSlider:
        row = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QLabel(label)
        lab.setMinimumWidth(92)
        sl = QtWidgets.QSlider(H)
        sl.setRange(lo, hi)
        val = QtWidgets.QLabel()
        val.setMinimumWidth(46)
        val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        if tip:
            lab.setToolTip(tip)
            sl.setToolTip(tip)

        def on(v: int) -> None:
            val.setText(fmt(v / scale))
            self.s.set(key, v / scale)

        sl.valueChanged.connect(on)
        sl.setValue(int(round(float(self.s[key]) * scale)))
        val.setText(fmt(float(self.s[key])))
        row.addWidget(lab)
        row.addWidget(sl, 1)
        row.addWidget(val)
        parent.addLayout(row)
        return sl

    def _check(self, parent: QtWidgets.QVBoxLayout, text: str, key: str, tip: str = "") -> QtWidgets.QCheckBox:
        cb = QtWidgets.QCheckBox(text)
        cb.setChecked(bool(self.s[key]))
        cb.toggled.connect(lambda v, k=key: self.s.set(k, bool(v)))
        if tip:
            cb.setToolTip(tip)
        parent.addWidget(cb)
        return cb

    # --------------------------------------------------------------- groups
    def _build_voxel_group(self) -> None:
        v = self._group("Selected voxel")
        row = QtWidgets.QHBoxLayout()
        self.spin_ijk: List[QtWidgets.QSpinBox] = []
        for name in ("i", "j", "k"):
            row.addWidget(QtWidgets.QLabel(name))
            sp = QtWidgets.QSpinBox()
            sp.setRange(0, 0)
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(self._on_voxel_spin)
            self.spin_ijk.append(sp)
            row.addWidget(sp)
        v.addLayout(row)
        self.lbl_voxel = QtWidgets.QLabel("-")
        self.lbl_voxel.setWordWrap(True)
        v.addWidget(self.lbl_voxel)
        hint = QtWidgets.QLabel("(i, j, k) are the indices into the array exactly as stored in the file.")
        hint.setStyleSheet("color: #999;")
        hint.setWordWrap(True)
        v.addWidget(hint)

    def _build_layers_group(self) -> None:
        v = self._group("Layers")
        self.cmb_layer = QtWidgets.QComboBox()
        self.cmb_layer.addItem("MRI + segmentation", "both")
        self.cmb_layer.addItem("MRI only", "mri")
        self.cmb_layer.addItem("Segmentation only", "seg")
        self.cmb_layer.currentIndexChanged.connect(lambda _: self.s.set("layer_mode", self.cmb_layer.currentData()))
        v.addWidget(self.cmb_layer)

    def _build_seg_group(self) -> None:
        self.grp_seg = QtWidgets.QGroupBox("Segmentation")
        v = QtWidgets.QVBoxLayout(self.grp_seg)
        self.lay.addWidget(self.grp_seg)
        self.cmb_seg = QtWidgets.QComboBox()
        self.cmb_seg.currentIndexChanged.connect(self._on_seg_mode)
        v.addWidget(QtWidgets.QLabel("Show:"))
        v.addWidget(self.cmb_seg)
        self._check(v, "Show 2D overlay (all three views)", "overlay_2d",
                    "Each 2D view also has its own checkbox.")
        self._slider(v, "Overlay opacity", "overlay_opacity", 0, 100, 100.0, lambda x: f"{x * 100:.0f} %")
        self.colors_box = QtWidgets.QGridLayout()
        v.addWidget(QtWidgets.QLabel("Colours:"))
        v.addLayout(self.colors_box)

    def _build_win2d_group(self) -> None:
        v = self._group("2D image window / level")
        self.sl_w_lo = self._slider(v, "Window low", "win2d_lo", 0, 1000, 1000.0, lambda x: f"{x * 100:.0f} %")
        self.sl_w_hi = self._slider(v, "Window high", "win2d_hi", 0, 1000, 1000.0, lambda x: f"{x * 100:.0f} %")
        self.sl_w_lo.valueChanged.connect(lambda val: self._keep_order(self.sl_w_lo, self.sl_w_hi, True))
        self.sl_w_hi.valueChanged.connect(lambda val: self._keep_order(self.sl_w_lo, self.sl_w_hi, False))
        b = QtWidgets.QPushButton("Auto window")
        b.clicked.connect(self._auto_window)
        v.addWidget(b)

    def _build_vol_group(self) -> None:
        v = self._group("3D: MRI volume rendering")
        self._check(v, "Show MRI volume", "vol_visible")
        self._slider(v, "Opacity", "vol_opacity", 0, 100, 100.0, lambda x: f"{x * 100:.0f} %",
                     "Overall opacity of the rendered volume.")
        self.sl_v_lo = self._slider(v, "Contrast low", "vol_win_lo", 0, 1000, 1000.0, lambda x: f"{x * 100:.0f} %",
                                    "Intensities below this are transparent and black.")
        self.sl_v_hi = self._slider(v, "Contrast high", "vol_win_hi", 0, 1000, 1000.0, lambda x: f"{x * 100:.0f} %",
                                    "Intensities above this are fully bright and fully opaque (x Opacity).")
        self.sl_v_lo.valueChanged.connect(lambda val: self._keep_order(self.sl_v_lo, self.sl_v_hi, True))
        self.sl_v_hi.valueChanged.connect(lambda val: self._keep_order(self.sl_v_lo, self.sl_v_hi, False))
        self._slider(v, "Opacity curve", "vol_gamma", 20, 400, 100.0, lambda x: f"{x:.2f}",
                     "1 = linear ramp; >1 hides faint tissue and emphasises bright structures.")

    def _build_surf_group(self) -> None:
        v = self._group("3D: segmentation surface")
        self._check(v, "Show 3D segmentation surface", "surf_visible")
        self._slider(v, "Surface opacity", "surf_opacity", 0, 100, 100.0, lambda x: f"{x * 100:.0f} %",
                     "Below 100 % the cartilage is see-through. Enable depth peeling for correct blending.")
        self._check(v, "Depth peeling (better transparency)", "depth_peeling",
                    "Correct blending of transparent surfaces with the volume. May be slower or\n"
                    "misbehave on some GPUs/drivers; leave off if the volume turns black.")

    def _build_res_group(self) -> None:
        v = self._group("3D: resolution (downsampling)")
        grid = QtWidgets.QGridLayout()
        self.spin_vds = QtWidgets.QSpinBox()
        self.spin_vds.setRange(1, 6)
        self.spin_vds.setValue(int(self.s["vol_ds"]))
        self.spin_vds.setToolTip("MRI volume rendering only: 1 = full resolution, 2 = block-average every 2 voxels, ...\n"
                                 "The 2D slices are never downsampled.")
        self.spin_sds = QtWidgets.QSpinBox()
        self.spin_sds.setRange(1, 4)
        self.spin_sds.setValue(int(self.s["surf_ds"]))
        self.spin_sds.setToolTip("Segmentation surfaces: 1 = full resolution (best; cartilage is thin,\n"
                                 "so larger values lose detail).")
        grid.addWidget(QtWidgets.QLabel("MRI volume"), 0, 0)
        grid.addWidget(self.spin_vds, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Surfaces"), 1, 0)
        grid.addWidget(self.spin_sds, 1, 1)
        v.addLayout(grid)
        self.btn_apply_ds = QtWidgets.QPushButton("Apply 3D resolution")
        self.btn_apply_ds.clicked.connect(self._apply_ds)
        v.addWidget(self.btn_apply_ds)

    def _build_planes_group(self) -> None:
        v = self._group("3D: slice planes")
        self._check(v, "Sagittal plane", "plane_sag")
        self._check(v, "Coronal plane", "plane_cor")
        self._check(v, "Transverse plane", "plane_axi")

    # --------------------------------------------------------------- callbacks
    def _keep_order(self, lo: QtWidgets.QSlider, hi: QtWidgets.QSlider, moved_lo: bool) -> None:
        """Keep low < high by pushing the other handle."""
        if lo.value() >= hi.value():
            if moved_lo:
                hi.setValue(min(lo.value() + 5, hi.maximum()))
                if lo.value() >= hi.value():
                    lo.setValue(max(hi.value() - 5, 0))
            else:
                lo.setValue(max(hi.value() - 5, 0))
                if lo.value() >= hi.value():
                    hi.setValue(min(lo.value() + 5, hi.maximum()))

    def _auto_window(self) -> None:
        if self.model is None:
            return
        self.sl_w_lo.setValue(int(self.model.default_win_lo * 1000))
        self.sl_w_hi.setValue(int(self.model.default_win_hi * 1000))

    def _apply_ds(self) -> None:
        self.s.set("surf_ds", int(self.spin_sds.value()))
        self.s.set("vol_ds", int(self.spin_vds.value()))

    def _on_seg_mode(self) -> None:
        d = self.cmb_seg.currentData()
        if d is not None:
            self.s.set("seg_mode", d)

    def _on_voxel_spin(self) -> None:
        self.voxelEdited.emit(*(sp.value() for sp in self.spin_ijk))

    # --------------------------------------------------------------- model
    def set_model(self, model: VolumeModel, reset_window: bool = True) -> None:
        self.model = model
        for a, sp in enumerate(self.spin_ijk):
            sp.blockSignals(True)
            sp.setRange(0, model.file_shape[a] - 1)
            sp.blockSignals(False)

        has_seg = model.seg is not None
        self.grp_seg.setEnabled(has_seg)
        n = model.n_channels
        self.s.ensure_channels(max(n, 6))

        # segmentation mode combo (rebuilt for this case's channel count)
        self.cmb_seg.blockSignals(True)
        self.cmb_seg.clear()
        self.cmb_seg.addItem("Combined (logical OR of all channels)", "combined")
        self.cmb_seg.addItem("All channels, each in its own colour", "all")
        for c in range(n):
            self.cmb_seg.addItem(f"Channel {c}: {model.channel_name(c)}" if model.config.channel_names
                                 else f"Channel {c}", c)
        cur = self.s["seg_mode"]
        if cur not in ("combined", "all") and not (isinstance(cur, int) and 0 <= cur < n):
            self.s.set("seg_mode", "combined")
            cur = "combined"
        self.cmb_seg.setCurrentIndex(max(0, self.cmb_seg.findData(cur)))
        self.cmb_seg.blockSignals(False)

        # colour buttons: combined + one per channel
        while self.colors_box.count():
            item = self.colors_box.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._color_buttons = []
        entries: List[Tuple[str, Callable[[Tuple[int, int, int]], None], Tuple[int, int, int]]] = [
            ("Combined", lambda rgb: self.s.set("combined_color", tuple(rgb)), tuple(self.s["combined_color"]))
        ]
        for c in range(n):
            entries.append((f"Ch {c}", lambda rgb, ch=c: self.s.set_channel_color(ch, rgb),
                            tuple(self.s["channel_colors"][c])))
        for i, (text, cb, rgb) in enumerate(entries):
            btn = ColorButton(rgb)
            btn.colorChanged.connect(cb)
            self._color_buttons.append(btn)
            r, c = divmod(i, 2)
            self.colors_box.addWidget(QtWidgets.QLabel(text), r, c * 2)
            self.colors_box.addWidget(btn, r, c * 2 + 1)

        if reset_window:
            self._auto_window()

    # --------------------------------------------------------------- readout
    def set_voxel_info(self, file_idx: Sequence[int], canon_idx: Sequence[int],
                       intensity: float, channels: Sequence[int]) -> None:
        for sp, v in zip(self.spin_ijk, file_idx):
            if sp.value() != v:
                sp.blockSignals(True)
                sp.setValue(int(v))
                sp.blockSignals(False)
        if channels and self.model is not None:
            seg = ", ".join(f"{c}" for c in channels)
        else:
            seg = "none"
        self.lbl_voxel.setText(
            f"(i, j, k) = ({file_idx[0]}, {file_idx[1]}, {file_idx[2]})\n"
            f"slices: sagittal {canon_idx[0]}, coronal {canon_idx[1]}, transverse {canon_idx[2]}\n"
            f"intensity = {intensity:.4g}\n"
            f"segmentation channels: {seg}"
        )

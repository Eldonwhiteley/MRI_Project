"""Dialogs: choose the .im / .seg files, and edit orientation + voxel spacing."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PySide6 import QtCore, QtWidgets

from .config import PLANE_NAMES, ROLE_NAMES, OrientationConfig, ViewOptions
from .data_io import guess_seg_path
from .volume_model import DisplayTransform


# =============================================================== open case
class OpenCaseDialog(QtWidgets.QDialog):
    """Pick an MRI (.im) and its segmentation (.seg)."""

    def __init__(self, parent=None, mri: str = "", seg: str = "", start_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Open MRI + segmentation")
        self.setMinimumWidth(560)
        self._dir = start_dir

        self.ed_mri = QtWidgets.QLineEdit(mri)
        self.ed_seg = QtWidgets.QLineEdit(seg)
        self.ed_mri.setPlaceholderText("e.g. train_001_V00.im")
        self.ed_seg.setPlaceholderText("e.g. train_001_V00.seg  (optional)")
        b1 = QtWidgets.QPushButton("Browse ...")
        b2 = QtWidgets.QPushButton("Browse ...")
        b1.clicked.connect(self._browse_mri)
        b2.clicked.connect(self._browse_seg)

        self.spin_thr = QtWidgets.QDoubleSpinBox()
        self.spin_thr.setDecimals(3)
        self.spin_thr.setRange(0.0, 1e6)
        self.spin_thr.setValue(0.0)
        self.spin_thr.setToolTip("A voxel belongs to a channel if value > threshold. 0 reproduces `segmentation > 0`.\n"
                                 "Use e.g. 0.5 when loading soft U-Net probabilities.")
        self.chk_meta = QtWidgets.QCheckBox("Use voxel spacing / orientation stored in the HDF5 file (if any)")
        self.chk_meta.setChecked(True)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("MRI (.im):"), 0, 0)
        grid.addWidget(self.ed_mri, 0, 1)
        grid.addWidget(b1, 0, 2)
        grid.addWidget(QtWidgets.QLabel("Segmentation (.seg):"), 1, 0)
        grid.addWidget(self.ed_seg, 1, 1)
        grid.addWidget(b2, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Binarise: value >"), 2, 0)
        grid.addWidget(self.spin_thr, 2, 1)
        grid.addWidget(self.chk_meta, 3, 0, 1, 3)

        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                         | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(grid)
        lay.addWidget(box)

    def _browse_mri(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select MRI file", self._dir, "MRI volume (*.im *.h5 *.hdf5);;All files (*)")
        if path:
            self.ed_mri.setText(path)
            self._dir = str(Path(path).parent)
            if not self.ed_seg.text().strip():
                g = guess_seg_path(path)
                if g:
                    self.ed_seg.setText(str(g))

    def _browse_seg(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select segmentation file", self._dir, "Segmentation (*.seg *.h5 *.hdf5);;All files (*)")
        if path:
            self.ed_seg.setText(path)
            self._dir = str(Path(path).parent)

    def _accept(self) -> None:
        if not self.ed_mri.text().strip():
            QtWidgets.QMessageBox.warning(self, "Missing file", "Please choose an MRI (.im) file.")
            return
        self.accept()

    @property
    def mri_path(self) -> str:
        return self.ed_mri.text().strip()

    @property
    def seg_path(self) -> Optional[str]:
        t = self.ed_seg.text().strip()
        return t or None

    @property
    def threshold(self) -> float:
        return float(self.spin_thr.value())

    @property
    def use_metadata(self) -> bool:
        return self.chk_meta.isChecked()


# =========================================================== orientation
class OrientationDialog(QtWidgets.QDialog):
    """Edit which array axis is which anatomical direction, flips, spacing and view rotations.

    ``applied`` is emitted with a validated OrientationConfig every time the user
    presses OK or Apply.
    """

    applied = QtCore.Signal(object)

    def __init__(self, config: OrientationConfig, file_shape: Optional[Tuple[int, int, int]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Orientation and voxel spacing")
        self.setMinimumWidth(640)
        self._file_shape = file_shape
        self._channel_names = None   # must exist before any widget signal fires

        # ---- anatomical axes ---------------------------------------------------
        g_axes = QtWidgets.QGroupBox("Which array axis is which anatomical direction?")
        ga = QtWidgets.QGridLayout(g_axes)
        ga.addWidget(QtWidgets.QLabel("Slices stacked along ..."), 0, 0)
        ga.addWidget(QtWidgets.QLabel("File axis"), 0, 1)
        ga.addWidget(QtWidgets.QLabel("Reverse"), 0, 2)
        self.cmb_axis: List[QtWidgets.QComboBox] = []
        self.chk_flip: List[QtWidgets.QCheckBox] = []
        labels = ("Sagittal slices (left-right axis)", "Coronal slices (anterior-posterior axis)",
                  "Transverse slices (superior-inferior axis)")
        for r in range(3):
            cb = QtWidgets.QComboBox()
            for a in range(3):
                n = f" ({file_shape[a]} voxels)" if file_shape else ""
                cb.addItem(f"axis {a}{n}", a)
            ck = QtWidgets.QCheckBox("flip")
            self.cmb_axis.append(cb)
            self.chk_flip.append(ck)
            ga.addWidget(QtWidgets.QLabel(labels[r]), r + 1, 0)
            ga.addWidget(cb, r + 1, 1)
            ga.addWidget(ck, r + 1, 2)

        # ---- spacing -------------------------------------------------------------
        g_sp = QtWidgets.QGroupBox("Voxel spacing in mm (per FILE axis)")
        gs = QtWidgets.QHBoxLayout(g_sp)
        self.spin_sp: List[QtWidgets.QDoubleSpinBox] = []
        for a in range(3):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setDecimals(4)
            sp.setRange(0.0001, 1000.0)
            sp.setSingleStep(0.01)
            sp.setSuffix(" mm")
            self.spin_sp.append(sp)
            gs.addWidget(QtWidgets.QLabel(f"axis {a}:"))
            gs.addWidget(sp)

        # ---- per-view display ----------------------------------------------------
        g_view = QtWidgets.QGroupBox("Display of each 2D view (applied in this order: transpose, rotate, flip)")
        gv = QtWidgets.QGridLayout(g_view)
        for c, t in enumerate(("View", "Transpose", "Rotate (counter-clockwise)", "Flip horizontally", "Flip vertically")):
            gv.addWidget(QtWidgets.QLabel(f"<b>{t}</b>"), 0, c)
        self.chk_t: List[QtWidgets.QCheckBox] = []
        self.cmb_rot: List[QtWidgets.QComboBox] = []
        self.chk_fh: List[QtWidgets.QCheckBox] = []
        self.chk_fv: List[QtWidgets.QCheckBox] = []
        for p in range(3):
            ct, cr, ch, cv = QtWidgets.QCheckBox(), QtWidgets.QComboBox(), QtWidgets.QCheckBox(), QtWidgets.QCheckBox()
            for k in range(4):
                cr.addItem(f"{90 * k} deg", k)
            self.chk_t.append(ct)
            self.cmb_rot.append(cr)
            self.chk_fh.append(ch)
            self.chk_fv.append(cv)
            gv.addWidget(QtWidgets.QLabel(PLANE_NAMES[p]), p + 1, 0)
            for c, w in enumerate((ct, cr, ch, cv)):
                gv.addWidget(w, p + 1, c + 1)

        # ---- live description ------------------------------------------------
        self.lbl_desc = QtWidgets.QLabel()
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_desc.setStyleSheet("color: #aaa;")

        # ---- buttons -----------------------------------------------------------
        b_ok = QtWidgets.QPushButton("OK")
        b_apply = QtWidgets.QPushButton("Apply")
        b_cancel = QtWidgets.QPushButton("Cancel")
        b_reset = QtWidgets.QPushButton("Defaults")
        b_load = QtWidgets.QPushButton("Load JSON ...")
        b_save = QtWidgets.QPushButton("Save JSON ...")
        b_ok.setDefault(True)
        b_ok.clicked.connect(lambda: self._apply(close=True))
        b_apply.clicked.connect(lambda: self._apply(close=False))
        b_cancel.clicked.connect(self.reject)
        b_reset.clicked.connect(lambda: self.set_config(OrientationConfig()))
        b_load.clicked.connect(self._load)
        b_save.clicked.connect(self._save)
        row = QtWidgets.QHBoxLayout()
        for b in (b_reset, b_load, b_save):
            row.addWidget(b)
        row.addStretch(1)
        for b in (b_ok, b_apply, b_cancel):
            row.addWidget(b)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(g_axes)
        lay.addWidget(g_sp)
        lay.addWidget(g_view)
        lay.addWidget(QtWidgets.QLabel("<b>What each screen direction currently means:</b>"))
        lay.addWidget(self.lbl_desc)
        lay.addLayout(row)

        for w in (*self.cmb_axis, *self.cmb_rot):
            w.currentIndexChanged.connect(self._update_desc)
        for w in (*self.chk_flip, *self.chk_t, *self.chk_fh, *self.chk_fv):
            w.toggled.connect(self._update_desc)
        self.set_config(config)

    # ------------------------------------------------------------------ state
    def set_config(self, cfg: OrientationConfig) -> None:
        for r, (axis, flip) in enumerate(zip(cfg.perm, cfg.flips)):
            self.cmb_axis[r].setCurrentIndex(int(axis))
            self.chk_flip[r].setChecked(bool(flip))
        for a in range(3):
            self.spin_sp[a].setValue(float(cfg.spacing_mm[a]))
        for p in range(3):
            v = cfg.views[p]
            self.chk_t[p].setChecked(bool(v.transpose))
            self.cmb_rot[p].setCurrentIndex(int(v.rot90) % 4)
            self.chk_fh[p].setChecked(bool(v.flip_h))
            self.chk_fv[p].setChecked(bool(v.flip_v))
        self._channel_names = cfg.channel_names
        self._update_desc()

    def get_config(self) -> OrientationConfig:
        cfg = OrientationConfig(
            axis_sagittal=self.cmb_axis[0].currentData(),
            axis_coronal=self.cmb_axis[1].currentData(),
            axis_transverse=self.cmb_axis[2].currentData(),
            flip_sagittal=self.chk_flip[0].isChecked(),
            flip_coronal=self.chk_flip[1].isChecked(),
            flip_transverse=self.chk_flip[2].isChecked(),
            spacing_mm=tuple(float(s.value()) for s in self.spin_sp),  # type: ignore[arg-type]
            views=[ViewOptions(self.chk_t[p].isChecked(), int(self.cmb_rot[p].currentData()),
                               self.chk_fh[p].isChecked(), self.chk_fv[p].isChecked()) for p in range(3)],
            channel_names=self._channel_names,
        )
        return cfg

    def _update_desc(self) -> None:
        try:
            cfg = self.get_config()
            cfg.validate()
        except ValueError as exc:
            self.lbl_desc.setText(f"<span style='color:#e66'>{exc}</span>")
            return
        perm, flips = cfg.perm, cfg.flips
        lines = []
        for p in range(3):
            tr = DisplayTransform.build(p, cfg.views[p])

            def part(axis: int, dflip: bool, fwd: str, back: str) -> str:
                rev = dflip ^ bool(flips[axis])
                return f"file axis {perm[axis]} [{ROLE_NAMES[axis]}] increases {back if rev else fwd}"

            lines.append(f"<b>{PLANE_NAMES[p]}</b>: horizontal = {part(tr.col_axis, tr.col_flip, 'to the right', 'to the left')}; "
                         f"vertical = {part(tr.row_axis, tr.row_flip, 'downward', 'upward')}")
        self.lbl_desc.setText("<br>".join(lines))

    # ------------------------------------------------------------------ buttons
    def _apply(self, close: bool) -> None:
        try:
            cfg = self.get_config()
            cfg.validate()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid orientation", str(exc))
            return
        self.applied.emit(cfg)
        if close:
            self.accept()

    def _load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load orientation config", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.set_config(OrientationConfig.load(path))
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot load config", str(exc))

    def _save(self) -> None:
        try:
            cfg = self.get_config()
            cfg.validate()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid orientation", str(exc))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save orientation config",
                                                        "orientation_config.json", "JSON (*.json)")
        if path:
            cfg.save(path)

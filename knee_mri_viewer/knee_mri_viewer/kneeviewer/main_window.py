"""Main window: builds the 2x2 grid and connects the shared state to the views.

Synchronisation design
----------------------
                +-------------------+
   slider /     |    CursorModel    |  one selected voxel, canonical (i, j, k)
   click  ----> |  .set(idx) -> emit|----> SliceView x3 .set_cursor(idx)
                +-------------------+----> View3D       .set_cursor(idx)
                                      ---> voxel read-out (panel + status bar)

   control panel --> ViewerSettings.set(key, value) --emit changed(key)--> all views

A view only redraws what the change affects: a SliceView re-extracts its slice
only if the index along ITS OWN normal axis changed; View3D moves only the plane
whose index changed.
"""
from __future__ import annotations

import gc
import traceback
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .config import OrientationConfig
from .controls import ControlPanel
from .data_io import DataLoadError, apply_metadata, load_case
from .dialogs import OpenCaseDialog, OrientationDialog
from .slice_view import SliceView
from .state import CursorModel, ViewerSettings
from .view3d import View3D
from .volume_model import VolumeModel

HELP_TEXT = """<h3>Mouse and keyboard</h3>
<b>2D views</b><ul>
<li>Left click: move the crosshair (all views follow)</li>
<li>Shift + left drag: move the crosshair continuously</li>
<li>Mouse wheel: previous / next slice</li>
<li>Ctrl + wheel or right drag: zoom; left or middle drag: pan</li>
<li>"Reset view" button: fit the slice again</li></ul>
<b>3D view</b><ul>
<li>Left drag: rotate &nbsp; Middle drag or Shift + left drag: pan &nbsp; Wheel or right drag: zoom</li>
<li>Iso / Sag / Cor / Ax buttons: preset camera directions</li></ul>
<b>Coordinates</b><br>
(i, j, k) are indices into the array as stored in the file (0-based).
Slice numbers are 0-based as well."""


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Knee MRI Viewer")
        self.resize(1560, 960)

        self.settings = ViewerSettings()
        self.cursor = CursorModel()
        self.model: Optional[VolumeModel] = None
        self.config = OrientationConfig()
        self._last_dir = ""
        self._last_paths = ("", "")
        if config_path:
            try:
                self.config = OrientationConfig.load(config_path)
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Orientation config ignored", str(exc))

        # ---- 2 x 2 grid -----------------------------------------------------
        self.view3d = View3D(self.settings)
        self.views = [SliceView(p, self.settings) for p in range(3)]   # sagittal, coronal, transverse
        central = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(central)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(4)
        grid.addWidget(self.view3d, 0, 0)       # top left
        grid.addWidget(self.views[0], 0, 1)     # top right: sagittal
        grid.addWidget(self.views[1], 1, 0)     # bottom left: coronal
        grid.addWidget(self.views[2], 1, 1)     # bottom right: transverse
        for i in range(2):
            grid.setRowStretch(i, 1)
            grid.setColumnStretch(i, 1)
        self.setCentralWidget(central)

        # ---- control panel ----------------------------------------------------
        self.panel = ControlPanel(self.settings)
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setWidget(self.panel)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # ---- menus / status bar ---------------------------------------------
        self._build_menus()
        self.lbl_status = QtWidgets.QLabel("Open a case with File > Open MRI + segmentation ...")
        self.statusBar().addPermanentWidget(self.lbl_status)

        # ---- wiring -------------------------------------------------------------
        self.settings.changed.connect(self._on_setting)
        for v in self.views:
            v.cursorRequested.connect(self._on_cursor_requested)
        self.cursor.changed.connect(self._on_cursor_changed)
        self.panel.voxelEdited.connect(self._on_voxel_edited)

    # ------------------------------------------------------------------ menus
    def _build_menus(self) -> None:
        mb = self.menuBar()
        m_file = mb.addMenu("&File")
        a_open = m_file.addAction("&Open MRI + segmentation ...")
        a_open.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Open))
        a_open.triggered.connect(self.open_case_dialog)
        m_file.addSeparator()
        a_quit = m_file.addAction("&Quit")
        a_quit.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Quit))
        a_quit.triggered.connect(self.close)

        m_view = mb.addMenu("&View")
        a_orient = m_view.addAction("&Orientation and voxel spacing ...")
        a_orient.setShortcut("Ctrl+E")
        a_orient.triggered.connect(self.orientation_dialog)
        a_reset2d = m_view.addAction("Reset all &2D views")
        a_reset2d.triggered.connect(lambda: [v.reset_view() for v in self.views])
        a_reset3d = m_view.addAction("Reset &3D camera")
        a_reset3d.triggered.connect(lambda: self.view3d._camera("view_isometric"))

        m_help = mb.addMenu("&Help")
        a_help = m_help.addAction("Mouse and keyboard")
        a_help.triggered.connect(lambda: QtWidgets.QMessageBox.information(self, "Controls", HELP_TEXT))

    # ------------------------------------------------------------------ loading
    def open_case_dialog(self) -> None:
        dlg = OpenCaseDialog(self, *self._last_paths, start_dir=self._last_dir)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.load_case(dlg.mri_path, dlg.seg_path, dlg.threshold, dlg.use_metadata)

    def load_case(self, mri_path: str, seg_path: Optional[str] = None,
                  threshold: float = 0.0, use_metadata: bool = True) -> bool:
        self._last_paths = (mri_path, seg_path or "")
        self._last_dir = str(Path(mri_path).parent)

        dlg = QtWidgets.QProgressDialog("Loading ...", "", 0, 100, self)
        dlg.setCancelButton(None)
        dlg.setWindowTitle("Loading")
        dlg.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)

        def progress(msg: str, frac: Optional[float] = None) -> None:
            dlg.setLabelText(msg)
            if frac is not None:
                dlg.setValue(int(frac * 90))
            QtWidgets.QApplication.processEvents()

        try:
            self.model = None                      # let the previous volume be freed first
            self.panel.model = None
            for v in self.views:
                v.release()
            self.view3d.release()
            gc.collect()
            case = load_case(mri_path, seg_path, threshold, progress)
            cfg = self.config
            notes = []
            if use_metadata:
                cfg, notes = apply_metadata(cfg, case.metadata)
            progress("Building 2D / 3D views ...", None)
            dlg.setValue(92)
            self.model = VolumeModel(case.mri, case.seg, case.mask, cfg)
            self.config = self.model.config
            self._refresh_all()
        except DataLoadError as exc:
            dlg.close()
            QtWidgets.QMessageBox.critical(self, "Cannot load data", str(exc))
            return False
        except Exception as exc:  # unexpected: show it, and print the traceback to the console
            traceback.print_exc()
            dlg.close()
            QtWidgets.QMessageBox.critical(self, "Unexpected error while loading",
                                           f"{type(exc).__name__}: {exc}\n\nSee the console for details.")
            return False
        dlg.close()

        shown = [w for w in case.warnings if "binarised" not in w]
        msg = f"Loaded {Path(mri_path).name}" + (f" + {Path(seg_path).name}" if seg_path else " (no segmentation)")
        if notes:
            msg += "  |  " + "; ".join(notes)
        self.statusBar().showMessage(msg, 15000)
        self.setWindowTitle(f"Knee MRI Viewer - {Path(mri_path).name}")
        if shown:
            QtWidgets.QMessageBox.warning(self, "Please check", "\n\n".join(shown))
        return True

    # ------------------------------------------------------------------ orientation
    def orientation_dialog(self) -> None:
        shape = self.model.file_shape if self.model else None
        dlg = OrientationDialog(self.config, shape, self)
        dlg.applied.connect(self.apply_config)
        dlg.exec()

    def apply_config(self, cfg: OrientationConfig) -> None:
        """Re-interpret the loaded arrays with a new orientation. No reload from disk."""
        if self.model is None:
            self.config = cfg
            return
        keep = self.model.canon_to_file(self.cursor.idx)     # stay on the same physical voxel
        try:
            self.model.set_config(cfg)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid orientation", str(exc))
            return
        self.config = self.model.config
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self._refresh_all(file_idx=keep, reset_window=False)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # ------------------------------------------------------------------ refresh
    def _refresh_all(self, file_idx=None, reset_window: bool = True) -> None:
        m = self.model
        canon = m.file_to_canon(file_idx) if file_idx is not None else None
        self.cursor.reset(m.shape, canon)

        # The panel updates settings (window defaults, channel colours...). Do that silently
        # first so the views read consistent values when they are given the model.
        self.settings.blockSignals(True)
        try:
            self.panel.set_model(m, reset_window)
        finally:
            self.settings.blockSignals(False)

        for v in self.views:
            v.set_model(m, self.cursor.idx)
        self.view3d.set_model(m, self.cursor.idx)
        self._update_readout()

    def _update_readout(self) -> None:
        m = self.model
        if m is None:
            return
        fidx, value, channels = m.point_info(self.cursor.idx)
        self.panel.set_voxel_info(fidx, self.cursor.idx, value, channels)
        seg = ",".join(str(c) for c in channels) if channels else "-"
        self.lbl_status.setText(
            f"voxel (i, j, k) = ({fidx[0]}, {fidx[1]}, {fidx[2]})   intensity {value:.4g}   segmentation channels: {seg}")

    # ------------------------------------------------------------------ slots
    def _on_setting(self, key: str) -> None:
        for v in self.views:
            v.on_setting(key)
        self.view3d.on_setting(key)

    def _on_cursor_requested(self, i: int, j: int, k: int) -> None:
        self.cursor.set((i, j, k))

    def _on_cursor_changed(self, idx) -> None:
        for v in self.views:
            v.set_cursor(idx)
        self.view3d.set_cursor(idx)
        self._update_readout()

    def _on_voxel_edited(self, i: int, j: int, k: int) -> None:
        if self.model is not None:
            self.cursor.set(self.model.file_to_canon((i, j, k)))

    # ------------------------------------------------------------------ closing
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.view3d.shutdown()
        super().closeEvent(event)

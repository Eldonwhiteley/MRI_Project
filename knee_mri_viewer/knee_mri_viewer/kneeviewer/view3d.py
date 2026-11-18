"""Interactive 3D window (PyVistaQt interactor + plain VTK pipeline).

Scene contents (world units = millimetres, x = L-R, y = A-P, z = S-I):

* MRI volume rendering   vtkSmartVolumeMapper (GPU ray casting when available,
                         automatic CPU fallback), built from a downsampled uint8
                         copy; opacity / contrast are transfer functions.
* Segmentation surfaces  flying-edges + windowed-sinc smoothing, built lazily per
                         channel (and for the combined mask) and cached.
* Three slice planes     textured quads, one per anatomical plane. Moving a
                         slider only rewrites that plane's 3 corner points and its
                         texture pixels - nothing is rebuilt.

Rendering is coalesced: any number of updates inside ~15 ms cause ONE render.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import vtk
from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor
from vtkmodules.util import numpy_support

from .overlays import compose_rgb, overlay_rgba, window_to_u8
from .state import PLANE_COLORS, ViewerSettings
from .volume_model import VolumeModel, inplane_axes
from .vtk_utils import build_surface, make_vtk_image

TEX_KEYS = {"layer_mode", "seg_mode", "overlay_opacity", "overlay_2d",
            "combined_color", "channel_colors", "win2d_lo", "win2d_hi"}
SURF_KEYS = {"layer_mode", "surf_visible", "surf_opacity", "seg_mode",
             "combined_color", "channel_colors"}
TF_KEYS = {"vol_opacity", "vol_win_lo", "vol_win_hi", "vol_gamma"}
VIS_KEYS = {"layer_mode", "vol_visible", "plane_sag", "plane_cor", "plane_axi"}
PLANE_KEYS = ("plane_sag", "plane_cor", "plane_axi")


class _Plane:
    """VTK objects of one slice plane."""

    def __init__(self) -> None:
        self.source = vtk.vtkPlaneSource()
        self.tex_img = vtk.vtkImageData()
        self.tex_arr: Optional[np.ndarray] = None
        self.texture = vtk.vtkTexture()
        self.actor = vtk.vtkActor()
        self.frame = vtk.vtkActor()
        self.shown_idx: Optional[int] = None


class View3D(QtWidgets.QFrame):
    def __init__(self, settings: ViewerSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.model: Optional[VolumeModel] = None
        self._cursor: Tuple[int, int, int] = (0, 0, 0)
        self._planes: List[_Plane] = []
        self._vol: Optional[vtk.vtkVolume] = None
        self._vol_mapper = None
        self._vol_prop = None
        self._ctf = None
        self._otf = None
        self._vol_image = None
        self._surf_actors: Dict[object, Optional[vtk.vtkActor]] = {}
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        self.title = QtWidgets.QLabel("3D knee volume")
        self.title.setStyleSheet("font-weight: 600;")
        self.status = QtWidgets.QLabel("")
        head = QtWidgets.QHBoxLayout()
        head.addWidget(self.title)
        head.addStretch(1)
        head.addWidget(self.status)
        for text, tip, name in (("Iso", "Isometric view", "view_isometric"),
                                ("Sag", "View along the L-R axis", "view_yz"),
                                ("Cor", "View along the A-P axis", "view_xz"),
                                ("Ax", "View along the S-I axis", "view_xy")):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, n=name: self._camera(n))
            head.addWidget(b)
        b = QtWidgets.QToolButton()
        b.setText("Reset camera")
        b.clicked.connect(lambda _=False: self._camera("view_isometric"))
        head.addWidget(b)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("black")
        try:
            self.plotter.add_axes(xlabel="L-R", ylabel="A-P", zlabel="S-I")
        except Exception:
            self.plotter.add_axes()
        self.renderer = self.plotter.renderer

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)
        lay.addLayout(head)
        lay.addWidget(self.plotter.interactor, 1)

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(15)
        self._timer.timeout.connect(self._do_render)

    # ------------------------------------------------------------ rendering
    def request_render(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def _do_render(self) -> None:
        try:
            self.plotter.render()
        except Exception:
            pass

    def _camera(self, name: str) -> None:
        if self.model is None:
            return
        try:
            getattr(self.plotter, name)()
            self.plotter.reset_camera()
        except Exception:
            pass
        self.request_render()

    def shutdown(self) -> None:
        self._timer.stop()
        try:
            self.plotter.close()
        except Exception:
            pass

    # ------------------------------------------------------------ scene lifecycle
    def clear_scene(self) -> None:
        for p in self._planes:
            self.renderer.RemoveActor(p.actor)
            self.renderer.RemoveActor(p.frame)
        self._planes = []
        self._remove_volume()
        self._clear_surfaces()

    def _remove_volume(self) -> None:
        if self._vol is not None:
            self.renderer.RemoveVolume(self._vol)
        self._vol = self._vol_mapper = self._vol_prop = self._ctf = self._otf = self._vol_image = None

    def _clear_surfaces(self) -> None:
        for actor in self._surf_actors.values():
            if actor is not None:
                self.renderer.RemoveActor(actor)
        self._surf_actors = {}

    def release(self) -> None:
        """Remove everything from the scene and drop the volume reference."""
        self.clear_scene()
        self.model = None
        self.request_render()

    def set_model(self, model: VolumeModel, cursor: Tuple[int, int, int]) -> None:
        self._busy(True)
        try:
            self.clear_scene()
            self.model = model
            self._cursor = tuple(cursor)  # type: ignore[assignment]
            self._build_planes()
            self._build_volume()
            self.refresh_surfaces()
            self._update_textures()
            self._apply_visibility()
            self._apply_depth_peeling()
            self.plotter.reset_camera()
            self._camera("view_isometric")
        finally:
            self._busy(False)
        self.request_render()

    @staticmethod
    def _busy(on: bool) -> None:
        if on:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        else:
            QtWidgets.QApplication.restoreOverrideCursor()

    # ------------------------------------------------------------ slice planes
    def _build_planes(self) -> None:
        m = self.model
        self._planes = []
        for p in range(3):
            a0, a1 = inplane_axes(p)
            n0, n1 = m.shape[a0], m.shape[a1]
            pl = _Plane()
            pl.tex_img.SetDimensions(n1, n0, 1)          # width = columns (a1), height = rows (a0)
            pl.tex_img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
            pl.tex_arr = numpy_support.vtk_to_numpy(pl.tex_img.GetPointData().GetScalars()).reshape(n0, n1, 3)
            pl.tex_arr[...] = 0
            pl.texture.SetInputData(pl.tex_img)
            pl.texture.InterpolateOff()                  # crisp voxels
            pl.texture.RepeatOff()
            pl.source.SetXResolution(1)
            pl.source.SetYResolution(1)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(pl.source.GetOutputPort())
            pl.actor.SetMapper(mapper)
            pl.actor.SetTexture(pl.texture)
            pl.actor.GetProperty().SetLighting(False)

            fmapper = vtk.vtkPolyDataMapper()
            fmapper.SetInputConnection(pl.source.GetOutputPort())
            pl.frame.SetMapper(fmapper)
            fp = pl.frame.GetProperty()
            fp.SetRepresentationToWireframe()
            fp.SetColor(*(c / 255.0 for c in PLANE_COLORS[p]))
            fp.SetLineWidth(2.0)
            fp.SetLighting(False)
            try:  # keep the outline from z-fighting with the textured quad
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(1.0, 1.0)
                fmapper.SetResolveCoincidentTopologyToPolygonOffset()
            except Exception:
                pass

            self.renderer.AddActor(pl.actor)
            self.renderer.AddActor(pl.frame)
            self._planes.append(pl)
            self._place_plane(p)

    def _place_plane(self, p: int) -> None:
        """Move plane p to the cursor position along its normal axis (3 points + Update)."""
        m = self.model
        pl = self._planes[p]
        a0, a1 = inplane_axes(p)
        sp, sh = m.spacing, m.shape
        lo = [-0.5 * sp[a] for a in range(3)]
        hi = [(sh[a] - 0.5) * sp[a] for a in range(3)]
        pos = self._cursor[p] * sp[p]

        def pt(v0: float, v1: float) -> List[float]:
            q = [0.0, 0.0, 0.0]
            q[p] = pos
            q[a0] = v0
            q[a1] = v1
            return q

        # texture s runs along a1 (columns), t along a0 (rows) - same layout as tex_arr
        pl.source.SetOrigin(*pt(lo[a0], lo[a1]))
        pl.source.SetPoint1(*pt(lo[a0], hi[a1]))
        pl.source.SetPoint2(*pt(hi[a0], lo[a1]))
        pl.source.Update()

    def _update_textures(self, only: Optional[int] = None, force: bool = False) -> None:
        m = self.model
        if m is None:
            return
        s = self.settings
        mode = s["layer_mode"]
        lo, hi = m.window_levels(s["win2d_lo"], s["win2d_hi"])
        show_ovl = m.seg is not None and bool(s["overlay_2d"]) and mode in ("both", "seg")
        for p, pl in enumerate(self._planes):
            if only is not None and p != only:
                continue
            idx = self._cursor[p]
            if not force and pl.shown_idx == idx and only is None:
                continue
            pl.shown_idx = idx
            gray = window_to_u8(m.mri_slice(p, idx), lo, hi) if mode != "seg" else None
            rgba = None
            if show_ovl:
                rgba = overlay_rgba(m, p, idx, s["seg_mode"], s["combined_color"], s["channel_colors"])
            opacity = 1.0 if mode == "seg" else float(s["overlay_opacity"])
            shape = m.mri_slice(p, idx).shape
            pl.tex_arr[...] = compose_rgb(gray, rgba, opacity, shape)
            pl.tex_img.GetPointData().GetScalars().Modified()
            pl.tex_img.Modified()
            pl.texture.Modified()

    # ------------------------------------------------------------ volume rendering
    def _build_volume(self) -> None:
        m = self.model
        self._remove_volume()
        f = int(self.settings["vol_ds"])
        u8 = m.volume_u8(f)
        spacing = tuple(s * f for s in m.spacing)
        origin = tuple(0.5 * (f - 1) * s for s in m.spacing)   # centre of the first block
        self._vol_image = make_vtk_image(u8, spacing, origin)
        del u8

        self._ctf = vtk.vtkColorTransferFunction()
        self._otf = vtk.vtkPiecewiseFunction()
        prop = vtk.vtkVolumeProperty()
        prop.SetColor(self._ctf)
        prop.SetScalarOpacity(self._otf)
        prop.SetInterpolationTypeToLinear()
        prop.ShadeOn()
        prop.SetAmbient(0.35)
        prop.SetDiffuse(0.65)
        prop.SetSpecular(0.15)
        # opacity is defined "per this many mm" -> independent of the downsampling factor
        prop.SetScalarOpacityUnitDistance(8.0 * float(np.mean(m.spacing)))
        self._vol_prop = prop

        mapper = vtk.vtkSmartVolumeMapper()      # GPU ray casting when supported, CPU otherwise
        mapper.SetInputData(self._vol_image)
        mapper.SetBlendModeToComposite()
        self._vol_mapper = mapper

        self._vol = vtk.vtkVolume()
        self._vol.SetMapper(mapper)
        self._vol.SetProperty(prop)
        self._update_transfer_functions()
        self.renderer.AddVolume(self._vol)

    def _update_transfer_functions(self) -> None:
        if self._ctf is None:
            return
        s = self.settings
        lo = float(s["vol_win_lo"]) * 255.0
        hi = max(float(s["vol_win_hi"]) * 255.0, lo + 1.0)
        gamma = max(float(s["vol_gamma"]), 0.05)
        scale = float(s["vol_opacity"])

        self._ctf.RemoveAllPoints()
        self._ctf.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        if lo > 0.0:
            self._ctf.AddRGBPoint(lo, 0.0, 0.0, 0.0)
        self._ctf.AddRGBPoint(hi, 1.0, 1.0, 1.0)
        if hi < 255.0:
            self._ctf.AddRGBPoint(255.0, 1.0, 1.0, 1.0)

        self._otf.RemoveAllPoints()
        for x in np.linspace(0.0, 255.0, 64):
            t = min(max((x - lo) / (hi - lo), 0.0), 1.0)
            self._otf.AddPoint(float(x), scale * (t ** gamma))

    # ------------------------------------------------------------ surfaces
    def _surface_keys(self) -> List[object]:
        m = self.model
        mode = self.settings["seg_mode"]
        if mode == "combined":
            return ["combined"]
        if mode == "all":
            return list(range(m.n_channels))
        return [int(mode)]

    def _get_surface(self, key) -> Optional[vtk.vtkActor]:
        """Build (once) and cache the surface actor for 'combined' or a channel number."""
        if key in self._surf_actors:
            return self._surf_actors[key]
        m = self.model
        if key == "combined":
            mask = m.mask
        else:
            if key < 0 or key >= m.n_channels:
                return None
            mask = m.seg[..., key]
        self.status.setText("building surface ...")
        self.status.repaint()          # paint just this label now (no event-loop re-entry)
        self._busy(True)
        try:
            poly = build_surface(mask, m.spacing, int(self.settings["surf_ds"]))
        finally:
            self._busy(False)
            self.status.setText("")
        if poly is None:
            self._surf_actors[key] = None
            return None
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetSpecular(0.3)
        prop.SetSpecularPower(25.0)
        self.renderer.AddActor(actor)
        self._surf_actors[key] = actor
        return actor

    def refresh_surfaces(self) -> None:
        m = self.model
        if m is None:
            return
        for a in self._surf_actors.values():
            if a is not None:
                a.SetVisibility(False)
        s = self.settings
        if m.seg is None or not s["surf_visible"] or s["layer_mode"] not in ("both", "seg"):
            self.request_render()
            return
        for key in self._surface_keys():
            actor = self._get_surface(key)
            if actor is None:
                continue
            col = s["combined_color"] if key == "combined" else s["channel_colors"][key]
            prop = actor.GetProperty()
            prop.SetColor(*(c / 255.0 for c in col))
            prop.SetOpacity(float(s["surf_opacity"]))
            actor.SetVisibility(True)
        self.request_render()

    # ------------------------------------------------------------ visibility / peeling
    def _apply_visibility(self) -> None:
        s = self.settings
        if self._vol is not None:
            self._vol.SetVisibility(bool(s["vol_visible"]) and s["layer_mode"] != "seg")
        for p, pl in enumerate(self._planes):
            vis = bool(s[PLANE_KEYS[p]])
            pl.actor.SetVisibility(vis)
            pl.frame.SetVisibility(vis)

    def _apply_depth_peeling(self) -> None:
        on = bool(self.settings["depth_peeling"])
        try:
            if on:
                self.plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)
            else:
                self.plotter.disable_depth_peeling()
            self.renderer.SetUseDepthPeelingForVolumes(on)
        except Exception:
            pass

    # ------------------------------------------------------------ external events
    def set_cursor(self, idx: Tuple[int, int, int]) -> None:
        if self.model is None or not self._planes:
            return
        old = self._cursor
        self._cursor = tuple(idx)  # type: ignore[assignment]
        for p in range(3):
            if old[p] != self._cursor[p]:
                self._place_plane(p)
                self._update_textures(only=p, force=True)
        self.request_render()

    def on_setting(self, key: str) -> None:
        if self.model is None:
            return
        if key in TEX_KEYS:
            self._update_textures(force=True)
        if key in TF_KEYS:
            self._update_transfer_functions()
        if key in VIS_KEYS:
            self._apply_visibility()
        if key in SURF_KEYS:
            self.refresh_surfaces()
        if key == "vol_ds":
            self._busy(True)
            try:
                self._build_volume()
                self._apply_visibility()
            finally:
                self._busy(False)
        if key == "surf_ds":
            self._clear_surfaces()
            self.refresh_surfaces()
        if key == "depth_peeling":
            self._apply_depth_peeling()
        self.request_render()

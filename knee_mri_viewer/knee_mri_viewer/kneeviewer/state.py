"""Shared state objects. Views never talk to each other directly; they talk to these.

* ``ViewerSettings`` - every user-adjustable display option. Widgets in the
  control panel write to it, views listen to ``changed(key)``.
* ``CursorModel``    - the one selected voxel (canonical indices). Every view
  reads it and every view's slider / click writes to it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from PySide6 import QtCore

Signal = QtCore.Signal

# Colour of each anatomical plane (sagittal, coronal, transverse). Used for the
# crosshair lines in the 2D views and the plane outlines in the 3D view.
PLANE_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (255, 214, 10),   # sagittal   - yellow
    (64, 220, 80),    # coronal    - green
    (255, 80, 80),    # transverse - red
)

CHANNEL_PALETTE: List[Tuple[int, int, int]] = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40),
]

DEFAULTS: Dict[str, Any] = {
    # what is shown: "both" | "mri" | "seg"
    "layer_mode": "both",
    # segmentation: "combined" | "all" | channel number
    "seg_mode": "combined",
    "overlay_2d": True,
    "overlay_opacity": 0.5,
    "combined_color": (0, 200, 255),
    "channel_colors": list(CHANNEL_PALETTE[:6]),
    # 2D window/level as fractions of the robust intensity range
    "win2d_lo": 0.0,
    "win2d_hi": 1.0,
    # 3D MRI volume rendering
    "vol_visible": True,
    "vol_opacity": 0.5,
    "vol_win_lo": 0.15,
    "vol_win_hi": 0.70,
    "vol_gamma": 1.5,
    "vol_ds": 2,
    # 3D segmentation surfaces
    "surf_visible": True,
    "surf_opacity": 1.0,
    "surf_ds": 1,
    "depth_peeling": False,
    # 3D slice planes (sagittal, coronal, transverse)
    "plane_sag": True,
    "plane_cor": True,
    "plane_axi": True,
}


class ViewerSettings(QtCore.QObject):
    changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._v: Dict[str, Any] = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}

    def get(self, key: str) -> Any:
        return self._v[key]

    __getitem__ = get

    def set(self, key: str, value: Any) -> None:
        if key not in self._v:
            raise KeyError(key)
        if self._v[key] == value:
            return
        self._v[key] = value
        self.changed.emit(key)

    def ensure_channels(self, n: int) -> None:
        """Make sure there is a colour for each of ``n`` channels."""
        cols = list(self._v["channel_colors"])
        i = len(cols)
        while len(cols) < n:
            cols.append(CHANNEL_PALETTE[i % len(CHANNEL_PALETTE)])
            i += 1
        if len(cols) != len(self._v["channel_colors"]):
            self.set("channel_colors", cols)

    def set_channel_color(self, channel: int, rgb: Tuple[int, int, int]) -> None:
        cols = list(self._v["channel_colors"])
        cols[channel] = tuple(rgb)
        self.set("channel_colors", cols)


class CursorModel(QtCore.QObject):
    """The selected voxel in canonical (sagittal, coronal, transverse) indices."""

    changed = Signal(tuple)

    def __init__(self) -> None:
        super().__init__()
        self.shape: Tuple[int, int, int] = (1, 1, 1)
        self.idx: Tuple[int, int, int] = (0, 0, 0)

    def _clamp(self, idx) -> Tuple[int, int, int]:
        return tuple(int(min(max(int(v), 0), n - 1)) for v, n in zip(idx, self.shape))  # type: ignore[return-value]

    def reset(self, shape, idx=None) -> None:
        """New volume: set shape and position WITHOUT emitting (callers refresh views explicitly)."""
        self.shape = tuple(int(s) for s in shape)  # type: ignore[assignment]
        if idx is None:
            idx = tuple(s // 2 for s in self.shape)
        self.idx = self._clamp(idx)

    def set(self, idx) -> None:
        new = self._clamp(idx)
        if new != self.idx:
            self.idx = new
            self.changed.emit(new)

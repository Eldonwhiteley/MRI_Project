"""Orientation and voxel-spacing configuration.

Terminology
-----------
*File axes* are the axes of the NumPy array exactly as stored in the HDF5 file
(for your data: axis 0, 1 -> 384 voxels, axis 2 -> 160 voxels).

*Canonical axes* are the three anatomical axes the viewer works with internally:

    canonical axis 0  =  left-right      (sagittal slices are stacked along it)
    canonical axis 1  =  anterior-posterior  (coronal slices are stacked along it)
    canonical axis 2  =  superior-inferior   (transverse/axial slices stack along it)

``OrientationConfig`` says which file axis plays which anatomical role, whether
the axis must be reversed, what the voxel spacing is, and how each 2D view
should be transposed / rotated / flipped for display.

Nothing here modifies the data on disk and nothing is copied: the viewer builds
NumPy *views* (``transpose`` + negative-stride slicing) from this description.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

PLANE_NAMES = ("Sagittal", "Coronal", "Transverse")
ROLE_NAMES = ("L-R", "A-P", "S-I")

# Typical OAI DESS knee voxel size (mm). This is only a DEFAULT -- verify it
# against your dataset documentation and edit it in Orientation > Voxel spacing.
DEFAULT_SPACING_MM = (0.3646, 0.3646, 0.7)


@dataclass
class ViewOptions:
    """Extra display-only operations for one 2D view (applied in this order:
    transpose, rotate, flip horizontally, flip vertically)."""

    transpose: bool = False
    rot90: int = 0  # number of counter-clockwise quarter turns, 0..3
    flip_h: bool = False
    flip_v: bool = False


def _default_views() -> List[ViewOptions]:
    return [ViewOptions(), ViewOptions(), ViewOptions()]


@dataclass
class OrientationConfig:
    # --- which FILE axis carries which anatomical direction -------------------
    axis_sagittal: int = 2    # sagittal slices are stacked along this file axis (L-R)
    axis_coronal: int = 1     # coronal slices are stacked along this file axis (A-P)
    axis_transverse: int = 0  # transverse slices are stacked along this file axis (S-I)

    # --- reverse the direction of an axis (index 0 <-> index n-1) -------------
    flip_sagittal: bool = False
    flip_coronal: bool = False
    flip_transverse: bool = False

    # --- voxel spacing in mm, one value per FILE axis (axis0, axis1, axis2) ---
    spacing_mm: Tuple[float, float, float] = DEFAULT_SPACING_MM

    # --- per-view display options: [sagittal, coronal, transverse] ------------
    views: List[ViewOptions] = field(default_factory=_default_views)

    # --- optional names for the segmentation channels -------------------------
    channel_names: Optional[List[str]] = None

    # ------------------------------------------------------------------ helpers
    @property
    def perm(self) -> Tuple[int, int, int]:
        """File axis used for canonical axis 0, 1, 2."""
        return (self.axis_sagittal, self.axis_coronal, self.axis_transverse)

    @property
    def flips(self) -> Tuple[bool, bool, bool]:
        return (self.flip_sagittal, self.flip_coronal, self.flip_transverse)

    def validate(self) -> None:
        if sorted(self.perm) != [0, 1, 2]:
            raise ValueError(
                "Each anatomical direction needs a different file axis "
                f"(got sagittal={self.axis_sagittal}, coronal={self.axis_coronal}, "
                f"transverse={self.axis_transverse}); they must be a permutation of 0, 1, 2."
            )
        if len(self.spacing_mm) != 3 or not all(float(s) > 0 for s in self.spacing_mm):
            raise ValueError(f"Voxel spacing must be three positive numbers, got {self.spacing_mm}.")
        if len(self.views) != 3:
            raise ValueError("Exactly three view option sets (sagittal, coronal, transverse) are required.")
        for v in self.views:
            if int(v.rot90) not in (0, 1, 2, 3):
                raise ValueError(f"rot90 must be 0, 1, 2 or 3 (got {v.rot90}).")

    def copy(self) -> "OrientationConfig":
        return copy.deepcopy(self)

    # --------------------------------------------------------------------- JSON
    def to_dict(self) -> dict:
        d = asdict(self)
        d["spacing_mm"] = [float(s) for s in self.spacing_mm]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "OrientationConfig":
        kw = {}
        for name in ("axis_sagittal", "axis_coronal", "axis_transverse"):
            if name in d:
                kw[name] = int(d[name])
        for name in ("flip_sagittal", "flip_coronal", "flip_transverse"):
            if name in d:
                kw[name] = bool(d[name])
        if "spacing_mm" in d:
            kw["spacing_mm"] = tuple(float(x) for x in d["spacing_mm"])
        if d.get("views") is not None:
            allowed = set(ViewOptions.__dataclass_fields__)
            views = [ViewOptions(**{k: v for k, v in vd.items() if k in allowed}) for vd in d["views"]]
            while len(views) < 3:
                views.append(ViewOptions())
            kw["views"] = views[:3]
        if d.get("channel_names"):
            kw["channel_names"] = [str(x) for x in d["channel_names"]]
        cfg = cls(**kw)
        cfg.validate()
        return cfg

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "OrientationConfig":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read orientation config '{path}': {exc}") from exc
        return cls.from_dict(data)

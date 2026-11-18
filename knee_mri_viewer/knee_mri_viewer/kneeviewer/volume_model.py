"""The data model shared by all views (no Qt / VTK imports).

``VolumeModel`` holds the arrays exactly as read from disk ("file order") and
exposes *canonical* zero-copy views of them:

    canonical axis 0 = left-right   (index of the SAGITTAL slice)
    canonical axis 1 = anterior-posterior (index of the CORONAL slice)
    canonical axis 2 = superior-inferior  (index of the TRANSVERSE slice)

A canonical view is ``np.transpose`` + negative-stride slicing of the file
array, so changing the orientation configuration never copies the volume.

World (millimetre) coordinates used by the 2D and 3D views:

    world = canonical_index * spacing        (voxel *centres*)

so a voxel occupies [(i-0.5)*s, (i+0.5)*s] along each axis, in every view.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .config import PLANE_NAMES, ROLE_NAMES, OrientationConfig, ViewOptions


def inplane_axes(plane: int) -> Tuple[int, int]:
    """The two canonical axes that remain inside plane ``plane`` (ascending)."""
    a0, a1 = [a for a in range(3) if a != plane]
    return a0, a1


# ----------------------------------------------------------- display transform
@dataclass
class DisplayTransform:
    """How a canonical (a0, a1) slice is turned into the picture shown on screen.

    ``row_axis`` / ``col_axis`` tell which canonical axis runs down / across the
    screen; ``row_flip`` / ``col_flip`` say whether that axis is reversed.
    The same description is used to draw the image, to place the crosshair and
    to convert mouse clicks back to voxel indices, so they can never disagree.
    """

    plane: int
    opts: ViewOptions
    row_axis: int
    col_axis: int
    row_flip: bool
    col_flip: bool

    @classmethod
    def build(cls, plane: int, opts: ViewOptions) -> "DisplayTransform":
        a0, a1 = inplane_axes(plane)
        # Base layout: the later axis runs vertically, the earlier one horizontally
        # (equivalent to showing slice.T). For the default orientation this gives
        # S-I vertical in the sagittal and coronal views and A-P vertical in the
        # transverse view.
        row_axis, col_axis = a1, a0
        row_flip = col_flip = False
        if opts.transpose:
            row_axis, col_axis = col_axis, row_axis
            row_flip, col_flip = col_flip, row_flip
        for _ in range(int(opts.rot90) % 4):
            # np.rot90 (counter-clockwise): new rows <- old columns reversed,
            #                               new columns <- old rows.
            row_axis, col_axis = col_axis, row_axis
            row_flip, col_flip = (not col_flip), row_flip
        if opts.flip_h:
            col_flip = not col_flip
        if opts.flip_v:
            row_flip = not row_flip
        return cls(plane, opts, row_axis, col_axis, row_flip, col_flip)

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Canonical (a0, a1[, channels]) array -> screen array (rows, cols[, channels]). View only."""
        out = np.swapaxes(img, 0, 1)
        if self.opts.transpose:
            out = np.swapaxes(out, 0, 1)
        k = int(self.opts.rot90) % 4
        if k:
            out = np.rot90(out, k, axes=(0, 1))
        if self.opts.flip_h:
            out = out[:, ::-1]
        if self.opts.flip_v:
            out = out[::-1]
        return out

    def to_display(self, idx: Tuple[int, int, int], shape: Tuple[int, int, int]) -> Tuple[int, int]:
        """Canonical voxel index -> (row, col) on screen."""
        n_rows, n_cols = shape[self.row_axis], shape[self.col_axis]
        r, c = idx[self.row_axis], idx[self.col_axis]
        if self.row_flip:
            r = n_rows - 1 - r
        if self.col_flip:
            c = n_cols - 1 - c
        return int(r), int(c)

    def from_display(self, row: int, col: int, shape: Tuple[int, int, int]) -> Tuple[int, int]:
        """(row, col) on screen -> (index along row_axis, index along col_axis)."""
        n_rows, n_cols = shape[self.row_axis], shape[self.col_axis]
        r = n_rows - 1 - row if self.row_flip else row
        c = n_cols - 1 - col if self.col_flip else col
        return int(r), int(c)


# ------------------------------------------------------------------ helpers
def estimate_intensity_range(mri: np.ndarray) -> Tuple[float, float, float, float]:
    """Robust display range from a sparse sample (no full-volume pass).

    Returns (range_min, range_max, default_lo_fraction, default_hi_fraction).
    """
    sample = np.asarray(mri[::4, ::4, ::2], dtype=np.float32).ravel()
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.0, 1.0, 0.0, 1.0
    rmin = float(sample.min())
    p1, p995, p9995 = (float(x) for x in np.percentile(sample, [1.0, 99.5, 99.95]))
    rmax = p9995 if p9995 > rmin else float(sample.max())
    if rmax <= rmin:
        rmax = rmin + 1.0
    span = rmax - rmin
    flo = min(max((p1 - rmin) / span, 0.0), 0.95)
    fhi = min(max((p995 - rmin) / span, flo + 0.02), 1.0)
    return rmin, rmax, flo, fhi


def downsample_mean(
    arr: np.ndarray,
    f: int,
    out_dtype=np.float32,
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Block-average a 3D array by integer factor ``f`` along every axis.

    Works slab-by-slab so only ~16 rows of float32 exist at once; the input may
    be a strided / transposed / flipped view. Trailing voxels that do not fill a
    whole block are dropped. ``transform`` (optional) is applied to every
    averaged slab (used to rescale to uint8).
    """
    f = max(int(f), 1)
    n0, n1, n2 = (s // f * f for s in arr.shape)
    out = np.empty((n0 // f, n1 // f, n2 // f), dtype=out_dtype)
    rows = f * max(1, 16 // f)
    for r0 in range(0, n0, rows):
        r1 = min(r0 + rows, n0)
        blk = np.asarray(arr[r0:r1, :n1, :n2], dtype=np.float32)
        if f > 1:
            blk = blk.reshape((r1 - r0) // f, f, n1 // f, f, n2 // f, f).mean(axis=(1, 3, 5))
        if transform is not None:
            blk = transform(blk)
        out[r0 // f: r1 // f] = blk
    return out


# --------------------------------------------------------------------- model
class VolumeModel:
    def __init__(
        self,
        mri: np.ndarray,
        seg: Optional[np.ndarray],
        mask: Optional[np.ndarray],
        config: OrientationConfig,
    ):
        self.mri_file = mri
        self.seg_file = seg
        self.mask_file = mask
        self.file_shape: Tuple[int, int, int] = tuple(int(s) for s in mri.shape)  # type: ignore[assignment]
        self.n_channels = 0 if seg is None else int(seg.shape[3])
        (self.range_min, self.range_max,
         self.default_win_lo, self.default_win_hi) = estimate_intensity_range(mri)
        self.set_config(config)

    # ------------------------------------------------------------ orientation
    def set_config(self, config: OrientationConfig) -> None:
        """(Re)build the canonical views. Cheap: no array data is copied."""
        config.validate()
        self.config = config.copy()
        perm = config.perm
        flips = config.flips
        self._perm = perm
        self._flips = flips

        def orient(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if a is None:
                return None
            extra = tuple(range(3, a.ndim))
            v = np.transpose(a, perm + extra)
            sl = tuple(slice(None, None, -1) if fl else slice(None) for fl in flips)
            return v[sl]

        self.mri = orient(self.mri_file)
        self.seg = orient(self.seg_file)
        self.mask = orient(self.mask_file)
        self.shape: Tuple[int, int, int] = tuple(int(s) for s in self.mri.shape)  # type: ignore[assignment]
        self.spacing: Tuple[float, float, float] = tuple(  # type: ignore[assignment]
            float(config.spacing_mm[perm[a]]) for a in range(3)
        )
        self.display: List[DisplayTransform] = [
            DisplayTransform.build(p, config.views[p]) for p in range(3)
        ]

    def canon_to_file(self, idx) -> Tuple[int, int, int]:
        out = [0, 0, 0]
        for a in range(3):
            v = int(idx[a])
            if self._flips[a]:
                v = self.shape[a] - 1 - v
            out[self._perm[a]] = v
        return tuple(out)  # type: ignore[return-value]

    def file_to_canon(self, fidx) -> Tuple[int, int, int]:
        out = [0, 0, 0]
        for a in range(3):
            v = int(fidx[self._perm[a]])
            if self._flips[a]:
                v = self.shape[a] - 1 - v
            out[a] = v
        return tuple(out)  # type: ignore[return-value]

    # ----------------------------------------------------------------- slices
    @staticmethod
    def _sl(plane: int, idx: int) -> tuple:
        s = [slice(None)] * 3
        s[plane] = int(idx)
        return tuple(s)

    def mri_slice(self, plane: int, idx: int) -> np.ndarray:
        """Canonical (a0, a1) view of one MRI slice (no copy)."""
        return self.mri[self._sl(plane, idx)]

    def mask_slice(self, plane: int, idx: int) -> np.ndarray:
        return self.mask[self._sl(plane, idx)]

    def channel_slice(self, plane: int, idx: int, channel: int) -> np.ndarray:
        return self.seg[self._sl(plane, idx) + (int(channel),)]

    def window_levels(self, frac_lo: float, frac_hi: float) -> Tuple[float, float]:
        span = self.range_max - self.range_min
        return self.range_min + frac_lo * span, self.range_min + frac_hi * span

    # -------------------------------------------------------------- 3D helper
    def volume_u8(self, f: int) -> np.ndarray:
        """Downsampled canonical MRI scaled to uint8 over [range_min, range_max]."""
        lo, hi = self.range_min, self.range_max
        scale = 255.0 / max(hi - lo, 1e-6)

        def tf(x: np.ndarray) -> np.ndarray:
            return np.clip((x - lo) * scale, 0, 255).astype(np.uint8)

        return downsample_mean(self.mri, f, np.uint8, tf)

    # --------------------------------------------------------------- queries
    def point_info(self, idx) -> Tuple[Tuple[int, int, int], float, List[int]]:
        """(file index, MRI intensity, list of segmentation channels that contain the voxel)."""
        idx = tuple(int(v) for v in idx)
        fidx = self.canon_to_file(idx)
        value = float(self.mri[idx])
        channels: List[int] = []
        if self.seg is not None:
            vec = np.asarray(self.seg[idx + (slice(None),)])
            channels = [int(c) for c in np.nonzero(vec > 0)[0]]
        return fidx, value, channels

    def channel_name(self, c: int) -> str:
        names = self.config.channel_names
        if names and c < len(names):
            return str(names[c])
        return f"Channel {c}"

    def describe_view(self, plane: int) -> str:
        """Human-readable statement of what each screen direction means (for verification)."""
        tr = self.display[plane]
        perm, flips = self._perm, self._flips

        def part(canon_axis: int, display_flip: bool, forward: str, backward: str) -> str:
            rev = display_flip ^ bool(flips[canon_axis])
            return (f"file axis {perm[canon_axis]} [{ROLE_NAMES[canon_axis]}] "
                    f"increases {backward if rev else forward}")

        horiz = part(tr.col_axis, tr.col_flip, "to the right", "to the left")
        vert = part(tr.row_axis, tr.row_flip, "downward", "upward")
        return f"{PLANE_NAMES[plane]}: horizontal = {horiz}; vertical = {vert}"

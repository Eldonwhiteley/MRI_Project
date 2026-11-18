"""Loading and validating the .im / .seg HDF5 files.

Both files must contain an HDF5 dataset called ``"data"``:

    .im   MRI           (X, Y, Z)       e.g. (384, 384, 160)
    .seg  segmentation  (X, Y, Z, C)    e.g. (384, 384, 160, 6)

Memory behaviour
----------------
* The MRI is read once, in its stored dtype (float16 is widened to float32).
* The segmentation is read slab-by-slab. 1-byte integer/bool data is kept as
  is (no conversion copy); anything else is binarised straight into ``uint8``
  while reading, so a float32 segmentation never exists in memory in full.
* The combined cartilage mask
      ``np.any(segmentation > 0, axis=-1).astype(np.uint8)``
  is computed ONCE here, slab-by-slab (no full-size temporary), and cached.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import h5py
import numpy as np

from .config import OrientationConfig

EXPECTED_MRI_SHAPE = (384, 384, 160)
EXPECTED_SEG_CHANNELS = 6
_SLAB = 16  # slices read per step along file axis 2

ProgressFn = Callable[[str, Optional[float]], None]


class DataLoadError(Exception):
    """Raised with a user-readable message when the files cannot be used."""


@dataclass
class FileMetadata:
    spacing: Optional[Tuple[float, float, float]] = None  # per FILE axis, mm
    axis_codes: Optional[Tuple[str, str, str]] = None      # direction each file axis increases toward
    notes: List[str] = field(default_factory=list)


@dataclass
class RawCase:
    mri: np.ndarray                      # (X, Y, Z) file order
    seg: Optional[np.ndarray]            # (X, Y, Z, C) file order, or None
    mask: Optional[np.ndarray]           # (X, Y, Z) uint8 combined mask, or None
    mri_path: Path
    seg_path: Optional[Path]
    metadata: FileMetadata
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- IO
def _noop(msg: str, frac: Optional[float] = None) -> None:
    pass


def _open_dataset(path: Path, kind: str):
    if not path.is_file():
        raise DataLoadError(f"{kind} file not found:\n{path}")
    try:
        f = h5py.File(path, "r")
    except OSError as exc:
        raise DataLoadError(
            f"{kind} file '{path.name}' could not be opened as an HDF5 file.\n"
            f"({exc})\nCheck that you selected the right file and that it is not corrupted."
        ) from exc
    if "data" not in f:
        keys = list(f.keys())
        f.close()
        raise DataLoadError(
            f"{kind} file '{path.name}' has no dataset named \"data\".\n"
            f"Datasets found at the top level: {keys if keys else 'none'}."
        )
    ds = f["data"]
    if not isinstance(ds, h5py.Dataset):
        f.close()
        raise DataLoadError(f"In '{path.name}', \"data\" is not a dataset (it is a group).")
    return f, ds


def load_case(
    mri_path,
    seg_path=None,
    threshold: float = 0.0,
    progress: Optional[ProgressFn] = None,
) -> RawCase:
    """Read, validate and return an MRI (+ optional segmentation) pair."""
    progress = progress or _noop
    mri_path = Path(mri_path)
    seg_path = Path(seg_path) if seg_path else None
    warnings: List[str] = []

    # ---------------------------------------------------------------- MRI
    progress("Reading MRI ...", 0.02)
    f, ds = _open_dataset(mri_path, "MRI")
    with f:
        if ds.ndim != 3:
            raise DataLoadError(
                f"MRI '{mri_path.name}': expected a 3D array (X, Y, Z) such as {EXPECTED_MRI_SHAPE}, "
                f"but \"data\" has shape {tuple(ds.shape)}."
            )
        if ds.dtype.kind not in "iuf":
            raise DataLoadError(f"MRI '{mri_path.name}': unsupported dtype {ds.dtype} (need integer or float).")
        mri = ds[...]
        meta = read_metadata(f, ds)
    if mri.dtype == np.float16:
        mri = mri.astype(np.float32)
    if tuple(mri.shape) != EXPECTED_MRI_SHAPE:
        warnings.append(f"MRI shape is {tuple(mri.shape)}, not the expected {EXPECTED_MRI_SHAPE}; continuing anyway.")
    if not np.isfinite(mri[::4, ::4, ::2]).all():
        warnings.append("MRI contains NaN/Inf values; they are ignored for intensity scaling.")

    # ---------------------------------------------------------------- SEG
    seg = None
    mask = None
    if seg_path is not None:
        progress("Reading segmentation ...", 0.15)
        f, ds = _open_dataset(seg_path, "Segmentation")
        with f:
            if ds.ndim not in (3, 4):
                raise DataLoadError(
                    f"Segmentation '{seg_path.name}': expected (X, Y, Z, C) such as "
                    f"{EXPECTED_MRI_SHAPE + (EXPECTED_SEG_CHANNELS,)}, but \"data\" has shape {tuple(ds.shape)}."
                )
            spatial = tuple(ds.shape[:3])
            if spatial != tuple(mri.shape):
                raise DataLoadError(
                    "MRI and segmentation do not have the same spatial size.\n"
                    f"  MRI '{mri_path.name}': {tuple(mri.shape)}\n"
                    f"  Segmentation '{seg_path.name}': {spatial}  (full shape {tuple(ds.shape)})\n"
                    "Make sure the .seg file belongs to the selected .im file."
                )
            if ds.dtype.kind not in "biuf":
                raise DataLoadError(f"Segmentation '{seg_path.name}': unsupported dtype {ds.dtype}.")

            keep_native = ds.dtype.kind in "biu" and ds.dtype.itemsize == 1 and threshold == 0.0
            out_dtype = ds.dtype if keep_native else np.uint8
            seg = np.empty(ds.shape, dtype=out_dtype)
            n2 = ds.shape[2]
            for s in range(0, n2, _SLAB):
                e = min(s + _SLAB, n2)
                blk = ds[:, :, s:e]
                seg[:, :, s:e] = blk if keep_native else (blk > threshold)
                progress("Reading segmentation ...", 0.15 + 0.55 * e / n2)
            if not keep_native:
                warnings.append(
                    f"Segmentation dtype {ds.dtype} was binarised with (value > {threshold:g}) while loading."
                )
            if ds.ndim == 3:
                seg = seg[..., None]
                warnings.append("Segmentation has no channel axis; treated as a single channel.")

        n_ch = seg.shape[3]
        if n_ch != EXPECTED_SEG_CHANNELS:
            warnings.append(f"Segmentation has {n_ch} channels (expected {EXPECTED_SEG_CHANNELS}); continuing anyway.")

        # Combined mask: identical result to  np.any(seg > 0, axis=-1).astype(np.uint8)
        # but computed slab-wise so no full-size boolean temporary is created.
        progress("Combining channels ...", 0.75)
        mask = np.empty(seg.shape[:3], dtype=np.uint8)
        for s in range(0, n2, _SLAB):
            e = min(s + _SLAB, n2)
            mask[:, :, s:e] = np.any(seg[:, :, s:e] > 0, axis=-1)
        if not mask.any():
            warnings.append("The segmentation is empty (no voxel > 0 in any channel).")

    progress("Loaded.", 1.0)
    return RawCase(mri, seg, mask, mri_path, seg_path, meta, warnings)


def guess_seg_path(mri_path) -> Optional[Path]:
    """train_001_V00.im -> train_001_V00.seg (if that file exists)."""
    p = Path(mri_path)
    cand = p.with_suffix(".seg")
    return cand if cand.is_file() else None


# ---------------------------------------------------------------- HDF5 metadata
_SPACING_KEYS = ("spacing", "voxel_spacing", "voxelspacing", "voxel_size", "voxelsize",
                 "pixel_spacing", "pixelspacing", "resolution", "element_spacing")
_ORIENT_KEYS = ("orientation", "axcodes", "axis_codes", "anatomical_orientation")
_AFFINE_KEYS = ("affine", "affine_matrix", "vox2ras")
_ROLE_OF = {"L": 0, "R": 0, "A": 1, "P": 1, "S": 2, "I": 2}


def _to_text(v) -> Optional[str]:
    if isinstance(v, bytes):
        return v.decode("utf-8", "ignore")
    if isinstance(v, str):
        return v
    if isinstance(v, np.ndarray) and v.dtype.kind in "SUO":
        return " ".join(_to_text(x) or "" for x in v.ravel().tolist())
    return None


def _candidates(f, ds):
    """Yield (lower-case name, value) from dataset attrs, root attrs, and small root datasets."""
    for attrs in (ds.attrs, f.attrs):
        for k in attrs.keys():
            try:
                yield k.lower(), attrs[k]
            except Exception:
                continue
    for k in f.keys():
        if k == "data":
            continue
        try:
            obj = f[k]
            if isinstance(obj, h5py.Dataset) and obj.size <= 64:
                yield k.lower(), obj[()]
        except Exception:
            continue


def _parse_axcodes(value) -> Optional[Tuple[str, str, str]]:
    text = _to_text(value)
    if text is None:
        return None
    letters = re.sub(r"[^A-Za-z]", "", text).upper()
    if len(letters) == 3 and all(c in _ROLE_OF for c in letters):
        return (letters[0], letters[1], letters[2])
    return None


def read_metadata(f, ds) -> FileMetadata:
    """Look for voxel spacing / axis codes / an affine in the HDF5 file.

    Most research files (including the usual .im/.seg pairs) contain none of
    this; in that case an empty FileMetadata is returned and the manual
    configuration is used.
    """
    meta = FileMetadata()
    for name, value in _candidates(f, ds):
        try:
            if meta.spacing is None and name in _SPACING_KEYS:
                arr = np.asarray(value, dtype=float).ravel()
                if arr.size == 3 and np.all(np.isfinite(arr)) and np.all(arr > 0):
                    meta.spacing = tuple(float(x) for x in arr)
                    meta.notes.append(f"voxel spacing {meta.spacing} mm from HDF5 entry '{name}'")
            elif meta.axis_codes is None and name in _ORIENT_KEYS:
                codes = _parse_axcodes(value)
                if codes and sorted(_ROLE_OF[c] for c in codes) == [0, 1, 2]:
                    meta.axis_codes = codes
                    meta.notes.append(f"axis codes {''.join(codes)} from HDF5 entry '{name}'")
            elif name in _AFFINE_KEYS:
                a = np.asarray(value, dtype=float)
                if a.shape == (4, 4) and np.all(np.isfinite(a)):
                    cols = a[:3, :3]
                    norms = np.linalg.norm(cols, axis=0)
                    if meta.spacing is None and np.all(norms > 0):
                        meta.spacing = tuple(float(x) for x in norms)
                        meta.notes.append(f"voxel spacing {meta.spacing} mm from affine '{name}'")
                    if meta.axis_codes is None:
                        codes = []
                        for c in range(3):
                            k = int(np.argmax(np.abs(cols[:, c])))
                            pos = cols[k, c] > 0
                            codes.append(("LR", "AP", "SI")[k][1 if pos else 0])
                        if sorted(_ROLE_OF[c] for c in codes) == [0, 1, 2]:
                            meta.axis_codes = tuple(codes)  # type: ignore[assignment]
                            meta.notes.append(f"axis codes {''.join(codes)} from affine '{name}' (assumed RAS+)")
        except Exception:
            continue
    return meta


def apply_metadata(cfg: OrientationConfig, meta: FileMetadata) -> Tuple[OrientationConfig, List[str]]:
    """Return a copy of ``cfg`` updated with whatever the file metadata provides."""
    out = cfg.copy()
    notes: List[str] = []
    if meta.spacing is not None:
        out.spacing_mm = meta.spacing
        notes.append("voxel spacing taken from file metadata")
    if meta.axis_codes is not None:
        file_axis_of_role = {}
        flip_of_role = {}
        for file_axis, code in enumerate(meta.axis_codes):
            role = _ROLE_OF[code]
            file_axis_of_role[role] = file_axis
            # canonical axes increase toward R, A, S; codes L, P, I need a flip
            flip_of_role[role] = code in "LPI"
        out.axis_sagittal = file_axis_of_role[0]
        out.axis_coronal = file_axis_of_role[1]
        out.axis_transverse = file_axis_of_role[2]
        out.flip_sagittal = flip_of_role[0]
        out.flip_coronal = flip_of_role[1]
        out.flip_transverse = flip_of_role[2]
        notes.append(f"axis orientation taken from file metadata ({''.join(meta.axis_codes)})")
    return out, notes

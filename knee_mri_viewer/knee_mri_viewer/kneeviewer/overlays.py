"""Segmentation overlay + image composition helpers (NumPy only)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .volume_model import VolumeModel

RGB = Tuple[int, int, int]
SegMode = Union[str, int]  # "combined" | "all" | channel number


def window_to_u8(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear window/level -> uint8 (used for the textures on the 3D slice planes)."""
    out = np.asarray(img, dtype=np.float32) - np.float32(lo)
    out *= np.float32(255.0 / max(hi - lo, 1e-6))
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def overlay_rgba(
    model: VolumeModel,
    plane: int,
    idx: int,
    seg_mode: SegMode,
    combined_color: RGB,
    channel_colors: Sequence[RGB],
) -> Optional[np.ndarray]:
    """RGBA (uint8, alpha is 0 or 255) overlay of one slice in canonical (a0, a1) order.

    * ``"combined"`` : the cached logical-OR mask, in ``combined_color``
    * ``"all"``      : every channel in its own colour (later channels are drawn on top)
    * ``int c``      : only channel ``c`` in its own colour
    Returns ``None`` if the case has no segmentation.
    """
    if model.seg is None:
        return None
    if seg_mode == "combined":
        m = model.mask_slice(plane, idx) > 0
        rgba = np.zeros(m.shape + (4,), dtype=np.uint8)
        rgba[m] = (*combined_color, 255)
        return rgba
    if seg_mode == "all":
        channels = list(range(model.n_channels))
    else:
        channels = [int(seg_mode)]
    rgba = np.zeros(model.mri_slice(plane, idx).shape + (4,), dtype=np.uint8)
    for c in channels:
        if c < 0 or c >= model.n_channels:
            continue
        m = model.channel_slice(plane, idx, c) > 0
        col = channel_colors[c] if c < len(channel_colors) else (255, 255, 255)
        rgba[m] = (*col, 255)
    return rgba


def compose_rgb(gray_u8: Optional[np.ndarray], rgba: Optional[np.ndarray], opacity: float,
                shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Blend an RGBA overlay (binary alpha) onto a grayscale uint8 image -> (h, w, 3) uint8.

    If ``gray_u8`` is None the background is black (segmentation-only mode).
    """
    if gray_u8 is None:
        if shape is None and rgba is not None:
            shape = rgba.shape[:2]
        rgb = np.zeros(tuple(shape) + (3,), dtype=np.uint8)  # type: ignore[arg-type]
    else:
        rgb = np.repeat(gray_u8[..., None], 3, axis=2)
    if rgba is not None and opacity > 0:
        m = rgba[..., 3] > 0
        if m.any():
            col = rgba[..., :3][m].astype(np.float32)
            base = rgb[m].astype(np.float32)
            rgb[m] = (base * (1.0 - opacity) + col * opacity).astype(np.uint8)
    return rgb

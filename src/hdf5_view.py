"""hdf5_view.py
Class-based viewer and utility interface for 3D MRI HDF5 volumes (.im) and
multi-channel segmentation masks (.seg).
"""

from __future__ import annotations

import os
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np


class HDF5View:
    """A high-level interface for loading, inspecting, and visualizing 3D MRI scans

    and multi-channel segmentation masks stored in HDF5 format (.im and .seg).
    """

    # Default anatomical labels for OAI (Osteoarthritis Initiative) Knee 3D DESS MRI
    DEFAULT_CHANNEL_LABELS: List[str] = [
        "Femoral Cartilage",
        "Lateral Tibial Cartilage",
        "Medial Tibial Cartilage",
        "Patellar Cartilage",
        "Lateral Meniscus",
        "Medial Meniscus",
    ]

    # Distinct RGB colors for mask overlays
    DEFAULT_CHANNEL_COLORS: List[List[float]] = [
        [1.0, 0.2, 0.2],  # Ch 0: Red
        [0.2, 0.8, 0.2],  # Ch 1: Green
        [0.2, 0.5, 1.0],  # Ch 2: Blue
        [1.0, 0.8, 0.0],  # Ch 3: Yellow/Gold
        [0.8, 0.2, 0.8],  # Ch 4: Magenta
        [0.2, 0.9, 0.9],  # Ch 5: Cyan
    ]

    def __init__(
        self,
        im_path: str,
        seg_path: Optional[str] = None,
        dataset_key: str = "data",
        channel_labels: Optional[List[str]] = None,
        channel_colors: Optional[List[List[float]]] = None,
        preload: bool = False,
    ) -> None:
        """Initialize the HDF5 viewer.

        Args:
            im_path: Path to the .im (image) HDF5 file.
            seg_path: Optional path to the matching .seg (segmentation) HDF5 file.
            dataset_key: The HDF5 dataset name containing the array (default: "data").
            channel_labels: Optional list of channel label names for the segmentation mask.
            channel_colors: Optional list of RGB colors (0.0 to 1.0) for mask overlays.
            preload: If True, immediately load the entire 3D volume into RAM.
        """
        if not os.path.isfile(im_path):
            raise FileNotFoundError(f"Image file not found: {im_path}")
        if seg_path is not None and not os.path.isfile(seg_path):
            raise FileNotFoundError(f"Segmentation file not found: {seg_path}")

        self.im_path: str = os.path.abspath(im_path)
        self.seg_path: Optional[str] = os.path.abspath(seg_path) if seg_path else None
        self.dataset_key: str = dataset_key

        self.channel_labels: List[str] = channel_labels or list(self.DEFAULT_CHANNEL_LABELS)
        self.channel_colors: List[List[float]] = channel_colors or list(self.DEFAULT_CHANNEL_COLORS)

        # In-memory cached volumes
        self._volume_im: Optional[np.ndarray] = None
        self._volume_seg: Optional[np.ndarray] = None

        # Inspect basic metadata without loading full data
        with h5py.File(self.im_path, "r") as f_im:
            if self.dataset_key not in f_im:
                raise KeyError(f"Dataset '{self.dataset_key}' not found in {im_path}. Available: {list(f_im.keys())}")
            ds_im = f_im[self.dataset_key]
            self.im_shape: Tuple[int, ...] = ds_im.shape
            self.im_dtype: np.dtype = ds_im.dtype

        if self.seg_path is not None:
            with h5py.File(self.seg_path, "r") as f_seg:
                if self.dataset_key not in f_seg:
                    raise KeyError(f"Dataset '{self.dataset_key}' not found in {seg_path}. Available: {list(f_seg.keys())}")
                ds_seg = f_seg[self.dataset_key]
                self.seg_shape: Optional[Tuple[int, ...]] = ds_seg.shape
                self.seg_dtype: Optional[np.dtype] = ds_seg.dtype
                self.num_channels: int = self.seg_shape[-1] if len(self.seg_shape) > 3 else 1
        else:
            self.seg_shape = None
            self.seg_dtype = None
            self.num_channels = 0

        # Number of slices along depth dimension (typically axis 2)
        self.num_slices: int = self.im_shape[2] if len(self.im_shape) >= 3 else self.im_shape[0]

        if preload:
            self.load_volume()

    # -------------------------------------------------------------------------
    # Data Loading & Memory Management
    # -------------------------------------------------------------------------

    def load_volume(self) -> HDF5View:
        """Load the entire 3D image and segmentation volume into RAM."""
        if self._volume_im is None:
            with h5py.File(self.im_path, "r") as f_im:
                self._volume_im = f_im[self.dataset_key][:]
        if self.seg_path is not None and self._volume_seg is None:
            with h5py.File(self.seg_path, "r") as f_seg:
                self._volume_seg = f_seg[self.dataset_key][:]
        return self

    def clear_cache(self) -> None:
        """Free loaded in-memory volume cache from RAM."""
        self._volume_im = None
        self._volume_seg = None

    @property
    def volume_im(self) -> np.ndarray:
        """Return the loaded 3D image array, loading it into RAM if not already cached."""
        if self._volume_im is None:
            self.load_volume()
        return self._volume_im  # type: ignore

    @property
    def volume_seg(self) -> Optional[np.ndarray]:
        """Return the loaded segmentation array, loading it into RAM if not already cached."""
        if self.seg_path is not None and self._volume_seg is None:
            self.load_volume()
        return self._volume_seg

    def get_slice(self, slice_idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Retrieve a 2D image slice and corresponding segmentation slice.

        If volumes are preloaded, slices directly from RAM.
        Otherwise, reads only the single slice directly from disk.

        Args:
            slice_idx: The slice index along the 3rd dimension (0 <= slice_idx < num_slices).

        Returns:
            Tuple of (im_slice, seg_slice). seg_slice is None if no segmentation was provided.
        """
        if not (0 <= slice_idx < self.num_slices):
            raise IndexError(f"slice_idx {slice_idx} is out of bounds for volume with {self.num_slices} slices.")

        # From memory if cached
        if self._volume_im is not None:
            im_slice = self._volume_im[:, :, slice_idx]
            seg_slice = self._volume_seg[:, :, slice_idx, :] if self._volume_seg is not None else None
            return im_slice, seg_slice

        # Directly from HDF5 file
        with h5py.File(self.im_path, "r") as f_im:
            im_slice = f_im[self.dataset_key][:, :, slice_idx]

        seg_slice = None
        if self.seg_path is not None:
            with h5py.File(self.seg_path, "r") as f_seg:
                seg_slice = f_seg[self.dataset_key][:, :, slice_idx, :]

        return im_slice, seg_slice

    # -------------------------------------------------------------------------
    # Visualization Helpers
    # -------------------------------------------------------------------------

    def create_overlay(
        self,
        im_slice: np.ndarray,
        seg_slice: Optional[np.ndarray],
        alpha: float = 0.5,
        selected_channels: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Create an RGB image by blending semi-transparent colored masks onto the grayscale MRI slice.

        Args:
            im_slice: 2D numpy array of image intensities.
            seg_slice: 3D numpy array of masks (H, W, num_channels) or None.
            alpha: Opacity factor for the segmentation masks (0.0 = invisible, 1.0 = solid).
            selected_channels: Optional list of channel indices to overlay (defaults to all).

        Returns:
            RGB float32 numpy array with shape (H, W, 3) in [0.0, 1.0].
        """
        im_min, im_max = float(im_slice.min()), float(im_slice.max())
        im_norm = (im_slice - im_min) / (im_max - im_min + 1e-8)
        rgb = np.stack([im_norm] * 3, axis=-1).astype(np.float32)

        if seg_slice is None:
            return rgb

        channels_to_plot = selected_channels if selected_channels is not None else list(range(seg_slice.shape[-1]))

        for c in channels_to_plot:
            if c >= seg_slice.shape[-1]:
                continue
            mask = seg_slice[:, :, c] > 0
            if np.any(mask):
                color = self.channel_colors[c % len(self.channel_colors)]
                for ch in range(3):
                    rgb[mask, ch] = (1.0 - alpha) * rgb[mask, ch] + alpha * color[ch]

        return np.clip(rgb, 0.0, 1.0)

    # -------------------------------------------------------------------------
    # Plotting & Interactivity
    # -------------------------------------------------------------------------

    def plot_slice(
        self,
        slice_idx: int = 75,
        alpha: float = 0.5,
        show_channels: bool = True,
        figsize: Tuple[int, int] = (16, 8),
    ) -> plt.Figure:
        """Plot a static multi-panel view for a given slice:

        1. Grayscale MRI slice.
        2. Composite color overlay.
        3. Individual channel subplots (if segmentation is present and show_channels=True).

        Args:
            slice_idx: Slice index (0 to num_slices - 1).
            alpha: Mask opacity for the overlay.
            show_channels: Whether to display individual channel panels.
            figsize: Figure size tuple (width, height).

        Returns:
            The matplotlib Figure object.
        """
        im_slice, seg_slice = self.get_slice(slice_idx)

        if seg_slice is None or not show_channels:
            fig, axes = plt.subplots(1, 2 if seg_slice is not None else 1, figsize=(10, 5))
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])

            axes[0].imshow(im_slice, cmap="gray")
            axes[0].set_title(f"MRI Slice (z={slice_idx})")
            axes[0].axis("off")

            if seg_slice is not None:
                overlay = self.create_overlay(im_slice, seg_slice, alpha=alpha)
                axes[1].imshow(overlay)
                axes[1].set_title(f"Segmentation Overlay (z={slice_idx})")
                axes[1].axis("off")

            plt.tight_layout()
            return fig

        # Multi-panel grid: 2 rows x 4 columns
        fig = plt.figure(figsize=figsize)

        # 1. Grayscale MRI
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(im_slice, cmap="gray")
        ax1.set_title(f"MRI Slice (z={slice_idx})", fontsize=12)
        ax1.axis("off")

        # 2. Composite Overlay
        ax2 = plt.subplot(2, 4, 2)
        overlay = self.create_overlay(im_slice, seg_slice, alpha=alpha)
        ax2.imshow(overlay)
        ax2.set_title("Composite Overlay", fontsize=12)
        ax2.axis("off")

        # 3-8. Individual channels
        num_c = seg_slice.shape[-1]
        for c in range(min(num_c, 6)):
            ax = plt.subplot(2, 4, c + 3)
            ax.imshow(im_slice, cmap="gray")
            mask = seg_slice[:, :, c]
            if np.any(mask):
                color_mask = np.zeros((*mask.shape, 4), dtype=np.float32)
                color = self.channel_colors[c % len(self.channel_colors)]
                color_mask[mask > 0] = [*color, 0.7]
                ax.imshow(color_mask)
                status = f"({np.sum(mask)} px)"
            else:
                status = "(empty)"

            label = self.channel_labels[c] if c < len(self.channel_labels) else f"Channel {c}"
            ax.set_title(f"{label}\n{status}", fontsize=10)
            ax.axis("off")

        plt.tight_layout()
        return fig

    def interactive_viewer(
        self,
        default_slice: Optional[int] = None,
        default_alpha: float = 0.5,
        figsize: Tuple[int, int] = (7, 7),
    ) -> None:
        """Launch an interactive slice viewer widget using ipywidgets.

        Works seamlessly in Jupyter Notebook, JupyterLab, VS Code, and Google Colab.

        Args:
            default_slice: Initial slice index to display (defaults to center slice).
            default_alpha: Initial mask opacity.
            figsize: Figure size for the interactive view.
        """
        try:
            import ipywidgets as widgets
            from ipywidgets import interact
        except ImportError as e:
            raise ImportError(
                "ipywidgets is required for the interactive viewer. "
                "Install it using: pip install ipywidgets"
            ) from e

        # Pre-load into memory for instantaneous slider performance
        self.load_volume()
        assert self._volume_im is not None

        if default_slice is None:
            default_slice = self.num_slices // 2

        @interact(
            slice_idx=widgets.IntSlider(
                min=0,
                max=self.num_slices - 1,
                step=1,
                value=default_slice,
                description="Slice:",
                continuous_update=False,
            ),
            show_overlay=widgets.Checkbox(value=True, description="Show Overlay"),
            alpha=widgets.FloatSlider(
                min=0.1, max=1.0, step=0.05, value=default_alpha, description="Opacity:"
            ),
        )
        def _view_slice(slice_idx: int, show_overlay: bool, alpha: float) -> None:
            im_slice = self._volume_im[:, :, slice_idx]
            seg_slice = self._volume_seg[:, :, slice_idx, :] if self._volume_seg is not None else None

            fig, ax = plt.subplots(figsize=figsize)
            if show_overlay and seg_slice is not None:
                overlay = self.create_overlay(im_slice, seg_slice, alpha=alpha)
                ax.imshow(overlay)
            else:
                ax.imshow(im_slice, cmap="gray")

            title = f"Slice {slice_idx} / {self.num_slices - 1}"
            if seg_slice is not None:
                active_info = []
                for c in range(seg_slice.shape[-1]):
                    cnt = np.sum(seg_slice[:, :, c])
                    if cnt > 0:
                        name = self.channel_labels[c] if c < len(self.channel_labels) else f"Ch {c}"
                        active_info.append(f"{name}: {cnt} px")
                if active_info:
                    title += "\n" + " | ".join(active_info)
                else:
                    title += "\n(No segmented structures in this slice)"

            ax.set_title(title, fontsize=11)
            ax.axis("off")
            plt.show()

    def plot_orthogonal(
        self,
        sag_idx: Optional[int] = None,
        cor_idx: Optional[int] = None,
        ax_idx: Optional[int] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> plt.Figure:
        """Plot the 3 orthogonal planar views (Sagittal, Coronal, Axial).

        Args:
            sag_idx: Slice index along depth axis 2 (Sagittal). Defaults to center.
            cor_idx: Slice index along axis 1 (Coronal). Defaults to center.
            ax_idx: Slice index along axis 0 (Axial). Defaults to center.
            figsize: Figure size tuple.

        Returns:
            The matplotlib Figure object.
        """
        vol = self.volume_im
        h, w, d = vol.shape

        sag_idx = sag_idx if sag_idx is not None else d // 2
        cor_idx = cor_idx if cor_idx is not None else w // 2
        ax_idx = ax_idx if ax_idx is not None else h // 2

        sagittal = vol[:, :, sag_idx]
        coronal = vol[:, cor_idx, :].T
        axial = vol[ax_idx, :, :].T

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].imshow(sagittal, cmap="gray")
        axes[0].set_title(f"Sagittal View (Slice z={sag_idx})")
        axes[0].axis("off")

        axes[1].imshow(coronal, cmap="gray", aspect=sagittal.shape[0] / coronal.shape[0])
        axes[1].set_title(f"Coronal Reslice (y={cor_idx})")
        axes[1].axis("off")

        axes[2].imshow(axial, cmap="gray", aspect=sagittal.shape[1] / axial.shape[0])
        axes[2].set_title(f"Axial Reslice (x={ax_idx})")
        axes[2].axis("off")

        plt.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    # Utilities & Static Methods
    # -------------------------------------------------------------------------

    def info(self) -> Dict[str, Union[str, Tuple[int, ...], int]]:
        """Return a dictionary summarizing the dataset specifications."""
        summary: Dict[str, Union[str, Tuple[int, ...], int]] = {
            "image_file": os.path.basename(self.im_path),
            "image_shape": self.im_shape,
            "image_dtype": str(self.im_dtype),
            "num_slices": self.num_slices,
            "has_segmentation": self.seg_path is not None,
        }
        if self.seg_path is not None and self.seg_shape is not None:
            summary["seg_file"] = os.path.basename(self.seg_path)
            summary["seg_shape"] = self.seg_shape
            summary["seg_dtype"] = str(self.seg_dtype)
            summary["num_channels"] = self.num_channels
        return summary

    def __repr__(self) -> str:
        seg_str = f", seg_shape={self.seg_shape}" if self.seg_path else ""
        return f"<HDF5View: im_shape={self.im_shape}{seg_str}, slices={self.num_slices}>"

    @classmethod
    def find_pairs(cls, data_dir: str) -> List[Tuple[str, str]]:
        """Search a directory recursively or flat for corresponding (.im, .seg) file pairs.

        Filters out AppleDouble metadata files (._*).

        Args:
            data_dir: Path to directory containing .im and .seg files.

        Returns:
            List of (im_path, seg_path) tuples where both files exist.
        """
        all_im = [
            f for f in glob(os.path.join(data_dir, "**", "*.im"), recursive=True)
            if not os.path.basename(f).startswith("._")
        ]
        pairs = []
        for im in sorted(all_im):
            base, _ = os.path.splitext(im)
            seg = base + ".seg"
            if os.path.isfile(seg):
                pairs.append((im, seg))
        return pairs


# Alias for convenience
HDF5Viewer = HDF5View


if __name__ == "__main__":
    import sys

    # Quick test if run directly
    sample_dir = "data/train-shifted/train-shifted"
    if not os.path.isdir(sample_dir):
        sample_dir = "../data/train-shifted/train-shifted"

    pairs = HDF5View.find_pairs(sample_dir)
    print(f"Found {len(pairs)} matching (.im, .seg) pairs in {sample_dir}")
    if pairs:
        viewer = HDF5View(pairs[0][0], pairs[0][1])
        print("Initialized viewer:", viewer)
        print("Summary info:", viewer.info())

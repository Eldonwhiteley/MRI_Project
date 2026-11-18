"""Small VTK helpers (pure VTK + NumPy, no PyVista API needed)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import vtk
from vtkmodules.util import numpy_support

from .volume_model import downsample_mean


def make_vtk_image(arr: np.ndarray, spacing: Sequence[float], origin: Sequence[float]) -> vtk.vtkImageData:
    """3D NumPy array (nx, ny, nz) -> vtkImageData. VTK stores x fastest, hence order='F'.

    The data are deep-copied into VTK memory, so the NumPy array can be freed.
    """
    arr = np.asarray(arr)
    nx, ny, nz = arr.shape
    flat = np.ascontiguousarray(arr.ravel(order="F"))
    vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True)
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(*(float(s) for s in spacing))
    img.SetOrigin(*(float(o) for o in origin))
    img.GetPointData().SetScalars(vtk_arr)
    return img


def build_surface(
    mask: np.ndarray,
    spacing: Sequence[float],
    ds: int = 1,
    smooth_iterations: int = 15,
) -> Optional[vtk.vtkPolyData]:
    """Binary mask (any 3D view, canonical order) -> smoothed triangle surface in millimetres.

    * ``ds`` > 1 block-averages the mask first and contours at a slightly
      lower level (0.35) so thin cartilage does not vanish.
    * The mask is zero-padded by one voxel so surfaces touching the volume
      border are closed.
    Returns ``None`` if the mask is empty.
    """
    binary = np.asarray(mask) > 0
    if not binary.any():
        return None
    spacing = tuple(float(s) for s in spacing)
    if ds <= 1:
        vol = binary.astype(np.uint8)
        sp = spacing
        shift = (0.0, 0.0, 0.0)
        iso = 0.5
    else:
        vol = downsample_mean(binary, ds, np.float32)
        sp = tuple(s * ds for s in spacing)
        shift = tuple(0.5 * (ds - 1) * s for s in spacing)  # centre of the first block
        iso = 0.35
    if vol.size == 0 or vol.max() <= 0:
        return None
    vol = np.pad(vol, 1)
    origin = tuple(sh - s for sh, s in zip(shift, sp))  # one padded cell before the first voxel

    img = make_vtk_image(vol, sp, origin)
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(img)
    fe.SetValue(0, iso)
    fe.ComputeNormalsOff()
    fe.Update()
    poly = fe.GetOutput()
    if poly.GetNumberOfPoints() == 0:
        return None

    last = fe
    if smooth_iterations > 0 and poly.GetNumberOfPoints() < 3_000_000:
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(fe.GetOutputPort())
        smoother.SetNumberOfIterations(int(smooth_iterations))
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetPassBand(0.1)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        last = smoother

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(last.GetOutputPort())
    normals.SetFeatureAngle(60.0)
    normals.SplittingOff()
    normals.ComputePointNormalsOn()
    normals.ConsistencyOn()
    normals.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(normals.GetOutput())
    return out

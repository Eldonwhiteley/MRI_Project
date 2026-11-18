# Knee MRI + Cartilage Segmentation Viewer

A desktop (PySide6) four-panel viewer for 3D knee MRI (`.im`) and its segmentation (`.seg`), laid out like a simplified 3D Slicer:

| | |
|---|---|
| **3D knee volume** (volume rendering + cartilage surfaces + 3 slice planes) | **Sagittal** plane |
| **Coronal** plane | **Transverse (axial)** plane |

All four windows share one selected voxel, so sliders, clicks and the 3D planes stay in sync.

---------------------------------------------------------------------------

## 1. Local desktop vs. Google Colab

I built this as a **local desktop application** because reliability matters more than Colab compatibility.

| | Local desktop (this app) | Google Colab |
|---|---|---|
| GUI window | Real native window (Qt) | None - Colab has no display; only notebook widgets |
| 3D rendering | GPU ray-casting via VTK/OpenGL | Remote, software-rendered or streamed (trame); laggy, fragile, no GPU OpenGL context |
| Interaction | Immediate: rotate/zoom/scrub at screen refresh rate | Every interaction round-trips to a server |
| Four synchronised panels | Straightforward | Possible only with ipywidgets/trame glue that often breaks between versions |
| Large data | Reads from your disk | Upload / Drive mount each session |
| Setup | One `pip install` on Windows | Nothing to install, but GUI libraries do not work |
| Best for | Inspecting volumes, checking alignment, screenshots | Training the U-Net on a GPU |

**Suggested workflow:** train in Colab, save the predicted masks as `.seg`-format HDF5 (dataset `"data"`, shape `(384, 384, 160, 6)`), download them, and inspect them locally with this viewer. Loading a prediction instead of the ground truth needs no code change. If your network outputs probabilities, use *Binarise: value >* 0.5 in the Open dialog.

---------------------------------------------------------------------------

## 2. Installation (Windows)

Python **3.10 - 3.12** is recommended (VTK wheels lag behind the newest Python releases).

```bat
cd knee_mri_viewer
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to activate the environment:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

`requirements.txt` contains: `numpy h5py PySide6 pyqtgraph pyvista pyvistaqt vtk`.

macOS / Linux work the same way (`python3 -m venv .venv && source .venv/bin/activate`).

### First run

```bat
python selftest.py                 :: numerical core checks, no window
python make_demo_data.py --small   :: writes demo_001.im / demo_001.seg (synthetic "knee")
python main.py --mri demo_001.im --seg demo_001.seg
```

Use the demo to confirm the window, 3D rendering and synchronisation work on your machine before loading real data.

---------------------------------------------------------------------------

## 3. Loading your files

* **GUI:** `File > Open MRI + segmentation...` (Ctrl+O). Pick `train_001_V00.im`; the matching `train_001_V00.seg` is filled in automatically if it sits next to it. The `.seg` is optional (MRI-only viewing).
* **Command line:**
  `python main.py --mri C:\data\train_001_V00.im --seg C:\data\train_001_V00.seg`
  (`--seg` is guessed from `--mri` if omitted; `--config file.json` loads an orientation file; `--threshold t` binarises with `value > t`.)

What the loader does (`kneeviewer/data_io.py`):

1. Opens each file with h5py and reads the dataset `"data"`.
2. Validates: MRI must be 3D; segmentation 3D or 4D; **spatial dimensions of MRI and segmentation must match** (otherwise an error names both shapes). A shape different from `(384, 384, 160)` / 6 channels only produces a warning.
3. Reads the segmentation slab by slab (1-byte data kept as is, anything else binarised on the fly) so no huge temporary array exists.
4. Computes the combined mask **once**:
   `np.any(segmentation > 0, axis=-1).astype(np.uint8)` (done slab-wise; identical result) and caches it.
5. Looks for voxel spacing / axis codes / an affine in the HDF5 file; uses them if present (option in the Open dialog), otherwise your manual configuration applies.
6. Initialises all four windows and the control panel.

Clear error messages are shown for: missing file, not an HDF5 file, no `"data"` dataset, wrong number of dimensions, unsupported dtype, mismatching shapes.

---------------------------------------------------------------------------

## 4. Using the viewer

**2D views (each has a slider, spin box, slice number, overlay checkbox, Reset view):**

| Action | Result |
|---|---|
| Left click | Move the crosshair; the other two 2D views and the 3D planes follow |
| Shift + left drag | Move the crosshair continuously |
| Mouse wheel | Previous / next slice |
| Ctrl + wheel, or right drag | Zoom |
| Left or middle drag | Pan |

**3D view:** left drag = rotate, middle drag (or Shift + left drag) = pan, wheel / right drag = zoom. Buttons *Iso / Sag / Cor / Ax / Reset camera* set standard camera directions.

**Control panel (right):**

* *Selected voxel* - editable `(i, j, k)` (indices into the array **as stored in the file**, 0-based), plus slice numbers, intensity and the segmentation channels containing the voxel. The same text is in the status bar.
* *Layers* - MRI + segmentation / MRI only / segmentation only (applies to 2D and 3D).
* *Segmentation* - show: **Combined (logical OR)**, **All channels (each in its own colour)**, or **Channel 0...5**; overlay checkbox (all views at once); overlay opacity; a colour button for the combined mask and for every channel.
* *2D window / level* - contrast of the 2D images and 3D slice planes.
* *3D: MRI volume rendering* - show/hide, opacity, contrast low/high, opacity curve.
* *3D: segmentation surface* - show/hide, opacity, depth peeling (see Troubleshooting).
* *3D: resolution* - separate downsampling factors for the volume (default 2) and the surfaces (default 1). The 2D slices always use the full-resolution data.
* *3D: slice planes* - show/hide each of the three planes (they are opaque and can hide the volume behind them).

Crosshair / plane colours: sagittal = yellow, coronal = green, transverse = red. In a 2D view the vertical line is coloured like the plane it marks.

---------------------------------------------------------------------------

## 5. How the four views are synchronised

```
slider / click / typed voxel  --> CursorModel.set(i, j, k)  --changed-->  SliceView x3 .set_cursor
                                                                    -->  View3D       .set_cursor
                                                                    -->  voxel read-out
control panel widgets  --> ViewerSettings.set(key, value)  --changed(key)-->  every view
```

* There is exactly **one** selected voxel (`CursorModel`, canonical sagittal/coronal/transverse indices). No view talks to another view.
* Moving the sagittal slider changes `i` only: the sagittal view re-slices, the other two views just move a crosshair line, and the 3D sagittal plane moves (its 3 corner points + its texture are rewritten).
* Clicking in a 2D image converts the mouse position back to voxel indices through the *same* `DisplayTransform` that drew the image, so clicks, crosshairs and images cannot disagree, whatever rotation/flip you configured. The click changes the two in-plane indices and leaves the view's own slice unchanged.
* All views use the same world coordinates (millimetres, voxel **centres** at `index * spacing`), so the segmentation is aligned with the MRI in every view, including the 3D slice planes.
* Rendering of the 3D view is coalesced: bursts of updates (dragging a slider) cause at most one render every ~15 ms.

---------------------------------------------------------------------------

## 6. Orientation and voxel spacing

**Nothing is assumed silently.** The orientation is a configuration (`OrientationConfig`, `View > Orientation and voxel spacing...`, Ctrl+E):

* **Which file axis is which anatomical direction** - for sagittal (left-right), coronal (anterior-posterior) and transverse (superior-inferior) slices, choose the array axis (0, 1 or 2) along which those slices are stacked.
* **Reverse (flip)** each axis if its direction is the wrong way round.
* **Voxel spacing in mm** for each *file* axis.
* **Per-view display:** transpose, rotate (0/90/180/270 deg), flip horizontal, flip vertical.
* The dialog prints, live, what each screen direction currently means ("horizontal = file axis 1 [A-P] increases to the right ...") so you can verify it.
* `Save JSON...` / `Load JSON...` store the settings. A file named `orientation_config.json` in the folder you start the viewer from (or next to `main.py`) is loaded automatically. An example is included.
* Changing the orientation does not reload data: the viewer builds NumPy *views* (transpose + negative-stride slicing) of the file arrays, so it is instant and uses no extra memory. The selected physical voxel is preserved.

**Defaults (a starting guess, please verify):** axis 2 (160 slices) = sagittal stack, axis 1 = anterior-posterior, axis 0 = superior-inferior, spacing `(0.3646, 0.3646, 0.7)` mm - typical for sagittally acquired knee DESS data. Neither the orientation nor the spacing is verified by the code; only your dataset documentation or inspection can do that.

**How to check the orientation with your data:**

1. Open a case and look at the **Sagittal** view. It should show a side profile of the knee: femur above, tibia below, patella at the front. If it looks like a front view (two condyles side by side) or a round cross-section, the sagittal stack axis is wrong: choose a different axis for "Sagittal slices" and re-check the other views.
2. The **Coronal** view should show both condyles and the tibial plateau side by side; the **Transverse** view should be a roughly round/oval cross-section.
3. Make sure the bone is upright (superior at the top); use *flip vertical*, *rotate 180*, or the axis flip to correct it.
4. Wrong spacing shows up as stretching (round structures look oval). Correct the three spacing values.
5. Scroll through the slices with the segmentation on: cartilage should hug the bone surfaces in **all** views. If it is displaced or mirrored in only one view, that is a display-flip issue; if it is displaced in all views, the segmentation and MRI do not share an orientation (a data problem, not a viewer problem).

If the HDF5 file contains spacing / axis codes / an affine, they are used automatically (see `read_metadata` in `data_io.py`). For affines the RAS+ convention is assumed. Uncheck the option in the Open dialog to ignore them.

Optional channel names (shown in the UI) can be given in the JSON file: `"channel_names": ["femoral cartilage", ...]`.

---------------------------------------------------------------------------

## 7. Code overview

| File | Contents |
|---|---|
| `main.py` | Entry point, command-line arguments, dark theme. |
| `kneeviewer/config.py` | `OrientationConfig`, `ViewOptions` dataclasses; JSON load/save; validation. |
| `kneeviewer/data_io.py` | `load_case()` (read, validate, combine channels), `read_metadata()`, `apply_metadata()`, `DataLoadError`. |
| `kneeviewer/volume_model.py` | `VolumeModel` (zero-copy canonical views, slice extraction, index mapping), `DisplayTransform` (screen <-> voxel mapping), `downsample_mean()`, `estimate_intensity_range()`. No Qt/VTK imports. |
| `kneeviewer/overlays.py` | `overlay_rgba()` (combined / all channels / one channel), `window_to_u8()`, `compose_rgb()` (used for the 3D plane textures). |
| `kneeviewer/state.py` | `ViewerSettings` (all display options + change signal), `CursorModel` (the selected voxel). |
| `kneeviewer/slice_view.py` | `SliceView` (one 2D window: pyqtgraph image + overlay + crosshair + slider), `SliceViewBox` (click/scroll behaviour). |
| `kneeviewer/view3d.py` | `View3D`: PyVistaQt interactor, VTK volume rendering, lazily built cached surfaces, textured slice planes, render coalescing. |
| `kneeviewer/vtk_utils.py` | `make_vtk_image()`, `build_surface()` (flying edges + smoothing). |
| `kneeviewer/controls.py` | `ControlPanel`, `ColorButton`. |
| `kneeviewer/dialogs.py` | `OpenCaseDialog`, `OrientationDialog`. |
| `kneeviewer/main_window.py` | `MainWindow`: 2x2 layout, menus, loading with progress, orientation changes, wiring of cursor/settings to the views. |
| `make_demo_data.py` | Writes a synthetic case in your file format for testing. |
| `selftest.py` | Window-free tests of the numerical core. |

Maintenance hints: a new display option = add a key to `DEFAULTS` in `state.py`, a widget in `controls.py`, and handle the key in `SliceView.on_setting` / `View3D.on_setting`. A new overlay (e.g. a second segmentation for prediction vs ground truth) belongs next to `overlay_rgba`.

---------------------------------------------------------------------------

## 8. Performance notes

* No full-volume copies after loading: orientation changes, slicing and channel selection are NumPy views. The combined mask is computed once.
* Moving a slider re-extracts one slice (a view) and updates existing `ImageItem`s / VTK textures; nothing is recreated.
* A view re-slices only when its **own** slice index changes; crosshair moves are two `setPos` calls.
* 3D volume: block-averaged, uint8, GPU ray casting (`vtkSmartVolumeMapper`, automatic CPU fallback). Surfaces: built on first display per channel, cached until the resolution or orientation changes.
* Typical memory for one case: MRI (float32 ~95 MB; float64 ~190 MB) + segmentation (uint8 ~140 MB) + combined mask (~23 MB) + the 3D copies.

---------------------------------------------------------------------------

## 9. Limitations and assumptions

* **Orientation and spacing defaults are guesses** (section 6). The viewer does not draw anatomical direction letters (R/L, A/P, S/I), because without trustworthy metadata they could mislead; use the live description in the orientation dialog.
* **"Combined" = logical OR of all channels**, exactly as in your formula. If some of your six channels are not cartilage (e.g. menisci), the combined mask includes them. Use *Channel n* to view them separately.
* In *All channels* mode, overlapping channels are drawn in channel order (later channels on top). The *Combined* mask is exact.
* Channel names default to "Channel 0...5"; set them in the JSON file if you know them.
* Segmentation is treated as binary (`value > threshold`, default 0).
* The 3D slice planes are opaque and can hide the volume behind them; hide individual planes with the checkboxes.
* Downsampling by `f` drops trailing voxels that do not fill a whole block (at most `f-1` voxels at the far end of an axis) for the 3D copy only.
* Translucent surfaces inside a translucent volume need *depth peeling* to blend correctly; it can misbehave on some GPUs/drivers (see below).
* Surfaces at full resolution can take several seconds to build the first time (cached afterwards). "All channels" builds up to six.
* One case is open at a time. Orientation changes rebuild the 3D scene.
* Not supported: Google Colab (see section 1), DICOM, NIfTI input (only HDF5 `.im`/`.seg`), editing the segmentation.

### Testing status

The numerical core (orientation maths, all rotate/flip/transpose combinations, index mapping, downsampling, overlays, loading and error messages) is covered by `selftest.py`. The Qt/VTK layers were checked by static analysis and by running the real view/controller code against recording stand-ins for Qt, pyqtgraph and VTK (click-to-voxel mapping, crosshair placement, plane geometry, texture contents, settings and cursor synchronisation). They have **not** been run against a real GPU/OpenGL stack, so the first launch on your machine is the real test. If something fails, send the console traceback and your versions (`pip list | findstr /i "pyside pyvista vtk pyqtgraph"`).

---------------------------------------------------------------------------

## 10. Troubleshooting

* **`ModuleNotFoundError` / Qt binding errors:** make sure the virtual environment is active and only PySide6 is used (`main.py` sets `QT_API=pyside6`). If you also have PyQt5/6 installed and see conflicts, uninstall them from this environment.
* **3D view is black or crashes at start:** update the graphics driver; on remote desktop / virtual machines OpenGL may be software-only (slow but usually works). Keep *Depth peeling* off.
* **Volume turns black after enabling depth peeling:** turn depth peeling off. Opaque surfaces (opacity 100 %) already show correctly inside the translucent volume.
* **3D interaction is slow:** raise the volume downsample factor (3 or 4) and press *Apply 3D resolution*; reduce volume opacity; hide the volume and look at the surfaces only.
* **Volume rendering looks like a milky blob:** raise *Contrast low* and the *Opacity curve*, lower *Opacity*.
* **Cartilage surface looks blocky or too thin:** keep surface downsample at 1.
* **A window shows a sliver / is empty:** use *Reset view* in that window.
* **Wheel scrolls slices instead of zooming:** that is intentional; hold Ctrl to zoom.

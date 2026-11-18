"""Non-GUI self-test of the numerical core. Run:  python selftest.py

Checks (no window is opened):
  * display transforms vs. real array operations (all rotate/flip/transpose combos)
  * every axis permutation / flip: canonical <-> file index round trip, zero-copy views
  * block-average downsampling, overlay colouring, HDF5 loading + error messages (needs h5py)
"""
import itertools
import os
import sys
import tempfile

import numpy as np

from kneeviewer.config import OrientationConfig, ViewOptions
from kneeviewer.overlays import compose_rgb, overlay_rgba, window_to_u8
from kneeviewer.volume_model import DisplayTransform, VolumeModel, downsample_mean


def test_display_transform():
    shape = (7, 11, 5)
    n = 0
    for plane in range(3):
        for t, rot, fh, fv in itertools.product([False, True], range(4), [False, True], [False, True]):
            dt = DisplayTransform.build(plane, ViewOptions(t, rot, fh, fv))
            for idx in [(0, 0, 0), (3, 4, 2), (6, 10, 4), (1, 9, 3)]:
                vol = np.zeros(shape, np.uint8)
                vol[idx] = 9
                sl = [slice(None)] * 3
                sl[plane] = idx[plane]
                img = dt.apply(vol[tuple(sl)])
                r, c = np.argwhere(img == 9)[0]
                assert (r, c) == dt.to_display(idx, shape)
                assert dt.from_display(r, c, shape) == (idx[dt.row_axis], idx[dt.col_axis])
                n += 1
    return f"display transforms: {n} checks"


def test_orientation():
    rng = np.random.default_rng(0)
    mri = rng.random((8, 9, 6)).astype(np.float32)
    seg = (rng.random((8, 9, 6, 3)) > 0.7).astype(np.uint8)
    mask = np.any(seg > 0, axis=-1).astype(np.uint8)
    n = 0
    for perm in itertools.permutations(range(3)):
        for flips in itertools.product([False, True], repeat=3):
            cfg = OrientationConfig(*perm, *flips, spacing_mm=(0.5, 0.6, 0.7))
            m = VolumeModel(mri, seg, mask, cfg)
            assert np.shares_memory(m.mri, mri) and np.shares_memory(m.seg, seg)
            for f in [(0, 0, 0), (7, 8, 5), (3, 4, 2)]:
                c = m.file_to_canon(f)
                assert m.canon_to_file(c) == f and m.mri[c] == mri[f]
            n += 1
    return f"orientation: {n} axis/flip configurations, views are zero-copy"


def test_downsample_and_overlay():
    rng = np.random.default_rng(1)
    a = rng.random((13, 10, 7)).astype(np.float32)
    view = a.transpose(1, 0, 2)[::-1]
    for f in (1, 2, 3):
        n0, n1, n2 = (s // f * f for s in view.shape)
        ref = np.asarray(view[:n0, :n1, :n2]).reshape(n0 // f, f, n1 // f, f, n2 // f, f).mean(axis=(1, 3, 5))
        assert np.allclose(downsample_mean(view, f), ref, atol=1e-6)
    seg = (rng.random((6, 7, 5, 3)) > 0.6).astype(np.uint8)
    mask = np.any(seg > 0, axis=-1).astype(np.uint8)
    m = VolumeModel(rng.random((6, 7, 5)).astype(np.float32), seg, mask, OrientationConfig(spacing_mm=(1, 1, 1)))
    o = overlay_rgba(m, 2, 3, "combined", (255, 0, 0), [])
    assert ((o[..., 3] == 255) == (m.mask_slice(2, 3) > 0)).all()
    assert window_to_u8(np.array([[0, 5, 10]], np.float32), 0, 10).tolist() == [[0, 127, 255]]
    assert compose_rgb(None, o, 1.0).shape == o.shape[:2] + (3,)
    return "downsampling and overlays"


def test_hdf5():
    try:
        import h5py
    except ImportError:
        return "HDF5 loading: SKIPPED (h5py not installed)"
    from kneeviewer import data_io
    rng = np.random.default_rng(2)
    tmp = tempfile.mkdtemp()
    im, sg = os.path.join(tmp, "a.im"), os.path.join(tmp, "a.seg")
    segf = (rng.random((10, 12, 8, 6)) > 0.8).astype(np.float32)
    with h5py.File(im, "w") as f:
        f["data"] = rng.random((10, 12, 8))
    with h5py.File(sg, "w") as f:
        f["data"] = segf
    case = data_io.load_case(im, sg)
    assert (case.mask == np.any(segf > 0, axis=-1).astype(np.uint8)).all()

    def expect(fn, text):
        try:
            fn()
        except data_io.DataLoadError as e:
            assert text in str(e), str(e)
            return
        raise AssertionError("expected DataLoadError: " + text)

    expect(lambda: data_io.load_case(os.path.join(tmp, "missing.im")), "not found")
    with h5py.File(sg, "w") as f:
        f["data"] = np.zeros((10, 12, 9, 6), np.uint8)
    expect(lambda: data_io.load_case(im, sg), "same spatial size")
    with h5py.File(sg, "w") as f:
        f["other"] = np.zeros(3)
    expect(lambda: data_io.load_case(im, sg), 'no dataset named "data"')
    return "HDF5 loading, mask combination and error messages"


if __name__ == "__main__":
    ok = True
    for t in (test_display_transform, test_orientation, test_downsample_and_overlay, test_hdf5):
        try:
            print("OK  ", t())
        except Exception as exc:  # noqa: BLE001
            ok = False
            print("FAIL", t.__name__, "->", repr(exc))
    sys.exit(0 if ok else 1)

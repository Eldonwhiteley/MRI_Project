"""Write a synthetic knee-like case (demo_001.im / demo_001.seg) in the same format as your data.

    python make_demo_data.py            # 384 x 384 x 160, 6 channels (about 20 MB on disk)
    python make_demo_data.py --small    # 192 x 192 x 80, much faster
    python make_demo_data.py --out C:\\temp

Layout of the synthetic file (matches the viewer's default orientation guess):
    axis 0 = superior-inferior   axis 1 = anterior-posterior   axis 2 = left-right
The "knee" is made of ellipsoids (femur, tibia, patella) with thin cartilage shells
(channels 0-3) and two meniscus-like blobs (channels 4-5). It is NOT anatomically
accurate; it exists only so you can test loading, synchronisation, overlays and 3D.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np


def ell(x, y, z, c, r):
    return np.sqrt(((x - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((z - c[2]) / r[2]) ** 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    shape = (192, 192, 80) if args.small else (384, 384, 160)
    spacing = (0.7292, 0.7292, 1.4) if args.small else (0.3646, 0.3646, 0.7)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    im_path, seg_path = out / "demo_001.im", out / "demo_001.seg"

    # millimetre coordinate of every index (centred); SI grows downward (towards the foot)
    si = ((np.arange(shape[0]) - shape[0] / 2) * spacing[0])[:, None, None]
    ap_ = ((np.arange(shape[1]) - shape[1] / 2) * spacing[1])[None, :, None]
    rng = np.random.default_rng(1)

    with h5py.File(im_path, "w") as fi, h5py.File(seg_path, "w") as fs:
        d_im = fi.create_dataset("data", shape, dtype="float32", chunks=(64, 64, 16),
                                 compression="gzip", compression_opts=1)
        d_sg = fs.create_dataset("data", shape + (6,), dtype="uint8", chunks=(64, 64, 16, 6),
                                 compression="gzip", compression_opts=1)
        step = 16
        for z0 in range(0, shape[2], step):
            z1 = min(z0 + step, shape[2])
            lr = ((np.arange(z0, z1) - shape[2] / 2) * spacing[2])[None, None, :]
            femur = ell(si, ap_, lr, (-47, 0, 0), (50, 25, 30))
            tibia = ell(si, ap_, lr, (53, 0, 0), (45, 25, 32))
            patella = ell(si, ap_, lr, (-15, -38, 0), (12, 6, 14))
            soft = ell(si, ap_, lr, (0, 0, 0), (70, 45, 50))

            img = np.zeros(soft.shape, np.float32)
            img[soft <= 1] = 0.45
            bone = (femur <= 1) | (tibia <= 1) | (patella <= 1)
            img[bone] = 0.25
            for b in (femur, tibia, patella):
                img[np.abs(b - 1) < 0.04] = 0.08          # dark cortex

            seg = np.zeros(soft.shape + (6,), np.uint8)
            seg[..., 0] = (femur > 1) & (femur <= 1 + 2.4 / 50) & (si > -20)             # femoral cartilage
            seg[..., 1] = (tibia > 1) & (tibia <= 1 + 2.4 / 45) & (si < 25) & (lr < 0)   # medial tibial
            seg[..., 2] = (tibia > 1) & (tibia <= 1 + 2.4 / 45) & (si < 25) & (lr >= 0)  # lateral tibial
            seg[..., 3] = (patella > 1) & (patella <= 1 + 0.25) & (ap_ > -38)            # patellar
            seg[..., 4] = ell(si, ap_, lr, (5.5, 0, 17), (3.5, 20, 8)) <= 1              # lateral meniscus
            seg[..., 5] = ell(si, ap_, lr, (5.5, 0, -17), (3.5, 20, 8)) <= 1             # medial meniscus
            img[seg[..., :4].any(axis=-1) > 0] = 0.85      # cartilage is bright
            img[seg[..., 4:].any(axis=-1) > 0] = 0.15      # meniscus is dark
            img += rng.normal(0, 0.02, img.shape).astype(np.float32)

            d_im[:, :, z0:z1] = img * 1000.0
            d_sg[:, :, z0:z1, :] = seg
            print(f"\rslab {z1}/{shape[2]}", end="", flush=True)
    print(f"\nWrote {im_path}\n      {seg_path}")


if __name__ == "__main__":
    main()

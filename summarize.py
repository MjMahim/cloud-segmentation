"""Extra summaries of the test results: median per-scene IoU and IoU over all test pixels pooled together."""
import csv
from pathlib import Path

import numpy as np

from data import TEST_DIR
from evaluate import read_scene_gt

OUT = Path(__file__).parent / "results"
rows = list(csv.DictReader(open(OUT / "test_scenes.csv")))
methods = sorted({r["method"] for r in rows}, key=lambda m: (m != "baseline", m))
scenes = sorted({r["scene"] for r in rows})

pooled = {m: [0, 0, 0] for m in methods}  # tp, fp, fn over all scenes
for s in scenes:
    gt = read_scene_gt(TEST_DIR / "Entire_scene_gts" / f"edited_corrected_gts_{s}.TIF")
    for m in methods:
        z = np.load(OUT / "preds" / f"{s}_{m}.npz")
        pred = np.unpackbits(z["mask"])[: gt.size].reshape(gt.shape).astype(bool)
        pooled[m][0] += np.count_nonzero(pred & gt)
        pooled[m][1] += np.count_nonzero(pred & ~gt)
        pooled[m][2] += np.count_nonzero(~pred & gt)

print(f"{'method':22s} {'mean IoU':>9s} {'median IoU':>11s} {'pooled IoU':>11s}")
for m in methods:
    j = [float(r["jaccard"]) for r in rows if r["method"] == m]
    tp, fp, fn = pooled[m]
    print(f"{m:22s} {100*np.mean(j):9.2f} {100*np.median(j):11.2f} {100*tp/(tp+fp+fn):11.2f}")

rgb = {r["scene"]: float(r["jaccard"]) for r in rows if r["method"] == "unet_resnet34_rgb"}
nir = {r["scene"]: float(r["jaccard"]) for r in rows if r["method"] == "unet_resnet34_rgbn"}
d = np.array([nir[s] - rgb[s] for s in scenes])
print(f"\nNIR better in {np.sum(d > 0.001)} scenes, worse in {np.sum(d < -0.001)}, equal in {np.sum(abs(d) <= 0.001)}")

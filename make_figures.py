"""Figure: satellite image, ground truth, and colour-coded error maps for a few test scenes.

Error map colours:  white = cloud found correctly,  black = clear found correctly,
                    red = false cloud (predicted cloud, but clear),  blue = missed cloud.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data import TEST_DIR
from evaluate import read_scene_gt

Image.MAX_IMAGE_PIXELS = None
OUT = Path(__file__).parent / "results"
STEP = 8  # downsample full scenes (~8000 px) for plotting

SCENES = ["LC08_L1TP_034037_20160520_20170223_01_T1"]
FS = 19  # font size: about the same as the report text when the figure is scaled to text width
METHODS = [("baseline", "Brightness rule"), ("unet_resnet34_rgb", "U-Net RGB"), ("unet_resnet34_rgbn", "U-Net RGB+NIR")]
iou = {(r["scene"], r["method"]): float(r["jaccard"]) for r in csv.DictReader(open(OUT / "test_scenes.csv"))}


def load_pred(scene, method, shape):
    z = np.load(OUT / "preds" / f"{scene}_{method}.npz")
    return np.unpackbits(z["mask"])[: shape[0] * shape[1]].reshape(shape).astype(bool)


def error_map(pred, gt):
    rgb = np.zeros(gt.shape + (3,), np.uint8)
    rgb[pred & gt] = (255, 255, 255)   # correct cloud
    rgb[pred & ~gt] = (230, 40, 40)    # false cloud
    rgb[~pred & gt] = (40, 120, 255)   # missed cloud
    return rgb


fig, axes = plt.subplots(len(SCENES), 2 + len(METHODS), figsize=(16, 4.2 * len(SCENES)), squeeze=False)
for row, scene in enumerate(SCENES):
    gt = read_scene_gt(TEST_DIR / "Entire_scene_gts" / f"edited_corrected_gts_{scene}.TIF")
    img = np.asarray(Image.open(TEST_DIR / "Natural_False_Color" / f"{scene}.TIF"))[::STEP, ::STEP]
    panels = [(img, "Satellite image\n"), (gt[::STEP, ::STEP], "Ground truth\n")]
    for m, label in METHODS:
        pred = load_pred(scene, m, gt.shape)
        panels.append((error_map(pred[::STEP, ::STEP], gt[::STEP, ::STEP]), f"{label}\nIoU {100 * iou[(scene, m)]:.1f}%"))
    for ax, (im, title) in zip(axes[row], panels):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None, interpolation="nearest")
        ax.set_title(title, fontsize=FS)
        ax.axis("off")

fig.tight_layout()  # colour legend is given in the report's figure caption
fig.savefig(OUT / "fig_scenes.png", dpi=110, bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUT / "fig_scenes.pdf", bbox_inches="tight", pad_inches=0.05)
print("saved", OUT / "fig_scenes.png")

"""Evaluate trained models (and a brightness-threshold baseline) on the 20 full 38-Cloud test scenes.

Follows the dataset authors' evaluation (evaluation.m in the 38-Cloud GitHub repository):
patch predictions are stitched row by row into a full-scene mask, the zero padding is removed
by a centred crop to the ground-truth size, and precision, recall, specificity, Jaccard (IoU)
and accuracy are computed per scene and averaged over the 20 scenes.

Usage:
    python evaluate.py                     # evaluates every run in runs/ plus the baseline
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import tifffile
import torch
from PIL import Image
from tqdm import tqdm

from data import BAND_SETS, ROOT, TEST_DIR, nonempty_train_patches, split_by_scene

HERE = Path(__file__).parent
RUNS = HERE / "runs"
OUT = HERE / "results"
PATCH = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- baseline
def brightness(img_rgbn):
    """Mean of red, green and blue reflectance: clouds are bright in all visible bands."""
    return img_rgbn[..., :3, :, :].mean(axis=-3)


def tune_baseline_threshold():
    """Pick the brightness threshold that maximises IoU on the validation scenes (no test data used)."""
    _, val_names, _ = split_by_scene(nonempty_train_patches())
    cache = ROOT / "cache"
    index = {n: i for i, n in enumerate((cache / "train_names.txt").read_text().split("\n"))}
    images = np.load(cache / "train_images.npy", mmap_mode="r")
    masks = np.load(cache / "train_masks.npy", mmap_mode="r")
    idx = sorted(index[n] for n in val_names)
    b = np.concatenate([brightness(images[i:i + 1].astype(np.float32) / 65535.0).ravel() for i in idx])
    g = np.concatenate([masks[i, 0].ravel() for i in idx]).astype(bool)
    best = (0.0, 0.0)
    for t in np.linspace(0.05, 0.6, 111):
        p = b > t
        iou = (p & g).sum() / max((p | g).sum(), 1)
        best = max(best, (iou, t))
    print(f"baseline: brightness threshold {best[1]:.3f} (val IoU {best[0]:.4f})")
    return best[1]


# ---------------------------------------------------------------- models
def load_models():
    models = {}
    for ckpt_path in sorted(RUNS.glob("*/best.pt")):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        args = ckpt["args"]
        in_ch = len(BAND_SETS[args["bands"]])
        model = smp.Unet(args["encoder"], encoder_weights=None, in_channels=in_ch, classes=1)
        model.load_state_dict(ckpt["model"])
        models[ckpt_path.parent.name] = (model.to(DEVICE).eval(), in_ch)
        print(f"loaded {ckpt_path.parent.name} (epoch {ckpt['epoch']}, val IoU {ckpt['val']['iou']:.4f})")
    return models


# ---------------------------------------------------------------- metrics
def scene_metrics(pred, gt):
    tp = np.count_nonzero(pred & gt)
    fp = np.count_nonzero(pred & ~gt)
    fn = np.count_nonzero(~pred & gt)
    tn = pred.size - tp - fp - fn
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "jaccard": tp / max(tp + fp + fn, 1),
        "accuracy": (tp + tn) / pred.size,
    }


def read_scene_gt(path):
    """Read a full-scene ground truth as a boolean cloud mask.

    The full-scene masks are CCITT-compressed with PhotometricInterpretation = MinIsWhite,
    i.e. a stored 0 means white (cloud). tifffile returns the raw stored values, so the
    mask must be inverted for those files (the patch masks use MinIsBlack and need no change).
    """
    with tifffile.TiffFile(path) as t:
        page = t.pages[0]
        raw = page.asarray().astype(bool)
        return ~raw if page.photometric == tifffile.PHOTOMETRIC.MINISWHITE else raw


def unzeropad(full, shape):
    """Centred crop back to the original scene size (same as unzeropad in evaluation.m)."""
    ny, nx = full.shape
    nys, nxs = shape
    oy, ox = (ny - nys) // 2, (nx - nxs) // 2
    return full[oy:oy + nys, ox:ox + nxs]


def save_preview(mask_or_img, path, size=900):
    im = Image.fromarray(mask_or_img)
    im.thumbnail((size, size))
    im.save(path)


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "scenes").mkdir(exist_ok=True)
    (OUT / "preds").mkdir(exist_ok=True)
    threshold = tune_baseline_threshold()
    models = load_models()
    methods = ["baseline"] + list(models)

    pat = re.compile(r"red_patch_(\d+)_(\d+)_by_(\d+)_(.+)\.TIF")
    scenes = defaultdict(list)
    for f in (TEST_DIR / "test_red").iterdir():
        m = pat.match(f.name)
        scenes[m.group(4)].append((int(m.group(2)), int(m.group(3)), f.name[len("red_"):-len(".TIF")]))

    rows_out = []
    for scene in tqdm(sorted(scenes), desc="test scenes"):
        cells = scenes[scene]
        R, C = max(c[0] for c in cells), max(c[1] for c in cells)
        full = {m: np.zeros((R * PATCH, C * PATCH), bool) for m in methods}

        batch, where = [], []

        def flush():
            x = np.stack(batch).astype(np.float32) / 65535.0          # (B, 4, H, W)
            pb = brightness(x) > threshold
            for k, (r, c) in enumerate(where):
                full["baseline"][(r - 1) * PATCH:r * PATCH, (c - 1) * PATCH:c * PATCH] = pb[k]
            xt = torch.from_numpy(x).to(DEVICE)
            with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
                for name, (model, in_ch) in models.items():
                    pred = (model(xt[:, :in_ch]) > 0).squeeze(1).cpu().numpy()  # probability > 0.5
                    for k, (r, c) in enumerate(where):
                        full[name][(r - 1) * PATCH:r * PATCH, (c - 1) * PATCH:c * PATCH] = pred[k]
            batch.clear()
            where.clear()

        for r, c, patch in cells:
            batch.append(np.stack([tifffile.imread(TEST_DIR / f"test_{b}" / f"{b}_{patch}.TIF")
                                   for b in BAND_SETS["rgbn"]]))
            where.append((r, c))
            if len(batch) == 16:
                flush()
        if batch:
            flush()

        gt = read_scene_gt(TEST_DIR / "Entire_scene_gts" / f"edited_corrected_gts_{scene}.TIF")
        save_preview((gt * 255).astype(np.uint8), OUT / "scenes" / f"{scene}_gt.png")
        for m in methods:
            pred = unzeropad(full[m], gt.shape)
            np.savez_compressed(OUT / "preds" / f"{scene}_{m}.npz", mask=np.packbits(pred), shape=pred.shape)
            rows_out.append({"scene": scene, "method": m, **scene_metrics(pred, gt)})
            save_preview((pred * 255).astype(np.uint8), OUT / "scenes" / f"{scene}_{m}.png")

    keys = ["precision", "recall", "specificity", "jaccard", "accuracy"]
    with open(OUT / "test_scenes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "method", *keys])
        w.writeheader()
        w.writerows(rows_out)

    print("\nAverage over the 20 test scenes (%):")
    print(f"{'method':24s}" + "".join(f"{k:>12s}" for k in keys))
    summary = []
    for m in methods:
        vals = {k: 100 * np.mean([r[k] for r in rows_out if r["method"] == m]) for k in keys}
        summary.append({"method": m, **vals})
        print(f"{m:24s}" + "".join(f"{vals[k]:12.2f}" for k in keys))
    with open(OUT / "test_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", *keys])
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()

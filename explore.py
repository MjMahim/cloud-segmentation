"""Quick look at the 38-Cloud training data: file format, value ranges, cloud coverage."""
from pathlib import Path

import numpy as np
import tifffile

DATA = Path(__file__).parent / "data" / "38-Cloud_training"
BANDS = ["red", "green", "blue", "nir"]

names = (Path(__file__).parent / "data" / "training_patches_38-cloud_nonempty.csv").read_text().split()[1:]
print(f"non-empty training patches: {len(names)} of 8400")

# One example patch
name = names[0]
for b in BANDS:
    img = tifffile.imread(DATA / f"train_{b}" / f"{b}_{name}.TIF")
    print(f"{b:5s} shape={img.shape} dtype={img.dtype} min={img.min()} max={img.max()}")
gt = tifffile.imread(DATA / "train_gt" / f"gt_{name}.TIF")
print(f"gt    shape={gt.shape} dtype={gt.dtype} values={np.unique(gt)}")

# Cloud coverage over a random sample of non-empty patches
rng = np.random.default_rng(0)
sample = rng.choice(names, size=500, replace=False)
fractions = np.array([(tifffile.imread(DATA / "train_gt" / f"gt_{n}.TIF") > 0).mean() for n in sample])
print(f"cloud fraction (500 patches): mean={fractions.mean():.2f}  "
      f"clear(<1%)={np.mean(fractions < 0.01):.0%}  mostly cloud(>90%)={np.mean(fractions > 0.9):.0%}")

# Scenes in the training set
scenes = sorted({n.split("_by_")[1].split("_", 2)[2] for n in names})
print(f"training scenes: {len(scenes)}")

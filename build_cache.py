"""Convert the non-empty training patches into two memory-mapped .npy files (one-time, ~7 GB).

Reading one big array is much faster than opening 5 TIF files per patch during training.
"""
import numpy as np
import tifffile
from tqdm import tqdm

from data import BAND_SETS, ROOT, TRAIN_DIR, nonempty_train_patches

names = nonempty_train_patches()
bands = BAND_SETS["rgbn"]
cache = ROOT / "cache"
cache.mkdir(exist_ok=True)

images = np.lib.format.open_memmap(cache / "train_images.npy", mode="w+", dtype=np.uint16,
                                   shape=(len(names), len(bands), 384, 384))
masks = np.lib.format.open_memmap(cache / "train_masks.npy", mode="w+", dtype=np.uint8,
                                  shape=(len(names), 1, 384, 384))
for i, name in enumerate(tqdm(names, desc="caching")):
    for c, b in enumerate(bands):
        images[i, c] = tifffile.imread(TRAIN_DIR / f"train_{b}" / f"{b}_{name}.TIF")
    masks[i, 0] = tifffile.imread(TRAIN_DIR / "train_gt" / f"gt_{name}.TIF") > 0
images.flush()
masks.flush()
(cache / "train_names.txt").write_text("\n".join(names))
print("done:", images.shape, masks.shape)

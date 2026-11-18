"""Find where the original scene sits inside the stitched (zero-padded) test patch grid.

Compares the no-data region of the stitched red band with the no-data region of the
full-scene 'Natural_False_Color' image (which has the original scene size) for candidate offsets.
"""
import re
from collections import defaultdict

import numpy as np
import tifffile
from PIL import Image

from data import TEST_DIR

Image.MAX_IMAGE_PIXELS = None
pat = re.compile(r"red_patch_(\d+)_(\d+)_by_(\d+)_(.+)\.TIF")
grid = defaultdict(list)
for f in (TEST_DIR / "test_red").iterdir():
    m = pat.match(f.name)
    grid[m.group(4)].append((int(m.group(1)), int(m.group(2)), int(m.group(3))))

print("NFC files:", [p.name for p in (TEST_DIR / "Natural_False_Color").iterdir()][:2])
for scene in sorted(grid)[:3]:
    cells = grid[scene]
    rows, cols = max(c[1] for c in cells), max(c[2] for c in cells)
    gt = tifffile.imread(TEST_DIR / "Entire_scene_gts" / f"edited_corrected_gts_{scene}.TIF")
    canvas = np.zeros((rows * 384, cols * 384), np.uint16)
    for idx, r, c in cells:
        canvas[(r - 1) * 384:r * 384, (c - 1) * 384:c * 384] = tifffile.imread(
            TEST_DIR / "test_red" / f"red_patch_{idx}_{r}_by_{c}_{scene}.TIF")
    nfc_path = next((TEST_DIR / "Natural_False_Color").glob(f"*{scene}*"))
    nfc = np.asarray(Image.open(nfc_path))
    nodata_ref = nfc.reshape(nfc.shape[0], nfc.shape[1], -1).sum(-1) == 0
    H, W = gt.shape
    ph, pw = canvas.shape[0] - H, canvas.shape[1] - W
    print(scene, "gt", gt.shape, "nfc", nfc.shape)
    results = []
    # coarse search on a 1/8 subsample over the whole padding range, then refine at full resolution
    sub_ref = nodata_ref[::8, ::8]
    for oy in range(0, ph + 1, 2):
        for ox in range(0, pw + 1, 2):
            crop = canvas[oy:oy + H:8, ox:ox + W:8] == 0
            results.append(((crop == sub_ref).mean(), oy, ox))
    _, cy, cx = max(results)
    results = []
    for oy in range(max(0, cy - 2), min(ph, cy + 2) + 1):
        for ox in range(max(0, cx - 2), min(pw, cx + 2) + 1):
            crop = canvas[oy:oy + H, ox:ox + W] == 0
            results.append(((crop == nodata_ref).mean(), oy, ox))
    best = max(results)
    print(f"  pad ({ph},{pw})  centre-floor ({ph // 2},{pw // 2})  best offset ({best[1]},{best[2]}) agreement {best[0]:.5f}")

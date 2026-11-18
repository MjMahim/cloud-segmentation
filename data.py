"""38-Cloud dataset loading: stacks the Landsat 8 band patches into one image tensor."""
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).parent / "data"
TRAIN_DIR = ROOT / "38-Cloud_training"
TEST_DIR = ROOT / "38-Cloud_test"

BAND_SETS = {
    "rgb": ["red", "green", "blue"],
    "rgbn": ["red", "green", "blue", "nir"],
}


def scene_of(patch_name: str) -> str:
    """'patch_100_5_by_12_LC08_L1TP_...' -> 'LC08_L1TP_...' (the Landsat scene ID)."""
    return patch_name.split("_by_")[1].split("_", 1)[1]


def nonempty_train_patches() -> list[str]:
    """Training patches that contain image data (the rest are black border padding)."""
    lines = (ROOT / "training_patches_38-cloud_nonempty.csv").read_text().split()
    return lines[1:]  # skip header


def split_by_scene(names: list[str], n_val_scenes: int = 3, seed: int = 42):
    """Hold out whole scenes for validation, so the model is validated on unseen places."""
    scenes = sorted({scene_of(n) for n in names})
    rng = np.random.default_rng(seed)
    val_scenes = set(rng.choice(scenes, size=n_val_scenes, replace=False))
    train = [n for n in names if scene_of(n) not in val_scenes]
    val = [n for n in names if scene_of(n) in val_scenes]
    return train, val, sorted(val_scenes)


def load_patch(folder: Path, split: str, name: str, bands: list[str]) -> np.ndarray:
    """Stack the band files of one patch into a (C, H, W) float32 array scaled to [0, 1]."""
    layers = [tifffile.imread(folder / f"{split}_{b}" / f"{b}_{name}.TIF") for b in bands]
    return np.stack(layers).astype(np.float32) / 65535.0


def augment(img, mask):
    """Random 90-degree rotation and horizontal flip (clouds have no fixed orientation)."""
    k = np.random.randint(4)
    img, mask = np.rot90(img, k, axes=(1, 2)), np.rot90(mask, k, axes=(1, 2))
    if np.random.rand() < 0.5:
        img, mask = img[:, :, ::-1], mask[:, :, ::-1]
    return img, mask


class CachedCloudDataset(Dataset):
    """Training patches read from the memory-mapped cache built by build_cache.py (fast)."""

    def __init__(self, names, bands="rgbn", augment=False):
        cache = ROOT / "cache"
        index = {n: i for i, n in enumerate((cache / "train_names.txt").read_text().split("\n"))}
        self.idx = [index[n] for n in names]
        self.channels = [BAND_SETS["rgbn"].index(b) for b in BAND_SETS[bands]]
        self.augment = augment
        self.cache = cache
        self.images = self.masks = None  # opened lazily inside each worker process

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.images is None:
            self.images = np.load(self.cache / "train_images.npy", mmap_mode="r")
            self.masks = np.load(self.cache / "train_masks.npy", mmap_mode="r")
        j = self.idx[i]
        img = self.images[j, self.channels].astype(np.float32) / 65535.0
        mask = self.masks[j].astype(np.float32)
        if self.augment:
            img, mask = augment(img, mask)
        return torch.from_numpy(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(mask))


class CloudDataset(Dataset):
    def __init__(self, names, bands="rgbn", augment=False, folder=TRAIN_DIR, split="train", with_mask=True):
        self.names = list(names)
        self.bands = BAND_SETS[bands]
        self.augment = augment
        self.folder = folder
        self.split = split
        self.with_mask = with_mask

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = load_patch(self.folder, self.split, name, self.bands)
        mask = None
        if self.with_mask:
            mask = (tifffile.imread(self.folder / f"{self.split}_gt" / f"gt_{name}.TIF") > 0).astype(np.float32)[None]

        if self.augment:  # random flips and 90-degree rotations (clouds have no fixed orientation)
            k = np.random.randint(4)
            img = np.rot90(img, k, axes=(1, 2))
            if mask is not None:
                mask = np.rot90(mask, k, axes=(1, 2))
            if np.random.rand() < 0.5:
                img = img[:, :, ::-1]
                if mask is not None:
                    mask = mask[:, :, ::-1]

        img = torch.from_numpy(np.ascontiguousarray(img))
        if mask is None:
            return img, name
        return img, torch.from_numpy(np.ascontiguousarray(mask))

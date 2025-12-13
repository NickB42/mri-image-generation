import os
from pathlib import Path
import random

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


def _normalize_volume(vol, eps=1e-6, clip_val=5.0):
    """
    vol: (D, H, W) np.float32
    - Z-score over non-zero voxels
    - Clip to [-clip_val, clip_val]
    - Rescale to [-1, 1]
    """
    mask = vol != 0  # ignore pure background
    if mask.any():
        vals = vol[mask]
        mean = vals.mean()
        std = vals.std()
        if std < eps:
            std = 1.0
        vol[mask] = (vol[mask] - mean) / std
    else:
        # fallback: z-score over all voxels
        mean = vol.mean()
        std = vol.std()
        if std < eps:
            std = 1.0
        vol = (vol - mean) / std

    # Clip extremes
    vol = np.clip(vol, -clip_val, clip_val)

    # Map [-clip_val, clip_val] -> [0, 1] -> [-1, 1]
    vol = (vol + clip_val) / (2.0 * clip_val)  # [0, 1]
    vol = vol * 2.0 - 1.0                      # [-1, 1]

    return vol


def _pad_to_min_shape(vol, target_shape):
    """
    vol: (C, D, H, W)
    target_shape: (D, H, W)
    Symmetric zero-padding if needed.
    """
    c, d, h, w = vol.shape
    td, th, tw = target_shape

    pd = max(td - d, 0)
    ph = max(th - h, 0)
    pw = max(tw - w, 0)

    pad_before_d = pd // 2
    pad_after_d = pd - pad_before_d
    pad_before_h = ph // 2
    pad_after_h = ph - pad_before_h
    pad_before_w = pw // 2
    pad_after_w = pw - pad_before_w

    if pd > 0 or ph > 0 or pw > 0:
        vol = np.pad(
            vol,
            (
                (0, 0),
                (pad_before_d, pad_after_d),
                (pad_before_h, pad_after_h),
                (pad_before_w, pad_after_w),
            ),
            mode="constant",
        )
    return vol


def _random_or_center_crop(vol, patch_size, random_crop=True):
    """
    vol: (C, D, H, W), already padded to at least patch_size.
    patch_size: (D, H, W)
    """
    c, d, h, w = vol.shape
    pd, ph, pw = patch_size

    if d < pd or h < ph or w < pw:
        raise ValueError("Volume is smaller than patch even after padding.")

    if random_crop:
        max_z = d - pd
        max_y = h - ph
        max_x = w - pw
        start_z = random.randint(0, max_z) if max_z > 0 else 0
        start_y = random.randint(0, max_y) if max_y > 0 else 0
        start_x = random.randint(0, max_x) if max_x > 0 else 0
    else:
        start_z = (d - pd) // 2
        start_y = (h - ph) // 2
        start_x = (w - pw) // 2

    end_z = start_z + pd
    end_y = start_y + ph
    end_x = start_x + pw

    return vol[:, start_z:end_z, start_y:end_y, start_x:end_x]


class BraTS3DVolumeDataset(Dataset):
    """
    Loads 4 modalities (FLAIR, T1, T1ce, T2) as channels of a 3D volume.

    Returns:
      volume: torch.float32 tensor of shape (4, D, H, W)
    """

    def __init__(
        self,
        root_dir,
        patch_size=(128, 160, 160),
        random_crop=True,
        modalities=("flair", "t1", "t1ce", "t2"),
    ):
        """
        root_dir: path to BraTS root (recursively searched for *_flair.nii.gz)
        patch_size: tuple (D, H, W) for 3D patches
        random_crop: if False, uses center crop
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.modalities = modalities

        self.cases = self._find_cases()
        if len(self.cases) == 0:
            raise ValueError(f"No BraTS cases found in {root_dir}")

        print(f"Found {len(self.cases)} BraTS subjects.")

    def _find_cases(self):
        cases = []
        flair_files = list(self.root_dir.rglob("*_flair.nii.gz"))
        for flair_path in flair_files:
            flair_path = Path(flair_path)
            base = str(flair_path).replace("_flair.nii.gz", "")
            paths = {
                "flair": flair_path,
                "t1": Path(base + "_t1.nii.gz"),
                "t1ce": Path(base + "_t1ce.nii.gz"),
                "t2": Path(base + "_t2.nii.gz"),
            }

            if all(p.exists() for p in paths.values()):
                cases.append(tuple(paths[m] for m in self.modalities))
        return cases

    def __len__(self):
        return len(self.cases)

    def _load_volume(self, paths):
        """
        paths: tuple of 4 paths (one per modality)
        Returns np.array (4, D, H, W)
        """
        vols = []
        for p in paths:
            img = nib.load(str(p))
            vol = img.get_fdata().astype(np.float32)

            # Expect shape (H, W, D) -> reorder to (D, H, W)
            if vol.ndim == 4:
                # just in case, drop singleton
                vol = vol[..., 0]
            vol = np.transpose(vol, (2, 0, 1))  # (D, H, W)

            vol = _normalize_volume(vol)
            vols.append(vol)

        vol = np.stack(vols, axis=0)  # (C=4, D, H, W)

        # Pad if needed
        vol = _pad_to_min_shape(vol, self.patch_size)

        # Crop
        vol = _random_or_center_crop(vol, self.patch_size, self.random_crop)

        return vol

    def __getitem__(self, idx):
        paths = self.cases[idx]
        vol = self._load_volume(paths)
        vol = torch.from_numpy(vol).float()  # (4, D, H, W)
        return vol

"""
mm_dataset.py — Memory-mapped dataset for 2.5D sequential DDPM.

Adapted to:
  - Match the return signature of the existing BraTSSliceDataset
    (returns a tuple: x_center, x_context, z_pos, fg_frac)
  - Use the same multi-modality 4-channel convention (but memmap
    stores 1-channel FLAIR, so we replicate to 4 if needed or keep 1)
  - Support separate train/val memmap files
  - Use correct workspace paths

The memmap layout is (N_volumes, 155, 1, H, W) float32.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

NUM_SLICES = 155
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMMAP_DIR = PROJECT_ROOT / "datasets" / "memmap"


class MemmapDataset(Dataset):
    """
    2.5D sequential dataset backed by a numpy memmap.

    Returns the same 4-tuple as the NIfTI-based BraTSSliceDataset so it
    can be used as a drop-in replacement in train.py:

        x_center:  (C, H, W)           — the target slice
        x_context: (C * slice_radius, H, W) — neighboring context slices
        z_pos:     float in [0, 1]      — normalised slice position
        fg_frac:   float in [0, 1]      — foreground fraction of center slice
    """

    def __init__(
        self,
        path: str | Path,
        shape: tuple | None = None,
        channels: int = 1,
        image_size: int = 256,
        slice_radius: int = 2,
    ):
        """
        Args:
            path:         Path to the .npy memmap file.
            shape:        Explicit shape (N, D, C, H, W).  If None it is
                          inferred by reading a tiny header written by prep_all.
            channels:     Number of channels stored per slice (1 for FLAIR-only).
            image_size:   Spatial size H=W used in memmap if shape is inferred.
            slice_radius: How many preceding slices to include as context.
        """
        self.path = Path(path)
        self.channels = channels
        self.image_size = image_size
        self.slice_radius = slice_radius

        if shape is None:
            # Infer from file size (assumes float32, C=channels, H=W=image_size, D=155)
            file_bytes = self.path.stat().st_size
            per_volume = NUM_SLICES * channels * image_size * image_size * 4  # 4 bytes per float32
            n_vols = file_bytes // per_volume
            shape = (n_vols, NUM_SLICES, channels, image_size, image_size)

        self.data = np.memmap(path, mode="r", shape=shape, dtype=np.float32)
        self.n_volumes = self.data.shape[0]
        self.num_slices = self.data.shape[1]
        self.H = self.data.shape[3]
        self.W = self.data.shape[4]

        # Usable z-range: need room for slice_radius neighbors below
        # and avoid boundary slices (mirror the 10%/90% convention)
        self.z_start = int(0.1 * self.num_slices) + self.slice_radius
        self.z_end = int(0.9 * self.num_slices)
        self.usable_slices = list(range(self.z_start, self.z_end))
        self.num_usable = len(self.usable_slices)

        # Build flat index -> (vol_idx, z) mapping for fast __getitem__
        # We also store (path, z, D) tuples for compatibility with the
        # WeightedRandomSampler logic in train.py
        self.slice_tuples = []
        for v in range(self.n_volumes):
            for z in self.usable_slices:
                self.slice_tuples.append((v, z, self.num_slices))

        print(f"MemmapDataset: {self.n_volumes} volumes, "
              f"{self.num_usable} slices/vol, "
              f"{len(self.slice_tuples)} total samples, "
              f"z_range=[{self.z_start}, {self.z_end}), "
              f"slice_radius={self.slice_radius}")

    def __len__(self):
        return len(self.slice_tuples)

    def __getitem__(self, idx):
        vol_idx, z, D = self.slice_tuples[idx]

        # Center slice: (C, H, W)
        x_center = torch.from_numpy(
            self.data[vol_idx, z].copy()
        )  # (C, H, W)

        # Context: preceding slices z-slice_radius .. z-1  (like the NIfTI dataset)
        context_slices = []
        for dz in range(-self.slice_radius, 0):
            zz = z + dz
            zz = max(0, min(zz, self.num_slices - 1))
            s = torch.from_numpy(self.data[vol_idx, zz].copy())  # (C, H, W)
            context_slices.append(s)
        x_context = torch.cat(context_slices, dim=0)  # (C * slice_radius, H, W)

        # Normalised slice position
        z_pos = np.float32(z / (D - 1))

        # Foreground fraction (fraction of voxels above ~background threshold)
        center_np = self.data[vol_idx, z]  # (C, H, W)
        fg_frac = np.float32((center_np > -0.999).mean())

        return x_center, x_context, z_pos, fg_frac


# -----------------------------------------------------------------------
# Convenience constructors (match workspace layout)
# -----------------------------------------------------------------------
def get_train_dataset(slice_radius: int = 2) -> MemmapDataset:
    return MemmapDataset(
        MEMMAP_DIR / "train_flair_256.npy",
        image_size=256,
        slice_radius=slice_radius,
    )


def get_val_dataset(slice_radius: int = 2) -> MemmapDataset:
    return MemmapDataset(
        MEMMAP_DIR / "val_flair_256.npy",
        image_size=256,
        slice_radius=slice_radius,
    )


def get_debug_dataset(slice_radius: int = 2) -> MemmapDataset:
    debug_path = MEMMAP_DIR / "train_flair_debug_256.npy"
    if not debug_path.exists():
        debug_path = MEMMAP_DIR / "train_flair_256.npy"

    return MemmapDataset(
        debug_path,
        image_size=256,
        slice_radius=slice_radius,
    )

from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class BraTSSliceDataset(Dataset):
    """
    Multi-modal 2.5D dataset:
    - Uses 4 modalities: t1, t1ce, t2, flair
    - Uses central ~80% slices, but also needs neighbors (z-1, z+1)
    - Returns:
        x_center:  (4, H, W)   center slice, all modalities
        x_context: (8, H, W)   neighbors (z-1, z+1) for all modalities
        z_pos:     scalar in [0, 1]
    """

    def __init__(self, root_dir, image_size=128, slice_radius=1):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.slice_radius = slice_radius

        # We'll anchor on FLAIR files and infer other modalities
        self.flair_suffix = "_flair.nii.gz"
        self.modalities = [
            "_t1.nii.gz",
            "_t1ce.nii.gz",
            "_t2.nii.gz",
            "_flair.nii.gz",
        ]

        self.volume_paths = sorted(self.root_dir.rglob(f"*{self.flair_suffix}"))
        if not self.volume_paths:
            raise RuntimeError(f"No FLAIR files (*{self.flair_suffix}) found under {root_dir}")

        self.slice_tuples = []
        for p in self.volume_paths:
            img = nib.load(str(p))
            shape = img.shape  # (H, W, D)
            if len(shape) != 3:
                continue
            H, W, D = shape

            # Need room for neighbors, so shrink z-range by slice_radius
            z_start = int(0.1 * D) + self.slice_radius
            z_end   = int(0.9 * D) - self.slice_radius
            for z in range(z_start, z_end):
                self.slice_tuples.append((p, z))

        print(f"Found {len(self.volume_paths)} volumes.")
        print(f"Built {len(self.slice_tuples)} (volume, slice) pairs.")

        self._cache = OrderedDict()
        self._cache_size = 4

    
    def _load_volume(self, path):
        path = str(path)  # ensure hashable/consistent key
        if path in self._cache:
            # Move to end to mark as recently used
            vol = self._cache.pop(path)
            self._cache[path] = vol
            return vol

        img = nib.load(path)
        vol = np.asanyarray(img.dataobj).astype(np.float32)

        # Insert and evict oldest if needed
        self._cache[path] = vol
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)  # pop least-recently used

        return vol
    
    def _preprocess_slice(self, slice_2d: np.ndarray) -> torch.Tensor:
        """
        Normalize and resize a single 2D slice (numpy array) to tensor (1, H, W) in [-1, 1].
        """
        slice_2d = slice_2d.astype(np.float32)
        mask = slice_2d != 0
        if np.any(mask):
            mean = slice_2d[mask].mean()
            std = slice_2d[mask].std()
            std = std if std > 0 else 1.0
            slice_2d[mask] = (slice_2d[mask] - mean) / std

        slice_2d = np.clip(slice_2d, -5, 5)
        slice_2d = (slice_2d + 5) / 10.0  # -> [0, 1]

        slice_t = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        slice_t = F.interpolate(
            slice_t,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        slice_t = slice_t.squeeze(0)  # (1, H, W)
        slice_t = slice_t * 2.0 - 1.0  # -> [-1, 1]
        return slice_t

    def __getitem__(self, idx):
        path, z = self.slice_tuples[idx]

        vol = self._load_volume(path)
        slice_2d = vol[:, :, z]
        ...
        slice_t = slice_t * 2.0 - 1.0

        # z normalization:
        D = vol.shape[-1]
        z_pos = np.float32(z / (D - 1))

        return slice_t, z_pos

    def __len__(self):
        return len(self.slice_tuples)

    def __getitem__(self, idx):
        flair_path, z = self.slice_tuples[idx]

        # Load all four modalities for this subject
        vols = []
        for suffix in self.modalities:
            m_path = str(flair_path).replace(self.flair_suffix, suffix)
            vol = self._load_volume(m_path)
            vols.append(vol)  # each (H, W, D)

        D = vols[0].shape[-1]

        # Center slice: all modalities at z
        center_slices = []
        for vol in vols:
            slice_2d = vol[:, :, z]
            center_slices.append(self._preprocess_slice(slice_2d))  # (1, H, W)
        x_center = torch.cat(center_slices, dim=0)  # (4, H, W)

        # Context: neighbors z-1 and z+1 for all modalities
        context_slices = []
        for dz in range(-self.slice_radius, self.slice_radius + 1):
            if dz == 0:
                continue
            zz = z + dz
            for vol in vols:
                slice_2d = vol[:, :, zz]
                context_slices.append(self._preprocess_slice(slice_2d))  # (1, H, W)
        x_context = torch.cat(context_slices, dim=0)  # (8, H, W) if slice_radius=1

        # z normalization (center slice position)
        z_pos = np.float32(z / (D - 1))

        return x_center, x_context, z_pos
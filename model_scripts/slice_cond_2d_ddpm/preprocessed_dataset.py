# dataset.py (add below BraTSSliceDataset)
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PreprocessedBraTSSliceDataset(Dataset):
    """
    Loads preprocessed .pt files created by preprocess_brats.py.

    Each .pt file contains:
      - "slices": tensor (num_slices, 1, H, W) in [-1, 1]
      - "z_pos": tensor (num_slices,) in [0, 1]
    """

    def __init__(self, root_dir, image_size=128):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        # All preprocessed volume files
        self.volume_files = sorted(self.root_dir.rglob("*.pt"))
        if not self.volume_files:
            raise RuntimeError(f"No preprocessed .pt files found under {root_dir}")

        # Build global index: each dataset index -> (volume_idx, slice_idx)
        self.index = []
        self._num_slices_per_volume = []

        print(f"Found {len(self.volume_files)} preprocessed volumes.")

        # We need to know how many slices in each file
        for v_idx, path in enumerate(self.volume_files):
            data = torch.load(path, map_location="cpu")
            slices = data["slices"]
            num_slices = slices.shape[0]
            self._num_slices_per_volume.append(num_slices)
            for s_idx in range(num_slices):
                self.index.append((v_idx, s_idx))

        print(f"Total preprocessed slices: {len(self.index)}")

        # Optional small cache of last few loaded volumes
        self._cache = OrderedDict()
        self._cache_size = 4

    def __len__(self):
        return len(self.index)

    def _load_volume(self, path):
        path = str(path)
        if path in self._cache:
            vol = self._cache.pop(path)
            self._cache[path] = vol
            return vol

        vol = torch.load(path, map_location="cpu")  # dict with "slices" and "z_pos"
        self._cache[path] = vol
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return vol

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.index[idx]
        vol_path = self.volume_files[vol_idx]

        data = self._load_volume(vol_path)
        slices = data["slices"]     # (num_slices, 1, H, W)
        z_pos = data["z_pos"]       # (num_slices,)

        x = slices[slice_idx]       # (1, H, W), already in [-1, 1]
        z = z_pos[slice_idx].item() # float in [0, 1]

        return x, z
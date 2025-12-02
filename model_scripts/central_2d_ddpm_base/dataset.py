from torch.utils.data import Dataset
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

class BraTSSliceDataset(Dataset):
    """
    Simple dataset:
    - Finds all *flair.nii.gz files
    - Uses central 80% slices from each volume
    - Returns normalized 2D slice as tensor in [-1, 1], shape (1, H, W)
    """
    def __init__(self, root_dir, modality_suffix="_flair.nii.gz", image_size=128):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.modality_suffix = modality_suffix

        self.volume_paths = sorted(self.root_dir.rglob(f"*{modality_suffix}"))
        if not self.volume_paths:
            raise RuntimeError(f"No FLAIR files (*{modality_suffix}) found under {root_dir}")

        # Build (path, slice_index) list without loading data
        self.slice_tuples = []
        for p in self.volume_paths:
            img = nib.load(str(p))
            shape = img.shape  # (H, W, D)
            if len(shape) != 3:
                continue
            H, W, D = shape
            z_start = int(0.1 * D)
            z_end   = int(0.9 * D)
            for z in range(z_start, z_end):
                self.slice_tuples.append((p, z))

        print(f"Found {len(self.volume_paths)} volumes.")
        print(f"Built {len(self.slice_tuples)} (volume, slice) pairs.")

    def __len__(self):
        return len(self.slice_tuples)

    def __getitem__(self, idx):
        path, z = self.slice_tuples[idx]

        img = nib.load(str(path))
        vol = np.asanyarray(img.dataobj)  # shape: (H, W, D)
        slice_2d = vol[:, :, z].astype(np.float32)

        # Normalize using non-zero voxels
        mask = slice_2d != 0
        if np.any(mask):
            mean = slice_2d[mask].mean()
            std = slice_2d[mask].std()
            std = std if std > 0 else 1.0
            slice_2d[mask] = (slice_2d[mask] - mean) / std

        # Clip extreme values and map to [0,1]
        slice_2d = np.clip(slice_2d, -5, 5)
        slice_2d = (slice_2d + 5) / 10.0  # now in [0,1]

        # To tensor and resize to (image_size, image_size)
        slice_t = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        slice_t = F.interpolate(
            slice_t,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        slice_t = slice_t.squeeze(0)  # (1,H,W)

        # Map [0,1] -> [-1,1]
        slice_t = slice_t * 2.0 - 1.0

        z_pos = np.float32(z / 154.0)

        return slice_t, z_pos
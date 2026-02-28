import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

NUM_SLICES = 155

class MemmapDataset(Dataset):
    def __init__(self, path, shape=(1251, 155, 1, 128, 128)):
        self.data = np.memmap(path, mode='r', shape=shape, dtype=np.float32)
        self.num_slices = self.data.shape[1]
        self.num_datapoints_per_vol = 2 * (self.num_slices - 1)

    def __len__(self):
        return self.data.shape[0] * self.num_datapoints_per_vol # both directions, except for edges

    def __getitem__(self, idx):
        vol_idx = idx // self.num_datapoints_per_vol
        datapoint_idx = idx % self.num_datapoints_per_vol
        slice_idx = datapoint_idx // 2
        direction = (idx % 2) * 2 - 1 # 0,1 -> -1, 1
        if slice_idx == 0 and direction == -1:
            slice_idx = self.num_slices - 1
        neighbor_idx = slice_idx + direction
        neighbor_idx = max(0, min(neighbor_idx, NUM_SLICES - 1))
        return {
            "context": torch.from_numpy(self.data[vol_idx, slice_idx]),        # slice k
            "neighbor": torch.from_numpy(self.data[vol_idx, neighbor_idx]),    # slice k-1 or k+1 (clamped at boundaries)
            "direction": direction,         # -1 for above, +1 for below
            "slice_pos": slice_idx / self.num_slices - 1,        
        }

def get_debug_dataset():
    return MemmapDataset(Path(__file__).resolve().parents[2] / "data" / "preprocessed_all_debug.npy", (10, 155, 1, 128, 128))

def get_full_dataset():
    return MemmapDataset(Path(__file__).resolve().parents[2] / "data" / "preprocessed_all.npy", (1251, 155, 1, 128, 128))

def get_debug_masks_dataset():
    return MemmapDataset(Path(__file__).resolve().parents[2] / "data" / "preprocessed_all_masks_debug.npy", (10, 155, 1, 128, 128))

def get_full_masks_dataset():
    return MemmapDataset(Path(__file__).resolve().parents[2] / "data" / "preprocessed_all_masks.npy", (1251, 155, 1, 128, 128))
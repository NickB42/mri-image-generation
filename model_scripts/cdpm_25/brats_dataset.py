# brats_dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import nibabel as nib


_MOD_MAP = {
    "t1": ["_t1.nii.gz", "_t1n.nii.gz", "_0000.nii.gz"],
    "t1ce": ["_t1ce.nii.gz", "_t1gd.nii.gz", "_0001.nii.gz"],
    "t2": ["_t2.nii.gz", "_0002.nii.gz"],
    "flair": ["_flair.nii.gz", "_0003.nii.gz"],
}


def _find_cases(data_root: str, modality: str) -> List[Tuple[str, str]]:
    """
    Returns list of (case_id, filepath) for chosen modality.
    Works for:
      - subject folders with *_t1.nii.gz etc
      - flat nnU-Net imagesTr with *_0000.nii.gz etc
    """
    root = Path(data_root)
    if modality not in _MOD_MAP:
        raise ValueError(f"Unknown modality={modality}. Choose one of {list(_MOD_MAP.keys())}")

    suffixes = _MOD_MAP[modality]
    nii_files = list(root.rglob("*.nii.gz"))

    candidates = []
    for p in nii_files:
        name = p.name.lower()
        if any(name.endswith(suf) for suf in suffixes):
            candidates.append(p)

    # Case-id heuristics:
    # - If name is BraTS20_Training_001_t1.nii.gz -> case_id = BraTS20_Training_001
    # - If name is BRATS_001_0000.nii.gz -> case_id = BRATS_001
    cases: Dict[str, Path] = {}
    for p in candidates:
        base = p.name
        # strip known suffix
        case_id = base
        for suf in suffixes:
            if case_id.lower().endswith(suf):
                case_id = case_id[: -len(suf)]
                break
        case_id = case_id.rstrip("_-")
        cases[case_id] = p

    out = sorted([(cid, str(fp)) for cid, fp in cases.items()], key=lambda x: x[0])
    if len(out) == 0:
        raise RuntimeError(
            f"No NIfTI files found for modality='{modality}' under {data_root}. "
            f"Expected suffixes like: {suffixes}"
        )
    return out


def _load_nifti_as_DHW(path: str) -> torch.Tensor:
    """
    Loads NIfTI and returns tensor [D,H,W] (axial slices along D).
    Nibabel loads as (X,Y,Z) typically -> we convert to [Z,X,Y].
    """
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)  # float32
    # data shape: (H, W, D) or (X,Y,Z). We'll treat last axis as depth.
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={data.shape} for {path}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    # -> [D,H,W]
    data = np.transpose(data, (2, 0, 1))
    return torch.from_numpy(data)


def _normalize_01(vol: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize to [0,1]. Uses nonzero voxels if present (BraTS background is usually 0).
    Paper normalizes intensities to [0,1]. :contentReference[oaicite:6]{index=6}
    """
    v = vol
    mask = v != 0
    if mask.any():
        vmin = v[mask].min()
        vmax = v[mask].max()
    else:
        vmin = v.min()
        vmax = v.max()
    denom = (vmax - vmin).clamp_min(eps)
    v = (v - vmin) / denom
    return v.clamp(0.0, 1.0)


def _resize_to(vol_DHW: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    vol_DHW: [D,H,W] -> resize to target_shape (D,H,W) using trilinear.
    """
    D, H, W = vol_DHW.shape
    td, th, tw = target_shape
    x = vol_DHW[None, None, :, :, :]  # [1,1,D,H,W]
    x = F.interpolate(x, size=(td, th, tw), mode="trilinear", align_corners=False)
    return x[0, 0]


class BraTSVolumeDataset(Dataset):
    def __init__(
        self,
        data_root: str = "/dataset",
        modality: str = "t1",
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        resize: bool = True,
        limit: Optional[int] = None,
    ):
        self.data_root = data_root
        self.modality = modality
        self.target_shape = target_shape
        self.resize = resize

        self.cases = _find_cases(data_root, modality)
        if limit is not None:
            self.cases = self.cases[: int(limit)]

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int):
        case_id, path = self.cases[idx]
        vol = _load_nifti_as_DHW(path)          # [D,H,W]
        vol = _normalize_01(vol)               # [0,1]
        if self.resize:
            vol = _resize_to(vol, self.target_shape)  # [D,H,W]
        return vol, case_id

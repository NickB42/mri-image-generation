"""
eval_diffusion_braTS.py

Core evaluation metrics for 2D BraTS slice diffusion models:

1. FID + KID between real and generated slices
2. Dice + HD95 helpers for segmentation-based evaluation
3. Nearest-neighbour memorization score using cosine correlation

Usage (high-level):

- Make sure you can import your trained diffusion model and dataset, e.g.:

    from base_model import BraTSSliceDataset, diffusion, device, DATASET_ROOT

- Then in a separate script / notebook cell you can do:

    from eval_diffusion_braTS import (
        InceptionFeatureExtractor,
        compute_fid_kid_for_model,
        compute_memorization_stats,
        dice_per_label,
        hd95_per_label,
    )

    feature_extractor = InceptionFeatureExtractor(device)

    fid, kid = compute_fid_kid_for_model(
        diffusion=diffusion,
        dataset_root=DATASET_ROOT,
        feature_extractor=feature_extractor,
        device=device,
        num_real=2000,
        num_fake=2000,
        batch_size=32,
    )

    mem_stats = compute_memorization_stats(
        diffusion=diffusion,
        dataset_root=DATASET_ROOT,
        device=device,
        num_real=2000,
        num_fake=512,
        batch_size=32,
    )

Adapt paths / numbers as needed.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    # for FID and HD95
    from scipy import linalg
    from scipy.ndimage import distance_transform_edt
except ImportError as e:
    raise ImportError(
        "This script requires scipy. Install it with `pip install scipy`."
    ) from e

try:
    import torchvision
    from torchvision.models import Inception_V3_Weights
except ImportError as e:
    raise ImportError(
        "This script requires torchvision for FID/KID. Install it with `pip install torchvision`."
    ) from e


# ---------------------------------------------------------
# 1. Feature extractor for FID / KID (2D slices)
# ---------------------------------------------------------

class InceptionFeatureExtractor(nn.Module):
    """
    Wraps torchvision Inception-v3 to produce 2048-d features.

    - Expects input in [-1, 1], shape (B, 1, H, W) or (B, 3, H, W).
    - Repeats channel if needed and resizes to 299x299.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.inception = torchvision.models.inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=False,
        )
        # Replace final FC with identity to get 2048-d features
        self.inception.fc = nn.Identity()
        self.inception.to(device)
        self.inception.eval()
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor in [-1, 1], shape (B, C, H, W), C = 1 or 3
        returns: (B, 2048) features
        """
        x = x.to(self.device)

        if x.ndim != 4:
            raise ValueError(f"Expected (B, C, H, W), got shape {x.shape}")

        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)

        # Resize to 299x299
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        feats = self.inception(x)
        if isinstance(feats, tuple):
            feats = feats[0]
        return feats


# ---------------------------------------------------------
# 2. Generic helpers for FID & KID
# ---------------------------------------------------------

@torch.no_grad()
def _extract_features_from_loader(
    loader: DataLoader,
    feature_extractor: nn.Module,
    device: torch.device,
    max_images: Optional[int] = None,
) -> np.ndarray:
    feats_list: List[np.ndarray] = []
    n_total = 0

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(device)

        if max_images is not None and n_total >= max_images:
            break

        if max_images is not None and n_total + x.size(0) > max_images:
            x = x[: max_images - n_total]

        feats = feature_extractor(x)
        feats_list.append(feats.cpu().numpy())
        n_total += x.size(0)

    if not feats_list:
        raise RuntimeError("No features extracted; check your loader and max_images.")

    feats_all = np.concatenate(feats_list, axis=0)
    return feats_all


@torch.no_grad()
def _extract_features_from_generator(
    sample_fn: Callable[[int, torch.device], torch.Tensor],
    feature_extractor: nn.Module,
    device: torch.device,
    total_samples: int,
    batch_size: int = 32,
) -> np.ndarray:
    feats_list: List[np.ndarray] = []
    n_total = 0

    while n_total < total_samples:
        cur_bs = min(batch_size, total_samples - n_total)
        x = sample_fn(cur_bs, device=device)  # (B, C, H, W) in [-1, 1]
        feats = feature_extractor(x)
        feats_list.append(feats.cpu().numpy())
        n_total += cur_bs

    feats_all = np.concatenate(feats_list, axis=0)
    return feats_all


def _compute_fid_from_features(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
    eps: float = 1e-6,
) -> float:
    mu_real = np.mean(real_feats, axis=0)
    mu_fake = np.mean(fake_feats, axis=0)
    sigma_real = np.cov(real_feats, rowvar=False)
    sigma_fake = np.cov(fake_feats, rowvar=False)

    diff = mu_real - mu_fake

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))

    # Numerical error might give small imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return float(fid)


def _polynomial_kernel(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
    gamma: Optional[float] = None,
    coef0: float = 1.0,
) -> np.ndarray:
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * x.dot(y.T) + coef0) ** degree


def _compute_kid_from_features(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
    max_samples: int = 2000,
    degree: int = 3,
    gamma: Optional[float] = None,
    coef0: float = 1.0,
) -> float:
    """
    Computes an unbiased polynomial MMD^2 (KID) between two sets of features.

    We subsample at most `max_samples` from each set to keep it tractable.
    """
    rng = np.random.default_rng(42)

    n = real_feats.shape[0]
    m = fake_feats.shape[0]

    if n < 2 or m < 2:
        raise ValueError("Need at least 2 samples from each distribution for KID.")

    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        real_feats = real_feats[idx]
        n = max_samples

    if m > max_samples:
        idx = rng.choice(m, size=max_samples, replace=False)
        fake_feats = fake_feats[idx]
        m = max_samples

    k_xx = _polynomial_kernel(real_feats, real_feats, degree=degree, gamma=gamma, coef0=coef0)
    k_yy = _polynomial_kernel(fake_feats, fake_feats, degree=degree, gamma=gamma, coef0=coef0)
    k_xy = _polynomial_kernel(real_feats, fake_feats, degree=degree, gamma=gamma, coef0=coef0)

    # Unbiased estimator: remove diagonal
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)

    mmd2 = (
        k_xx.sum() / (n * (n - 1))
        + k_yy.sum() / (m * (m - 1))
        - 2.0 * k_xy.mean()
    )
    return float(mmd2)


# ---------------------------------------------------------
# 3. High-level FID/KID interface for your BraTS model
# ---------------------------------------------------------

def compute_fid_kid_for_model(
    diffusion: nn.Module,
    dataset_root: Path,
    feature_extractor: nn.Module,
    device: torch.device,
    num_real: int = 2000,
    num_fake: int = 2000,
    batch_size: int = 32,
    slice_dataset_class: Optional[type] = None,
) -> Tuple[float, float]:
    """
    Computes FID and KID between real BraTS FLAIR slices and samples from your diffusion model.

    Parameters
    ----------
    diffusion : nn.Module
        Your trained diffusion wrapper, expected to have a `.sample(batch_size)` method
        returning (B, 1, H, W) in [-1, 1].
    dataset_root : Path
        Root folder where BraTS NIfTI files live.
    feature_extractor : nn.Module
        Something like InceptionFeatureExtractor(device).
    device : torch.device
        CPU / CUDA / MPS device.
    num_real : int
        Number of real slices to use.
    num_fake : int
        Number of fake slices to generate.
    batch_size : int
        Batch size for data loading and sampling.
    slice_dataset_class : type, optional
        If None, we will try to import BraTSSliceDataset from the current namespace.
        Otherwise, pass your dataset class (must return image tensors in [-1, 1]).

    Returns
    -------
    (fid, kid)
    """
    # Lazy import to avoid circular dependencies
    if slice_dataset_class is None:
        try:
            from base_model import BraTSSliceDataset  # type: ignore
            slice_dataset_class = BraTSSliceDataset
        except ImportError:
            raise ImportError(
                "Could not import BraTSSliceDataset from base_model. "
                "Pass your dataset class via `slice_dataset_class=`."
            )

    # Build a dataset + loader of real slices
    real_dataset: Dataset = slice_dataset_class(root_dir=dataset_root, image_size=128)
    real_loader = DataLoader(
        real_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    def sample_fn(b: int, device: torch.device) -> torch.Tensor:
        diffusion.eval()
        with torch.no_grad():
            samples = diffusion.sample(batch_size=b)  # (B, 1, H, W)
        return samples

    # Extract features
    real_feats = _extract_features_from_loader(
        real_loader,
        feature_extractor=feature_extractor,
        device=device,
        max_images=num_real,
    )
    fake_feats = _extract_features_from_generator(
        sample_fn=sample_fn,
        feature_extractor=feature_extractor,
        device=device,
        total_samples=num_fake,
        batch_size=batch_size,
    )

    fid = _compute_fid_from_features(real_feats, fake_feats)
    kid = _compute_kid_from_features(real_feats, fake_feats)

    return fid, kid


# ---------------------------------------------------------
# 4. Memorization / nearest-neighbour statistics
# ---------------------------------------------------------

@torch.no_grad()
def compute_memorization_stats(
    diffusion: nn.Module,
    dataset_root: Path,
    device: torch.device,
    num_real: int = 2000,
    num_fake: int = 512,
    batch_size: int = 32,
    slice_dataset_class: Optional[type] = None,
) -> Dict[str, float]:
    """
    For each generated slice, compute its maximum cosine similarity to a pool of real training slices.

    Returns a dict with summary statistics and the raw per-sample maxima.

    NOTE:
    - To keep memory reasonable, we only use `num_real` slices as the pool
      and `num_fake` generated samples.
    """
    if slice_dataset_class is None:
        try:
            from base_model import BraTSSliceDataset  # type: ignore
            slice_dataset_class = BraTSSliceDataset
        except ImportError:
            raise ImportError(
                "Could not import BraTSSliceDataset from base_model. "
                "Pass your dataset class via `slice_dataset_class=`."
            )

    real_dataset: Dataset = slice_dataset_class(root_dir=dataset_root, image_size=128)
    real_loader = DataLoader(
        real_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Collect a pool of real slices
    real_images: List[torch.Tensor] = []
    n_real = 0
    for batch in real_loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        if n_real >= num_real:
            break
        if n_real + x.size(0) > num_real:
            x = x[: num_real - n_real]
        real_images.append(x)
        n_real += x.size(0)
    real_images_t = torch.cat(real_images, dim=0).to(device)  # (N, 1, H, W)

    # Flatten and normalize
    N = real_images_t.size(0)
    real_flat = real_images_t.view(N, -1)
    real_flat = F.normalize(real_flat, dim=1)

    # Generate fake slices
    diffusion.eval()
    fake_list: List[torch.Tensor] = []
    n_fake = 0
    while n_fake < num_fake:
        cur_bs = min(batch_size, num_fake - n_fake)
        samples = diffusion.sample(batch_size=cur_bs).to(device)
        fake_list.append(samples)
        n_fake += cur_bs

    fake_images_t = torch.cat(fake_list, dim=0)  # (M, 1, H, W)
    M = fake_images_t.size(0)
    fake_flat = fake_images_t.view(M, -1)
    fake_flat = F.normalize(fake_flat, dim=1)

    # Compute cosine similarity matrix in chunks along real axis
    max_sims: List[float] = []
    chunk_size = 512  # adjust if you run out of memory
    for i in range(M):
        f = fake_flat[i : i + 1]  # (1, D)
        max_sim_i = -1.0
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            sims = torch.mm(f, real_flat[start:end].T)  # (1, chunk)
            max_sim_i = max(max_sim_i, sims.max().item())
        max_sims.append(max_sim_i)

    max_sims_arr = np.array(max_sims, dtype=np.float32)
    stats: Dict[str, float] = {
        "mean_max_cosine": float(max_sims_arr.mean()),
        "std_max_cosine": float(max_sims_arr.std()),
        "p95_max_cosine": float(np.percentile(max_sims_arr, 95)),
        "p99_max_cosine": float(np.percentile(max_sims_arr, 99)),
    }
    # You can also inspect the full distribution if you want
    stats["all_max_cosine"] = max_sims_arr
    return stats


# ---------------------------------------------------------
# 5. Segmentation metrics: Dice & HD95
# ---------------------------------------------------------

def dice_per_label(
    pred: np.ndarray,
    target: np.ndarray,
    labels: Sequence[int],
    eps: float = 1e-5,
) -> Dict[int, float]:
    """
    Computes Dice score per label given integer segmentation masks.

    pred, target: np.ndarray of same shape, e.g. (H, W) or (D, H, W)
    labels: iterable of integer label ids (e.g. [1, 2, 4] for BraTS subregions)
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

    scores: Dict[int, float] = {}
    for lab in labels:
        pred_l = (pred == lab)
        targ_l = (target == lab)
        intersection = np.logical_and(pred_l, targ_l).sum()
        denom = pred_l.sum() + targ_l.sum()
        dice = (2.0 * intersection + eps) / (denom + eps)
        scores[int(lab)] = float(dice)
    return scores


def hd95_binary(
    pred_bin: np.ndarray,
    targ_bin: np.ndarray,
    voxelspacing: Optional[Sequence[float]] = None,
) -> float:
    """
    95th percentile Hausdorff distance between two binary masks.

    Uses distance_transform_edt; works for 2D or 3D masks.
    """
    pred_bin = np.asarray(pred_bin).astype(bool)
    targ_bin = np.asarray(targ_bin).astype(bool)

    if not pred_bin.any() and not targ_bin.any():
        return 0.0  # both empty => perfect match
    if not pred_bin.any() or not targ_bin.any():
        return float("inf")

    # Distance transform of complement
    dt_t = distance_transform_edt(~targ_bin, sampling=voxelspacing)
    dt_p = distance_transform_edt(~pred_bin, sampling=voxelspacing)

    # Distances from each mask to the other
    dist_p_to_t = dt_t[pred_bin]
    dist_t_to_p = dt_p[targ_bin]

    all_dists = np.concatenate([dist_p_to_t, dist_t_to_p])
    return float(np.percentile(all_dists, 95))


def hd95_per_label(
    pred: np.ndarray,
    target: np.ndarray,
    labels: Sequence[int],
    voxelspacing: Optional[Sequence[float]] = None,
) -> Dict[int, float]:
    """
    HD95 per label for integer segmentation masks.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

    scores: Dict[int, float] = {}
    for lab in labels:
        pred_l = (pred == lab)
        targ_l = (target == lab)
        scores[int(lab)] = hd95_binary(pred_l, targ_l, voxelspacing=voxelspacing)
    return scores


# ---------------------------------------------------------
# 6. Optional: quick CLI example
# ---------------------------------------------------------

if __name__ == "__main__":
    # This is just a usage example; adapt to your project structure.
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate BraTS diffusion model (2D slices).")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to BraTS dataset root.")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to diffusion checkpoint.")
    parser.add_argument("--num_real", type=int, default=2000)
    parser.add_argument("--num_fake", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # You need to provide these according to your training code:
    # from base_model import GaussianDiffusion, BraTSSliceDataset, device, TIMESTEPS, UNetModel, IMAGE_SIZE, DATASET_ROOT

    raise SystemExit(
        "This file is intended to be imported and used from your training notebook / script.\n"
        "See the docstring at the top for example usage."
    )
"""
eval_metrics_accelerate.py

Distributed evaluation for the memmap-backed 2.5D BraTS FLAIR DDPM using
free-running *full-volume* generation.

What this script does
---------------------
1. Loads the validation split and groups slices into full volumes.
2. Generates each validation volume sequentially from z=0..D-1 using the model's
   own previously generated slices as context.
3. Computes dataset-level slice-distribution metrics on the generated slices:
   - FID
   - KID (mean/std)
   - MiFID
4. Optionally computes aligned slice-wise metrics against the corresponding real
   validation volume:
   - SSIM
   - MS-SSIM
   - PSNR
   These are *not* pure generative metrics, but can still be useful as an
   auxiliary upper-bound style diagnostic.
5. Optionally computes a nearest-neighbor cosine diagnostic against the train
   split as a simple memorization check.

Important model-specific design choices
---------------------------------------
- Your model conditions on context slices, z-position, and fg_frac.
- For free-running generation, context is built autoregressively from the last
  generated slices.
- Because fg_frac is not available in a truly unconditional rollout, the default
  behavior is to use a training-set fg_frac profile as a function of normalized
  z-position. This is the cleanest default for your current checkpoint.
- You can switch this with --fg-frac-source if you want an ablation.

Example
-------
accelerate launch -m model_scripts.ddpm_25d_mm.eval_metrics_accelerate \
    --ckpt /path/to/ddpm_25d_mm_best.pt \
    --sampler sample_ddim \
    --sample-timesteps 50 \
    --fg-frac-source profile \
    --max-volumes 40 \
    --compute-paired \
    --compute-nn-cosine \
    --save-json eval_metrics.json \
    --save-sample-grid eval_grid.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from accelerate import Accelerator

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
    from torchmetrics.image import (
        StructuralSimilarityIndexMeasure,
        MultiScaleStructuralSimilarityIndexMeasure,
        PeakSignalNoiseRatio,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "torchmetrics image metrics are required. Install with: "
        "pip install 'torchmetrics[image]' torch-fidelity"
    ) from exc

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None

try:  # package mode
    from .mm_dataset import get_debug_dataset, get_train_dataset, get_val_dataset
    from .unet import UNet
    from .diffusion import GaussianDiffusion
except ImportError:  # script mode
    from mm_dataset import get_debug_dataset, get_train_dataset, get_val_dataset
    from unet import UNet
    from diffusion import GaussianDiffusion


# -----------------------------------------------------------------------------
# Defaults matched to train.py
# -----------------------------------------------------------------------------
EXPERIMENT_NAME = "ddpm_25d_mm_eval"
IMAGE_SIZE = 256
TIMESTEPS = 1000
CENTER_CHANNELS = 1
SLICE_RADIUS = 2
NUM_WORKERS = 8
BATCH_SIZE = 1  # volume batches should stay at 1


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate 2.5D DDPM metrics with Accelerate")
    p.add_argument("--ckpt", type=str, required=True, help="Path to saved checkpoint")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Volume batch size; keep at 1")
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--max-volumes", type=int, default=None, help="Evaluate at most this many complete validation volumes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug-fast", action="store_true")
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-use-ema", action="store_false", dest="use_ema")
    p.add_argument(
        "--data-range-mode",
        type=str,
        choices=["zero_one", "minus_one_one", "zscore_clip"],
        default="minus_one_one",
        help="Storage/output range before metric preprocessing",
    )
    p.add_argument("--zscore-clip-min", type=float, default=-5.0)
    p.add_argument("--zscore-clip-max", type=float, default=5.0)
    p.add_argument(
        "--compute-paired",
        action="store_true",
        help="Compute aligned slice-wise SSIM / MS-SSIM / PSNR against the corresponding real validation subject",
    )
    p.add_argument(
        "--compute-nn-cosine",
        action="store_true",
        help="Compute a simple nearest-neighbor cosine diagnostic against the train split",
    )
    p.add_argument("--nn-bank-size", type=int, default=2000)
    p.add_argument("--nn-resize", type=int, default=64)
    p.add_argument("--kid-subsets", type=int, default=20)
    p.add_argument("--kid-subset-size", type=int, default=100)
    p.add_argument("--save-json", type=str, default="eval_metrics.json")
    p.add_argument("--progress-ckpt", type=str, default="", help="Path to resumable progress checkpoint (.pt). Default: <save-json>.progress.pt")
    p.add_argument("--resume-progress", action="store_true", help="Resume from --progress-ckpt if it exists")
    p.add_argument("--save-progress-every-volumes", type=int, default=5, help="Save progress checkpoint every N processed volumes on main process; 0 disables")
    p.add_argument("--log-every-volumes", type=int, default=5, help="Print progress/ETA every N processed volumes on main process")
    p.add_argument("--keep-progress-ckpt", action="store_true", help="Keep progress checkpoint after successful completion")
    p.add_argument("--save-sample-grid", type=str, default="")
    p.add_argument("--mlflow", action="store_true")
    p.add_argument(
        "--final-sync",
        type=str,
        default="none",
        choices=["none", "barrier"],
        help=(
            "End-of-run synchronization strategy across ranks. "
            "Use 'none' (default) to avoid potential tail barrier deadlocks; "
            "use 'barrier' to enforce a final global sync."
        ),
    )
    p.add_argument(
        "--sampler",
        type=str,
        default="sample_ddim",
        choices=["sample", "sample_ddim", "p_sample_loop", "ddim_sample_loop"],
        help="Which diffusion sampler to use for generation",
    )
    p.add_argument("--sample-timesteps", type=int, default=50, help="Used for DDIM sampling")
    p.add_argument("--ddim-eta", type=float, default=0.0, help="Used for DDIM sampling")
    p.add_argument(
        "--fg-frac-source",
        type=str,
        default="profile",
        choices=["profile", "real", "zero", "constant"],
        help=(
            "How to provide fg_frac during free-running rollout: "
            "profile=train-set mean as function of z, real=use per-subject real fg_frac (ablation), "
            "zero=always 0, constant=use --fg-frac-constant"
        ),
    )
    p.add_argument("--fg-frac-constant", type=float, default=0.05)
    p.add_argument("--fg-profile-bins", type=int, default=128)
    p.add_argument("--fg-profile-smooth", type=int, default=7, help="Moving-average width in bins; 1 disables smoothing")
    p.add_argument(
        "--fg-profile-max-slices",
        type=int,
        default=None,
        help="Optional cap when estimating the fg_frac(z) profile from the train split",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def maybe_subset(ds, max_items: Optional[int]):
    if max_items is None or len(ds) <= max_items:
        return ds
    return Subset(ds, list(range(max_items)))



def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        drop_last=False,
    )



def unwrap_subset(dataset):
    if isinstance(dataset, Subset):
        return dataset.dataset, list(dataset.indices)
    return dataset, list(range(len(dataset)))


# -----------------------------------------------------------------------------
# Volume grouping
# -----------------------------------------------------------------------------
def build_volume_specs(dataset) -> List[Dict[str, Any]]:
    base_ds, active_indices = unwrap_subset(dataset)
    if not hasattr(base_ds, "slice_tuples"):
        raise RuntimeError("Dataset does not expose slice_tuples; cannot rebuild full volumes.")

    groups: "OrderedDict[Tuple[int, int], List[Tuple[int, int]]]" = OrderedDict()
    for ds_idx in active_indices:
        vol_id, z, d = base_ds.slice_tuples[ds_idx]
        key = (int(vol_id), int(d))
        groups.setdefault(key, []).append((int(z), int(ds_idx)))

    specs: List[Dict[str, Any]] = []
    for (vol_id, declared_depth), z_and_idx in groups.items():
        z_and_idx.sort(key=lambda x: x[0])
        z_values = [z for z, _ in z_and_idx]
        ds_indices = [idx for _, idx in z_and_idx]
        if len(ds_indices) != len(z_values):
            continue
        specs.append(
            {
                "volume_id": vol_id,
                "declared_depth": declared_depth,
                "actual_depth": len(ds_indices),
                "z_values": z_values,
                "dataset_indices": ds_indices,
            }
        )
    return specs



def maybe_limit_volume_specs(specs: List[Dict[str, Any]], max_volumes: Optional[int]) -> List[Dict[str, Any]]:
    if max_volumes is None or len(specs) <= max_volumes:
        return specs
    return specs[:max_volumes]


class VolumePlanDataset(Dataset):
    def __init__(self, base_dataset, volume_specs: List[Dict[str, Any]]):
        self.base_dataset = base_dataset
        self.volume_specs = volume_specs

    def __len__(self) -> int:
        return len(self.volume_specs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        spec = self.volume_specs[index]
        real_slices: List[torch.Tensor] = []
        z_positions: List[float] = []
        fg_fracs: List[float] = []

        d = int(spec["declared_depth"])
        for ds_idx in spec["dataset_indices"]:
            x_center, _x_context, z_pos, fg_frac = self.base_dataset[ds_idx]
            if x_center.ndim != 3:
                raise ValueError(f"Expected x_center [C,H,W], got {tuple(x_center.shape)}")
            real_slices.append(x_center.float())
            z_positions.append(float(z_pos))
            fg_fracs.append(float(fg_frac))

        real_stack = torch.stack(real_slices, dim=0)  # [D,1,H,W]
        z_pos_t = torch.tensor(z_positions, dtype=torch.float32)
        fg_frac_t = torch.tensor(fg_fracs, dtype=torch.float32)
        z_index_t = torch.tensor(spec["z_values"], dtype=torch.long)

        return {
            "volume_id": torch.tensor(int(spec["volume_id"]), dtype=torch.long),
            "declared_depth": torch.tensor(d, dtype=torch.long),
            "actual_depth": torch.tensor(real_stack.shape[0], dtype=torch.long),
            "real_slices": real_stack,
            "z_pos": z_pos_t,
            "z_index": z_index_t,
            "fg_frac_real": fg_frac_t,
        }


# -----------------------------------------------------------------------------
# Model construction / checkpoint loading
# -----------------------------------------------------------------------------
def build_diffusion() -> GaussianDiffusion:
    in_channels = CENTER_CHANNELS + CENTER_CHANNELS * SLICE_RADIUS
    out_channels = CENTER_CHANNELS

    unet = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256,
    )

    diffusion = GaussianDiffusion(
        model=unet,
        image_size=IMAGE_SIZE,
        channels=out_channels,
        timesteps=TIMESTEPS,
        schedule="cosine",
    )
    return diffusion



def _try_load_unet_state(unet: torch.nn.Module, state: Dict[str, Any]) -> bool:
    if not isinstance(state, dict):
        return False

    candidates: List[Dict[str, Any]] = [state]
    for key in ("shadow", "ema_model", "model", "state_dict", "ema_state_dict"):
        sub = state.get(key)
        if isinstance(sub, dict):
            candidates.append(sub)

    for cand in candidates:
        try:
            unet.load_state_dict(cand, strict=True)
            return True
        except Exception:
            pass
    return False



def load_checkpoint_into_diffusion(diffusion: GaussianDiffusion, ckpt_path: str, use_ema: bool) -> str:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if use_ema and isinstance(ckpt, dict) and "ema_unet" in ckpt:
        if _try_load_unet_state(diffusion.model, ckpt["ema_unet"]):
            return "ema_unet"

    if isinstance(ckpt, dict) and "diffusion" in ckpt:
        try:
            diffusion.load_state_dict(ckpt["diffusion"], strict=True)
            return "diffusion"
        except Exception:
            pass

    if isinstance(ckpt, dict) and _try_load_unet_state(diffusion.model, ckpt):
        return "raw_unet"

    raise RuntimeError(
        "Could not load checkpoint. I tried ckpt['ema_unet'] -> UNet, ckpt['diffusion'] -> GaussianDiffusion, and raw ckpt -> UNet."
    )


# -----------------------------------------------------------------------------
# Metric preprocessing
# -----------------------------------------------------------------------------
def to_zero_one(
    x: torch.Tensor,
    mode: str,
    zscore_clip_min: float,
    zscore_clip_max: float,
) -> torch.Tensor:
    x = x.float()
    if mode == "zero_one":
        return x.clamp(0.0, 1.0)
    if mode == "minus_one_one":
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    if mode == "zscore_clip":
        x = x.clamp(zscore_clip_min, zscore_clip_max)
        x = (x - zscore_clip_min) / max(1e-8, (zscore_clip_max - zscore_clip_min))
        return x.clamp(0.0, 1.0)
    raise ValueError(f"Unsupported mode: {mode}")



def to_rgb_for_inception(x01: torch.Tensor) -> torch.Tensor:
    if x01.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x01.shape)}")
    if x01.shape[1] == 1:
        x01 = x01.repeat(1, 3, 1, 1)
    elif x01.shape[1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {x01.shape[1]}")
    return x01.contiguous()



def lowres_l2_features(x01: torch.Tensor, out_size: int) -> torch.Tensor:
    x = F.interpolate(x01, size=(out_size, out_size), mode="bilinear", align_corners=False)
    x = x.flatten(1)
    return F.normalize(x, dim=1, p=2)


# -----------------------------------------------------------------------------
# fg_frac profile estimation
# -----------------------------------------------------------------------------
def _moving_average_1d(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return x
    k = int(k)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    kernel = torch.ones(1, 1, k, dtype=x.dtype) / float(k)
    x3 = x.view(1, 1, -1)
    xpad = F.pad(x3, (pad, pad), mode="replicate")
    return F.conv1d(xpad, kernel).view(-1)



def estimate_fg_frac_profile(
    dataset,
    *,
    bins: int,
    num_workers: int,
    max_slices: Optional[int],
    smooth_width: int,
) -> torch.Tensor:
    ds = maybe_subset(dataset, max_slices)
    loader = make_loader(ds, batch_size=128, num_workers=num_workers, shuffle=False)

    sums = torch.zeros(bins, dtype=torch.float64)
    counts = torch.zeros(bins, dtype=torch.float64)

    for _x_center, _x_context, z_pos, fg_frac in loader:
        z = z_pos.float().view(-1).clamp(0.0, 1.0)
        fg = fg_frac.float().view(-1)
        idx = torch.clamp(torch.round(z * (bins - 1)).long(), 0, bins - 1)
        ones = torch.ones_like(fg, dtype=torch.float64)
        sums.index_add_(0, idx.cpu(), fg.double().cpu())
        counts.index_add_(0, idx.cpu(), ones.cpu())

    valid = counts > 0
    if not valid.any():
        raise RuntimeError("Could not estimate fg_frac profile; no valid slices were seen.")

    profile = torch.zeros_like(sums)
    profile[valid] = sums[valid] / counts[valid]

    if not valid.all():
        xp = torch.arange(bins, dtype=torch.float64)
        x_valid = xp[valid]
        y_valid = profile[valid]
        if len(x_valid) == 1:
            profile[:] = y_valid[0]
        else:
            interp = np.interp(xp.numpy(), x_valid.numpy(), y_valid.numpy())
            profile = torch.from_numpy(interp).to(profile.dtype)

    profile = _moving_average_1d(profile.float(), smooth_width).clamp(min=0.0)
    return profile



def fg_from_profile(profile: torch.Tensor, z_pos: torch.Tensor) -> torch.Tensor:
    if profile.ndim != 1:
        raise ValueError("profile must be 1D")
    z = z_pos.float().clamp(0.0, 1.0)
    x = z * (len(profile) - 1)
    x0 = torch.floor(x).long().clamp(0, len(profile) - 1)
    x1 = torch.ceil(x).long().clamp(0, len(profile) - 1)
    w = (x - x0.float()).clamp(0.0, 1.0)
    return (1.0 - w) * profile[x0] + w * profile[x1]


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------
@torch.no_grad()
def sample_one_slice(
    diffusion: GaussianDiffusion,
    *,
    context: torch.Tensor,
    z_pos: torch.Tensor,
    fg_frac: Optional[torch.Tensor],
    sampler_name: str,
    sample_timesteps: int,
    ddim_eta: float,
) -> torch.Tensor:
    b = 1
    shape = (b, CENTER_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

    if sampler_name == "sample":
        out = diffusion.sample(batch_size=b, z_pos=z_pos, fg_frac=fg_frac, context=context)
    elif sampler_name == "sample_ddim":
        out = diffusion.sample_ddim(
            batch_size=b,
            z_pos=z_pos,
            fg_frac=fg_frac,
            context=context,
            sample_timesteps=sample_timesteps,
            eta=ddim_eta,
        )
    elif sampler_name == "p_sample_loop":
        out = diffusion.p_sample_loop(shape, z_pos=z_pos, fg_frac=fg_frac, context=context)
    elif sampler_name == "ddim_sample_loop":
        out = diffusion.ddim_sample_loop(
            shape,
            z_pos=z_pos,
            fg_frac=fg_frac,
            context=context,
            sample_timesteps=sample_timesteps,
            eta=ddim_eta,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported sampler: {sampler_name}")

    if out.ndim == 3:
        out = out.unsqueeze(1)
    if out.shape != shape:
        raise ValueError(f"Sampler returned shape {tuple(out.shape)}, expected {shape}")
    return out.clamp(-1.0, 1.0)


@torch.no_grad()
def generate_volume_autoregressive(
    diffusion: GaussianDiffusion,
    *,
    z_pos_seq: torch.Tensor,
    fg_frac_seq: Optional[torch.Tensor],
    sampler_name: str,
    sample_timesteps: int,
    ddim_eta: float,
    device: torch.device,
) -> torch.Tensor:
    history: deque[torch.Tensor] = deque(maxlen=SLICE_RADIUS)
    for _ in range(SLICE_RADIUS):
        history.append(torch.zeros(1, CENTER_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device))

    generated: List[torch.Tensor] = []
    z_pos_seq = z_pos_seq.to(device).float().view(-1)
    fg_frac_seq = None if fg_frac_seq is None else fg_frac_seq.to(device).float().view(-1)

    for i in range(len(z_pos_seq)):
        context = torch.cat(list(history), dim=1) if SLICE_RADIUS > 0 else None
        z_pos = z_pos_seq[i : i + 1]
        fg_frac = None if fg_frac_seq is None else fg_frac_seq[i : i + 1]

        fake = sample_one_slice(
            diffusion,
            context=context,
            z_pos=z_pos,
            fg_frac=fg_frac,
            sampler_name=sampler_name,
            sample_timesteps=sample_timesteps,
            ddim_eta=ddim_eta,
        )
        generated.append(fake[0].detach())
        history.append(fake.detach())

    return torch.stack(generated, dim=0)


# -----------------------------------------------------------------------------
# Optional memorization bank
# -----------------------------------------------------------------------------
@torch.no_grad()
def build_train_nn_bank(
    dataset,
    *,
    bank_size: int,
    batch_size: int,
    num_workers: int,
    data_range_mode: str,
    zscore_clip_min: float,
    zscore_clip_max: float,
    out_size: int,
) -> torch.Tensor:
    loader = make_loader(maybe_subset(dataset, bank_size), batch_size=batch_size, num_workers=num_workers, shuffle=False)
    bank: List[torch.Tensor] = []
    total = 0
    for x_center, _x_context, _z_pos, _fg_frac in loader:
        x01 = to_zero_one(x_center, data_range_mode, zscore_clip_min, zscore_clip_max)
        bank.append(lowres_l2_features(x01, out_size).cpu())
        total += x_center.shape[0]
        if total >= bank_size:
            break
    if not bank:
        raise RuntimeError("Nearest-neighbor bank is empty")
    return torch.cat(bank, dim=0)[:bank_size]


# -----------------------------------------------------------------------------
# Optional sample grid
# -----------------------------------------------------------------------------
def save_sample_grid(path: str, reals: torch.Tensor, fakes: torch.Tensor, n_rows: int = 8) -> None:
    import matplotlib.pyplot as plt

    n = min(n_rows, reals.shape[0], fakes.shape[0])
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = np.array([axes])
    for i in range(n):
        axes[i, 0].imshow(reals[i, 0].cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 0].set_title("real")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(fakes[i, 0].cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 1].set_title("fake")
        axes[i, 1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _resolve_progress_ckpt_path(progress_ckpt: str, save_json: str) -> Path:
    if progress_ckpt:
        return Path(progress_ckpt)
    save_path = Path(save_json)
    if save_path.suffix:
        return save_path.with_suffix(save_path.suffix + ".progress.pt")
    return Path(str(save_path) + ".progress.pt")


def _infer_job_id() -> str:
    val = os.environ.get("SLURM_JOB_ID")
    if val:
        return str(val)
    return "local"


def _route_output_to_job_dir(path_str: str, *, job_id: str, default_name: str) -> str:
    path = Path(path_str).expanduser() if path_str else Path(default_name)
    filename = path.name if path.name else default_name
    return str(Path("eval_out") / str(job_id) / filename)


def _save_progress_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _build_progress_payload(
    *,
    args: argparse.Namespace,
    load_mode: str,
    num_eval_volumes: int,
    num_eval_slices: int,
    completed_steps: int,
    generated_slice_count: int,
    processed_volume_count: int,
    preview_real_list: List[torch.Tensor],
    preview_fake_list: List[torch.Tensor],
    nn_scores: List[torch.Tensor],
    fid,
    kid,
    mifid,
    ssim,
    msssim,
    psnr,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "version": 1,
        "timestamp": time.time(),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "load_mode": load_mode,
        "num_eval_volumes": int(num_eval_volumes),
        "num_eval_slices": int(num_eval_slices),
        "completed_steps": int(completed_steps),
        "generated_slice_count": int(generated_slice_count),
        "processed_volume_count": int(processed_volume_count),
        "preview_real_list": [t.detach().cpu() for t in preview_real_list],
        "preview_fake_list": [t.detach().cpu() for t in preview_fake_list],
        "nn_scores": [t.detach().cpu() for t in nn_scores],
        "fid_state": fid.state_dict() if fid is not None else None,
        "kid_state": kid.state_dict() if kid is not None else None,
        "mifid_state": mifid.state_dict() if mifid is not None else None,
        "ssim_state": ssim.state_dict() if ssim is not None else None,
        "msssim_state": msssim.state_dict() if msssim is not None else None,
        "psnr_state": psnr.state_dict() if psnr is not None else None,
        "config": {
            "sampler": args.sampler,
            "sample_timesteps": int(args.sample_timesteps),
            "ddim_eta": float(args.ddim_eta),
            "data_range_mode": args.data_range_mode,
            "compute_paired": bool(args.compute_paired),
            "compute_nn_cosine": bool(args.compute_nn_cosine),
            "fg_frac_source": args.fg_frac_source,
            "fg_frac_constant": float(args.fg_frac_constant),
            "max_volumes": args.max_volumes,
            "seed": int(args.seed),
        },
    }
    return payload


def _safe_load_metric_state(metric, state: Optional[Dict[str, Any]]) -> None:
    if metric is None or state is None:
        return
    metric.load_state_dict(state)


def _disable_metric_dist_sync(metric) -> None:
    """Disable torchmetrics distributed sync for main-process-only metric usage."""
    if metric is None:
        return
    if hasattr(metric, "sync_on_compute"):
        metric.sync_on_compute = False
    if hasattr(metric, "dist_sync_on_step"):
        metric.dist_sync_on_step = False
    # Force local-only behavior even when a distributed process group exists.
    if hasattr(metric, "distributed_available_fn"):
        metric.distributed_available_fn = lambda: False
    if hasattr(metric, "process_group"):
        metric.process_group = None


def _is_metric_state_resumable(metric_state: Optional[Dict[str, Any]]) -> bool:
    """
    Check whether a serialized torchmetrics state contains accumulators needed
    to resume compute() without re-running updates.

    Some checkpoints only contain module weights/buffers (legacy behavior),
    which is not enough to resume FID/KID/MiFID statistics.
    """
    if not isinstance(metric_state, dict):
        return False
    expected_any_keys = {
        "real_features",
        "fake_features",
        "real_features_sum",
        "fake_features_sum",
        "real_features_num_samples",
        "fake_features_num_samples",
    }
    return any(k in metric_state for k in expected_any_keys)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process

    stop_requested = {"flag": False}

    def _mark_stop_requested(signum, _frame) -> None:
        stop_requested["flag"] = True
        if is_main:
            accelerator.print(f"Received signal {signum}. Will save progress and exit cleanly after current step.")

    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _mark_stop_requested)
        except Exception:
            pass

    if args.batch_size != 1 and is_main:
        accelerator.print("Warning: --batch-size > 1 is not recommended for variable-depth volume generation. Proceeding anyway.")

    # Route JSON/PNG outputs under eval_out/<job_id>/
    job_id = _infer_job_id()
    args.save_json = _route_output_to_job_dir(args.save_json, job_id=job_id, default_name="eval_metrics.json")
    if args.save_sample_grid:
        args.save_sample_grid = _route_output_to_job_dir(
            args.save_sample_grid,
            job_id=job_id,
            default_name="eval_grid.png",
        )
    if is_main:
        accelerator.print(f"Output directory: {Path(args.save_json).parent}")

    # Base datasets
    if args.debug_fast:
        val_slice_dataset = get_debug_dataset(slice_radius=SLICE_RADIUS)
        train_slice_dataset = get_debug_dataset(slice_radius=SLICE_RADIUS)
    else:
        val_slice_dataset = get_val_dataset(slice_radius=SLICE_RADIUS)
        train_slice_dataset = get_train_dataset(slice_radius=SLICE_RADIUS)

    val_base, _ = unwrap_subset(val_slice_dataset)

    # Build full-volume plans from the validation split
    val_specs = build_volume_specs(val_slice_dataset)
    val_specs = maybe_limit_volume_specs(val_specs, args.max_volumes)
    val_volume_dataset = VolumePlanDataset(val_base, val_specs)
    val_volume_loader = make_loader(
        val_volume_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # fg_frac profile for free-running rollout
    fg_profile = None
    if args.fg_frac_source == "profile":
        if is_main:
            accelerator.print("Estimating fg_frac(z) profile from the train split...")
        fg_profile = estimate_fg_frac_profile(
            train_slice_dataset,
            bins=args.fg_profile_bins,
            num_workers=args.num_workers,
            max_slices=args.fg_profile_max_slices,
            smooth_width=args.fg_profile_smooth,
        )

    # Model
    diffusion = build_diffusion()
    load_mode = load_checkpoint_into_diffusion(diffusion, args.ckpt, use_ema=args.use_ema)
    diffusion.eval()

    diffusion, val_volume_loader = accelerator.prepare(diffusion, val_volume_loader)
    base_diffusion = accelerator.unwrap_model(diffusion)

    num_eval_volumes = len(val_volume_dataset)
    num_eval_slices = int(sum(spec["actual_depth"] for spec in val_specs))

    if is_main:
        accelerator.print(f"Checkpoint load mode: {load_mode}")
        accelerator.print(f"Validation volumes: {num_eval_volumes}")
        accelerator.print(f"Validation slices:  {num_eval_slices}")

    progress_ckpt_path = _resolve_progress_ckpt_path(args.progress_ckpt, args.save_json)

    # Main-process metrics only; gathered tensors are routed here.
    if is_main:
        fid = FrechetInceptionDistance(
            feature=2048,
            normalize=True,
            reset_real_features=True,
            sync_on_compute=False,
        )

        kid = KernelInceptionDistance(
            feature=2048,
            normalize=True,
            subsets=args.kid_subsets,
            subset_size=args.kid_subset_size,
            reset_real_features=True,
            sync_on_compute=False,
        )

        mifid = MemorizationInformedFrechetInceptionDistance(
            feature=2048,
            normalize=True,
            reset_real_features=True,
            sync_on_compute=False,
        )

        ssim = (
            StructuralSimilarityIndexMeasure(data_range=1.0, sync_on_compute=False)
            if args.compute_paired
            else None
        )
        msssim = (
            MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, sync_on_compute=False)
            if args.compute_paired
            else None
        )
        psnr = PeakSignalNoiseRatio(data_range=1.0, sync_on_compute=False) if args.compute_paired else None

        # Metrics are updated only on main process after gathered tensors.
        # Disable internal distributed sync to avoid collective calls at compute().
        for _metric in (fid, kid, mifid, ssim, msssim, psnr):
            _disable_metric_dist_sync(_metric)

        nn_bank = None
        if args.compute_nn_cosine:
            accelerator.print("Building train-set NN bank on main process...")
            nn_bank = build_train_nn_bank(
                train_slice_dataset,
                bank_size=args.nn_bank_size,
                batch_size=32,
                num_workers=args.num_workers,
                data_range_mode=args.data_range_mode,
                zscore_clip_min=args.zscore_clip_min,
                zscore_clip_max=args.zscore_clip_max,
                out_size=args.nn_resize,
            )
            accelerator.print(f"NN bank shape: {tuple(nn_bank.shape)}")
        nn_scores: List[torch.Tensor] = []
    else:
        fid = kid = mifid = ssim = msssim = psnr = None
        nn_bank = None
        nn_scores = []

    preview_real_list: List[torch.Tensor] = []
    preview_fake_list: List[torch.Tensor] = []
    generated_slice_count = 0
    processed_volume_count = 0
    completed_steps = 0
    last_logged_volumes = 0
    last_saved_volumes = 0
    terminated_early = False

    if is_main and args.resume_progress and progress_ckpt_path.exists():
        accelerator.print(f"Loading progress checkpoint: {progress_ckpt_path}")
        progress_state = torch.load(progress_ckpt_path, map_location="cpu")
        completed_steps = int(progress_state.get("completed_steps", 0))
        generated_slice_count = int(progress_state.get("generated_slice_count", 0))
        processed_volume_count = int(progress_state.get("processed_volume_count", 0))
        preview_real_list = [t.float() for t in progress_state.get("preview_real_list", [])]
        preview_fake_list = [t.float() for t in progress_state.get("preview_fake_list", [])]
        nn_scores = [t.float() for t in progress_state.get("nn_scores", [])]
        fid_state = progress_state.get("fid_state")
        if _is_metric_state_resumable(fid_state):
            _safe_load_metric_state(fid, fid_state)
            _safe_load_metric_state(kid, progress_state.get("kid_state"))
            _safe_load_metric_state(mifid, progress_state.get("mifid_state"))
            _safe_load_metric_state(ssim, progress_state.get("ssim_state"))
            _safe_load_metric_state(msssim, progress_state.get("msssim_state"))
            _safe_load_metric_state(psnr, progress_state.get("psnr_state"))
            last_logged_volumes = min(num_eval_volumes, completed_steps * accelerator.num_processes)
            last_saved_volumes = last_logged_volumes
            accelerator.print(
                f"Resumed at loader step {completed_steps} (~{last_logged_volumes}/{num_eval_volumes} volumes)."
            )
        else:
            accelerator.print(
                "Progress checkpoint does not contain resumable metric accumulators. "
                "Falling back to full metric recomputation from step 0."
            )
            completed_steps = 0
            generated_slice_count = 0
            processed_volume_count = 0
            preview_real_list = []
            preview_fake_list = []
            nn_scores = []
            last_logged_volumes = 0
            last_saved_volumes = 0

    resume_steps_tensor = torch.tensor([completed_steps if is_main else 0], device=device, dtype=torch.long)
    completed_steps = int(accelerator.gather(resume_steps_tensor).max().item())

    start = time.time()

    for step, batch in enumerate(val_volume_loader, start=1):
        if step <= completed_steps:
            continue

        # batch_size should be 1 here to keep variable-depth handling simple and robust.
        real_slices = batch["real_slices"][0].to(device, non_blocking=True)  # [D,1,H,W]
        z_pos_seq = batch["z_pos"][0].to(device, non_blocking=True).float()  # [D]
        fg_frac_real = batch["fg_frac_real"][0].to(device, non_blocking=True).float()  # [D]
        actual_depth = int(batch["actual_depth"][0].item())
        declared_depth = int(batch["declared_depth"][0].item())

        if actual_depth != declared_depth and is_main:
            accelerator.print(
                f"Warning: volume with declared depth {declared_depth} had {actual_depth} loaded slices. Using actual depth."
            )

        if args.fg_frac_source == "profile":
            fg_frac_seq = fg_from_profile(fg_profile.to(device), z_pos_seq)
        elif args.fg_frac_source == "real":
            fg_frac_seq = fg_frac_real
        elif args.fg_frac_source == "zero":
            fg_frac_seq = torch.zeros_like(z_pos_seq)
        elif args.fg_frac_source == "constant":
            fg_frac_seq = torch.full_like(z_pos_seq, float(args.fg_frac_constant))
        else:  # pragma: no cover
            raise ValueError(f"Unsupported fg_frac_source: {args.fg_frac_source}")

        with torch.no_grad():
            fake_slices = generate_volume_autoregressive(
                base_diffusion,
                z_pos_seq=z_pos_seq,
                fg_frac_seq=fg_frac_seq,
                sampler_name=args.sampler,
                sample_timesteps=args.sample_timesteps,
                ddim_eta=args.ddim_eta,
                device=device,
            )

        real01 = to_zero_one(real_slices, args.data_range_mode, args.zscore_clip_min, args.zscore_clip_max)
        fake01 = to_zero_one(fake_slices, args.data_range_mode, args.zscore_clip_min, args.zscore_clip_max)

        valid = torch.ones(real01.shape[0], device=device, dtype=torch.long)
        real01_pad = accelerator.pad_across_processes(real01, dim=0, pad_index=0)
        fake01_pad = accelerator.pad_across_processes(fake01, dim=0, pad_index=0)
        valid_pad = accelerator.pad_across_processes(valid, dim=0, pad_index=0)

        all_real01, all_fake01, all_valid = accelerator.gather_for_metrics((real01_pad, fake01_pad, valid_pad))

        if is_main:
            keep = all_valid.bool().cpu()
            all_real01 = all_real01.cpu()[keep]
            all_fake01 = all_fake01.cpu()[keep]

            real_rgb = to_rgb_for_inception(all_real01)
            fake_rgb = to_rgb_for_inception(all_fake01)
            fid.update(real_rgb, real=True)
            fid.update(fake_rgb, real=False)
            kid.update(real_rgb, real=True)
            kid.update(fake_rgb, real=False)
            mifid.update(real_rgb, real=True)
            mifid.update(fake_rgb, real=False)

            if args.compute_paired:
                ssim.update(all_fake01, all_real01)
                msssim.update(all_fake01, all_real01)
                psnr.update(all_fake01, all_real01)

            if nn_bank is not None:
                fake_feat = lowres_l2_features(all_fake01, args.nn_resize)
                sims = fake_feat @ nn_bank.T
                nn_scores.append(sims.max(dim=1).values)

            if len(preview_real_list) < 8:
                take = min(8 - len(preview_real_list), all_real01.shape[0])
                preview_real_list.extend(list(all_real01[:take]))
                preview_fake_list.extend(list(all_fake01[:take]))

            generated_slice_count += int(all_real01.shape[0])
            processed_volume_count += int(accelerator.num_processes)

            done_volumes = min(num_eval_volumes, step * accelerator.num_processes)
            should_log = (
                args.log_every_volumes > 0
                and (done_volumes - last_logged_volumes >= args.log_every_volumes or done_volumes == num_eval_volumes)
            )
            if should_log:
                elapsed_so_far = max(1e-6, time.time() - start)
                rate = done_volumes / elapsed_so_far
                eta_sec = max(0.0, (num_eval_volumes - done_volumes) / max(1e-6, rate))
                eta_min = eta_sec / 60.0
                accelerator.print(
                    f"Progress: {done_volumes}/{num_eval_volumes} volumes "
                    f"({100.0 * done_volumes / max(1, num_eval_volumes):.1f}%) | "
                    f"{rate:.2f} vol/s | ETA {eta_min:.1f} min"
                )
                last_logged_volumes = done_volumes

            should_save_progress = (
                args.save_progress_every_volumes > 0
                and (done_volumes - last_saved_volumes >= args.save_progress_every_volumes)
            )
            if should_save_progress:
                payload = _build_progress_payload(
                    args=args,
                    load_mode=load_mode,
                    num_eval_volumes=num_eval_volumes,
                    num_eval_slices=num_eval_slices,
                    completed_steps=step,
                    generated_slice_count=generated_slice_count,
                    processed_volume_count=processed_volume_count,
                    preview_real_list=preview_real_list,
                    preview_fake_list=preview_fake_list,
                    nn_scores=nn_scores,
                    fid=fid,
                    kid=kid,
                    mifid=mifid,
                    ssim=ssim,
                    msssim=msssim,
                    psnr=psnr,
                )
                _save_progress_checkpoint(progress_ckpt_path, payload)
                accelerator.print(
                    f"Saved progress checkpoint at {done_volumes}/{num_eval_volumes} volumes -> {progress_ckpt_path}"
                )
                last_saved_volumes = done_volumes

        local_stop = torch.tensor([1 if stop_requested["flag"] else 0], device=device, dtype=torch.long)
        should_stop = bool(accelerator.gather(local_stop).max().item())
        if should_stop:
            terminated_early = True
            if is_main:
                done_volumes = min(num_eval_volumes, step * accelerator.num_processes)
                payload = _build_progress_payload(
                    args=args,
                    load_mode=load_mode,
                    num_eval_volumes=num_eval_volumes,
                    num_eval_slices=num_eval_slices,
                    completed_steps=step,
                    generated_slice_count=generated_slice_count,
                    processed_volume_count=processed_volume_count,
                    preview_real_list=preview_real_list,
                    preview_fake_list=preview_fake_list,
                    nn_scores=nn_scores,
                    fid=fid,
                    kid=kid,
                    mifid=mifid,
                    ssim=ssim,
                    msssim=msssim,
                    psnr=psnr,
                )
                _save_progress_checkpoint(progress_ckpt_path, payload)
                accelerator.print(
                    f"Stop requested. Saved final progress snapshot at {done_volumes}/{num_eval_volumes} volumes -> {progress_ckpt_path}"
                )
            break

    if args.final_sync == "barrier":
        accelerator.wait_for_everyone()

    if terminated_early:
        if is_main:
            accelerator.print("Evaluation stopped early after saving progress checkpoint. Re-run with --resume-progress to continue.")
        return

    elapsed = time.time() - start
    results: Dict[str, Any] = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "load_mode": load_mode,
        "data_range_mode": args.data_range_mode,
        "sampler": args.sampler,
        "sample_timesteps": int(args.sample_timesteps),
        "ddim_eta": float(args.ddim_eta),
        "use_ema": bool(args.use_ema),
        "generation_mode": "free_running_full_volume",
        "fg_frac_source": args.fg_frac_source,
        "num_val_volumes": int(num_eval_volumes),
        "num_val_slices": int(num_eval_slices),
        "generated_slices_used_for_metrics": int(generated_slice_count) if is_main else None,
        "elapsed_seconds": float(elapsed),
    }
    if args.fg_frac_source == "constant":
        results["fg_frac_constant"] = float(args.fg_frac_constant)
    if args.fg_frac_source == "profile":
        results["fg_profile_bins"] = int(args.fg_profile_bins)
        results["fg_profile_smooth"] = int(args.fg_profile_smooth)
        results["fg_profile_max_slices"] = args.fg_profile_max_slices

    if is_main:
        fid_value = float(fid.compute().item())
        kid_mean, kid_std = kid.compute()
        mifid_value = float(mifid.compute().item())

        results.update(
            {
                "fid": fid_value,
                "kid_mean": float(kid_mean.item()),
                "kid_std": float(kid_std.item()),
                "mifid": mifid_value,
            }
        )

        if args.compute_paired:
            results.update(
                {
                    "ssim": float(ssim.compute().item()),
                    "ms_ssim": float(msssim.compute().item()),
                    "psnr": float(psnr.compute().item()),
                }
            )

        if nn_scores:
            nn_cat = torch.cat(nn_scores, dim=0)
            results.update(
                {
                    "nn_cosine_mean": float(nn_cat.mean().item()),
                    "nn_cosine_median": float(nn_cat.median().item()),
                    "nn_cosine_p95": float(torch.quantile(nn_cat, 0.95).item()),
                    "nn_cosine_max": float(nn_cat.max().item()),
                }
            )

        if args.save_sample_grid and preview_real_list and preview_fake_list:
            preview_real = torch.stack(preview_real_list, dim=0)
            preview_fake = torch.stack(preview_fake_list, dim=0)
            save_sample_grid(args.save_sample_grid, preview_real, preview_fake)
            results["sample_grid"] = str(Path(args.save_sample_grid).resolve())

        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(results, indent=2))

        if progress_ckpt_path.exists() and not args.keep_progress_ckpt:
            try:
                progress_ckpt_path.unlink()
                accelerator.print(f"Removed progress checkpoint after successful completion: {progress_ckpt_path}")
            except Exception:
                accelerator.print(f"Warning: could not remove progress checkpoint: {progress_ckpt_path}")

        if args.mlflow:
            if mlflow is None:
                accelerator.print("MLflow is not installed; skipping MLflow logging.")
            else:
                mlflow.set_experiment(EXPERIMENT_NAME)
                with mlflow.start_run(run_name=f"eval_{Path(args.ckpt).stem}"):
                    mlflow.log_params(
                        {
                            "checkpoint": str(Path(args.ckpt).resolve()),
                            "generation_mode": "free_running_full_volume",
                            "data_range_mode": args.data_range_mode,
                            "use_ema": args.use_ema,
                            "sampler": args.sampler,
                            "sample_timesteps": args.sample_timesteps,
                            "ddim_eta": args.ddim_eta,
                            "fg_frac_source": args.fg_frac_source,
                            "num_val_volumes": num_eval_volumes,
                            "num_val_slices": num_eval_slices,
                        }
                    )
                    for k, v in results.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(k, v)
                    mlflow.log_artifact(str(save_path))
                    if args.save_sample_grid:
                        mlflow.log_artifact(args.save_sample_grid)

        accelerator.print("\n=== Evaluation results ===")
        for key, value in results.items():
            if isinstance(value, float):
                accelerator.print(f"{key}: {value:.6f}")
            else:
                accelerator.print(f"{key}: {value}")


if __name__ == "__main__":
    main()

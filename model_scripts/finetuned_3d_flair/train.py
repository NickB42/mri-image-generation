"""
Fine-tune a pretrained 3D Latent Diffusion Model (LDM) on BraTS 2021 FLAIR volumes.

- Multi-GPU with 🤗 accelerate
- MLflow logging (main process only)
- Perun energy metrics (per-process output dirs)
- Patch-based 3D training (gives 3D consistency; you can later add full-volume sampling/tiling)

Pretrained weights used by default:
  - Hugging Face: MONAI/brats_mri_generative_diffusion (contains models/model.pt and models/model_autoencoder.pt)

"""

from __future__ import annotations

import copy
import json
import time
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torch.distributed as dist

import mlflow
from accelerate import Accelerator
from accelerate.utils import set_seed

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityRangePercentilesd,
    SpatialPadd,
    Spacingd,
)
from monai.data import CacheDataset

from monai.networks.nets.autoencoderkl import AutoencoderKL
from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet
from monai.networks.schedulers.ddpm import DDPMScheduler


from ..helpers.perun_utils import run_with_perun

# -------------------------
# Configuration
# -------------------------
EXPERIMENT_NAME = "finetuned_3d_flair"
RUN_IDENTIFIER = os.environ.get("SLURM_JOB_ID") or str(uuid.uuid4())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / EXPERIMENT_NAME
DATASET_ROOT = (PROJECT_ROOT / "../datasets").resolve()
TRAIN_SET_ROOT = DATASET_ROOT / "train"
VAL_SET_ROOT = DATASET_ROOT / "val"
OUT_DIR = "./runs/brats_ldm_finetune"
BUNDLE_CACHE_DIR = "./_pretrained_bundles"

BATCH_SIZE = 1
NUM_WORKERS = 4

LEARNING_RATE = 5e-6 #1e-5
MIN_DELTA = 1e-4
NUM_EPOCHS = 80
PATIENCE = 30
GRAD_ACCUM = 4

CACHE_RATE = 0.0

PATCH_SIZE = (144, 176, 112)
SPACING = (1.1, 1.1, 1.1)
AXCODES = "RAS"

HF_REPO_ID = "MONAI/brats_mri_generative_diffusion"
HF_REVISION = "1.1.3"

FREEZE_AUTOENCODER = True
# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    # Data
    train_dir: str = TRAIN_SET_ROOT.as_posix()
    val_dir: str = VAL_SET_ROOT.as_posix()
    out_dir: str = OUT_DIR
    cache_rate: float = 0.1
    num_workers: int = 8

    # Preprocessing
    # BraTS is usually already aligned, but we keep these stable defaults.
    axcodes: str = "RAS"
    spacing: Tuple[float, float, float] = (1.1, 1.1, 1.1)

    # Patches (match MONAI bundle diffusion patch shape)
    patch_size: Tuple[int, int, int] = (144, 176, 112)

    # Training
    batch_size: int = 1  # 3D is heavy; increase if it fits
    lr: float = 1e-5
    weight_decay: float = 0.0
    num_epochs: int = 50
    grad_accum: int = 1
    max_grad_norm: float = 1.0

    # Diffusion
    num_train_timesteps: int = 1000
    beta_start: float = 0.0015
    beta_end: float = 0.0195
    schedule: str = "scaled_linear_beta"

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    mixed_precision: str = "no"

    # Pretrained bundle
    hf_repo_id: str = "MONAI/brats_mri_generative_diffusion"
    hf_revision: str = "1.1.3"  # known version with configs/models
    bundle_cache_dir: str = "./_pretrained_bundles"

    # Finetune strategy
    freeze_autoencoder: bool = True

    # MLflow
    mlflow_experiment: str = "brats_ldm"
    run_name: str = "finetune_flair"

# -------------------------
# Utilities
# -------------------------
def is_main_process(accelerator: Accelerator) -> bool:
    return accelerator.is_main_process

def _dbg(accelerator: Optional[Accelerator], *args):
    # Safe, process-aware printing; falls back to print if accelerator is not yet available
    try:
        if accelerator is not None:
            accelerator.print(f"[proc {accelerator.process_index}]", *args)
        else:
            print("[proc ?]", *args)
    except Exception:
        print("[proc ?]", *args)

# -------------------------
# Coordinate conditioning helpers (Fix 2: patch position conditioning)
# -------------------------
def make_coord_channels(latents: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Create normalised (x, y, z) coordinate channels matching latent spatial size."""
    _, _, lD, lH, lW = latents.shape
    z = torch.linspace(-1, 1, lD, device=device)
    y = torch.linspace(-1, 1, lH, device=device)
    x = torch.linspace(-1, 1, lW, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    coords = torch.stack([xx, yy, zz], dim=0)  # [3, lD, lH, lW]
    return coords[None].repeat(latents.shape[0], 1, 1, 1, 1)  # [B, 3, lD, lH, lW]


class CoordWrappedDiffusion(torch.nn.Module):
    """Wraps a diffusion UNet to automatically concat coordinate channels."""
    def __init__(self, diffusion: torch.nn.Module):
        super().__init__()
        self.diffusion = diffusion

    def forward(self, x, timesteps, **kwargs):
        coords = make_coord_channels(x, x.device)
        return self.diffusion(torch.cat([x, coords], dim=1), timesteps=timesteps, **kwargs)


# -------------------------
# EMA helper (Fix 5)
# -------------------------
def ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    """Exponential-moving-average update of *ema_model* towards *model*."""
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema_model.state_dict().items():
            v.copy_(v * decay + msd[k] * (1.0 - decay))


def list_flair_files(data_dir: str) -> List[str]:
    """
    Recursively find BraTS-style FLAIR files:
      *_flair.nii or *_flair.nii.gz
    """
    root = Path(data_dir)
    files = sorted([str(p) for p in root.rglob("*_flair.nii.gz")])
    files += sorted([str(p) for p in root.rglob("*_flair.nii") if str(p) not in files])
    if len(files) == 0:
        raise FileNotFoundError(
            f"No FLAIR files found under {data_dir}. Expected *_flair.nii(.gz)."
        )
    return files

def make_transforms(cfg: TrainConfig):
    # Deterministic part (good for CacheDataset)
    base = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"]),
        Orientationd(keys=["image"], axcodes=cfg.axcodes),
        Spacingd(keys=["image"], pixdim=cfg.spacing, mode="bilinear"),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.5, b_min=0.0, b_max=1.0),
        CropForegroundd(keys=["image"], source_key="image"),  # Fix 4: avoid mostly-background patches
        SpatialPadd(keys=["image"], spatial_size=cfg.patch_size),  # ensure volume is at least patch_size
    ]
    # Random crop for training
    train_tf = Compose(base + [RandSpatialCropd(keys=["image"], roi_size=cfg.patch_size, random_size=False)])
    # For val: center-ish behavior by using a deterministic crop (here: still random crop but seeded via set_seed)
    # If you prefer true center crop, swap in CenterSpatialCropd.
    val_tf = Compose(base + [RandSpatialCropd(keys=["image"], roi_size=cfg.patch_size, random_size=False)])
    return train_tf, val_tf


def instantiate_models() -> Tuple[AutoencoderKL, DiffusionModelUNet]:
    """
    Instantiate networks matching the MONAI BraTS generative diffusion bundle configs.
    (AutoencoderKL for 1-channel images, and a 3D diffusion UNet operating in latent space.)
    """
    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=8,
        channels=(64, 128, 256),
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=(False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        include_fc=False,
    )

    diffusion = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=8 + 3,   # 8 latent + 3 coordinate channels (Fix 2)
        out_channels=8,
        channels=(256, 256, 512),
        attention_levels=(False, True, True),
        num_head_channels=(0, 64, 64),
        num_res_blocks=2,
        include_fc=False,
        use_combined_linear=False,
    )

    return autoencoder, diffusion


def load_pretrained_weights(
    cfg: TrainConfig,
    autoencoder: AutoencoderKL,
    diffusion: DiffusionModelUNet,
    accelerator: Accelerator,
) -> Path:
    """
    Download the MONAI bundle snapshot from HF and load weights.
    Returns local bundle directory.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "Please install huggingface_hub: pip install huggingface_hub"
        ) from e

    bundle_root = Path(cfg.bundle_cache_dir) / cfg.hf_repo_id.replace("/", "__") / cfg.hf_revision
    if accelerator.is_main_process:
        bundle_root.mkdir(parents=True, exist_ok=True)
        _dbg(accelerator, f"Downloading HF bundle {cfg.hf_repo_id}@{cfg.hf_revision} to {bundle_root} ...")
        snapshot_download(
            repo_id=cfg.hf_repo_id,
            revision=cfg.hf_revision,
            local_dir=str(bundle_root),
            local_dir_use_symlinks=False,
        )
        _dbg(accelerator, "Download complete.")
    accelerator.wait_for_everyone()
    _dbg(accelerator, "All processes synchronized after download.")

    ae_path = bundle_root / "models" / "model_autoencoder.pt"
    dm_path = bundle_root / "models" / "model.pt"

    if not ae_path.exists() or not dm_path.exists():
        raise FileNotFoundError(
            f"Expected weights not found:\n  {ae_path}\n  {dm_path}\n"
            f"Check repo/revision: {cfg.hf_repo_id}@{cfg.hf_revision}"
        )

    _dbg(accelerator, "Loading pretrained weights to CPU...")

    # Load on CPU first
    ae_sd = torch.load(ae_path, map_location="cpu")
    dm_sd = torch.load(dm_path, map_location="cpu")

    # IMPORTANT: use load_old_state_dict for MONAI bundle checkpoints
    if hasattr(autoencoder, "load_old_state_dict"):
        autoencoder.load_old_state_dict(ae_sd)   # optionally: verbose=True if your MONAI supports it
    else:
        autoencoder.load_state_dict(ae_sd, strict=False)

    # The pretrained checkpoint has in_channels=8, but our model has in_channels=11
    # (8 latent + 3 coordinate channels). We need to handle the conv_in weight mismatch
    # by padding pretrained weights (8ch) into the first 8 channels of our 11ch conv_in
    # and zero-initializing the extra 3 coordinate channels.
    #
    # We can't use load_old_state_dict directly because it ultimately calls
    # load_state_dict(strict=True) which fails on the shape mismatch.
    # Instead, we call it on a temporary copy to get the key-remapped state dict,
    # then fix the mismatched weight, and load with strict=False.

    conv_in_key = "conv_in.conv.weight"
    model_sd = diffusion.state_dict()

    if hasattr(diffusion, "load_old_state_dict"):
        # Build a temporary model with the pretrained in_channels=8 to do the key remapping
        tmp_diffusion = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=8,
            out_channels=8,
            channels=(256, 256, 512),
            attention_levels=(False, True, True),
            num_head_channels=(0, 64, 64),
            num_res_blocks=2,
            include_fc=False,
            use_combined_linear=False,
        )
        tmp_diffusion.load_old_state_dict(dm_sd)
        remapped_sd = tmp_diffusion.state_dict()
        del tmp_diffusion

        # Pad conv_in.conv.weight from [256,8,3,3,3] -> [256,11,3,3,3]
        if conv_in_key in remapped_sd and remapped_sd[conv_in_key].shape != model_sd[conv_in_key].shape:
            pretrained_w = remapped_sd[conv_in_key]        # [256, 8, 3, 3, 3]
            padded = torch.zeros_like(model_sd[conv_in_key])  # [256, 11, 3, 3, 3]
            padded[:, :pretrained_w.shape[1]] = pretrained_w
            remapped_sd[conv_in_key] = padded
            _dbg(accelerator,
                 f"Padded {conv_in_key}: {pretrained_w.shape} -> {padded.shape}")

        diffusion.load_state_dict(remapped_sd, strict=False)
    else:
        # Pad conv_in weight if needed even without key remapping
        if conv_in_key in dm_sd and dm_sd[conv_in_key].shape != model_sd[conv_in_key].shape:
            pretrained_w = dm_sd[conv_in_key]
            padded = torch.zeros_like(model_sd[conv_in_key])
            padded[:, :pretrained_w.shape[1]] = pretrained_w
            dm_sd[conv_in_key] = padded
        diffusion.load_state_dict(dm_sd, strict=False)

    _dbg(accelerator, "Pretrained weights loaded.")

    return bundle_root


@torch.no_grad()
def compute_scale_factor(
    autoencoder,
    loader,
    device: torch.device,
    accelerator,
    n_batches: int = 100,
) -> float:
    """Compute a stable latent scale factor averaged over many batches (Fix 3)."""
    autoencoder.eval()
    stds = []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        x = batch["image"].to(device)
        z = autoencoder.encode_stage_2_inputs(x)
        stds.append(torch.std(z).detach())

    if len(stds) == 0:
        return 1.0

    std_t = torch.stack(stds).mean()
    # Gather across ranks for consistent value
    std_g = accelerator.gather_for_metrics(std_t[None]).mean()
    sf = (1.0 / std_g).item()
    return float(sf)


def save_checkpoint(
    cfg: TrainConfig,
    accelerator: Accelerator,
    diffusion: torch.nn.Module,
    out_dir: Path,
    name: str,
    extra: Optional[dict] = None,
    ema_diffusion: Optional[torch.nn.Module] = None,
):
    if not accelerator.is_main_process:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "diffusion_state_dict": accelerator.unwrap_model(diffusion).state_dict(),
        "cfg": asdict(cfg),
        "extra": extra or {},
    }
    if ema_diffusion is not None:
        ckpt["ema_diffusion_state_dict"] = ema_diffusion.state_dict()
    path = out_dir / name
    accelerator.save(ckpt, str(path))
    print(f"[ckpt] saved: {path}")


# -------------------------
# Training
# -------------------------
def train(cfg: TrainConfig, accelerator: Accelerator) -> float:
    device = accelerator.device
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _dbg(accelerator, f"Listing training FLAIR files in {cfg.train_dir} ...")
    train_flairs = list_flair_files(cfg.train_dir)
    _dbg(accelerator, f"Found {len(train_flairs)} train files.")
    _dbg(accelerator, f"Listing validation FLAIR files in {cfg.val_dir} ...")
    val_flairs   = list_flair_files(cfg.val_dir)
    _dbg(accelerator, f"Found {len(val_flairs)} val files.")

    train_items = [{"image": f} for f in train_flairs]
    val_items   = [{"image": f} for f in val_flairs]

    _dbg(accelerator, "Creating transforms...")
    train_tf, val_tf = make_transforms(cfg)
    _dbg(accelerator, "Transforms ready.")

    _dbg(accelerator, "Building CacheDatasets...")
    train_ds = CacheDataset(
        train_items, transform=train_tf,
        cache_rate=cfg.cache_rate, num_workers=cfg.num_workers
    )
    val_ds = CacheDataset(
        val_items, transform=val_tf,
        cache_rate=min(cfg.cache_rate, 0.2), num_workers=cfg.num_workers
    )
    _dbg(accelerator, f"Datasets built. train_ds={len(train_ds)} samples, val_ds={len(val_ds)} samples.")

    _dbg(accelerator, "Creating DataLoaders...")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    _dbg(accelerator, f"DataLoaders ready. train_batches≈{len(train_loader)}, val_batches≈{len(val_loader)}.")

    # Models
    _dbg(accelerator, "Instantiating models...")
    autoencoder, diffusion = instantiate_models()
    _dbg(accelerator, "Models instantiated.")

    # Load pretrained weights (strict=False because in_channels changed from 8 -> 11)
    _dbg(accelerator, "Loading pretrained bundle & weights...")
    bundle_root = load_pretrained_weights(cfg, autoencoder, diffusion, accelerator)
    _dbg(accelerator, f"Bundle ready at {bundle_root}.")

    if cfg.freeze_autoencoder:
        _dbg(accelerator, "Freezing autoencoder parameters.")
        autoencoder.requires_grad_(False)
        autoencoder.eval()

    # EMA model (Fix 5) — created *before* accelerator.prepare so it stays on CPU/device
    ema_decay = 0.999
    ema_diffusion = copy.deepcopy(diffusion).eval()
    for p in ema_diffusion.parameters():
        p.requires_grad_(False)
    _dbg(accelerator, "EMA diffusion model created.")

    # Scheduler & optimizer
    _dbg(accelerator, "Creating noise scheduler and optimizer...")
    noise_scheduler = DDPMScheduler(
        schedule=cfg.schedule,
        num_train_timesteps=cfg.num_train_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
    )
    optimizer = Adam(diffusion.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_sched = MultiStepLR(optimizer, milestones=[10, 30], gamma=0.1)  # fine-tune-friendly milestones

    # Prepare for distributed
    _dbg(accelerator, "Preparing models/loaders with Accelerator...")
    diffusion, optimizer, lr_sched, train_loader, val_loader = accelerator.prepare(
        diffusion, optimizer, lr_sched, train_loader, val_loader
    )

    autoencoder = autoencoder.to(device)
    ema_diffusion = ema_diffusion.to(device)
    _dbg(accelerator, f"Accelerator prepared. Device={device}, process_index={accelerator.process_index}.")

    # Scale factor
    scale_factor = compute_scale_factor(autoencoder, train_loader, device, accelerator)
    if accelerator.is_main_process:
        print(f"[scale_factor] {scale_factor:.6f}")
        (out_dir / "pretrained_bundle_info.json").write_text(
            json.dumps(
                {
                    "hf_repo_id": cfg.hf_repo_id,
                    "hf_revision": cfg.hf_revision,
                    "bundle_root": str(bundle_root),
                    "scale_factor": scale_factor,
                },
                indent=2,
            )
        )
        _dbg(accelerator, "Wrote pretrained_bundle_info.json.")

    # Early stopping
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    global_step = 0

    for epoch in range(1, cfg.num_epochs + 1):
        _dbg(accelerator, f"Starting epoch {epoch}/{cfg.num_epochs}...")
        diffusion.train()
        t0 = time.time()

        train_losses = []
        for bidx, batch in enumerate(train_loader):
            with accelerator.accumulate(diffusion):
                x = batch["image"].to(device)  # [B,1,D,H,W]
                if bidx == 0:
                    _dbg(accelerator, f"First train batch shape: {tuple(x.shape)}")
                with torch.no_grad():
                    latents = autoencoder.encode_stage_2_inputs(x) * scale_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, cfg.num_train_timesteps, (latents.shape[0],), device=device
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Fix 2: concatenate coordinate channels
                coords = make_coord_channels(noisy_latents, device)
                unet_in = torch.cat([noisy_latents, coords], dim=1)  # [B, 11, ...]

                with accelerator.autocast():
                    noise_pred = diffusion(unet_in, timesteps=timesteps)

                    # Fix 6: P2 (perception-prioritised) loss weighting
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)  # [T]
                    a = alphas_cumprod[timesteps]  # [B]
                    snr = a / (1.0 - a)
                    p2_k, p2_gamma = 1.0, 1.0
                    w = (p2_k + snr) ** (-p2_gamma)  # [B]
                    w = w.view(-1, 1, 1, 1, 1)  # broadcast over spatial dims

                    mse = (noise_pred.float() - noise.float()) ** 2
                    loss = (w * mse).mean()

                accelerator.backward(loss)
                if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(diffusion.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Fix 5: EMA update
                if accelerator.is_main_process:
                    ema_update(ema_diffusion, accelerator.unwrap_model(diffusion), ema_decay)

            # Gather loss across processes for correct logging
            loss_g = accelerator.gather_for_metrics(loss.detach())
            train_losses.append(loss_g.mean().item())

            global_step += 1
            if bidx % 25 == 0:
                _dbg(accelerator, f"Epoch {epoch} train step {bidx}/{len(train_loader)} loss={loss_g.mean().item():.6f}")

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        _dbg(accelerator, f"Epoch {epoch} finished training. avg_train_loss={train_loss:.6f}")

        # Validation
        diffusion.eval()
        val_losses = []
        with torch.no_grad():
            for vbidx, batch in enumerate(val_loader):
                x = batch["image"].to(device)
                if vbidx == 0:
                    _dbg(accelerator, f"First val batch shape: {tuple(x.shape)}")
                latents = autoencoder.encode_stage_2_inputs(x) * scale_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, cfg.num_train_timesteps, (latents.shape[0],), device=device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                # Fix 2: concatenate coordinate channels
                coords = make_coord_channels(noisy_latents, device)
                unet_in = torch.cat([noisy_latents, coords], dim=1)
                noise_pred = diffusion(unet_in, timesteps=timesteps)
                vloss = F.mse_loss(noise_pred.float(), noise.float())

                vloss_g = accelerator.gather_for_metrics(vloss.detach())
                val_losses.append(vloss_g.mean().item())
            _dbg(accelerator, f"Epoch {epoch} finished validation. val_batches={len(val_loader)}")

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        
        lr_sched.step()

        dt = time.time() - t0
        if accelerator.is_main_process:
            print(
                f"Epoch {epoch:04d}/{cfg.num_epochs} | "
                f"train {train_loss:.6f} | val {val_loss:.6f} | "
                f"time {dt:.1f}s | step {global_step}"
            )
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("epoch_time_sec", dt, step=epoch)
            mlflow.log_metric("scale_factor", scale_factor, step=epoch)
            mlflow.log_metric("lr", lr_sched.get_last_lr()[0], step=epoch)

        # Checkpointing + early stopping
        improved = (best_val - val_loss) > cfg.min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            _dbg(accelerator, f"Validation improved at epoch {epoch}. Saving best checkpoint.")
            save_checkpoint(
                cfg,
                accelerator,
                diffusion,
                out_dir,
                name="best_diffusion.pt",
                extra={"epoch": epoch, "val_loss": val_loss, "scale_factor": scale_factor},
                ema_diffusion=ema_diffusion,
            )
        else:
            bad_epochs += 1
            _dbg(accelerator, f"No improvement. bad_epochs={bad_epochs}/{cfg.patience}")

        if accelerator.is_main_process:
            mlflow.log_metric("best_val_loss_so_far", best_val, step=epoch)
            mlflow.log_metric("bad_epochs", bad_epochs, step=epoch)

        if bad_epochs >= cfg.patience:
            if accelerator.is_main_process:
                print(f"[early_stop] epoch={epoch} best_epoch={best_epoch} best_val={best_val:.6f}")
            _dbg(accelerator, "Early stopping triggered.")
            break

        # Save last each epoch (optional; adjust to taste)
        _dbg(accelerator, "Saving last checkpoint for this epoch.")
        save_checkpoint(
            cfg,
            accelerator,
            diffusion,
            out_dir,
            name="last_diffusion.pt",
            extra={"epoch": epoch, "val_loss": val_loss, "scale_factor": scale_factor},
            ema_diffusion=ema_diffusion,
        )

    _dbg(accelerator, f"Training loop finished. best_val={best_val:.6f} at epoch {best_epoch}.")
    return best_val


# -------------------------
# Main
# ------------------------
def main() -> None:
    cfg = TrainConfig(
        train_dir=TRAIN_SET_ROOT.as_posix(),
        val_dir=VAL_SET_ROOT.as_posix(),
        out_dir=OUT_DIR,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        grad_accum=GRAD_ACCUM,
        num_workers=NUM_WORKERS,
        cache_rate=CACHE_RATE,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        patch_size=PATCH_SIZE,
        spacing=SPACING,
        axcodes=AXCODES,
        hf_repo_id=HF_REPO_ID,
        hf_revision=HF_REVISION,
        bundle_cache_dir=BUNDLE_CACHE_DIR,
        freeze_autoencoder=FREEZE_AUTOENCODER,
        mlflow_experiment=EXPERIMENT_NAME,
        run_name=RUN_IDENTIFIER,
    )

    accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum)
    _dbg(accelerator, f"Accelerator initialized. num_processes={accelerator.num_processes}, local_process_index={accelerator.local_process_index}")
    if torch.cuda.is_available():
        torch.cuda.set_device(accelerator.local_process_index)
        _dbg(accelerator, f"CUDA available. Set device to local_process_index={accelerator.local_process_index}")

    cfg.mixed_precision = str(accelerator.mixed_precision)

    # Per-process perun output to avoid collisions (as in your snippet)
    PERUN_OUT_DIR = Path(cfg.out_dir) / "perun"
    perun_out = PERUN_OUT_DIR / f"proc_{accelerator.process_index}"
    perun_out.mkdir(parents=True, exist_ok=True)
    _dbg(accelerator, f"Perun output dir: {perun_out}")

    # Make runs reproducible-ish
    set_seed(0 + accelerator.process_index)
    _dbg(accelerator, f"Seed set to {0 + accelerator.process_index}")

    device = accelerator.device
    IS_MAIN_PROCESS = is_main_process(accelerator)
    _dbg(accelerator, f"Main process? {IS_MAIN_PROCESS}. Device={device}")

    if IS_MAIN_PROCESS:
        mlflow.set_experiment(cfg.mlflow_experiment)
        _dbg(accelerator, f"MLflow experiment set: {cfg.mlflow_experiment}")
        with mlflow.start_run(run_name=cfg.run_name):
            _dbg(accelerator, f"MLflow run started: {cfg.run_name}")
            mlflow.log_params(
                {
                    "patch_size": cfg.patch_size,
                    "batch_size": cfg.batch_size,
                    "timesteps": cfg.num_train_timesteps,
                    "learning_rate": cfg.lr,
                    "num_epochs": cfg.num_epochs,
                    "patience": cfg.patience,
                    "min_delta": cfg.min_delta,
                    "device": str(device),
                    "model": "Finetuned DiffusionModelUNet (latent)",
                    "dataset": "BraTS 2021 FLAIR (3D patches)",
                    "num_workers": cfg.num_workers,
                    "accelerate_num_processes": accelerator.num_processes,
                    "accelerate_mixed_precision": str(accelerator.mixed_precision),
                    "hf_repo_id": cfg.hf_repo_id,
                    "hf_revision": cfg.hf_revision,
                    "freeze_autoencoder": cfg.freeze_autoencoder,
                }
            )

            def _run():
                return train(cfg, accelerator)

            # best_val = run_with_perun(_run, data_out=str(perun_out))
            best_val = _run()
            _dbg(accelerator, f"Train returned best_val={best_val}")
            if best_val is not None and best_val != float("inf"):
                mlflow.log_metric("best_val_loss", float(best_val))
        _dbg(accelerator, "MLflow run closed.")
    else:
        def _run():
            return train(cfg, accelerator)

        # best_val = run_with_perun(_run, data_out=str(perun_out))
        best_val = _run()
        _dbg(accelerator, f"Non-main process train returned best_val={best_val}")

    accelerator.end_training()
    _dbg(accelerator, "Accelerator end_training called.")

if __name__ == "__main__":
    main()
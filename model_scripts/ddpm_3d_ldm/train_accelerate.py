from pathlib import Path
from typing import Union, Optional
import os
import uuid
import time
import random
import math

import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from accelerate import Accelerator

import mlflow
import mlflow.pytorch

from .dataset import BraTS3DVolumeDataset
from .vae import VAE3D
from .unet_attention import UNet3DModelWithAttention
from .diffusion import GaussianDiffusionLatent3D
from ..helpers.perun_utils import run_with_perun

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
EXPERIMENT_NAME = "ddpm_3d_ldm"
RUN_IDENTIFIER = os.environ.get("SLURM_JOB_ID") or str(uuid.uuid4())

# 3D patch size (D, H, W)
PATCH_SIZE = (128, 160, 160)
TIMESTEPS = 200

# VAE hyperparams
VAE_NUM_EPOCHS = 40
VAE_LEARNING_RATE = 1e-4
VAE_BASE_CHANNELS = 32
VAE_NUM_DOWN = 3
LATENT_CHANNELS = 16
VAE_KL_WEIGHT = 1e-4
LATENT_SCALE = 1.0  # will be estimated before LDM stage

# LDM hyperparams
LDM_NUM_EPOCHS = 60
LDM_LEARNING_RATE = 1e-4
PATIENCE = 10
MIN_DELTA = 1e-4

UNET_BASE_CHANNELS = 128
UNET_CHANNEL_MULTS = (1, 2, 4)

BATCH_SIZE = 2
NUM_WORKERS = 8

DEBUG_FAST = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / EXPERIMENT_NAME
DATASET_ROOT = (PROJECT_ROOT / "../datasets").resolve()
TRAIN_SET_ROOT = DATASET_ROOT / "train"
VAL_SET_ROOT = DATASET_ROOT / "val"
PERUN_OUT_DIR = EXPERIMENT_ROOT / "perun_results" / RUN_IDENTIFIER
MODELS_DIR = EXPERIMENT_ROOT / "models" / RUN_IDENTIFIER

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# -------------------------------------------------------------------
# Accelerate setup
# -------------------------------------------------------------------
accelerator = Accelerator()
device = accelerator.device
IS_MAIN_PROCESS = accelerator.is_main_process

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def maybe_set_epoch(dataloader: DataLoader, epoch: int) -> None:
    # When Accelerate prepares a dataloader in distributed mode, it will attach a DistributedSampler internally.
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

def reduce_avg_loss(running_loss_sum: float, count: int) -> float:
    # Reduce sums across all processes, then divide.
    loss_t = torch.tensor(running_loss_sum, device=device, dtype=torch.float64)
    cnt_t = torch.tensor(count, device=device, dtype=torch.float64)
    loss_t = accelerator.reduce(loss_t, reduction="sum")
    cnt_t = accelerator.reduce(cnt_t, reduction="sum")
    return (loss_t / torch.clamp(cnt_t, min=1.0)).item()

# -------------------------------------------------------------------
# Dataset and DataLoaders (no DistributedSampler needed)
# -------------------------------------------------------------------
train_dataset = BraTS3DVolumeDataset(
    TRAIN_SET_ROOT,
    patch_size=PATCH_SIZE,
    random_crop=True,
)

val_dataset = BraTS3DVolumeDataset(
    VAL_SET_ROOT,
    patch_size=PATCH_SIZE,
    random_crop=False,
)

if DEBUG_FAST:
    train_dataset = Subset(train_dataset, list(range(min(20, len(train_dataset)))))
    val_dataset = Subset(val_dataset, list(range(min(10, len(val_dataset)))))
    if IS_MAIN_PROCESS:
        accelerator.print(
            f"[DEBUG FAST] Using reduced datasets: "
            f"train size {len(train_dataset)}, val size {len(val_dataset)}"
        )

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    worker_init_fn=seed_worker,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    worker_init_fn=seed_worker,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

if IS_MAIN_PROCESS:
    accelerator.print(f"Train volumes: {len(train_dataset)}, Val volumes: {len(val_dataset)}")
    if torch.cuda.is_available():
        accelerator.print("=== PyTorch CUDA / Slurm info ===")
        accelerator.print("torch.cuda.is_available():", torch.cuda.is_available())
        accelerator.print("torch.cuda.device_count():", torch.cuda.device_count())
        accelerator.print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))

# -------------------------------------------------------------------
# Models: VAE + UNet + Diffusion
# -------------------------------------------------------------------
vae = VAE3D(
    in_channels=4,
    base_channels=VAE_BASE_CHANNELS,
    num_down=VAE_NUM_DOWN,
    latent_channels=LATENT_CHANNELS,
)

unet = UNet3DModelWithAttention(
    in_channels=LATENT_CHANNELS,
    base_channels=UNET_BASE_CHANNELS,
    channel_mults=UNET_CHANNEL_MULTS,
)

diffusion = GaussianDiffusionLatent3D(
    model=unet,
    channels=LATENT_CHANNELS,
    timesteps=TIMESTEPS,
)

optimizer_vae = torch.optim.Adam(vae.parameters(), lr=VAE_LEARNING_RATE)
optimizer_ldm = torch.optim.Adam(diffusion.model.parameters(), lr=LDM_LEARNING_RATE)

scheduler_ldm = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_ldm,
    mode="min",
    factor=0.5,
    patience=3
)

# Prepare everything for distributed + mixed precision
vae, diffusion, optimizer_vae, optimizer_ldm, train_loader, val_loader, scheduler_ldm = accelerator.prepare(
    vae, diffusion, optimizer_vae, optimizer_ldm, train_loader, val_loader, scheduler_ldm
)

# -------------------------------------------------------------------
# VAE Training helpers
# -------------------------------------------------------------------
def train_vae_one_epoch(epoch: int, max_steps: Union[int, None] = None) -> float:
    vae.train()
    running_loss = 0.0
    n_steps = 0
    count = 0

    start_time = time.time()

    for step, x in enumerate(train_loader, start=1):
        # (optional) keep explicit move; safe even if Accelerate already placed it
        x = x.to(device, non_blocking=True)

        optimizer_vae.zero_grad(set_to_none=True)

        with accelerator.autocast():
            recon, mu, logvar = vae(x)
            recon_loss = F.l1_loss(recon, x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + VAE_KL_WEIGHT * kl

        accelerator.backward(loss)
        optimizer_vae.step()

        if IS_MAIN_PROCESS and torch.cuda.is_available() and step == 1:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            accelerator.print(f"[VAE] Peak GPU memory after first step: {peak:.2f} GB")

        b = x.size(0)
        count += b
        running_loss += loss.item() * b
        n_steps += 1

        if IS_MAIN_PROCESS and step % 100 == 0:
            avg_local = running_loss / max(1, count)
            accelerator.print(f"[VAE epoch {epoch} | step {step}] avg loss (local): {avg_local:.4f}")

        if max_steps is not None and step >= max_steps:
            break

    elapsed = time.time() - start_time
    avg_loss = reduce_avg_loss(running_loss, count)
    steps_per_s = n_steps / max(elapsed, 1e-8)

    if IS_MAIN_PROCESS:
        accelerator.print(
            f"VAE Epoch {epoch} | Train loss: {avg_loss:.4f} | "
            f"steps: {n_steps} | time: {elapsed:.1f}s | {steps_per_s:.2f} steps/s"
        )

        run = mlflow.active_run()
        if run is not None:
            mlflow.log_metric("vae_train_steps_per_s", steps_per_s, step=epoch)
            mlflow.log_metric("vae_train_num_steps", n_steps, step=epoch)
            mlflow.log_metric("vae_epoch_time_s", elapsed, step=epoch)

    return avg_loss


@torch.no_grad()
def validate_vae(epoch: int, max_steps: Union[int, None] = None) -> float:
    vae.eval()
    running_loss = 0.0
    count = 0

    for step, x in enumerate(val_loader, start=1):
        x = x.to(device, non_blocking=True)

        with accelerator.autocast():
            recon, mu, logvar = vae(x)
            recon_loss = F.l1_loss(recon, x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + VAE_KL_WEIGHT * kl

        running_loss += loss.item() * x.size(0)
        count += x.size(0)

        if max_steps is not None and step >= max_steps:
            break

    avg_loss = reduce_avg_loss(running_loss, count)

    if IS_MAIN_PROCESS:
        accelerator.print(f"VAE Epoch {epoch} | Val loss: {avg_loss:.4f}")

    return avg_loss


@torch.no_grad()
def estimate_latent_scale(num_batches: int = 200) -> float:
    """
    Estimate latent scale across ALL processes:
    - each process computes (sum(var), n) over its shard
    - reduce sums and compute global mean var
    """
    vae.eval()
    base_vae = accelerator.unwrap_model(vae)

    local_sum = 0.0
    local_n = 0

    for i, x in enumerate(train_loader):
        if i >= num_batches:
            break
        x = x.to(device, non_blocking=True)
        z = base_vae.encode_to_latent(x).float()
        local_sum += float(z.var(unbiased=False).item())
        local_n += 1

    sum_t = torch.tensor(local_sum, device=device, dtype=torch.float64)
    n_t = torch.tensor(local_n, device=device, dtype=torch.float64)
    sum_t = accelerator.reduce(sum_t, reduction="sum")
    n_t = accelerator.reduce(n_t, reduction="sum")

    mean_var = (sum_t / torch.clamp(n_t, min=1.0)).item()
    return 1.0 / math.sqrt(max(mean_var, 1e-8))

# -------------------------------------------------------------------
# LDM Training helpers
# -------------------------------------------------------------------
def train_ldm_one_epoch(epoch: int, max_steps: Union[int, None] = None) -> float:
    diffusion.train()
    vae.eval()  # frozen in this stage
    base_vae = accelerator.unwrap_model(vae)

    running_loss = 0.0
    n_steps = 0
    count = 0
    start_time = time.time()

    for step, x in enumerate(train_loader, start=1):
        x = x.to(device, non_blocking=True)

        # Get latents without grad
        with torch.no_grad():
            with accelerator.autocast():
                z = base_vae.encode_to_latent(x).float() * LATENT_SCALE

        b = z.size(0)
        t = torch.randint(1, diffusion.timesteps, (b,), device=device)

        optimizer_ldm.zero_grad(set_to_none=True)

        with accelerator.autocast():
            loss = diffusion.p_losses(z, t, cond=None, min_snr_gamma=5.0)

        accelerator.backward(loss)
        optimizer_ldm.step()

        if IS_MAIN_PROCESS and torch.cuda.is_available() and step == 1:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            accelerator.print(f"[LDM] Peak GPU memory after first step: {peak:.2f} GB")

        running_loss += loss.item() * b
        count += b
        n_steps += 1

        if IS_MAIN_PROCESS and step % 100 == 0:
            avg_local = running_loss / max(1, count)
            accelerator.print(f"[LDM epoch {epoch} | step {step}] avg loss (local): {avg_local:.4f}")

        if max_steps is not None and step >= max_steps:
            break

    elapsed = time.time() - start_time
    avg_loss = reduce_avg_loss(running_loss, count)
    steps_per_s = n_steps / max(elapsed, 1e-8)

    if IS_MAIN_PROCESS:
        accelerator.print(
            f"LDM Epoch {epoch} | Train loss: {avg_loss:.4f} | "
            f"steps: {n_steps} | time: {elapsed:.1f}s | {steps_per_s:.2f} steps/s"
        )

        run = mlflow.active_run()
        if run is not None:
            mlflow.log_metric("ldm_train_steps_per_s", steps_per_s, step=epoch)
            mlflow.log_metric("ldm_train_num_steps", n_steps, step=epoch)
            mlflow.log_metric("ldm_epoch_time_s", elapsed, step=epoch)

    return avg_loss


@torch.no_grad()
def validate_ldm(epoch: int, max_steps: Union[int, None] = None) -> float:
    diffusion.eval()
    vae.eval()
    base_vae = accelerator.unwrap_model(vae)

    running_loss = 0.0
    count = 0

    fixed_ts = torch.linspace(1, diffusion.timesteps - 1, steps=8, device=device).long()

    for step, x in enumerate(val_loader, start=1):
        x = x.to(device, non_blocking=True)

        with torch.no_grad():
            with accelerator.autocast():
                z = base_vae.encode_to_latent(x).float() * LATENT_SCALE

        b = z.size(0)
        t = fixed_ts[(step - 1) % len(fixed_ts)].expand(b)

        with accelerator.autocast():
            loss = diffusion.p_losses(z, t, cond=None, min_snr_gamma=5.0)

        running_loss += loss.item() * b
        count += b

        if max_steps is not None and step >= max_steps:
            break

    avg_loss = reduce_avg_loss(running_loss, count)

    if IS_MAIN_PROCESS:
        accelerator.print(f"LDM Epoch {epoch} | Val loss: {avg_loss:.4f}")

    return avg_loss

# -------------------------------------------------------------------
# Main training loop: VAE then LDM
# -------------------------------------------------------------------
def train() -> float:
    global LATENT_SCALE

    if IS_MAIN_PROCESS:
        accelerator.print("Starting 3D VAE + LDM Training")
        accelerator.print("=== Stage 1: Training 3D VAE ===")

    # ------------------ Stage 1: VAE training ------------------
    for epoch in range(1, VAE_NUM_EPOCHS + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        maybe_set_epoch(train_loader, epoch)

        train_loss = train_vae_one_epoch(epoch, max_steps=5 if DEBUG_FAST else None)
        val_loss = validate_vae(epoch, max_steps=2 if DEBUG_FAST else None)

        if IS_MAIN_PROCESS:
            mlflow.log_metric("vae_train_loss", train_loss, step=epoch)
            mlflow.log_metric("vae_val_loss", val_loss, step=epoch)
            mlflow.log_metric("vae_learning_rate", optimizer_vae.param_groups[0]["lr"], step=epoch)

        # Save VAE weights (main process only)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        vae_path = MODELS_DIR / "vae3d_final.pt"

        accelerator.wait_for_everyone()
        if IS_MAIN_PROCESS:
            state = accelerator.get_state_dict(vae)
            accelerator.save(state, str(vae_path))
            mlflow.log_artifact(str(vae_path), artifact_path="checkpoints")
        accelerator.wait_for_everyone()

    # freeze VAE for LDM
    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()

    if IS_MAIN_PROCESS:
        accelerator.print("=== Stage 2: Training 3D latent diffusion (LDM) ===")

    # Estimate latent scale (distributed-safe)
    LATENT_SCALE = estimate_latent_scale(num_batches=20 if DEBUG_FAST else 200)

    if IS_MAIN_PROCESS:
        accelerator.print(f"[latent] LATENT_SCALE={LATENT_SCALE:.6f}")
        run = mlflow.active_run()
        if run is not None:
            mlflow.log_param("latent_scale", LATENT_SCALE)

    best_val = float("inf")
    epochs_without_improvement = 0
    best_ldm_path: Optional[Path] = None

    for epoch in range(1, LDM_NUM_EPOCHS + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        maybe_set_epoch(train_loader, epoch)

        train_loss = train_ldm_one_epoch(epoch, max_steps=5 if DEBUG_FAST else None)
        val_loss = validate_ldm(epoch, max_steps=2 if DEBUG_FAST else None)

        if IS_MAIN_PROCESS:
            mlflow.log_metric("ldm_train_loss", train_loss, step=epoch)
            mlflow.log_metric("ldm_val_loss", val_loss, step=epoch)
            mlflow.log_metric("ldm_learning_rate", optimizer_ldm.param_groups[0]["lr"], step=epoch)

        scheduler_ldm.step(val_loss)

        # Early stopping logic (runs identically on all processes because val_loss is reduced)
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            epochs_without_improvement = 0

            accelerator.wait_for_everyone()
            if IS_MAIN_PROCESS:
                best_ldm_path = MODELS_DIR / "3d_ldm_diffusion_best.pt"
                state = accelerator.get_state_dict(diffusion.model)
                accelerator.save(state, str(best_ldm_path))
                mlflow.log_artifact(str(best_ldm_path), artifact_path="checkpoints")
                accelerator.print(f"✅ New best LDM val loss: {best_val:.4f}")
            accelerator.wait_for_everyone()
        else:
            epochs_without_improvement += 1
            if IS_MAIN_PROCESS:
                accelerator.print(f"⚠️ No LDM improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= PATIENCE:
            if IS_MAIN_PROCESS:
                accelerator.print(
                    f"⏹ LDM early stopping at epoch {epoch} "
                    f"(no val improvement for {PATIENCE} epochs)."
                )
            break

    if IS_MAIN_PROCESS:
        # Log final models to MLflow (unwrap so you don't log wrappers)
        mlflow.pytorch.log_model(accelerator.unwrap_model(diffusion.model), artifact_path="final_ldm_unet")
        mlflow.pytorch.log_model(accelerator.unwrap_model(vae), artifact_path="final_vae")

        if best_val != float("inf"):
            mlflow.log_metric("best_ldm_val_loss", best_val)

    return best_val

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
def main() -> None:
    # Use per-process perun output to avoid collisions
    perun_out = PERUN_OUT_DIR / f"proc_{accelerator.process_index}"
    perun_out.mkdir(parents=True, exist_ok=True)

    if IS_MAIN_PROCESS:
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=RUN_IDENTIFIER):
            mlflow.log_params(
                {
                    "patch_size_d": PATCH_SIZE[0],
                    "patch_size_h": PATCH_SIZE[1],
                    "patch_size_w": PATCH_SIZE[2],
                    "batch_size": BATCH_SIZE,
                    "timesteps": TIMESTEPS,
                    "vae_learning_rate": VAE_LEARNING_RATE,
                    "vae_num_epochs": VAE_NUM_EPOCHS,
                    "ldm_learning_rate": LDM_LEARNING_RATE,
                    "ldm_num_epochs": LDM_NUM_EPOCHS,
                    "patience": PATIENCE,
                    "min_delta": MIN_DELTA,
                    "device": str(device),
                    "model_vae": "VAE3D",
                    "model_ldm_unet": "UNet3DModelWithAttention",
                    "dataset": "BraTS_3D_4modalities",
                    "debug_fast": DEBUG_FAST,
                    "run_identifier": RUN_IDENTIFIER,
                    "num_workers": NUM_WORKERS,
                    "vae_base_channels": VAE_BASE_CHANNELS,
                    "vae_num_down": VAE_NUM_DOWN,
                    "latent_channels": LATENT_CHANNELS,
                    "unet_base_channels": UNET_BASE_CHANNELS,
                    "unet_channel_mults": str(UNET_CHANNEL_MULTS),
                    "accelerate_num_processes": accelerator.num_processes,
                    "accelerate_mixed_precision": str(accelerator.mixed_precision),
                }
            )

            best_val = run_with_perun(train, data_out=str(perun_out))
            if best_val is not None and best_val != float("inf"):
                mlflow.log_metric("best_ldm_val_loss", best_val)
    else:
        run_with_perun(train, data_out=str(perun_out))

    accelerator.end_training()

if __name__ == "__main__":
    main()
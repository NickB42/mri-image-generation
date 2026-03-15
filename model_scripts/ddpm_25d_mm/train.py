"""
train.py — Training script for the memmap-backed 2.5D sequential DDPM.

Uses the MemmapDataset (FLAIR-only, 1 channel) instead of the NIfTI-based
BraTSSliceDataset (4 modalities).

Launch with Accelerate:
  accelerate launch -m model_scripts.ddpm_25d_mm.train
"""

from pathlib import Path
from typing import Union
import os
import uuid
import time
import random

import numpy as np

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from accelerate import Accelerator

import mlflow
import mlflow.pytorch

from .mm_dataset import MemmapDataset, get_train_dataset, get_val_dataset, get_debug_dataset
from .unet import UNet
from .diffusion import GaussianDiffusion
from .ema import EMA
from ..helpers.perun_utils import run_with_perun

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
EXPERIMENT_NAME = "ddpm_25d_mm"
RUN_IDENTIFIER = os.environ.get("SLURM_JOB_ID") or str(uuid.uuid4())

IMAGE_SIZE = 256
TIMESTEPS = 1000
PATIENCE = 15
LEARNING_RATE = 2e-4
MIN_DELTA = 1e-4
BATCH_SIZE = 1
NUM_EPOCHS = 80

# 1 channel (FLAIR only) instead of 4
CENTER_CHANNELS = 1
SLICE_RADIUS = 2

NUM_WORKERS = 8
DEBUG_FAST = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / EXPERIMENT_NAME
MEMMAP_DIR = (PROJECT_ROOT / "../datasets/memmap").resolve()

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
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def reduce_avg_loss(running_loss_sum: float, count: int) -> float:
    loss_t = torch.tensor(running_loss_sum, device=device, dtype=torch.float64)
    cnt_t = torch.tensor(count, device=device, dtype=torch.float64)
    loss_t = accelerator.reduce(loss_t, reduction="sum")
    cnt_t = accelerator.reduce(cnt_t, reduction="sum")
    return (loss_t / torch.clamp(cnt_t, min=1.0)).item()


# -------------------------------------------------------------------
# Dataset and DataLoaders
# -------------------------------------------------------------------
if IS_MAIN_PROCESS:
    accelerator.print("Using MEMMAP_DIR:", MEMMAP_DIR)

if DEBUG_FAST:
    train_dataset = get_debug_dataset(slice_radius=SLICE_RADIUS)
    val_dataset = get_debug_dataset(slice_radius=SLICE_RADIUS)
    train_dataset = Subset(train_dataset, list(range(min(64, len(train_dataset)))))
    val_dataset = Subset(val_dataset, list(range(min(64, len(val_dataset)))))
else:
    train_dataset = get_train_dataset(slice_radius=SLICE_RADIUS)
    val_dataset = get_val_dataset(slice_radius=SLICE_RADIUS)


# Oversample end-ish slices via WeightedRandomSampler
def z_weight(z_pos: float) -> float:
    if z_pos < 0.25 or z_pos > 0.75:
        return 2.0
    return 1.0


_base_train = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
if isinstance(train_dataset, Subset):
    _indices = train_dataset.indices
    weights = [
        z_weight(float(_base_train.slice_tuples[i][1]) / float(_base_train.slice_tuples[i][2] - 1))
        for i in _indices
    ]
else:
    weights = [z_weight(float(z) / float(D - 1)) for (_, z, D) in _base_train.slice_tuples]

train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    worker_init_fn=seed_worker,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    worker_init_fn=seed_worker,
)

if IS_MAIN_PROCESS:
    accelerator.print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    if torch.cuda.is_available():
        accelerator.print("=== PyTorch CUDA / Slurm info ===")
        accelerator.print("torch.cuda.is_available():", torch.cuda.is_available())
        accelerator.print("torch.cuda.device_count():", torch.cuda.device_count())
        accelerator.print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))

# -------------------------------------------------------------------
# Model, diffusion process, optimizer, scheduler
# -------------------------------------------------------------------
# 1 channel target + 1 * slice_radius context channels
IN_CHANNELS = CENTER_CHANNELS + CENTER_CHANNELS * SLICE_RADIUS
OUT_CHANNELS = CENTER_CHANNELS

unet = UNet(
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    base_channels=64,
    channel_mults=(1, 2, 4, 8),
    time_emb_dim=256,
)

diffusion = GaussianDiffusion(
    model=unet,
    image_size=IMAGE_SIZE,
    channels=OUT_CHANNELS,
    timesteps=TIMESTEPS,
    schedule="cosine",
)

optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=PATIENCE,
)

diffusion, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    diffusion, optimizer, train_loader, val_loader, scheduler
)

# Create EMA *after* accelerator.prepare so the shadow copy lives on GPU
ema = EMA(accelerator.unwrap_model(diffusion).model, decay=0.9999)


# -------------------------------------------------------------------
# Training helpers
# -------------------------------------------------------------------
def train_one_epoch(epoch: int, max_steps: Union[int, None] = None) -> float:
    diffusion.train()
    base_diffusion = accelerator.unwrap_model(diffusion)
    running_loss_sum = 0.0
    count = 0
    start_time = time.time()

    for step, (x_center, x_context, z_pos, fg_frac) in enumerate(train_loader, start=1):
        x_center = x_center.to(device, non_blocking=True)
        x_context = x_context.to(device, non_blocking=True)
        z_pos = z_pos.to(device).float()
        fg_frac = fg_frac.to(device).float()

        b = x_center.size(0)
        t = torch.randint(0, base_diffusion.timesteps, (b,), device=device).long()

        optimizer.zero_grad(set_to_none=True)

        with accelerator.autocast():
            loss = base_diffusion.p_losses(x_center, t, z_pos, fg_frac=fg_frac, context=x_context)

        accelerator.backward(loss)
        optimizer.step()

        ema.update(base_diffusion.model)

        running_loss_sum += float(loss.item()) * b
        count += b

        if IS_MAIN_PROCESS and step % 500 == 0:
            avg_local = running_loss_sum / max(1, count)
            accelerator.print(f"[epoch {epoch} | step {step}] avg loss (local): {avg_local:.4f}")

        if max_steps is not None and step >= max_steps:
            break

    elapsed = time.time() - start_time
    avg_loss = reduce_avg_loss(running_loss_sum, count)

    if IS_MAIN_PROCESS:
        accelerator.print(f"Epoch {epoch} | Train loss: {avg_loss:.4f} | time: {elapsed:.1f}s")

    return avg_loss


@torch.no_grad()
def validate(epoch: int, max_steps: Union[int, None] = None) -> float:
    diffusion.eval()
    base_diffusion = accelerator.unwrap_model(diffusion)
    running_loss_sum = 0.0
    count = 0

    for step, (x_center, x_context, z_pos, fg_frac) in enumerate(val_loader, start=1):
        x_center = x_center.to(device, non_blocking=True)
        x_context = x_context.to(device, non_blocking=True)
        z_pos = z_pos.to(device).float()
        fg_frac = fg_frac.to(device).float()

        b = x_center.size(0)
        t = torch.randint(0, base_diffusion.timesteps, (b,), device=device).long()

        with accelerator.autocast():
            loss = base_diffusion.p_losses(x_center, t, z_pos, fg_frac=fg_frac, context=x_context)

        running_loss_sum += float(loss.item()) * b
        count += b

        if max_steps is not None and step >= max_steps:
            break

    avg_loss = reduce_avg_loss(running_loss_sum, count)

    if IS_MAIN_PROCESS:
        accelerator.print(f"Epoch {epoch} | Val loss:   {avg_loss:.4f}")

    return avg_loss


# -------------------------------------------------------------------
# Main training loop
# -------------------------------------------------------------------
def train() -> float:
    if IS_MAIN_PROCESS:
        accelerator.print("Starting Training")

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        maybe_set_epoch(train_loader, epoch)

        train_loss = train_one_epoch(epoch, max_steps=10 if DEBUG_FAST else None)
        val_loss = validate(epoch, max_steps=5 if DEBUG_FAST else None)

        if IS_MAIN_PROCESS:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=epoch)

        scheduler.step(val_loss)

        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            epochs_without_improvement = 0

            accelerator.wait_for_everyone()
            if IS_MAIN_PROCESS:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                best_path = MODELS_DIR / "ddpm_25d_mm_best.pt"

                state = {
                    "diffusion": accelerator.get_state_dict(diffusion),
                    "ema_unet": ema.state_dict(),
                }
                accelerator.save(state, str(best_path))

                accelerator.print(f"✅ New best val loss: {best_val:.4f}")
                mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
            accelerator.wait_for_everyone()
        else:
            epochs_without_improvement += 1
            if IS_MAIN_PROCESS:
                accelerator.print(f"⚠️ No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= PATIENCE:
            if IS_MAIN_PROCESS:
                accelerator.print(
                    f"⏹ Early stopping at epoch {epoch} "
                    f"(no val improvement for {PATIENCE} epochs)."
                )
            break

    if IS_MAIN_PROCESS:
        base_diffusion = accelerator.unwrap_model(diffusion)
        mlflow.pytorch.log_model(base_diffusion.model, artifact_path="final_model_unet")

        if best_val != float("inf"):
            mlflow.log_metric("best_val_loss", best_val)

    return best_val


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
def main() -> None:
    perun_out = PERUN_OUT_DIR / f"proc_{accelerator.process_index}"
    perun_out.mkdir(parents=True, exist_ok=True)

    if IS_MAIN_PROCESS:
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=RUN_IDENTIFIER):
            mlflow.log_params(
                {
                    "image_size": IMAGE_SIZE,
                    "batch_size": BATCH_SIZE,
                    "timesteps": TIMESTEPS,
                    "learning_rate": LEARNING_RATE,
                    "num_epochs": NUM_EPOCHS,
                    "patience": PATIENCE,
                    "min_delta": MIN_DELTA,
                    "device": str(device),
                    "model": "UNet",
                    "dataset": "BraTS FLAIR memmap",
                    "debug_fast": DEBUG_FAST,
                    "run_identifier": RUN_IDENTIFIER,
                    "num_workers": NUM_WORKERS,
                    "channels": CENTER_CHANNELS,
                    "SLICE_RADIUS": SLICE_RADIUS,
                    "accelerate_num_processes": accelerator.num_processes,
                    "accelerate_mixed_precision": str(accelerator.mixed_precision),
                }
            )

            best_val = run_with_perun(train, data_out=str(perun_out))
            if best_val is not None and best_val != float("inf"):
                mlflow.log_metric("best_val_loss", best_val)
    else:
        run_with_perun(train, data_out=str(perun_out))

    accelerator.end_training()


if __name__ == "__main__":
    main()

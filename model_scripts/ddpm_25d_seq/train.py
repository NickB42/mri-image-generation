from pathlib import Path
from typing import Union
import os
import uuid
import time
import random

import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator

import mlflow
import mlflow.pytorch

from .dataset import BraTSSliceDataset
from .unet import UNet
from .diffusion import GaussianDiffusion
from ..helpers.perun_utils import run_with_perun

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
EXPERIMENT_NAME = "ddpm_25d_seq"
RUN_IDENTIFIER = os.environ.get("SLURM_JOB_ID") or str(uuid.uuid4())

IMAGE_SIZE = 128
TIMESTEPS = 1000
PATIENCE = 3
LEARNING_RATE = 2e-4
MIN_DELTA = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 60

CENTER_MODALITIES = 4
SLICE_RADIUS = 2

NUM_WORKERS = 4
DEBUG_FAST = True

# Make split deterministic across processes
SPLIT_SEED = 42

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
    # When Accelerate prepares a dataloader in distributed mode, it attaches a DistributedSampler internally.
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
# Dataset and DataLoaders
# -------------------------------------------------------------------
if IS_MAIN_PROCESS:
    accelerator.print("Using DATASET_ROOT:", DATASET_ROOT)

train_dataset = BraTSSliceDataset(
    TRAIN_SET_ROOT,
    image_size=IMAGE_SIZE,
    slice_radius=SLICE_RADIUS,
)

val_dataset = BraTSSliceDataset(
    VAL_SET_ROOT,
    image_size=IMAGE_SIZE,
    slice_radius=SLICE_RADIUS,
)

if DEBUG_FAST:
    train_dataset = Subset(train_dataset, list(range(min(64, len(train_dataset)))))
    val_dataset   = Subset(val_dataset,   list(range(min(64, len(val_dataset)))))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
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
    accelerator.print(f"Train slices: {len(train_dataset)}, Val slices: {len(val_dataset)}")
    if torch.cuda.is_available():
        accelerator.print("=== PyTorch CUDA / Slurm info ===")
        accelerator.print("torch.cuda.is_available():", torch.cuda.is_available())
        accelerator.print("torch.cuda.device_count():", torch.cuda.device_count())
        accelerator.print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))

# -------------------------------------------------------------------
# Model, diffusion process, optimizer, scheduler
# -------------------------------------------------------------------
IN_CHANNELS = CENTER_MODALITIES + CENTER_MODALITIES * SLICE_RADIUS
OUT_CHANNELS = CENTER_MODALITIES

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
)

optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=PATIENCE,
)

# Prepare everything for distributed + mixed precision
diffusion, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    diffusion, optimizer, train_loader, val_loader, scheduler
)

# -------------------------------------------------------------------
# Training helpers
# -------------------------------------------------------------------
def train_one_epoch(epoch: int, max_steps: Union[int, None] = None) -> float:
    diffusion.train()
    base_diffusion = accelerator.unwrap_model(diffusion)
    running_loss_sum = 0.0
    count = 0
    n_steps = 0

    start_time = time.time()

    for step, (x_center, x_context, z_pos) in enumerate(train_loader, start=1):
        x_center = x_center.to(device, non_blocking=True)
        x_context = x_context.to(device, non_blocking=True)
        z_pos = z_pos.to(device).float()

        b = x_center.size(0)
        t = torch.randint(0, base_diffusion.timesteps, (b,), device=device).long()

        optimizer.zero_grad(set_to_none=True)

        with accelerator.autocast():
            loss = base_diffusion.p_losses(x_center, t, z_pos, context=x_context)

        accelerator.backward(loss)
        optimizer.step()

        running_loss_sum += float(loss.item()) * b
        count += b
        n_steps += 1

        if IS_MAIN_PROCESS and step % 500 == 0:
            avg_local = running_loss_sum / max(1, count)
            accelerator.print(f"[epoch {epoch} | step {step}] avg loss (local): {avg_local:.4f}")

        if max_steps is not None and step >= max_steps:
            break

    elapsed = time.time() - start_time
    avg_loss = reduce_avg_loss(running_loss_sum, count)
    steps_per_s = n_steps / max(elapsed, 1e-8)

    if IS_MAIN_PROCESS:
        accelerator.print(
            f"Epoch {epoch} | Train loss: {avg_loss:.4f} | "
            f"steps: {n_steps} | time: {elapsed:.1f}s | {steps_per_s:.2f} steps/s"
        )
        run = mlflow.active_run()
        if run is not None:
            mlflow.log_metric("train_steps_per_s", steps_per_s, step=epoch)
            mlflow.log_metric("train_num_steps", n_steps, step=epoch)
            mlflow.log_metric("train_epoch_time_s", elapsed, step=epoch)

    return avg_loss

@torch.no_grad()
def validate(epoch: int, max_steps: Union[int, None] = None) -> float:
    diffusion.eval()
    base_diffusion = accelerator.unwrap_model(diffusion)
    running_loss_sum = 0.0
    count = 0

    for step, (x_center, x_context, z_pos) in enumerate(val_loader, start=1):
        x_center = x_center.to(device, non_blocking=True)
        x_context = x_context.to(device, non_blocking=True)
        z_pos = z_pos.to(device).float()

        b = x_center.size(0)
        t = torch.randint(0, base_diffusion.timesteps, (b,), device=device).long()

        with accelerator.autocast():
            loss = base_diffusion.p_losses(x_center, t, z_pos, context=x_context)

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

        # Early stopping logic (val_loss is reduced => identical decision on all processes)
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            epochs_without_improvement = 0

            accelerator.wait_for_everyone()
            if IS_MAIN_PROCESS:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                best_path = MODELS_DIR / "2d_central_ddpm_flair_best.pt"

                # Save diffusion (includes model + diffusion buffers)
                state = accelerator.get_state_dict(diffusion)
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

    # Log final model (main process only)
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
    # Use per-process perun output to avoid collisions
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
                    "dataset": "BraTS 2D slices",
                    "debug_fast": DEBUG_FAST,
                    "run_identifier": RUN_IDENTIFIER,
                    "num_workers": NUM_WORKERS,
                    "modalities": CENTER_MODALITIES,
                    "SLICE_RADIUS": SLICE_RADIUS,
                    "accelerate_num_processes": accelerator.num_processes,
                    "accelerate_mixed_precision": str(accelerator.mixed_precision),
                    "split_seed": SPLIT_SEED,
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
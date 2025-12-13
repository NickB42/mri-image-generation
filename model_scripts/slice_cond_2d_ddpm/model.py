from pathlib import Path
from typing import Union, Optional
import os
from datetime import datetime
import uuid
import time

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch.amp import autocast, GradScaler

import mlflow
import mlflow.pytorch

from .dataset import BraTSSliceDataset
from .unet import UNet
from .diffusion import GaussianDiffusion
from ..helpers.perun_utils import run_with_perun
from ..helpers.signals import install_signal_handlers, should_terminate

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
EXPERIMENT_NAME = "slice_cond_2d_ddpm"
RUN_IDENTIFIER = os.environ.get("SLURM_JOB_ID") or str(uuid.uuid4())

IMAGE_SIZE = 128
DATASET_SUBSAMPLE_PORTION = 3
TIMESTEPS = 1000
PATIENCE = 4
LEARNING_RATE = 2e-4
MIN_DELTA = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 20

NUM_WORKERS = 4
DEBUG_FAST = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / EXPERIMENT_NAME
DATASET_ROOT = (PROJECT_ROOT / "../datasets/dataset").resolve()
PERUN_OUT_DIR = EXPERIMENT_ROOT / "perun_results" / RUN_IDENTIFIER

print("Using DATASET_ROOT:", DATASET_ROOT)

torch.backends.cudnn.benchmark = True
# -------------------------------------------------------------------
# Device setup
# -------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

print("=== PyTorch CUDA / Slurm info ===")
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"[GPU {i}] {props.name}, {props.total_memory / (1024**3):.1f} GB")

# -------------------------------------------------------------------
# Dataset and DataLoaders
# -------------------------------------------------------------------
full_dataset = BraTSSliceDataset(DATASET_ROOT, image_size=IMAGE_SIZE)

full_dataset = Subset(full_dataset, list(range(len(full_dataset) // DATASET_SUBSAMPLE_PORTION)))

if DEBUG_FAST:
    indices = list(range(64))
    full_dataset = Subset(full_dataset, indices)

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

print(f"Train slices: {len(train_dataset)}, Val slices: {len(val_dataset)}")


# -------------------------------------------------------------------
# Model, diffusion process, optimizer, scheduler
# -------------------------------------------------------------------
model = UNet(
    img_channels=1,
    base_channels=64,
    channel_mults=(1, 2, 4, 8),
    time_emb_dim=256,
)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for model training.")
    model = torch.nn.DataParallel(model)

model = model.to(device)

diffusion = GaussianDiffusion(
    model=model,
    image_size=IMAGE_SIZE,
    channels=1,
    timesteps=TIMESTEPS,
).to(device)

optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

# -------------------------------------------------------------------
# Training helpers
# -------------------------------------------------------------------
def train_one_epoch(epoch: int, max_steps: Union[int, None] = None) -> float:
    diffusion.train()
    running_loss = 0.0
    n_steps = 0

    start_time = time.time()

    for step, (x, z_pos) in enumerate(train_loader, start=1):
        # if should_terminate():
        #     print(f"[train_one_epoch] Termination requested at epoch {epoch}, step {step}. Breaking.")
        #     break

        x = x.to(device, non_blocking=True)          # (B, 1, H, W)
        z_pos = z_pos.to(device).float()             # (B,)

        t = torch.randint(
            0,
            diffusion.timesteps,
            (x.size(0),),
            device=device,
        ).long()

        optimizer.zero_grad()
        
        with autocast("cuda", enabled=(device.type == "cuda")):
            loss = diffusion.p_losses(x, t, z_pos)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        n_steps += 1

        if step % 500 == 0:
            avg = running_loss / n_steps
            print(f"[epoch {epoch} | step {step}] avg loss: {avg:.4f}")

        if max_steps is not None and step >= max_steps:
            break

    elapsed = time.time() - start_time
    avg_loss = running_loss / max(1, n_steps)
    steps_per_s = n_steps / max(elapsed, 1e-8)

    print(
        f"Epoch {epoch} | Train loss: {avg_loss:.4f} | "
        f"steps: {n_steps} | time: {elapsed:.1f}s | {steps_per_s:.2f} steps/s"
    )

    run = mlflow.active_run()
    if run is not None:
        mlflow.log_metric("train_steps_per_s", steps_per_s, step=epoch)
        mlflow.log_metric("train_num_steps", n_steps, step=epoch)

    return avg_loss

@torch.no_grad()
def validate(epoch: int, max_steps: Union[int, None] = None) -> float:
    diffusion.eval()
    running_loss = 0.0
    n_steps = 0

    for step, (x, z_pos) in enumerate(val_loader, start=1):
        x = x.to(device, non_blocking=True)
        z_pos = z_pos.to(device).float()

        t = torch.randint(
            0,
            diffusion.timesteps,
            (x.size(0),),
            device=device,
        ).long()

        with autocast("cuda", enabled=(device.type == "cuda")):
            loss = diffusion.p_losses(x, t, z_pos)
        
        running_loss += loss.item()
        n_steps += 1

        if max_steps is not None and step >= max_steps:
            break

    avg_loss = running_loss / max(1, n_steps)
    print(f"Epoch {epoch} | Val loss:   {avg_loss:.4f}")
    return avg_loss


# -------------------------------------------------------------------
# Main training loop
# -------------------------------------------------------------------
def train() -> float:
    print("Starting Training")

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        if should_terminate():
            print(f"[train] Termination requested before epoch {epoch}, stopping.")
            break

        train_loss = train_one_epoch(epoch, max_steps=10 if DEBUG_FAST else None)
        
        if should_terminate():
            print(f"[train] Termination requested after epoch {epoch}, stopping before validation.")
            break

        val_loss = validate(epoch, max_steps=5 if DEBUG_FAST else None)

        if should_terminate():
            print(f"[train] Termination requested after epoch {epoch} validation, stopping.")
            break

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric(
            "learning_rate",
            optimizer.param_groups[0]["lr"],
            step=epoch,
        )

        scheduler.step(val_loss)

        # Check for improvement
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            epochs_without_improvement = 0

            models_dir = EXPERIMENT_ROOT / "models" / RUN_IDENTIFIER
            models_dir.mkdir(parents=True, exist_ok=True)
            model_path = models_dir / "2d_central_ddpm_flair_best.pt"

            torch.save(diffusion.state_dict(), str(model_path))
            print(f"✅ New best val loss: {best_val:.4f}")

            mlflow.log_artifact(
                str(model_path),
                artifact_path="checkpoints",
            )
        else:
            epochs_without_improvement += 1
            print(f"⚠️ No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= PATIENCE:
            print(
                f"⏹ Early stopping at epoch {epoch} "
                f"(no val improvement for {PATIENCE} epochs).",
            )
            break

    return best_val

def main() -> None:
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
                "dataset": "BraTS_2D_central_slice_flair",
                "debug_fast": DEBUG_FAST,
                "run_identifier": RUN_IDENTIFIER,
                "num_workers": NUM_WORKERS,
                "dataset_subsample_portion": DATASET_SUBSAMPLE_PORTION,
            }
        )

        best_val = run_with_perun(
            train,
            data_out=str(PERUN_OUT_DIR),
        )

        mlflow.pytorch.log_model(diffusion.model, artifact_path="final_model")

        if best_val is not None:
            mlflow.log_metric("best_val_loss", best_val)


if __name__ == "__main__":
    install_signal_handlers()
    main()
from pathlib import Path
from typing import Union, Optional
import json, os
from datetime import datetime
import sys
import uuid
import signal
import time

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch.amp import autocast, GradScaler
from torchvision.utils import save_image

import mlflow
import mlflow.pytorch

import perun
from perun.data_model.data import MetricType, DataNode
from perun.processing import processDataNode

from .dataset import BraTSSliceDataset
from .unet import UNet
from .diffusion import GaussianDiffusion

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
EXPERIMENT_NAME = "25d_ddpm_all_modalities"
RUN_IDENTIFIER = os.environ.get("SLURM_JOB_ID") or str(uuid.uuid4())

IMAGE_SIZE = 128
DATASET_SUBSAMPLE_PORTION = 2
TIMESTEPS = 1000
PATIENCE = 4
LEARNING_RATE = 2e-4
MIN_DELTA = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 50

CENTER_MODALITIES = 4
SLICE_RADIUS = 2
CONTEXT_SLICES = 2 * SLICE_RADIUS

NUM_WORKERS = 2
DEBUG_FAST = False
SHOULD_TERMINATE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / EXPERIMENT_NAME
DATASET_ROOT = (PROJECT_ROOT / "../dataset").resolve()

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
# Signal handlers for graceful shutdown
# -------------------------------------------------------------------
def _handle_sigusr1(signum, frame):
    global SHOULD_TERMINATE
    print(f"[Signal] Received SIGUSR1 ({signum}). Requested graceful shutdown.")
    SHOULD_TERMINATE = True

def _handle_sigterm(signum, frame):
    global SHOULD_TERMINATE
    print(f"[Signal] Received SIGTERM ({signum}). Emergency shutdown requested.")
    SHOULD_TERMINATE = True

def install_signal_handlers():
    signal.signal(signal.SIGUSR1, _handle_sigusr1)
    signal.signal(signal.SIGTERM, _handle_sigterm)
# -------------------------------------------------------------------
# Dataset and DataLoaders
# -------------------------------------------------------------------
full_dataset = BraTSSliceDataset(
    DATASET_ROOT,
    image_size=IMAGE_SIZE,
    slice_radius=SLICE_RADIUS,
)

full_dataset = Subset(full_dataset, list(range(len(full_dataset) // DATASET_SUBSAMPLE_PORTION)))
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
IN_CHANNELS = CENTER_MODALITIES + CENTER_MODALITIES * CONTEXT_SLICES
OUT_CHANNELS = CENTER_MODALITIES

model = UNet(
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
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
    channels=OUT_CHANNELS,
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
    global SHOULD_TERMINATE
    diffusion.train()
    running_loss = 0.0
    n_steps = 0

    start_time = time.time()

    for step, (x_center, x_context, z_pos) in enumerate(train_loader, start=1):
        # if SHOULD_TERMINATE:
        #     print(f"[train_one_epoch] Termination requested at epoch {epoch}, step {step}. Breaking.")
        #     break

        x_center = x_center.to(device, non_blocking=True)    # (B, 4, H, W)
        x_context = x_context.to(device, non_blocking=True)  # (B, 8, H, W)
        z_pos = z_pos.to(device).float()                     # (B,)

        t = torch.randint(
            0,
            diffusion.timesteps,
            (x_center.size(0),),
            device=device,
        ).long()

        optimizer.zero_grad()
        
        with autocast("cuda", enabled=(device.type == "cuda")):
            loss = diffusion.p_losses(x_center, t, z_pos, context=x_context)

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

    for step, (x_center, x_context, z_pos) in enumerate(val_loader, start=1):
        x_center = x_center.to(device, non_blocking=True)
        x_context = x_context.to(device, non_blocking=True)
        z_pos = z_pos.to(device).float()

        t = torch.randint(
            0,
            diffusion.timesteps,
            (x_center.size(0),),
            device=device,
        ).long()

        with autocast("cuda", enabled=(device.type == "cuda")):
            loss = diffusion.p_losses(x_center, t, z_pos, context=x_context)

        running_loss += loss.item()
        n_steps += 1

        if max_steps is not None and step >= max_steps:
            break

    avg_loss = running_loss / max(1, n_steps)
    print(f"Epoch {epoch} | Val loss:   {avg_loss:.4f}")
    return avg_loss

# -------------------------------------------------------------------
# Perun ↔ MLflow bridge
# -------------------------------------------------------------------
def log_perun_metrics_to_mlflow(root: DataNode) -> None:
    cfg = getattr(perun, "config", None)
    processed_root = processDataNode(root, cfg, force_process=False) if cfg is not None else root

    def find_first_metric(node: DataNode, metric_type: MetricType):
        metrics = getattr(node, "metrics", None)
        if metrics and metric_type in metrics:
            return float(metrics[metric_type].value)

        for child in getattr(node, "nodes", {}).values():
            val = find_first_metric(child, metric_type)
            if val is not None:
                return val
        return None

    run = mlflow.active_run()
    if run is None:
        return

    total_energy_j = find_first_metric(processed_root, MetricType.ENERGY)
    runtime_s = find_first_metric(processed_root, MetricType.RUNTIME)
    co2_kg = find_first_metric(processed_root, MetricType.CO2)
    money = find_first_metric(processed_root, MetricType.MONEY)

    def log_if_not_none(name: str, value):
        if value is not None:
            mlflow.log_metric(name, float(value))

    log_if_not_none("perun_energy_joules", total_energy_j)
    log_if_not_none("perun_runtime_seconds", runtime_s)
    log_if_not_none("perun_co2_kg", co2_kg)
    log_if_not_none("perun_cost", money)

    if total_energy_j is not None:
        energy_kwh = total_energy_j / 3.6e6
        log_if_not_none("perun_energy_kwh", energy_kwh)

    if total_energy_j is not None and runtime_s is not None and runtime_s > 0:
        avg_power_w = total_energy_j / runtime_s
        log_if_not_none("perun_avg_power_watts", avg_power_w)

# -------------------------------------------------------------------
# Main training loop
# -------------------------------------------------------------------
def train() -> float:
    global SHOULD_TERMINATE

    print("Starting Training")

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        if SHOULD_TERMINATE:
            print(f"[train] Termination requested before epoch {epoch}, stopping.")
            break

        train_loss = train_one_epoch(epoch, max_steps=10 if DEBUG_FAST else None)
        
        if SHOULD_TERMINATE:
            print(f"[train] Termination requested after epoch {epoch}, stopping before validation.")
            break

        val_loss = validate(epoch, max_steps=5 if DEBUG_FAST else None)

        if SHOULD_TERMINATE:
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


@perun.perun(
    data_out=str(EXPERIMENT_ROOT / "perun_results" / RUN_IDENTIFIER),
    format="json",
)
def train_with_perun():
    try:
        perun.register_callback(log_perun_metrics_to_mlflow)
    except Exception as e:
        print(f"Perun callback registration failed: {e}")

    return train()


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
                "modalities": CENTER_MODALITIES,
                "context_slices": CONTEXT_SLICES,
            }
        )

        best_val = train_with_perun()

        mlflow.pytorch.log_model(diffusion.model, artifact_path="final_model")

        if best_val is not None:
            mlflow.log_metric("best_val_loss", best_val)


if __name__ == "__main__":
    install_signal_handlers()
    main()
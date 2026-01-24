from __future__ import annotations

import os
import uuid
import random
import time
from pathlib import Path
from itertools import cycle
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from accelerate import Accelerator

import mlflow
import mlflow.pytorch

from .brats_dataset import BraTSVolumeDataset
from .cdpm25d import CDPM25D, GaussianDiffusion, train_step, generate_volume_staged

# -------------------------------------------------------------------
# Configuration (constants instead of argparse)
# -------------------------------------------------------------------
EXPERIMENT_NAME = "cdpm25d"
RUN_IDENTIFIER = os.environ.get("SLURM_JOB_ID") or str(uuid.uuid4())

# Data
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = (PROJECT_ROOT / "../datasets").resolve()
TRAIN_SET_ROOT = DATASET_ROOT / "train"
VAL_SET_ROOT   = DATASET_ROOT / "val"

MODALITY = os.environ.get("MODALITY", "t1")  # one of: t1, t1ce, t2, flair

TARGET_SHAPE = (128, 128, 128)  # (D,H,W) after preprocessing in dataset
RESIZE = True

BATCH_SIZE = 3
NUM_WORKERS = 4

# Training
TOTAL_STEPS = 200_000
LEARNING_RATE = 1e-4
TAU_MAX = 20
DIFFUSION_T = 1000

LOG_EVERY = 100
SAVE_EVERY = 5000

SAMPLE_EVERY = 0
STAGE_SIZE = 10

# Checkpoints / outputs
EXPERIMENT_ROOT = PROJECT_ROOT / "runs" / EXPERIMENT_NAME / RUN_IDENTIFIER
MODELS_DIR = EXPERIMENT_ROOT / "checkpoints"
SAMPLES_DIR = EXPERIMENT_ROOT / "samples"

# Resume (set to a checkpoint path string or leave empty)
RESUME_PATH = os.environ.get("RESUME_PATH", "")  # e.g. ".../ckpt_step_50000.pt"

# Debug
DEBUG_FAST = True
DEBUG_TRAIN_N = 20
DEBUG_VAL_N = 10
DEBUG_TOTAL_STEPS = 2000

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
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def reduce_avg_loss(running_loss_sum: float, count: int) -> float:
    # Reduce sums across all processes, then divide.
    loss_t = torch.tensor(running_loss_sum, device=device, dtype=torch.float64)
    cnt_t = torch.tensor(count, device=device, dtype=torch.float64)
    loss_t = accelerator.reduce(loss_t, reduction="sum")
    cnt_t = accelerator.reduce(cnt_t, reduction="sum")
    return (loss_t / torch.clamp(cnt_t, min=1.0)).item()

def save_checkpoint(path: Path, model: torch.nn.Module, opt: torch.optim.Optimizer, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only main process writes; accelerator.save is safe to call on all, but we gate anyway.
    if IS_MAIN_PROCESS:
        payload = {
            "model": accelerator.get_state_dict(model),
            "opt": opt.state_dict(),
            "step": int(step),
        }
        accelerator.save(payload, str(path))

def load_checkpoint(path: Path, model: torch.nn.Module, opt: Optional[torch.optim.Optimizer] = None) -> int:
    ckpt = torch.load(str(path), map_location="cpu")
    # Load into the *unwrapped* model to avoid wrapper key mismatches
    base_model = accelerator.unwrap_model(model)
    base_model.load_state_dict(ckpt["model"], strict=True)
    if opt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    return int(ckpt.get("step", 0))

@torch.no_grad()
def quick_sample(
    model: torch.nn.Module,
    diffusion: torch.nn.Module,
    out_path: Path,
    shape_dhw: Tuple[int, int, int],
    stage_size: int,
) -> None:
    # Only run sampling on main process to avoid duplicated files & GPU work
    if not IS_MAIN_PROCESS:
        return

    base_model = accelerator.unwrap_model(model)
    base_model.eval()

    D, H, W = shape_dhw
    vol = generate_volume_staged(
        model=base_model,
        diffusion=diffusion,
        D=D,
        H=H,
        W=W,
        stage_size=stage_size,
        device=device,
    )  # [1,D,H,W]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vol.detach().cpu(), str(out_path))
    accelerator.print(f"[sample] saved torch volume to: {out_path} (shape={tuple(vol.shape)})")

# -------------------------------------------------------------------
# Dataset and DataLoaders (no DistributedSampler needed)
# -------------------------------------------------------------------
train_dataset = BraTSVolumeDataset(
    data_root=str(TRAIN_SET_ROOT),
    modality=MODALITY,
    target_shape=TARGET_SHAPE,
    resize=RESIZE,
)

val_dataset = BraTSVolumeDataset(
    data_root=str(VAL_SET_ROOT),
    modality=MODALITY,
    target_shape=TARGET_SHAPE,
    resize=RESIZE,
)

train_ids = {cid for _, cid in train_dataset}
val_ids   = {cid for _, cid in val_dataset}
overlap = train_ids & val_ids
assert len(overlap) == 0, f"Train/Val overlap: {sorted(list(overlap))[:10]}"

if DEBUG_FAST:
    train_ds = Subset(train_dataset, list(range(min(DEBUG_TRAIN_N, len(train_dataset)))))
    val_ds = Subset(val_dataset, list(range(min(DEBUG_VAL_N, len(val_dataset)))))

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    worker_init_fn=seed_worker,
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

# Determine (D,H,W) after preprocessing
sample_vol, _ = train_ds[0]
D, H, W = tuple(sample_vol.shape)

if IS_MAIN_PROCESS:
    accelerator.print(f"Train volumes: {len(train_ds)}, Val volumes: {len(val_ds)}")
    accelerator.print(f"Volume shape (D,H,W): {(D,H,W)}")
    if torch.cuda.is_available():
        accelerator.print("=== PyTorch CUDA / Slurm info ===")
        accelerator.print("torch.cuda.is_available():", torch.cuda.is_available())
        accelerator.print("torch.cuda.device_count():", torch.cuda.device_count())
        accelerator.print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))

# -------------------------------------------------------------------
# Models / Diffusion / Optimizer
# -------------------------------------------------------------------
model = CDPM25D(
    tau_max=TAU_MAX,
    volume_depth=D,
    base_channels=64,
    channel_mults=(1, 2, 4, 8),
    num_res_blocks=2,
    time_emb_dim=256,
    attn_heads=4,
)

diffusion = GaussianDiffusion(T=DIFFUSION_T)  # keep same API as your original code

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

start_step = 0
if RESUME_PATH:
    resume_p = Path(RESUME_PATH)
    if resume_p.is_file():
        start_step = load_checkpoint(resume_p, model, optimizer)
        if IS_MAIN_PROCESS:
            accelerator.print(f"[resume] loaded {resume_p} at step={start_step}")
    elif IS_MAIN_PROCESS:
        accelerator.print(f"[resume] RESUME_PATH set but file not found: {resume_p}")

# Prepare everything for distributed + mixed precision
model, optimizer, train_loader, val_loader, diffusion = accelerator.prepare(
    model, optimizer, train_loader, val_loader, diffusion
)

# -------------------------------------------------------------------
# Training loop (step-based like your original script)
# -------------------------------------------------------------------
def train_steps() -> None:
    total_steps = DEBUG_TOTAL_STEPS if DEBUG_FAST else TOTAL_STEPS
    if start_step >= total_steps:
        if IS_MAIN_PROCESS:
            accelerator.print(f"[done] start_step ({start_step}) >= total_steps ({total_steps}); nothing to do.")
        return

    train_iter = cycle(train_loader)

    running_loss_sum = 0.0
    running_count = 0

    t0 = time.time()

    for step in range(start_step, total_steps):
        model.train()

        vols, _ = next(train_iter)  # vols: [B,D,H,W]
        vols = vols.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with accelerator.autocast():
            loss = train_step(model=model, diffusion=diffusion, volumes=vols)

        accelerator.backward(loss)
        optimizer.step()

        b = vols.size(0)
        running_loss_sum += float(loss.detach().item()) * b
        running_count += b

        # Logging
        if (step + 1) % LOG_EVERY == 0:
            avg_loss = reduce_avg_loss(running_loss_sum, running_count)
            running_loss_sum = 0.0
            running_count = 0

            if IS_MAIN_PROCESS:
                elapsed = time.time() - t0
                steps_per_s = (step + 1 - start_step) / max(elapsed, 1e-8)
                accelerator.print(f"step={step+1}  loss={avg_loss:.6f}  ({steps_per_s:.2f} steps/s)")
                run = mlflow.active_run()
                if run is not None:
                    mlflow.log_metric("train_loss", avg_loss, step=step + 1)
                    mlflow.log_metric("steps_per_s", steps_per_s, step=step + 1)
                    mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=step + 1)

        # Checkpoint
        if (step + 1) % SAVE_EVERY == 0:
            ckpt_path = MODELS_DIR / f"ckpt_step_{step+1}.pt"
            accelerator.wait_for_everyone()
            save_checkpoint(ckpt_path, model, optimizer, step + 1)
            accelerator.wait_for_everyone()

            if IS_MAIN_PROCESS:
                accelerator.print(f"[ckpt] saved: {ckpt_path}")
                run = mlflow.active_run()
                if run is not None:
                    mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")

        # Sample
        if SAMPLE_EVERY and (step + 1) % SAMPLE_EVERY == 0:
            sample_path = SAMPLES_DIR / f"sample_step_{step+1}.pt"
            accelerator.wait_for_everyone()
            quick_sample(
                model=model,
                diffusion=diffusion,
                out_path=sample_path,
                shape_dhw=(D, H, W),
                stage_size=STAGE_SIZE,
            )
            accelerator.wait_for_everyone()
            if IS_MAIN_PROCESS:
                run = mlflow.active_run()
                if run is not None:
                    mlflow.log_artifact(str(sample_path), artifact_path="samples")

    # Final
    final_path = MODELS_DIR / "ckpt_final.pt"
    accelerator.wait_for_everyone()
    save_checkpoint(final_path, model, optimizer, total_steps)
    accelerator.wait_for_everyone()

    if IS_MAIN_PROCESS:
        accelerator.print(f"[done] saved: {final_path}")
        run = mlflow.active_run()
        if run is not None:
            mlflow.log_artifact(str(final_path), artifact_path="checkpoints")
            # Optional: log model (unwrap so MLflow doesn't store wrappers)
            try:
                mlflow.pytorch.log_model(accelerator.unwrap_model(model), artifact_path="final_cdpm25d")
            except Exception as e:
                accelerator.print(f"[mlflow] could not log model object: {e!r}")

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
def main() -> None:
    if IS_MAIN_PROCESS:
        EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=RUN_IDENTIFIER):
            mlflow.log_params(
                {
                    "experiment_name": EXPERIMENT_NAME,
                    "run_identifier": RUN_IDENTIFIER,
                    "modality": MODALITY,
                    "target_shape_d": TARGET_SHAPE[0],
                    "target_shape_h": TARGET_SHAPE[1],
                    "target_shape_w": TARGET_SHAPE[2],
                    "resize": RESIZE,
                    "batch_size": BATCH_SIZE,
                    "num_workers": NUM_WORKERS,
                    "total_steps": (DEBUG_TOTAL_STEPS if DEBUG_FAST else TOTAL_STEPS),
                    "learning_rate": LEARNING_RATE,
                    "tau_max": TAU_MAX,
                    "diffusion_T": DIFFUSION_T,
                    "log_every": LOG_EVERY,
                    "save_every": SAVE_EVERY,
                    "sample_every": SAMPLE_EVERY,
                    "stage_size": STAGE_SIZE,
                    "debug_fast": DEBUG_FAST,
                    "accelerate_num_processes": accelerator.num_processes,
                    "accelerate_mixed_precision": str(accelerator.mixed_precision),
                    "device": str(device),
                    "model": "CDPM25D",
                    "dataset": "BraTSVolumeDataset",
                }
            )
            train_steps()
    else:
        train_steps()

    accelerator.end_training()

if __name__ == "__main__":
    main()
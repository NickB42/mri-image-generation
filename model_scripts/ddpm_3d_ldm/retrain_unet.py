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
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from accelerate import Accelerator

import mlflow
import mlflow.pytorch

from .dataset import BraTS3DVolumeDataset
from .vae import VAE3D
from .unet_attention import UNet3DModelWithAttention
from .diffusion import GaussianDiffusionLatent3D
from ..helpers.perun_utils import run_with_perun
from ..helpers.signals import install_signal_handlers, should_terminate

# -------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------
EXPERIMENT_NAME = "ddpm_3d_ldm"
RUN_IDENTIFIER = os.environ.get("SLURM_JOB_ID") or str(uuid.uuid4())

PATCH_SIZE = (128, 160, 160)
TIMESTEPS = 200

# VAE architecture must match checkpoint
VAE_BASE_CHANNELS = 32
VAE_NUM_DOWN = 3
LATENT_CHANNELS = 16

# UNet / LDM hyperparams
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
DATASET_ROOT = (PROJECT_ROOT / "../datasets").resolve()
TRAIN_SET_ROOT = DATASET_ROOT / "train"
VAL_SET_ROOT = DATASET_ROOT / "val"

EXPERIMENT_ROOT = PROJECT_ROOT / EXPERIMENT_NAME
PERUN_OUT_DIR = EXPERIMENT_ROOT / "perun_results" / RUN_IDENTIFIER
MODELS_DIR = EXPERIMENT_ROOT / "models" / RUN_IDENTIFIER

VAE_CKPT = EXPERIMENT_ROOT/ "models" / "1595833" / "vae3d_final.pt"
LATENT_SCALE = 1.0
ESTIMATE_LATENT_SCALE = True
LATENT_SCALE_BATCHES = 200

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# -------------------------------------------------------------------
# Device / DDP setup
# -------------------------------------------------------------------
def setup_distributed():
    """Return (device, rank, world_size, local_rank, is_distributed)."""
    if torch.cuda.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        is_distributed = world_size > 1
        print(f"[DDP] rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        rank = 0
        world_size = 1
        local_rank = 0
        is_distributed = False
        print(f"[Single process] Using device: {device}")

    return device, rank, world_size, local_rank, is_distributed


device, rank, world_size, local_rank, IS_DISTRIBUTED = setup_distributed()
IS_MAIN_PROCESS = (rank == 0)


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def ddp_broadcast_bool(flag: bool) -> bool:
    if not IS_DISTRIBUTED:
        return flag
    t = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    dist.broadcast(t, src=0)
    return bool(t.item())


def ddp_reduce_mean(sum_loss: float, count: int) -> float:
    if not IS_DISTRIBUTED:
        return sum_loss / max(1, count)
    t = torch.tensor([sum_loss, count], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / t[1]).item()


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_unwrapped_model(m):
    return m.module if hasattr(m, "module") else m


def _strip_module_prefix(state_dict):
    # Supports checkpoints saved from DDP where keys might start with "module."
    keys = list(state_dict.keys())
    if len(keys) > 0 and all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint_strict(model: torch.nn.Module, ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load to CPU first to avoid GPU memory spikes. :contentReference[oaicite:1]{index=1}
    state = torch.load(str(ckpt_path), map_location="cpu")
    state = _strip_module_prefix(state)

    incompat = model.load_state_dict(state, strict=True)
    # In current PyTorch, strict=True should raise on mismatch; incompat usually empty.
    if hasattr(incompat, "missing_keys") and (incompat.missing_keys or incompat.unexpected_keys):
        print("[WARN] IncompatibleKeys:", incompat)


# -------------------------------------------------------------------
# Dataset & loaders
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
        print(f"[DEBUG FAST] train size {len(train_dataset)}, val size {len(val_dataset)}")

if IS_DISTRIBUTED and device.type == "cuda":
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
else:
    train_sampler = None

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    worker_init_fn=seed_worker,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# Only rank0 validates in distributed mode (like your script)
if IS_DISTRIBUTED and device.type == "cuda":
    if rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            sampler=None,
            worker_init_fn=seed_worker,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
    else:
        val_loader = None
else:
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=None,
        worker_init_fn=seed_worker,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

if IS_MAIN_PROCESS:
    print(f"Train volumes: {len(train_dataset)}, Val volumes: {len(val_dataset)}")

# -------------------------------------------------------------------
# Latent scale helper
# -------------------------------------------------------------------
@torch.no_grad()
def estimate_latent_scale(vae: VAE3D, loader: DataLoader, device: torch.device, num_batches: int = 200) -> float:
    vae.eval()
    vars_ = []
    for i, x in enumerate(loader):
        if i >= num_batches:
            break
        x = x.to(device, non_blocking=True)
        z = vae.encode_to_latent(x).float()
        vars_.append(z.var(unbiased=False).item())

    v = float(np.mean(vars_)) if len(vars_) else 1.0
    return 1.0 / math.sqrt(max(v, 1e-8))


# -------------------------------------------------------------------
# LDM training/validation
# -------------------------------------------------------------------
def train_ldm_one_epoch(
    epoch: int,
    diffusion: GaussianDiffusionLatent3D,
    vae_frozen: VAE3D,
    optimizer_ldm: torch.optim.Optimizer,
    scaler_ldm: GradScaler,
    latent_scale: float,
    max_steps: Union[int, None] = None
) -> float:
    diffusion.train()
    vae_frozen.eval()

    running_loss = 0.0
    n_steps = 0
    count = 0
    start_time = time.time()

    for step, x in enumerate(train_loader, start=1):
        if should_terminate():
            print(f"[train_ldm_one_epoch] Termination requested at epoch {epoch}, step {step}. Breaking.")
            break

        x = x.to(device, non_blocking=True)

        # Encode with frozen VAE (no grads) to latents
        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                z = vae_frozen.encode_to_latent(x).float() * latent_scale

        b = z.size(0)
        t = torch.randint(1, diffusion.timesteps, (b,), device=device)

        optimizer_ldm.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss = diffusion.p_losses(z, t, cond=None, min_snr_gamma=5.0)

        scaler_ldm.scale(loss).backward()
        scaler_ldm.step(optimizer_ldm)
        scaler_ldm.update()

        running_loss += loss.item() * b
        count += b
        n_steps += 1

        if IS_MAIN_PROCESS and step % 100 == 0:
            print(f"[LDM epoch {epoch} | step {step}] avg loss: {(running_loss / count):.4f}")

        if max_steps is not None and step >= max_steps:
            break

    elapsed = time.time() - start_time
    avg_loss = ddp_reduce_mean(running_loss, count)
    steps_per_s = n_steps / max(elapsed, 1e-8)

    if IS_MAIN_PROCESS:
        print(
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
def validate_ldm(
    epoch: int,
    diffusion: GaussianDiffusionLatent3D,
    vae_frozen: VAE3D,
    latent_scale: float,
    max_steps: Union[int, None] = None
) -> float:
    diffusion.eval()
    vae_frozen.eval()

    if val_loader is None:
        return 0.0

    running_loss = 0.0
    count = 0

    fixed_ts = torch.linspace(1, diffusion.timesteps - 1, steps=8, device=device).long()

    for step, x in enumerate(val_loader, start=1):
        x = x.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            z = vae_frozen.encode_to_latent(x).float() * latent_scale

        b = z.size(0)
        t = fixed_ts[(step - 1) % len(fixed_ts)].expand(b)

        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss = diffusion.p_losses(z, t, cond=None, min_snr_gamma=5.0)

        running_loss += loss.item() * b
        count += b

        if max_steps is not None and step >= max_steps:
            break

    avg_loss = running_loss / max(1, count)

    if IS_MAIN_PROCESS:
        print(f"LDM Epoch {epoch} | Val loss: {avg_loss:.4f}")

    return avg_loss


# -------------------------------------------------------------------
# Main: load VAE, freeze it, train UNet/LDM only
# -------------------------------------------------------------------
def train_unet_only() -> float:
    # --- Build + load frozen VAE ---
    vae = VAE3D(
        in_channels=4,
        base_channels=VAE_BASE_CHANNELS,
        num_down=VAE_NUM_DOWN,
        latent_channels=LATENT_CHANNELS,
    ).to(device)

    load_checkpoint_strict(vae, Path(VAE_CKPT))

    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()

    # --- Build UNet (fresh) ---
    unet = UNet3DModelWithAttention(
        in_channels=LATENT_CHANNELS,
        base_channels=UNET_BASE_CHANNELS,
        channel_mults=UNET_CHANNEL_MULTS,
    ).to(device)

    if IS_DISTRIBUTED and device.type == "cuda":
        unet = DDP(unet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    diffusion = GaussianDiffusionLatent3D(
        model=unet,
        channels=LATENT_CHANNELS,
        timesteps=TIMESTEPS,
    ).to(device)

    optimizer_ldm = torch.optim.Adam(diffusion.model.parameters(), lr=LDM_LEARNING_RATE)
    scheduler_ldm = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ldm, mode="min", factor=0.5, patience=3
    )
    scaler_ldm = GradScaler("cuda", enabled=(device.type == "cuda"))

    # --- Latent scale ---
    latent_scale = float(LATENT_SCALE)

    if ESTIMATE_LATENT_SCALE:
        if IS_MAIN_PROCESS:
            scale_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                sampler=None,
                worker_init_fn=seed_worker,
                num_workers=NUM_WORKERS,
                pin_memory=True,
            )
            latent_scale = estimate_latent_scale(
                vae, scale_loader, device,
                num_batches=20 if DEBUG_FAST else LATENT_SCALE_BATCHES
            )
            print(f"[latent] LATENT_SCALE={latent_scale:.6f}")
            run = mlflow.active_run()
            if run is not None:
                mlflow.log_param("latent_scale", latent_scale)

        if IS_DISTRIBUTED:
            tscale = torch.tensor([latent_scale], device=device, dtype=torch.float32)
            dist.broadcast(tscale, src=0)
            latent_scale = float(tscale.item())

    # --- Train loop ---
    best_val = float("inf")
    epochs_without_improvement = 0
    best_path: Optional[Path] = None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, LDM_NUM_EPOCHS + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if should_terminate():
            print(f"[train] Termination requested before LDM epoch {epoch}, stopping.")
            break

        if IS_DISTRIBUTED and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss = train_ldm_one_epoch(
            epoch, diffusion, vae, optimizer_ldm, scaler_ldm, latent_scale,
            max_steps=5 if DEBUG_FAST else None
        )

        if IS_DISTRIBUTED:
            dist.barrier()

        val_loss = validate_ldm(
            epoch, diffusion, vae, latent_scale,
            max_steps=2 if DEBUG_FAST else None
        )

        if IS_DISTRIBUTED:
            t = torch.tensor([val_loss], device=device, dtype=torch.float32)
            dist.broadcast(t, src=0)
            val_loss = float(t.item())
            dist.barrier()

        if IS_MAIN_PROCESS:
            mlflow.log_metric("ldm_train_loss", train_loss, step=epoch)
            mlflow.log_metric("ldm_val_loss", val_loss, step=epoch)
            mlflow.log_metric("ldm_learning_rate", optimizer_ldm.param_groups[0]["lr"], step=epoch)

        scheduler_ldm.step(val_loss)

        # Best checkpoint
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            epochs_without_improvement = 0

            if IS_MAIN_PROCESS:
                best_path = MODELS_DIR / "3d_ldm_unet_best.pt"
                unet_to_save = get_unwrapped_model(diffusion.model)
                torch.save(unet_to_save.state_dict(), str(best_path))
                mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
                print(f"✅ New best UNet val loss: {best_val:.4f}")
        else:
            epochs_without_improvement += 1
            if IS_MAIN_PROCESS:
                print(f"⚠️ No UNet improvement for {epochs_without_improvement} epoch(s)")

        stop_now = False
        if IS_MAIN_PROCESS and epochs_without_improvement >= PATIENCE:
            stop_now = True
            print(f"⏹ Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs).")

        stop_now = ddp_broadcast_bool(stop_now)
        if stop_now:
            break

    if IS_MAIN_PROCESS:
        mlflow.pytorch.log_model(get_unwrapped_model(diffusion.model), artifact_path="final_ldm_unet_retrained")
        # Log the frozen VAE as reference (unchanged)
        mlflow.pytorch.log_model(vae, artifact_path="frozen_vae_reference")
        if best_val != float("inf"):
            mlflow.log_metric("best_ldm_val_loss", best_val)

    return best_val


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    if IS_MAIN_PROCESS:
        with mlflow.start_run(run_name=RUN_IDENTIFIER):
            mlflow.log_params(
                {
                    "patch_size_d": PATCH_SIZE[0],
                    "patch_size_h": PATCH_SIZE[1],
                    "patch_size_w": PATCH_SIZE[2],
                    "batch_size": BATCH_SIZE,
                    "timesteps": TIMESTEPS,
                    "ldm_learning_rate": LDM_LEARNING_RATE,
                    "ldm_num_epochs": LDM_NUM_EPOCHS,
                    "patience": PATIENCE,
                    "min_delta": MIN_DELTA,
                    "device": str(device),
                    "model_vae": "VAE3D (frozen)",
                    "model_ldm_unet": "UNet3DModelWithAttention (retrained)",
                    "dataset": "BraTS_3D_4modalities",
                    "debug_fast": DEBUG_FAST,
                    "run_identifier": RUN_IDENTIFIER,
                    "num_workers": NUM_WORKERS,
                    "vae_base_channels": VAE_BASE_CHANNELS,
                    "vae_num_down": VAE_NUM_DOWN,
                    "latent_channels": LATENT_CHANNELS,
                    "unet_base_channels": UNET_BASE_CHANNELS,
                    "unet_channel_mults": str(UNET_CHANNEL_MULTS),
                    "vae_ckpt": VAE_CKPT,
                    "estimate_latent_scale": ESTIMATE_LATENT_SCALE,
                    "retrain_unet_only": True,
                }
            )

            best_val = run_with_perun(train_unet_only, data_out=str(PERUN_OUT_DIR))

            if best_val is not None and best_val != float("inf"):
                mlflow.log_metric("best_ldm_val_loss", best_val)
    else:
        run_with_perun(train_unet_only, data_out=str(PERUN_OUT_DIR))

    cleanup_distributed()


if __name__ == "__main__":
    install_signal_handlers()
    main()

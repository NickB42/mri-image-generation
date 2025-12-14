from pathlib import Path
from typing import Union, Optional
import os
import uuid
import time

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch.amp import autocast, GradScaler
import torch.nn.functional as F

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import mlflow
import mlflow.pytorch

from .dataset import BraTS3DVolumeDataset
from .vae import VAE3D
from .unet_attention import UNet3DModelWithAttention
from .diffusion import GaussianDiffusionLatent3D
from ..helpers.perun_utils import run_with_perun
from ..helpers.signals import install_signal_handlers, should_terminate

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
EXPERIMENT_NAME = "ddpm_3d_ldm"
RUN_IDENTIFIER = os.environ.get("SLURM_JOB_ID") or str(uuid.uuid4())

# 3D patch size (D, H, W)
PATCH_SIZE = (128, 160, 160)
DATASET_SUBSAMPLE_PORTION = 2
TIMESTEPS = 400

# VAE hyperparams
VAE_NUM_EPOCHS = 40
VAE_LEARNING_RATE = 1e-4
VAE_BASE_CHANNELS = 32
VAE_NUM_DOWN = 3
LATENT_CHANNELS = 16
VAE_KL_WEIGHT = 1e-4

# LDM hyperparams
LDM_NUM_EPOCHS = 60
LDM_LEARNING_RATE = 1e-4
PATIENCE = 10
MIN_DELTA = 1e-4

UNET_BASE_CHANNELS = 128
UNET_CHANNEL_MULTS = (1, 2, 4)

BATCH_SIZE = 1
NUM_WORKERS = 2

DEBUG_FAST = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / EXPERIMENT_NAME
DATASET_ROOT = (PROJECT_ROOT / "../datasets/dataset").resolve()
PERUN_OUT_DIR = EXPERIMENT_ROOT / "perun_results" / RUN_IDENTIFIER

print("Using DATASET_ROOT:", DATASET_ROOT)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# -------------------------------------------------------------------
# Device setup
# -------------------------------------------------------------------
def setup_distributed():
    """Return (device, rank, world_size, local_rank, is_distributed)."""
    if torch.cuda.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Launched by torchrun
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        is_distributed = world_size > 1
        print(f"[DDP] rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        # Fallback: your original single-process logic
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

if IS_MAIN_PROCESS and torch.cuda.is_available():
    print("=== PyTorch CUDA / Slurm info ===")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"[GPU {i}] {props.name}, {props.total_memory / (1024**3):.1f} GB")

# -------------------------------------------------------------------
# DDP helpers
# -------------------------------------------------------------------
def get_unwrapped_model(m):
    return m.module if hasattr(m, "module") else m


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def ddp_broadcast_bool(flag: bool) -> bool:
    if not IS_DISTRIBUTED:
        return flag
    t = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    dist.broadcast(t, src=0)
    return bool(t.item())

# -------------------------------------------------------------------
# Dataset and DataLoaders
# -------------------------------------------------------------------
full_dataset = BraTS3DVolumeDataset(
    DATASET_ROOT,
    patch_size=PATCH_SIZE,
    random_crop=True,
)

if DATASET_SUBSAMPLE_PORTION > 1:
    full_dataset = Subset(
        full_dataset,
        list(range(len(full_dataset) // DATASET_SUBSAMPLE_PORTION))
    )

if DEBUG_FAST:
    indices = list(range(min(16, len(full_dataset))))
    full_dataset = Subset(full_dataset, indices)

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
split_generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=split_generator,
)

if IS_DISTRIBUTED and device.type == "cuda":
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    val_sampler = None
else:
    train_sampler = None
    val_sampler = None

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    sampler=val_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

if IS_MAIN_PROCESS:
    print(f"Train volumes: {len(train_dataset)}, Val volumes: {len(val_dataset)}")

# -------------------------------------------------------------------
# Models: VAE + UNet + Diffusion
# -------------------------------------------------------------------
vae = VAE3D(
    in_channels=4,
    base_channels=VAE_BASE_CHANNELS,
    num_down=VAE_NUM_DOWN,
    latent_channels=LATENT_CHANNELS,
).to(device)

unet = UNet3DModelWithAttention(
    in_channels=LATENT_CHANNELS,
    base_channels=UNET_BASE_CHANNELS,
    channel_mults=UNET_CHANNEL_MULTS,
).to(device)

if IS_DISTRIBUTED and device.type == "cuda":
    vae = DDP(vae, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

diffusion = GaussianDiffusionLatent3D(
    model=unet,
    channels=LATENT_CHANNELS,
    timesteps=TIMESTEPS,
).to(device)

# Optimizers & scaler
optimizer_vae = torch.optim.Adam(vae.parameters(), lr=VAE_LEARNING_RATE)
optimizer_ldm = torch.optim.Adam(diffusion.model.parameters(), lr=LDM_LEARNING_RATE)

scheduler_ldm = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_ldm,
    mode="min",
    factor=0.5,
    patience=3
)

scaler_vae = GradScaler("cuda", enabled=(device.type == "cuda"))
scaler_ldm = GradScaler("cuda", enabled=(device.type == "cuda"))

# -------------------------------------------------------------------
# VAE Training helpers
# -------------------------------------------------------------------
def train_vae_one_epoch(epoch: int, max_steps: Union[int, None] = None) -> float:
    vae.train()
    running_loss = 0.0
    n_steps = 0
    start_time = time.time()

    for step, x in enumerate(train_loader, start=1):
        if should_terminate():
            print(f"[train_vae_one_epoch] Termination requested at epoch {epoch}, step {step}. Breaking.")
            break

        x = x.to(device, non_blocking=True)  # (B, 4, D, H, W)

        optimizer_vae.zero_grad()

        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            recon, mu, logvar = vae(x)
            recon_loss = F.l1_loss(recon, x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + VAE_KL_WEIGHT * kl

        scaler_vae.scale(loss).backward()
        scaler_vae.step(optimizer_vae)
        scaler_vae.update()

        if torch.cuda.is_available() and step == 1:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[VAE] Peak GPU memory after first step: {peak:.2f} GB")

        running_loss += loss.item()
        n_steps += 1

        if step % 100 == 0:
            avg = running_loss / n_steps
            print(f"[VAE epoch {epoch} | step {step}] avg loss: {avg:.4f}")

        if max_steps is not None and step >= max_steps:
            break

    elapsed = time.time() - start_time
    avg_loss = running_loss / max(1, n_steps)
    steps_per_s = n_steps / max(elapsed, 1e-8)

    if IS_MAIN_PROCESS:
        print(
            f"VAE Epoch {epoch} | Train loss: {avg_loss:.4f} | "
            f"steps: {n_steps} | time: {elapsed:.1f}s | {steps_per_s:.2f} steps/s"
        )

    run = mlflow.active_run()
    if run is not None:
        mlflow.log_metric("vae_train_steps_per_s", steps_per_s, step=epoch)
        mlflow.log_metric("vae_train_num_steps", n_steps, step=epoch)

    return avg_loss


@torch.no_grad()
def validate_vae(epoch: int, max_steps: Union[int, None] = None) -> float:
    vae.eval()
    running_loss = 0.0
    n_steps = 0

    for step, x in enumerate(val_loader, start=1):
        x = x.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            recon, mu, logvar = vae(x)
            recon_loss = F.l1_loss(recon, x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + VAE_KL_WEIGHT * kl

        running_loss += loss.item()
        n_steps += 1

        if max_steps is not None and step >= max_steps:
            break

    avg_loss = running_loss / max(1, n_steps)

    if IS_MAIN_PROCESS:
        print(f"VAE Epoch {epoch} | Val loss: {avg_loss:.4f}")

    return avg_loss

# -------------------------------------------------------------------
# LDM Training helpers (latent diffusion)
# -------------------------------------------------------------------
def train_ldm_one_epoch(epoch: int, max_steps: Union[int, None] = None) -> float:
    diffusion.train()
    vae.eval()  # VAE is frozen in this stage
    base_vae = get_unwrapped_model(vae)

    running_loss = 0.0
    n_steps = 0
    start_time = time.time()

    for step, x in enumerate(train_loader, start=1):
        if should_terminate():
            print(f"[train_ldm_one_epoch] Termination requested at epoch {epoch}, step {step}. Breaking.")
            break

        x = x.to(device, non_blocking=True)  # (B, 4, D, H, W)

        # Get latents (mean of posterior) without grad
        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                z = base_vae.encode_to_latent(x)

        b = z.size(0)
        
        u = torch.rand((b,), device=device)
        t = (u * u * (diffusion.timesteps - 1)).long()  # squares bias toward 0
        t = (diffusion.timesteps - 1) - t               # flip -> bias toward high t

        optimizer_ldm.zero_grad()

        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss = diffusion.p_losses(z, t, cond=None, min_snr_gamma=5.0)

        scaler_ldm.scale(loss).backward()
        scaler_ldm.step(optimizer_ldm)
        scaler_ldm.update()

        if torch.cuda.is_available() and step == 1:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[LDM] Peak GPU memory after first step: {peak:.2f} GB")

        running_loss += loss.item()
        n_steps += 1

        if IS_MAIN_PROCESS and step % 100 == 0:
            avg = running_loss / n_steps
            print(f"[LDM epoch {epoch} | step {step}] avg loss: {avg:.4f}")

        if max_steps is not None and step >= max_steps:
            break

    elapsed = time.time() - start_time
    avg_loss = running_loss / max(1, n_steps)
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

    return avg_loss


@torch.no_grad()
def validate_ldm(epoch: int, max_steps: Union[int, None] = None) -> float:
    diffusion.eval()
    vae.eval()
    base_vae = get_unwrapped_model(vae)

    running_loss = 0.0
    n_steps = 0

    for step, x in enumerate(val_loader, start=1):
        x = x.to(device, non_blocking=True)

        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                z = base_vae.encode_to_latent(x)

        b = z.size(0)
        u = torch.rand((b,), device=device)
        t = (u * u * (diffusion.timesteps - 1)).long()  # squares bias toward 0
        t = (diffusion.timesteps - 1) - t               # flip -> bias toward high t


        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss = diffusion.p_losses(z, t, cond=None, min_snr_gamma=5.0)

        running_loss += loss.item()
        n_steps += 1

        if max_steps is not None and step >= max_steps:
            break

    avg_loss = running_loss / max(1, n_steps)

    if IS_MAIN_PROCESS:
        print(f"LDM Epoch {epoch} | Val loss: {avg_loss:.4f}")

    return avg_loss

# -------------------------------------------------------------------
# Main training loop: VAE then LDM
# -------------------------------------------------------------------
def train() -> float:
    print("Starting 3D VAE + LDM Training")

    # ------------------ Stage 1: VAE training ------------------
    print("=== Stage 1: Training 3D VAE ===")

    for epoch in range(1, VAE_NUM_EPOCHS + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if should_terminate():
            print(f"[train] Termination requested before VAE epoch {epoch}, stopping.")
            break

        if IS_DISTRIBUTED and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss = train_vae_one_epoch(epoch, max_steps=5 if DEBUG_FAST else None)

        if should_terminate():
            print(f"[train] Termination requested after VAE epoch {epoch}, stopping.")
            break

        val_loss = validate_vae(epoch, max_steps=2 if DEBUG_FAST else None)

        if IS_MAIN_PROCESS:
            mlflow.log_metric("vae_train_loss", train_loss, step=epoch)
            mlflow.log_metric("vae_val_loss", val_loss, step=epoch)
            mlflow.log_metric("vae_learning_rate", optimizer_vae.param_groups[0]["lr"], step=epoch)

        # Save VAE weights
        models_dir = EXPERIMENT_ROOT / "models" / RUN_IDENTIFIER
        models_dir.mkdir(parents=True, exist_ok=True)

        vae_path = models_dir / "vae3d_final.pt"

        if IS_MAIN_PROCESS:
            vae_to_save = get_unwrapped_model(vae)
            torch.save(vae_to_save.state_dict(), str(vae_path))
            mlflow.log_artifact(str(vae_path), artifact_path="checkpoints")

    # freeze VAE for LDM
    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()

    # ------------------ Stage 2: LDM training ------------------
    print("=== Stage 2: Training 3D latent diffusion (LDM) ===")

    best_val = float("inf")
    epochs_without_improvement = 0
    best_ldm_path: Optional[Path] = None

    for epoch in range(1, LDM_NUM_EPOCHS + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if should_terminate():
            print(f"[train] Termination requested before LDM epoch {epoch}, stopping.")
            break

        if IS_DISTRIBUTED and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss = train_ldm_one_epoch(epoch, max_steps=5 if DEBUG_FAST else None)

        if should_terminate():
            print(f"[train] Termination requested after LDM epoch {epoch}, stopping before validation.")
            break

        val_loss = validate_ldm(epoch, max_steps=2 if DEBUG_FAST else None)

        if should_terminate():
            print(f"[train] Termination requested after LDM epoch {epoch} validation, stopping.")
            break

        if IS_MAIN_PROCESS:
            mlflow.log_metric("ldm_train_loss", train_loss, step=epoch)
            mlflow.log_metric("ldm_val_loss", val_loss, step=epoch)
            mlflow.log_metric(
                "ldm_learning_rate", optimizer_ldm.param_groups[0]["lr"], step=epoch,
            )

        scheduler_ldm.step(val_loss)

        # Check for improvement
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            epochs_without_improvement = 0

            if IS_MAIN_PROCESS:
                best_ldm_path = models_dir / "3d_ldm_diffusion_best.pt"
                diff_to_save = diffusion
                torch.save(diff_to_save.state_dict(), str(best_ldm_path))
                mlflow.log_artifact(str(best_ldm_path), artifact_path="checkpoints")

                print(f"✅ New best LDM val loss: {best_val:.4f}")
        else:
            epochs_without_improvement += 1
            if IS_MAIN_PROCESS:
                print(f"⚠️ No LDM improvement for {epochs_without_improvement} epoch(s)")

        stop_now = False
        if IS_MAIN_PROCESS and epochs_without_improvement >= PATIENCE:
            stop_now = True
            print(
                f"⏹ LDM early stopping at epoch {epoch} "
                f"(no val improvement for {PATIENCE} epochs).",
            )

        stop_now = ddp_broadcast_bool(stop_now)
        if stop_now:
            if IS_MAIN_PROCESS:
                print(
                    f"⏹ LDM early stopping at epoch {epoch} (synced across ranks)."
                    f"(no val improvement for {PATIENCE} epochs)."
                )
            break

    if IS_MAIN_PROCESS:
        mlflow.pytorch.log_model(get_unwrapped_model(diffusion.model), artifact_path="final_ldm_unet")
        mlflow.pytorch.log_model(get_unwrapped_model(vae), artifact_path="final_vae")

        if best_val is not None and not (best_val == float("inf")):
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
                    "dataset_subsample_portion": DATASET_SUBSAMPLE_PORTION,
                    "vae_base_channels": VAE_BASE_CHANNELS,
                    "vae_num_down": VAE_NUM_DOWN,
                    "latent_channels": LATENT_CHANNELS,
                    "unet_base_channels": UNET_BASE_CHANNELS,
                    "unet_channel_mults": str(UNET_CHANNEL_MULTS),
                }
            )

            best_val = run_with_perun(
                train,
                data_out=str(PERUN_OUT_DIR),
            )

            if best_val is not None and best_val != float("inf"):
                mlflow.log_metric("best_ldm_val_loss", best_val)
    else:
        run_with_perun(
            train,
            data_out=str(PERUN_OUT_DIR),
        )

    cleanup_distributed()


if __name__ == "__main__":
    install_signal_handlers()
    main()
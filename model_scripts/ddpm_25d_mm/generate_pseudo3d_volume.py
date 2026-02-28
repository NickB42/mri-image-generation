"""
generate_pseudo3d_volume.py — Sequential 3D volume generation using
the memmap-backed 2.5D DDPM (FLAIR-only, 1 channel).

Usage:
  python -m model_scripts.ddpm_25d_mm.generate_pseudo3d_volume
"""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import nibabel as nib

import torch
from torchvision.utils import save_image

from .mm_dataset import MemmapDataset, get_train_dataset
from .unet import UNet
from .diffusion import GaussianDiffusion


# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # -> /home/.../mri

DATASET_MEMMAP = PROJECT_ROOT / "datasets" / "memmap" / "train_flair_256.npy"
# Update this to your actual checkpoint path after training:
CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "models" / "1615499" / "ddpm_25d_mm_best.pt"
)

OUT_DIR = Path(__file__).resolve().parent / "pseudo3d_mm_out"

SUBJECT_IDX = 10
NUM_SAMPLES = 2
RESET_EVERY = 144

MODE = "sequential"  # "sequential" or "teacher_forced"
SEED: Union[int, None] = 1234

SAVE_PNG_GRIDS = True
PNG_NROW = 16

IMAGE_SIZE = 256
TIMESTEPS = 1000

CENTER_CHANNELS = 1
SLICE_RADIUS = 2
NUM_SLICES = 155

MODALITY_NAMES = ["flair"]

# =============================================================================
IN_CHANNELS = CENTER_CHANNELS + CENTER_CHANNELS * SLICE_RADIUS
OUT_CHANNELS = CENTER_CHANNELS


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------
# Checkpoint loading
# ----------------------------------------------------------------------
def load_diffusion_from_checkpoint(checkpoint_path: Path, device: torch.device) -> GaussianDiffusion:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")

    model = UNet(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256,
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        image_size=IMAGE_SIZE,
        channels=OUT_CHANNELS,
        timesteps=TIMESTEPS,
        schedule="cosine",
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, dict) and "diffusion" in state:
        diffusion_sd = state["diffusion"]
        ema_sd = state.get("ema_unet", None)
    else:
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        diffusion_sd = state
        ema_sd = None

    def _strip_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith("model.module."):
                nk = nk.replace("model.module.", "model.", 1)
            if nk.startswith("module."):
                nk = nk.replace("module.", "", 1)
            out[nk] = v
        return out

    diffusion_sd = _strip_prefix(diffusion_sd)

    looks_like_diffusion = any(
        k.startswith(("betas", "alphas", "sqrt_", "posterior_", "model."))
        for k in diffusion_sd.keys()
    )

    if looks_like_diffusion:
        missing, unexpected = diffusion.load_state_dict(diffusion_sd, strict=False)
        print("Loaded into diffusion (strict=False).")
    else:
        missing, unexpected = diffusion.model.load_state_dict(diffusion_sd, strict=False)
        print("Loaded into diffusion.model (strict=False).")

    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    if ema_sd is not None:
        diffusion.model.load_state_dict(ema_sd, strict=True)
        print("Loaded EMA weights into diffusion.model for sampling.")

    diffusion.eval()
    return diffusion


# ----------------------------------------------------------------------
# NIfTI saving helpers
# ----------------------------------------------------------------------
def save_volume_as_nifti(
    volume: torch.Tensor,
    out_dir: Path,
    subject_idx: int,
    z_indices: List[int],
    full_depth: int,
    modality_names: List[str] = MODALITY_NAMES,
) -> None:
    """
    volume: (S, C, H, W) generated slices
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vol = (volume.clamp(-1, 1) + 1.0) / 2.0  # -> [0, 1]
    vol = vol.float().detach().cpu()
    S, C, H, W = vol.shape

    # Fill into full-depth array
    full = torch.zeros((C, H, W, full_depth), dtype=vol.dtype)
    for k in range(S):
        z = int(z_indices[k])
        if 0 <= z < full_depth:
            full[:, :, :, z] = vol[k]

    affine = np.eye(4, dtype=np.float32)

    for c_idx in range(C):
        name = modality_names[c_idx] if c_idx < len(modality_names) else f"mod{c_idx}"
        vol_hwz = full[c_idx].numpy()  # (H, W, D)
        img = nib.Nifti1Image(vol_hwz.astype(np.float32), affine)
        out_path = out_dir / f"subject{subject_idx:03d}_{name}.nii.gz"
        nib.save(img, str(out_path))
        print(f"Saved NIfTI: {out_path}  shape={vol_hwz.shape}")


# ----------------------------------------------------------------------
# Core generation
# ----------------------------------------------------------------------
@torch.inference_mode()
def generate_volume_for_subject(
    diffusion: GaussianDiffusion,
    dataset: MemmapDataset,
    subject_idx: int,
    out_dir: Path,
    mode: str,
    num_samples: int,
    save_png_grids: bool,
    png_nrow: int,
    seed: Union[int, None],
) -> torch.Tensor:
    if mode not in {"sequential", "teacher_forced"}:
        raise ValueError("mode must be 'sequential' or 'teacher_forced'")

    device = next(diffusion.parameters()).device
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get usable slice indices for this subject
    n_slices = dataset.num_usable
    z_start = dataset.z_start
    z_end = dataset.z_end
    z_indices = list(range(z_start, z_end))

    H = W = IMAGE_SIZE
    C = CENTER_CHANNELS
    full_depth = NUM_SLICES
    last_volume = None

    for sample_idx in range(num_samples):
        if seed is not None:
            torch.manual_seed(int(seed) + sample_idx)
            np.random.seed(int(seed) + sample_idx)

        sample_out_dir = out_dir / f"sample_{sample_idx:03d}"
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        volume_gen = torch.zeros(n_slices, C, H, W, device=device)
        generated_mask = [False] * n_slices

        print(f"\n=== subject {subject_idx:03d} | sample {sample_idx:03d} | mode={mode} ===")

        for k, z in enumerate(z_indices):
            do_reset = (mode == "sequential") and (RESET_EVERY is not None) and (k % RESET_EVERY == 0)

            # Build context: preceding slices
            context_channels = []
            for dz in range(-SLICE_RADIUS, 0):
                zz = z + dz
                zz = max(0, min(zz, NUM_SLICES - 1))

                if mode == "teacher_forced" or do_reset:
                    # Use real data from memmap
                    src = torch.from_numpy(
                        dataset.data[subject_idx, zz].copy()
                    ).to(device)  # (C, H, W)
                else:
                    # Find generated slice
                    kk = zz - z_start if z_start <= zz < z_end else None
                    if kk is not None and 0 <= kk < n_slices and generated_mask[kk]:
                        src = volume_gen[kk].detach()
                    else:
                        src = torch.from_numpy(
                            dataset.data[subject_idx, zz].copy()
                        ).to(device)

                src = src.clamp(-1, 1)
                context_channels.append(src)

            x_context = torch.cat(context_channels, dim=0).unsqueeze(0)  # (1, C*R, H, W)

            z_pos = torch.tensor([float(z) / float(full_depth - 1)], device=device)

            # fg_frac from real data
            real_slice = dataset.data[subject_idx, z]  # (C, H, W)
            fg_frac = torch.tensor(
                [float((real_slice > -0.999).mean())], device=device
            )

            t0 = time.time()
            sample = diffusion.sample(
                batch_size=1,
                z_pos=z_pos,
                fg_frac=fg_frac,
                context=x_context,
            )  # (1, C, H, W)
            dt = time.time() - t0

            volume_gen[k] = sample[0]
            generated_mask[k] = True

            if (k + 1) % 10 == 0 or (k + 1) == n_slices:
                print(f"[slice {k + 1:4d}/{n_slices}] z={z:4d}  time={dt:6.2f}s")

        if save_png_grids:
            volume_vis = (volume_gen.clamp(-1, 1) + 1.0) / 2.0
            for c in range(C):
                mod_name = MODALITY_NAMES[c] if c < len(MODALITY_NAMES) else f"mod{c}"
                mod_vol = volume_vis[:, c:c + 1, :, :]
                grid_path = sample_out_dir / f"subject{subject_idx:03d}_{mode}_all_slices_{mod_name}.png"
                save_image(mod_vol, grid_path, nrow=int(png_nrow))
                print(f"Saved grid -> {grid_path}")

        save_volume_as_nifti(
            volume=volume_gen,
            out_dir=sample_out_dir,
            subject_idx=subject_idx,
            z_indices=z_indices,
            full_depth=full_depth,
        )

        last_volume = volume_gen.detach().cpu()

    return last_volume


def main() -> None:
    device = get_device()
    print("Device:", device)

    diffusion = load_diffusion_from_checkpoint(CHECKPOINT_PATH, device=device)
    dataset = get_train_dataset(slice_radius=SLICE_RADIUS)

    generate_volume_for_subject(
        diffusion=diffusion,
        dataset=dataset,
        subject_idx=SUBJECT_IDX,
        out_dir=OUT_DIR,
        mode=MODE,
        num_samples=NUM_SAMPLES,
        save_png_grids=SAVE_PNG_GRIDS,
        png_nrow=PNG_NROW,
        seed=SEED,
    )


if __name__ == "__main__":
    main()

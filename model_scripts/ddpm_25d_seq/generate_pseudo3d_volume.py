from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import nibabel as nib

import torch
from torchvision.utils import save_image

from .dataset import BraTSSliceDataset
from .unet import UNet
from .diffusion import GaussianDiffusion


# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (PROJECT_ROOT / "../datasets/train").resolve()
CHECKPOINT_PATH = (PROJECT_ROOT / "ddpm_25d_seq" / "models" / "1611221" / "2d_central_ddpm_flair_best.pt").resolve()

OUT_DIR = (PROJECT_ROOT / "ddpm_25d_seq" / "pseudo3d_seq_ddpm_out").resolve()

SUBJECT_IDX = 10
NUM_SAMPLES = 2

RESET_EVERY = 144

# Mode: "sequential" or "teacher_forced"
MODE = "sequential" # "teacher_forced"

SEED: Union[int, None] = 1234

SAVE_PNG_GRIDS = True
PNG_NROW = 16

IMAGE_SIZE = 128
TIMESTEPS = 1000

CENTER_MODALITIES = 4
SLICE_RADIUS = 2

MODALITY_NAMES = ["t1", "t1ce", "t2", "flair"]
# =============================================================================
IN_CHANNELS = CENTER_MODALITIES + CENTER_MODALITIES * SLICE_RADIUS
OUT_CHANNELS = CENTER_MODALITIES

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
# ----------------------------------------------------------------------
# Subject helpers
# ----------------------------------------------------------------------
def _get_subject_flair_path(dataset: BraTSSliceDataset, subject_idx: int) -> Path:
    all_paths = [p for (p, z, D) in dataset.slice_tuples]
    unique_paths = sorted(set(all_paths))
    if subject_idx < 0 or subject_idx >= len(unique_paths):
        raise IndexError(f"subject_idx {subject_idx} out of range (have {len(unique_paths)} subjects)")
    return Path(unique_paths[subject_idx])


def get_subject_indices(dataset: BraTSSliceDataset, subject_idx: int = 0) -> List[int]:
    all_paths = [p for (p, z, D) in dataset.slice_tuples]
    unique_paths = sorted(set(all_paths))
    if subject_idx < 0 or subject_idx >= len(unique_paths):
        raise IndexError(f"subject_idx {subject_idx} out of range (have {len(unique_paths)} subjects)")

    target_path = unique_paths[subject_idx]
    print(f"Using subject {subject_idx} with FLAIR volume: {target_path}")

    indices = [i for i, (p, z, D) in enumerate(dataset.slice_tuples) if p == target_path]
    indices = sorted(indices, key=lambda i: dataset.slice_tuples[i][1])  # sort by z
    print(f"Subject has {len(indices)} usable center slices.")
    return indices


def make_subject_slice_getter(
    dataset: BraTSSliceDataset,
    flair_path: Path,
) -> Tuple[callable, int]:
    """
    Returns:
      get_real_slice(z:int) -> torch.Tensor (4,H,W) preprocessed in [-1,1]
      full_depth D
    """
    vols = []
    for suffix in dataset.modalities:
        m_path = str(flair_path).replace(dataset.flair_suffix, suffix)
        vol = dataset._load_volume(m_path)  # (H,W,D), cached
        vols.append(vol)

    D = int(vols[0].shape[-1])

    def get_real_slice(z: int) -> torch.Tensor:
        z = int(z)
        if z < 0:
            z = 0
        if z >= D:
            z = D - 1
        slices = []
        for vol in vols:
            slice_2d = vol[:, :, z]
            slices.append(dataset._preprocess_slice(slice_2d))  # (1,H,W)
        return torch.cat(slices, dim=0)  # (4,H,W)

    return get_real_slice, D


# ----------------------------------------------------------------------
# NIfTI saving helpers
# ----------------------------------------------------------------------
def _tensor_volume_to_full_depth_numpy(
    vol_sc_hw: torch.Tensor,
    z_indices: List[int],
    full_depth: int,
) -> np.ndarray:
    """
    vol_sc_hw: (S, C, H, W)
    returns: (C, H, W, full_depth) filled at z_indices, zeros elsewhere
    """
    vol_sc_hw = vol_sc_hw.detach().cpu()
    S, C, H, W = vol_sc_hw.shape

    out = torch.zeros((C, H, W, full_depth), dtype=vol_sc_hw.dtype)
    for k in range(S):
        z = int(z_indices[k])
        if 0 <= z < full_depth:
            out[:, :, :, z] = vol_sc_hw[k]
    return out.numpy()


def save_brats_like_nifti(
    vol_sc_hw: torch.Tensor,
    out_dir: Path,
    subject_idx: int,
    modality_names: List[str] = MODALITY_NAMES,
    value_range: str = "0_1",   # "0_1" or "-1_1"
    reference_nifti_path: Union[Path, None] = None,
    z_indices: Union[List[int], None] = None,
    pad_to_reference_depth: bool = True,
    also_save_4d: bool = True,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if value_range == "0_1":
        vol = (vol_sc_hw.clamp(-1, 1) + 1.0) / 2.0
    elif value_range == "-1_1":
        vol = vol_sc_hw.clamp(-1, 1)
    else:
        raise ValueError("value_range must be '0_1' or '-1_1'")

    vol = vol.float()
    S, C, H, W = vol.shape

    ref_img = None
    if reference_nifti_path is not None and Path(reference_nifti_path).is_file():
        ref_img = nib.load(str(reference_nifti_path))

    if ref_img is not None and pad_to_reference_depth and z_indices is not None:
        ref_shape = ref_img.shape
        ref_depth = int(ref_shape[2]) if len(ref_shape) >= 3 else S
        full_c_hwz = _tensor_volume_to_full_depth_numpy(vol, z_indices=z_indices, full_depth=ref_depth)
    else:
        full_c_hwz = vol.permute(1, 2, 3, 0).detach().cpu().numpy()  # (C,H,W,S)

    if ref_img is not None:
        affine = ref_img.affine
        header = deepcopy(ref_img.header)
    else:
        affine = np.eye(4, dtype=np.float32)
        header = nib.Nifti1Header()

    header.set_data_dtype(np.float32)

    for c_idx in range(C):
        name = modality_names[c_idx] if c_idx < len(modality_names) else f"mod{c_idx}"
        vol_hwz = full_c_hwz[c_idx]  # (H,W,Z)
        img = nib.Nifti1Image(vol_hwz.astype(np.float32), affine, header=header)
        out_path = out_dir / f"subject{subject_idx:03d}_{name}.nii.gz"
        nib.save(img, str(out_path))
        print(f"Saved NIfTI: {out_path}  shape={vol_hwz.shape}")

    if also_save_4d:
        vol_hwzc = np.stack([full_c_hwz[c] for c in range(C)], axis=-1)  # (H,W,Z,C)
        img4d = nib.Nifti1Image(vol_hwzc.astype(np.float32), affine, header=header)
        out_path_4d = out_dir / f"subject{subject_idx:03d}_allmods.nii.gz"
        nib.save(img4d, str(out_path_4d))
        print(f"Saved 4D NIfTI: {out_path_4d}  shape={vol_hwzc.shape}")


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

    # New checkpoint format: dict with "diffusion" + "ema_unet" keys
    if isinstance(state, dict) and "diffusion" in state:
        diffusion_sd = state["diffusion"]
        ema_sd = state.get("ema_unet", None)
    else:
        # Legacy checkpoint format
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
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
        k.startswith("betas")
        or k.startswith("alphas")
        or k.startswith("alphas_cumprod")
        or k.startswith("sqrt_alphas_cumprod")
        or k.startswith("posterior_variance")
        or k.startswith("model.")
        for k in diffusion_sd.keys()
    )

    if looks_like_diffusion:
        missing, unexpected = diffusion.load_state_dict(diffusion_sd, strict=False)
        print("Loaded into diffusion (strict=False).")
    else:
        missing, unexpected = diffusion.model.load_state_dict(diffusion_sd, strict=False)
        print("Loaded into diffusion.model (strict=False).")

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    # B) Load EMA weights into model for sampling (sharper / more stable samples)
    if ema_sd is not None:
        diffusion.model.load_state_dict(ema_sd, strict=True)
        print("Loaded EMA weights into diffusion.model for sampling.")

    diffusion.eval()
    return diffusion


# ----------------------------------------------------------------------
# Core generation
# ----------------------------------------------------------------------
@torch.inference_mode()
def generate_volume_for_subject(
    diffusion: GaussianDiffusion,
    dataset_root: Path,
    subject_idx: int,
    out_dir: Path,
    mode: str,
    num_samples: int,
    save_png_grids: bool,
    png_nrow: int,
    seed: Union[int, None],
) -> torch.Tensor:
    """
    mode:
      - "sequential": use generated slices as context when available, else real slices
      - "teacher_forced": always use real slices for context
    """
    if mode not in {"sequential", "teacher_forced"}:
        raise ValueError("MODE must be 'sequential' or 'teacher_forced'")

    device = next(diffusion.parameters()).device
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = BraTSSliceDataset(
        dataset_root,
        image_size=IMAGE_SIZE,
        slice_radius=SLICE_RADIUS,
    )

    indices = get_subject_indices(dataset, subject_idx=subject_idx)
    num_slices = len(indices)
    z_indices = [int(dataset.slice_tuples[i][1]) for i in indices]  # actual z in original volume (tuple is (path, z, D))

    flair_path = _get_subject_flair_path(dataset, subject_idx)
    get_real_slice, full_depth = make_subject_slice_getter(dataset, flair_path)
    z_to_k = {z: k for k, z in enumerate(z_indices)}

    H = W = IMAGE_SIZE
    last_volume = None

    for sample_idx in range(num_samples):
        if seed is not None:
            torch.manual_seed(int(seed) + sample_idx)
            np.random.seed(int(seed) + sample_idx)

        sample_out_dir = out_dir / f"sample_{sample_idx:03d}"
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        volume_gen = torch.zeros(num_slices, OUT_CHANNELS, H, W, device=device)
        generated_mask = [False] * num_slices

        print(f"\n=== subject {subject_idx:03d} | sample {sample_idx:03d} | mode={mode} ===")

        for k, z in enumerate(z_indices):
            # Build context exactly like training: (z-2 modalities), then (z-1 modalities)
            # In sequential mode, every RESET_EVERY slices we "reset" and use GT context
            do_reset = (mode == "sequential") and (RESET_EVERY is not None) and (k % RESET_EVERY == 0)

            context_channels = []
            for dz in (-2, -1):
                zz = z + dz

                if mode == "teacher_forced" or do_reset:
                    src = get_real_slice(zz)  # (4,H,W)
                else:
                    kk = z_to_k.get(zz, None)
                    if kk is not None and generated_mask[kk]:
                        # NOTE: keep on device to avoid cpu->gpu churn
                        src = volume_gen[kk].detach()  # (4,H,W) on device
                    else:
                        src = get_real_slice(zz).to(device)

                # optional: clamp helps keep generated context in expected range
                src = src.clamp(-1, 1)

                for m in range(CENTER_MODALITIES):
                    context_channels.append(src[m:m+1, :, :])

            x_context = torch.cat(context_channels, dim=0).unsqueeze(0)  # (1,8,H,W) already on device

            # z_pos uses ORIGINAL depth D
            z_pos = torch.tensor([float(z) / float(full_depth - 1)], device=device)

            # D1) Compute fg_frac from real center slice (FLAIR channel, index 3)
            real_center = get_real_slice(z).to(device)  # (4,H,W) in [-1,1]
            fg_frac = ((real_center[3:4] > -0.999).float().mean()).unsqueeze(0)  # (1,)

            t0 = time.time()
            sample = diffusion.sample(
                batch_size=1,
                z_pos=z_pos,
                fg_frac=fg_frac,
                context=x_context,
            )  # (1,4,H,W) in [-1,1]
            dt = time.time() - t0

            volume_gen[k] = sample[0]
            generated_mask[k] = True

            if (k + 1) % 10 == 0 or (k + 1) == num_slices:
                print(f"[slice {k+1:4d}/{num_slices}] z={z:4d}  time={dt:6.2f}s")

        if save_png_grids:
            volume_vis = (volume_gen.clamp(-1, 1) + 1.0) / 2.0
            for c in range(OUT_CHANNELS):
                mod_name = MODALITY_NAMES[c] if c < len(MODALITY_NAMES) else f"mod{c}"
                mod_vol = volume_vis[:, c:c+1, :, :]
                grid_path = sample_out_dir / f"subject{subject_idx:03d}_{mode}_all_slices_{mod_name}.png"
                save_image(mod_vol, grid_path, nrow=int(png_nrow))
                print(f"Saved grid {mod_name} -> {grid_path}")

        save_brats_like_nifti(
            vol_sc_hw=volume_gen,
            out_dir=sample_out_dir,
            subject_idx=subject_idx,
            reference_nifti_path=flair_path,
            z_indices=z_indices,
            pad_to_reference_depth=True,
            value_range="0_1",
            also_save_4d=True,
        )

        last_volume = volume_gen.detach().cpu()

    return last_volume


def main() -> None:
    device = get_device()
    print("Device:", device)

    diffusion = load_diffusion_from_checkpoint(CHECKPOINT_PATH, device=device)

    generate_volume_for_subject(
        diffusion=diffusion,
        dataset_root=DATASET_ROOT,
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
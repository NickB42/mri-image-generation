#!/usr/bin/env python3
"""
Sanity checks for patch-trained MONAI 3D LDM (BraTS FLAIR).

Checks:
  A) Autoencoder reconstruction: real patch -> encode -> decode -> save NIfTI + PNG grids
  B) Scale factor: compute 1/std(z) across several patches
  C) Single patch sampling: sample ONE patch -> save NIfTI + PNG grid

References:
- MONAI LatentDiffusionInferer.sample signature and usage. :contentReference[oaicite:2]{index=2}
- nibabel NIfTI save pattern (Nifti1Image(data, affine), nib.save). :contentReference[oaicite:3]{index=3}
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)
from monai.networks.nets.autoencoderkl import AutoencoderKL
from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet
from monai.networks.schedulers.ddpm import DDPMScheduler
from monai.inferers import LatentDiffusionInferer


# ============================================================
# CONFIG (edit these)
# ============================================================
CKPT_PATH = "./runs/brats_ldm_finetune/best_diffusion.pt"
BUNDLE_CACHE_DIR = "./_pretrained_bundles"  # or leave; can be overridden by ckpt cfg

REFERENCE_NIFTI = "./datasets/train/BraTS2021_00005/BraTS2021_00005_flair.nii.gz"

OUT_DIR = "./sanity_out"

# Which checks to run
RUN_AUTOENC_RECON = True
RUN_SCALE_FACTOR = True
RUN_SINGLE_PATCH_SAMPLE = True

# Patch size: should match training (or leave None to read from ckpt cfg)
PATCH_SIZE_DHW = None  # e.g. (144, 176, 112)

# Preprocessing defaults if ckpt cfg not present
AXCODES_FALLBACK = "RAS"
SPACING_FALLBACK = (1.1, 1.1, 1.1)

# Scale factor check parameters
NUM_SF_PATCHES = 8  # how many patches to estimate sf stats from
SF_SEED = 0

# Sampling parameters
NUM_INFERENCE_STEPS = 1000
SAMPLE_SEED = 123
DEVICE = "cuda"  # "cuda" or "cpu"
AMP = "fp16"     # "no" | "fp16" | "bf16"

# PNG preview
GRID_AXIS = "z"
GRID_NUM_SLICES = 24
GRID_NCOLS = 6
GRID_CLIP_RANGE = (0.0, 1.0)
# ============================================================


# -------------------------
# Models (must match training)
# -------------------------
def instantiate_models() -> Tuple[AutoencoderKL, DiffusionModelUNet]:
    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=8,
        channels=(64, 128, 256),
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=(False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        include_fc=False,
    )

    diffusion = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=8,
        out_channels=8,
        channels=(256, 256, 512),
        attention_levels=(False, True, True),
        num_head_channels=(0, 64, 64),
        num_res_blocks=2,
        include_fc=False,
        use_combined_linear=False,
    )
    return autoencoder, diffusion


# -------------------------
# HF bundle loading (autoencoder weights)
# -------------------------
def snapshot_bundle(repo_id: str, revision: str, cache_dir: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("Please install huggingface_hub: pip install huggingface_hub") from e

    bundle_root = Path(cache_dir) / repo_id.replace("/", "__") / revision
    bundle_root.mkdir(parents=True, exist_ok=True)

    ae_path = bundle_root / "models" / "model_autoencoder.pt"
    dm_path = bundle_root / "models" / "model.pt"
    if ae_path.exists() and dm_path.exists():
        return bundle_root

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(bundle_root),
        local_dir_use_symlinks=False,
    )
    return bundle_root


def load_autoencoder_weights(autoencoder: AutoencoderKL, bundle_root: Path) -> None:
    ae_path = bundle_root / "models" / "model_autoencoder.pt"
    if not ae_path.exists():
        raise FileNotFoundError(f"Missing autoencoder weights: {ae_path}")

    ae_sd = torch.load(ae_path, map_location="cpu")
    if hasattr(autoencoder, "load_old_state_dict"):
        autoencoder.load_old_state_dict(ae_sd)
    else:
        autoencoder.load_state_dict(ae_sd, strict=False)


def load_diffusion_ckpt(diffusion: DiffusionModelUNet, ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "diffusion_state_dict" not in ckpt:
        raise ValueError(
            f"{ckpt_path} missing 'diffusion_state_dict'. "
            "Use best_diffusion.pt/last_diffusion.pt from your training."
        )
    diffusion.load_state_dict(ckpt["diffusion_state_dict"], strict=True)
    return ckpt


# -------------------------
# Preprocessing (match training deterministic part)
# -------------------------
def make_preproc(axcodes: str, spacing: Sequence[float]):
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image"]),
            Orientationd(keys=["image"], axcodes=axcodes),
            Spacingd(keys=["image"], pixdim=tuple(spacing), mode="bilinear"),
            ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.5, b_min=0.0, b_max=1.0),
        ]
    )


def center_crop_3d(x: torch.Tensor, roi_dhw: Tuple[int, int, int]) -> torch.Tensor:
    """
    x: [1, D, H, W] or [B,1,D,H,W]; returns same rank with center-cropped spatial dims.
    """
    if x.ndim == 4:
        # [C, D, H, W] (C=1)
        c, D, H, W = x.shape
        pD, pH, pW = roi_dhw
        sd = max(0, (D - pD) // 2)
        sh = max(0, (H - pH) // 2)
        sw = max(0, (W - pW) // 2)
        return x[:, sd : sd + pD, sh : sh + pH, sw : sw + pW]
    elif x.ndim == 5:
        # [B, C, D, H, W]
        B, c, D, H, W = x.shape
        pD, pH, pW = roi_dhw
        sd = max(0, (D - pD) // 2)
        sh = max(0, (H - pH) // 2)
        sw = max(0, (W - pW) // 2)
        return x[:, :, sd : sd + pD, sh : sh + pH, sw : sw + pW]
    else:
        raise ValueError(f"Unexpected tensor shape for center_crop_3d: {tuple(x.shape)}")


# -------------------------
# Visualization helpers
# -------------------------

def describe_and_window_png(
    vol_dhw: np.ndarray,
    out_png: Path,
    save_slice_grid_png_fn,  # pass your existing save_slice_grid_png
    axis: str = "z",
    num_slices: int = 24,
    ncols: int = 6,
    p_lo: float = 0.5,
    p_hi: float = 99.5,
    title: str = "",
):
    """
    Prints stats and saves a grid using percentile windowing instead of fixed [0,1].
    """
    x = vol_dhw.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    p1 = float(np.percentile(x, 1))
    p50 = float(np.percentile(x, 50))
    p99 = float(np.percentile(x, 99))
    vlo = float(np.percentile(x, p_lo))
    vhi = float(np.percentile(x, p_hi))
    print(f"[stats] min={mn:.4f} p1={p1:.4f} p50={p50:.4f} p99={p99:.4f} max={mx:.4f}")
    print(f"[window] p{p_lo}={vlo:.4f}  p{p_hi}={vhi:.4f}")

    save_slice_grid_png_fn(
        x,
        out_png,
        axis=axis,
        num_slices=num_slices,
        ncols=ncols,
        vmin=vlo,
        vmax=vhi,
        title=title or f"window p{p_lo}-p{p_hi}",
    )

def save_slice_grid_png(
    vol_dhw: np.ndarray,
    out_png: Path,
    axis: str = "z",
    num_slices: int = 24,
    ncols: int = 6,
    vmin: float = 0.0,
    vmax: float = 1.0,
    title: Optional[str] = None,
) -> None:
    assert vol_dhw.ndim == 3, f"Expected [D,H,W], got {vol_dhw.shape}"
    axis = axis.lower()
    if axis not in ("z", "y", "x"):
        raise ValueError("axis must be 'z', 'y', or 'x'")

    D, H, W = vol_dhw.shape
    axis_len = {"z": D, "y": H, "x": W}[axis]
    num_slices = int(min(max(1, num_slices), axis_len))
    idxs = np.linspace(0, axis_len - 1, num_slices, dtype=int)

    ncols = int(max(1, ncols))
    nrows = int(math.ceil(num_slices / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))
    axes = np.array(axes).reshape(-1)

    if title:
        fig.suptitle(title, fontsize=12)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= num_slices:
            continue

        k = int(idxs[i])
        if axis == "z":
            sl = vol_dhw[k, :, :]
        elif axis == "y":
            sl = vol_dhw[:, k, :]
        else:
            sl = vol_dhw[:, :, k]

        ax.imshow(sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"{axis}={k}", fontsize=8)

    plt.tight_layout(pad=0.2)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_nifti(vol_dhw: np.ndarray, affine: np.ndarray, out_path: Path) -> None:
    # nibabel: Nifti1Image(data, affine) then save :contentReference[oaicite:4]{index=4}
    nib.save(nib.Nifti1Image(vol_dhw.astype(np.float32), affine.astype(np.float32)), str(out_path))


def _amp_dtype_from_str(amp: str) -> Optional[torch.dtype]:
    if amp == "fp16":
        return torch.float16
    if amp == "bf16":
        return torch.bfloat16
    return None


# -------------------------
# Checks
# -------------------------
@torch.no_grad()
def check_autoencoder_recon(
    autoencoder: AutoencoderKL,
    patch: torch.Tensor,        # [1,1,D,H,W] in ~[0,1]
    affine: np.ndarray,
    out_dir: Path,
) -> None:
    """
    AE sanity check: if this looks bad, either AE weights/preproc are wrong.
    """
    autoencoder.eval()
    z = autoencoder.encode_stage_2_inputs(patch)

    # decode API differs slightly across versions; try the usual stage-2 decode if available
    if hasattr(autoencoder, "decode_stage_2_outputs"):
        recon = autoencoder.decode_stage_2_outputs(z)
    else:
        # fallback: many AutoencoderKL expose decode(z)
        recon = autoencoder.decode(z)

    x = patch.detach().cpu().numpy().squeeze()        # [D,H,W]
    r = recon.detach().cpu().numpy().squeeze()        # [D,H,W]
    d = np.abs(r - x)

    save_nifti(x, affine, out_dir / "ae_input_patch.nii.gz")
    save_nifti(r, affine, out_dir / "ae_recon_patch.nii.gz")
    save_nifti(d, affine, out_dir / "ae_absdiff_patch.nii.gz")

    vmin, vmax = GRID_CLIP_RANGE
    save_slice_grid_png(x, out_dir / "ae_input_grid.png", axis=GRID_AXIS, num_slices=GRID_NUM_SLICES, ncols=GRID_NCOLS,
                        vmin=vmin, vmax=vmax, title="AE input patch")
    save_slice_grid_png(r, out_dir / "ae_recon_grid.png", axis=GRID_AXIS, num_slices=GRID_NUM_SLICES, ncols=GRID_NCOLS,
                        vmin=vmin, vmax=vmax, title="AE recon patch")
    # diff usually smaller dynamic range; auto-scale a bit
    save_slice_grid_png(d, out_dir / "ae_absdiff_grid.png", axis=GRID_AXIS, num_slices=GRID_NUM_SLICES, ncols=GRID_NCOLS,
                        vmin=float(d.min()), vmax=float(np.percentile(d, 99.0)), title="AE |recon-input|")

    mse = float(np.mean((r - x) ** 2))
    mae = float(np.mean(np.abs(r - x)))
    print(f"[AE recon] MSE={mse:.6e} MAE={mae:.6e}")
    print(f"[AE recon] Saved NIfTI + PNG grids to: {out_dir}")


@torch.no_grad()
def check_scale_factor(
    autoencoder: AutoencoderKL,
    vol_1dhw: torch.Tensor,      # [1,D,H,W] in ~[0,1]
    patch_size: Tuple[int, int, int],
    device: torch.device,
    out_dir: Path,
) -> None:
    """
    Compute 1/std(z) over NUM_SF_PATCHES random-ish center+offset crops.
    Your training used a single batch; this gives a quick distribution sanity check.
    """
    autoencoder.eval()
    torch.manual_seed(int(SF_SEED))

    _, D, H, W = vol_1dhw.shape
    pD, pH, pW = patch_size

    def rand_crop():
        # random crop indices (clamped)
        sd = int(torch.randint(0, max(1, D - pD + 1), (1,)).item())
        sh = int(torch.randint(0, max(1, H - pH + 1), (1,)).item())
        sw = int(torch.randint(0, max(1, W - pW + 1), (1,)).item())
        return vol_1dhw[:, sd:sd+pD, sh:sh+pH, sw:sw+pW]

    sfs = []
    for i in range(int(NUM_SF_PATCHES)):
        x = rand_crop().unsqueeze(0).to(device)  # [1,1,D,H,W]
        z = autoencoder.encode_stage_2_inputs(x)
        sf = (1.0 / (torch.std(z) + 1e-8)).item()
        sfs.append(sf)

    sfs = np.array(sfs, dtype=np.float32)
    stats = {
        "n": int(len(sfs)),
        "mean": float(sfs.mean()),
        "std": float(sfs.std()),
        "min": float(sfs.min()),
        "max": float(sfs.max()),
        "values": [float(x) for x in sfs.tolist()],
    }
    (out_dir / "scale_factor_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[scale_factor] mean={stats['mean']:.6f} std={stats['std']:.6f} min={stats['min']:.6f} max={stats['max']:.6f}")
    print(f"[scale_factor] wrote: {out_dir / 'scale_factor_stats.json'}")


@torch.no_grad()
def check_single_patch_sampling(
    inferer: LatentDiffusionInferer,
    autoencoder: AutoencoderKL,
    diffusion: DiffusionModelUNet,
    patch_size: Tuple[int, int, int],
    affine: np.ndarray,
    device: torch.device,
    out_dir: Path,
) -> None:
    """
    Sample ONE patch using LatentDiffusionInferer.sample(input_noise, autoencoder_model, diffusion_model).
    :contentReference[oaicite:5]{index=5}
    """
    autoencoder.eval()
    diffusion.eval()

    # Determine latent noise shape robustly by encoding a dummy patch
    pD, pH, pW = patch_size
    dummy = torch.zeros((1, 1, pD, pH, pW), device=device)
    z = autoencoder.encode_stage_2_inputs(dummy)
    noise_shape = tuple(z.shape)  # [1, latent_ch, lD, lH, lW]

    # Seeded noise
    g = torch.Generator(device=device)
    g.manual_seed(int(SAMPLE_SEED))
    noise = torch.randn(noise_shape, generator=g, device=device)

    amp_dtype = _amp_dtype_from_str(AMP)
    inferer.scheduler.set_timesteps(int(NUM_INFERENCE_STEPS))

    if amp_dtype is not None and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            patch = inferer.sample(
                input_noise=noise,
                autoencoder_model=autoencoder,
                diffusion_model=diffusion,
                verbose=False,
            )
    else:
        patch = inferer.sample(
            input_noise=noise,
            autoencoder_model=autoencoder,
            diffusion_model=diffusion,
            verbose=False,
        )

    if isinstance(patch, (tuple, list)):
        patch = patch[0]

    patch_np = patch.detach().cpu().numpy().squeeze().astype(np.float32)  # [D,H,W]

    # IMPORTANT: do NOT clip yet. First inspect.
    describe_and_window_png(
        patch_np,
        out_dir / "sample_single_patch_grid_windowed.png",
        save_slice_grid_png_fn=save_slice_grid_png,
        axis=GRID_AXIS,
        num_slices=GRID_NUM_SLICES,
        ncols=GRID_NCOLS,
        p_lo=0.5,
        p_hi=99.5,
        title="Single patch (percentile windowed)",
    )

    patch_np = np.clip(patch_np, 0.0, 1.0)

    save_nifti(patch_np, affine, out_dir / "sample_single_patch.nii.gz")
    vmin, vmax = GRID_CLIP_RANGE
    save_slice_grid_png(
        patch_np,
        out_dir / "sample_single_patch_grid.png",
        axis=GRID_AXIS,
        num_slices=GRID_NUM_SLICES,
        ncols=GRID_NCOLS,
        vmin=vmin,
        vmax=vmax,
        title="Single sampled patch",
    )
    print(f"[sample] Saved: {out_dir / 'sample_single_patch.nii.gz'}")
    print(f"[sample] Saved: {out_dir / 'sample_single_patch_grid.png'}")


# -------------------------
# Main
# -------------------------
def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(CKPT_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing CKPT_PATH: {ckpt_path}")

    if REFERENCE_NIFTI is None or not str(REFERENCE_NIFTI).strip():
        raise ValueError("Set REFERENCE_NIFTI to a real flair nifti for these sanity checks.")

    # Device
    if DEVICE == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[warn] CUDA not available; using CPU.")
    else:
        device = torch.device(DEVICE)

    # Instantiate models + load ckpt
    autoencoder, diffusion = instantiate_models()
    ckpt = load_diffusion_ckpt(diffusion, ckpt_path)
    cfg = ckpt.get("cfg", {}) or {}
    extra = ckpt.get("extra", {}) or {}

    # Preproc params from ckpt cfg (preferred)
    axcodes = cfg.get("axcodes", AXCODES_FALLBACK)
    spacing = tuple(cfg.get("spacing", SPACING_FALLBACK))

    # Patch size from config
    if PATCH_SIZE_DHW is not None:
        patch_size = tuple(PATCH_SIZE_DHW)
    else:
        patch_size = tuple(cfg.get("patch_size", (144, 176, 112)))

    # HF bundle parameters (from ckpt cfg if present, else fallback)
    hf_repo_id = cfg.get("hf_repo_id", "MONAI/brats_mri_generative_diffusion")
    hf_revision = cfg.get("hf_revision", "1.1.3")
    bundle_cache_dir = cfg.get("bundle_cache_dir", BUNDLE_CACHE_DIR)

    # Scale factor (prefer ckpt extra, else pretrained_bundle_info.json)
    scale_factor = 1.0
    if "scale_factor" in extra:
        scale_factor = float(extra["scale_factor"])
    else:
        info_path = ckpt_path.parent / "pretrained_bundle_info.json"
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text())
                scale_factor = float(info.get("scale_factor", 1.0))
            except Exception:
                pass

    # Load autoencoder from HF bundle snapshot
    bundle_root = snapshot_bundle(hf_repo_id, hf_revision, bundle_cache_dir)
    load_autoencoder_weights(autoencoder, bundle_root)

    autoencoder = autoencoder.to(device).eval()
    diffusion = diffusion.to(device).eval()

    # Load + preprocess reference image
    preproc = make_preproc(axcodes=axcodes, spacing=spacing)
    data = preproc({"image": REFERENCE_NIFTI})
    img = data["image"]  # [1,D,H,W] MetaTensor (usually)
    try:
        affine = np.array(img.affine, dtype=np.float32)
    except Exception:
        # fallback affine
        sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
        affine = np.array([[sx,0,0,0],[0,sy,0,0],[0,0,sz,0],[0,0,0,1]], dtype=np.float32)

    # Center crop to patch size for checks
    patch_1dhw = center_crop_3d(img, patch_size)  # [1,D,H,W]
    patch = patch_1dhw.unsqueeze(0).to(device)    # [1,1,D,H,W]

    # Scheduler / inferer for sampling
    scheduler = DDPMScheduler(
        schedule=str(cfg.get("schedule", "scaled_linear_beta")),
        num_train_timesteps=int(cfg.get("num_train_timesteps", 1000)),
        beta_start=float(cfg.get("beta_start", 0.0015)),
        beta_end=float(cfg.get("beta_end", 0.0195)),
    )
    inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=scale_factor)

    print(f"[cfg] patch_size={patch_size} axcodes={axcodes} spacing={spacing}")
    print(f"[cfg] scale_factor={scale_factor:.6f} device={device} amp={AMP}")

    # Run checks
    if RUN_AUTOENC_RECON:
        check_autoencoder_recon(autoencoder, patch, affine, out_dir)

    if RUN_SCALE_FACTOR:
        # use full preprocessed vol for random crops
        check_scale_factor(autoencoder, img.to(torch.float32), patch_size, device, out_dir)

    if RUN_SINGLE_PATCH_SAMPLE:
        check_single_patch_sampling(inferer, autoencoder, diffusion, patch_size, affine, device, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Patch-trained 3D Latent Diffusion inference (BraTS FLAIR) -> full volume via tiled sampling + blending,
PLUS a PNG slice grid for quick inspection.

Notes:
- Sampling uses MONAI LatentDiffusionInferer.sample(input_noise, autoencoder_model, diffusion_model, ...) :contentReference[oaicite:2]{index=2}
- NIfTI writing uses nibabel Nifti1Image(data, affine) + nib.save :contentReference[oaicite:3]{index=3}
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
# Coordinate conditioning helpers (Fix 2)
# ============================================================
def make_coord_channels(latents: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Create normalised (x, y, z) coordinate channels matching latent spatial size."""
    _, _, lD, lH, lW = latents.shape
    z = torch.linspace(-1, 1, lD, device=device)
    y = torch.linspace(-1, 1, lH, device=device)
    x = torch.linspace(-1, 1, lW, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    coords = torch.stack([xx, yy, zz], dim=0)  # [3, lD, lH, lW]
    return coords[None].repeat(latents.shape[0], 1, 1, 1, 1)  # [B, 3, lD, lH, lW]


class CoordWrappedDiffusion(torch.nn.Module):
    """Wraps a diffusion UNet to automatically concat coordinate channels."""
    def __init__(self, diffusion: torch.nn.Module):
        super().__init__()
        self.diffusion = diffusion

    def forward(self, x, timesteps, **kwargs):
        coords = make_coord_channels(x, x.device)
        return self.diffusion(torch.cat([x, coords], dim=1), timesteps=timesteps, **kwargs)


# ============================================================
# Global-noise helper (Fix 1)
# ============================================================
def latent_shape_from_image_shape(autoencoder, image_shape_dhw, device):
    """Compute the full latent shape for a given image shape by encoding a dummy."""
    D, H, W = image_shape_dhw
    dummy = torch.zeros((1, 1, D, H, W), device=device)
    z = autoencoder.encode_stage_2_inputs(dummy)
    return tuple(z.shape)  # (1, C, lD, lH, lW)


# ============================================================
# CONFIG (edit these)
# ============================================================
CKPT_PATH = "./runs/brats_ldm_finetune/best_diffusion.pt"
OUT_DIR = "./model_scripts/finetuned_3d_flair/samples"

# Option A (recommended): use reference scan for output shape + affine (after preprocessing)
REFERENCE_NIFTI = None # "/path/to/some_subject_flair.nii.gz"  # set to None to use SHAPE instead

USE_PRETRAINED_DIFFUSION = True  # True = HF bundle models/model.pt, False = your CKPT_PATH
USE_EMA_WEIGHTS = True   # If True, load EMA weights from checkpoint when available (Fix 5)

# Option B: explicitly specify output shape (D,H,W) in preprocessed space if no reference
SHAPE_DHW = (240, 240, 155)  # used only if REFERENCE_NIFTI is None

# Patch sampling / tiling
PATCH_SIZE_DHW = None  # None = read from ckpt cfg if present, else fallback to (144,176,112)
OVERLAP = 0.5          # in [0,1). Higher overlap reduces seams but costs more compute
NUM_INFERENCE_STEPS = 1000  # often <=1000; fewer steps = faster but may reduce quality

# Sampling count & determinism
N_SAMPLES = 1
BASE_SEED = 123  # set to None for nondeterministic sampling

# Device / precision
DEVICE = "cuda"  # "cuda" or "cpu"
AMP = "fp16"     # "no" | "fp16" | "bf16"

# HF bundle (defaults taken from ckpt cfg or fallback values)
HF_REPO_ID_OVERRIDE = None   # e.g. "MONAI/brats_mri_generative_diffusion"
HF_REVISION_OVERRIDE = None  # e.g. "1.1.3"
BUNDLE_CACHE_DIR_OVERRIDE = None  # e.g. "./_pretrained_bundles"

# Preprocessing defaults (used if ckpt cfg missing)
AXCODES_FALLBACK = "RAS"
SPACING_FALLBACK = (1.1, 1.1, 1.1)

# PNG grid preview settings
SAVE_PNG_GRID = True
GRID_AXIS = "z"        # "z" (axial-like), "y" (coronal-like), "x" (sagittal-like) in array indexing
GRID_NUM_SLICES = 24
GRID_NCOLS = 6
GRID_CLIP_RANGE = (0.0, 1.0)  # because training scaled intensities into ~[0,1]
# ============================================================


# -------------------------
# Model definitions (must match training)
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
        in_channels=8 + 3,   # 8 latent + 3 coordinate channels (Fix 2)
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
# HF bundle loading (same logic as training)
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

def load_pretrained_diffusion_weights(diffusion: DiffusionModelUNet, bundle_root: Path) -> None:
    dm_path = bundle_root / "models" / "model.pt"
    if not dm_path.exists():
        raise FileNotFoundError(f"Missing pretrained diffusion weights: {dm_path}")

    dm_sd = torch.load(dm_path, map_location="cpu")

    # MONAI bundle checkpoints often use load_old_state_dict (also used in the bundle configs).
    # First, remap keys if needed (load_old_state_dict does this internally).
    if hasattr(diffusion, "load_old_state_dict"):
        # Use a temporary DiffusionModelUNet with the *original* in_channels=8
        # just to remap the old key names, then handle the shape mismatch ourselves.
        from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet as _DMUNET
        tmp = _DMUNET(
            spatial_dims=3, in_channels=8, out_channels=8,
            channels=(256, 256, 512), attention_levels=(False, True, True),
            num_head_channels=(0, 64, 64), num_res_blocks=2,
            include_fc=False, use_combined_linear=False,
        )
        tmp.load_old_state_dict(dm_sd)
        dm_sd = tmp.state_dict()

    # Handle conv_in shape mismatch: pretrained has 8 in-channels,
    # our model has 11 (8 latent + 3 coordinate channels).
    # Zero-initialise the extra coordinate channels (consistent with training script).
    conv_in_key = "conv_in.conv.weight"
    model_sd = diffusion.state_dict()
    if conv_in_key in dm_sd and dm_sd[conv_in_key].shape != model_sd[conv_in_key].shape:
        pretrained_w = dm_sd[conv_in_key]                    # [256, 8, 3, 3, 3]
        padded = torch.zeros_like(model_sd[conv_in_key])     # [256, 11, 3, 3, 3]
        padded[:, :pretrained_w.shape[1]] = pretrained_w
        dm_sd[conv_in_key] = padded
        print(f"[info] Padded {conv_in_key}: {pretrained_w.shape} -> {padded.shape}")

    diffusion.load_state_dict(dm_sd, strict=False)

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
            f"Checkpoint {ckpt_path} missing 'diffusion_state_dict'. "
            "Use best_diffusion.pt/last_diffusion.pt produced by your training script."
        )
    # Fix 5: prefer EMA weights for best sample quality
    if USE_EMA_WEIGHTS and "ema_diffusion_state_dict" in ckpt:
        print("[info] Loading EMA diffusion weights.")
        diffusion.load_state_dict(ckpt["ema_diffusion_state_dict"], strict=False)
    else:
        diffusion.load_state_dict(ckpt["diffusion_state_dict"], strict=False)
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
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0, upper=99.5, b_min=0.0, b_max=1.0
            ),
        ]
    )


# -------------------------
# Patch tiling helpers
# -------------------------
def _compute_starts(full: int, patch: int, stride: int):
    if full <= patch:
        return [0]
    starts = list(range(0, full - patch + 1, stride))
    if starts[-1] != full - patch:
        starts.append(full - patch)
    return starts


def gaussian_weight_map_3d(patch_size: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """
    Smooth center-weighted blend mask.
    Output shape: [1, 1, D, H, W]
    """
    dz, dy, dx = patch_size
    z = torch.linspace(-1.0, 1.0, steps=dz, device=device, dtype=torch.float32)
    y = torch.linspace(-1.0, 1.0, steps=dy, device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, steps=dx, device=device, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    sigma = 0.5
    w = torch.exp(-(zz**2 + yy**2 + xx**2) / (2 * sigma * sigma))
    w = w / (w.max() + 1e-8)
    return w[None, None, ...]


def _amp_dtype_from_str(amp: str) -> Optional[torch.dtype]:
    if amp == "fp16":
        return torch.float16
    if amp == "bf16":
        return torch.bfloat16
    return None


@torch.no_grad()
def sample_patch(
    inferer: LatentDiffusionInferer,
    autoencoder: AutoencoderKL,
    diffusion: DiffusionModelUNet,
    noise_shape: Tuple[int, int, int, int, int],
    device: torch.device,
    num_inference_steps: int,
    seed: Optional[int] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        noise = torch.randn(noise_shape, generator=g, device=device)
    else:
        noise = torch.randn(noise_shape, device=device)

    inferer.scheduler.set_timesteps(int(num_inference_steps))

    if amp_dtype is not None and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = inferer.sample(
                input_noise=noise,
                autoencoder_model=autoencoder,
                diffusion_model=diffusion,
                verbose=False,
            )
    else:
        out = inferer.sample(
            input_noise=noise,
            autoencoder_model=autoencoder,
            diffusion_model=diffusion,
            verbose=False,
        )

    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


@torch.no_grad()
def generate_full_volume(
    inferer: LatentDiffusionInferer,
    autoencoder: AutoencoderKL,
    diffusion,
    out_shape_dhw: Tuple[int, int, int],
    patch_size_dhw: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
    num_inference_steps: int,
    base_seed: Optional[int] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    assert 0.0 <= overlap < 1.0, "OVERLAP must be in [0, 1)."

    D, H, W = out_shape_dhw
    pD, pH, pW = patch_size_dhw

    sD = max(1, int(round(pD * (1.0 - overlap))))
    sH = max(1, int(round(pH * (1.0 - overlap))))
    sW = max(1, int(round(pW * (1.0 - overlap))))

    starts_d = _compute_starts(D, pD, sD)
    starts_h = _compute_starts(H, pH, sH)
    starts_w = _compute_starts(W, pW, sW)

    # Determine latent patch shape robustly (encode dummy patch)
    dummy = torch.zeros((1, 1, pD, pH, pW), device=device)
    z = autoencoder.encode_stage_2_inputs(dummy)
    latent_patch_shape = tuple(z.shape)  # (1, C, lpD, lpH, lpW)

    # Fix 1: compute full latent shape and create ONE global noise volume
    full_latent_shape = latent_shape_from_image_shape(autoencoder, (D, H, W), device)

    if base_seed is not None:
        g = torch.Generator(device=device).manual_seed(int(base_seed))
        global_noise = torch.randn(full_latent_shape, generator=g, device=device)
    else:
        global_noise = torch.randn(full_latent_shape, device=device)

    # Map image-space patch starts -> latent-space patch starts
    _, _, lpD, lpH, lpW = latent_patch_shape
    _, _, lD,  lH,  lW  = full_latent_shape
    rz = lD / D
    ry = lH / H
    rx = lW / W

    def to_lat_start(s, r, lp, lfull):
        ls = int(round(s * r))
        ls = min(max(ls, 0), max(lfull - lp, 0))
        return ls

    weight = gaussian_weight_map_3d((pD, pH, pW), device=device)

    acc = torch.zeros((1, 1, D, H, W), device=device, dtype=torch.float32)
    wsum = torch.zeros((1, 1, D, H, W), device=device, dtype=torch.float32)

    for sd in starts_d:
        for sh in starts_h:
            for sw in starts_w:
                # Crop noise patch from global noise (Fix 1)
                lsd = to_lat_start(sd, rz, lpD, lD)
                lsh = to_lat_start(sh, ry, lpH, lH)
                lsw = to_lat_start(sw, rx, lpW, lW)
                noise_patch = global_noise[:, :, lsd:lsd+lpD, lsh:lsh+lpH, lsw:lsw+lpW]

                inferer.scheduler.set_timesteps(int(num_inference_steps))

                if amp_dtype is not None and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        out = inferer.sample(
                            input_noise=noise_patch,
                            autoencoder_model=autoencoder,
                            diffusion_model=diffusion,
                            verbose=False,
                        )
                else:
                    out = inferer.sample(
                        input_noise=noise_patch,
                        autoencoder_model=autoencoder,
                        diffusion_model=diffusion,
                        verbose=False,
                    )

                patch = out[0] if isinstance(out, (tuple, list)) else out
                patch = patch.float()

                acc[:, :, sd : sd + pD, sh : sh + pH, sw : sw + pW] += patch * weight
                wsum[:, :, sd : sd + pD, sh : sh + pH, sw : sw + pW] += weight

    out = acc / (wsum + 1e-8)
    return out.clamp(0.0, 1.0)


# -------------------------
# PNG grid preview
# -------------------------
def save_slice_grid_png(
    vol_dhw: np.ndarray,
    out_png: Path,
    axis: str = "z",
    num_slices: int = 24,
    ncols: int = 6,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """
    Save a grid of evenly-spaced slices as PNG.
    vol_dhw: shape [D,H,W]
    axis:
      - "z": slices along D (vol[k,:,:])
      - "y": slices along H (vol[:,k,:])
      - "x": slices along W (vol[:,:,k])
    """
    assert vol_dhw.ndim == 3, f"Expected 3D array [D,H,W], got {vol_dhw.shape}"
    axis = axis.lower()
    if axis not in ("z", "y", "x"):
        raise ValueError("GRID_AXIS must be one of: 'z','y','x'")

    D, H, W = vol_dhw.shape
    axis_len = {"z": D, "y": H, "x": W}[axis]
    if axis_len <= 1:
        raise ValueError(f"Volume too small along axis {axis}: {axis_len}")

    num_slices = int(min(max(1, num_slices), axis_len))
    idxs = np.linspace(0, axis_len - 1, num_slices, dtype=int)

    ncols = int(max(1, ncols))
    nrows = int(math.ceil(num_slices / ncols))

    fig_w = 2.2 * ncols
    fig_h = 2.2 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= num_slices:
            continue

        k = int(idxs[i])
        if axis == "z":
            sl = vol_dhw[k, :, :]
        elif axis == "y":
            sl = vol_dhw[:, k, :]
        else:  # "x"
            sl = vol_dhw[:, :, k]

        ax.imshow(sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"{axis}={k}", fontsize=8)

    plt.tight_layout(pad=0.2)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    ckpt_path = Path(CKPT_PATH)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ckpt + cfg
    # Instantiate models
    autoencoder, diffusion = instantiate_models()

    cfg = {}
    extra = {}

    # If using finetuned checkpoint, load it and pull cfg/extra
    if not USE_PRETRAINED_DIFFUSION:
        ckpt_path = Path(CKPT_PATH)
        ckpt = load_diffusion_ckpt(diffusion, ckpt_path)
        cfg = ckpt.get("cfg", {}) or {}
        extra = ckpt.get("extra", {}) or {}
    else:
        # Using pretrained diffusion: cfg/extra may not exist, fall back to defaults + overrides
        ckpt_path = None

    axcodes = cfg.get("axcodes", AXCODES_FALLBACK)
    spacing = tuple(cfg.get("spacing", SPACING_FALLBACK))

    hf_repo_id = HF_REPO_ID_OVERRIDE or cfg.get("hf_repo_id", "MONAI/brats_mri_generative_diffusion")
    hf_revision = HF_REVISION_OVERRIDE or cfg.get("hf_revision", "1.1.3")
    bundle_cache_dir = BUNDLE_CACHE_DIR_OVERRIDE or cfg.get("bundle_cache_dir", "./_pretrained_bundles")

    # Patch size: config override > ckpt cfg > fallback
    if PATCH_SIZE_DHW is not None:
        patch_size = tuple(PATCH_SIZE_DHW)
    else:
        patch_size = tuple(cfg.get("patch_size", (144, 176, 112)))

    # Output shape and affine: from reference after preprocessing, else from SHAPE_DHW
    ref_affine = None
    out_shape = None
    if REFERENCE_NIFTI is not None and str(REFERENCE_NIFTI).strip():
        preproc = make_preproc(axcodes=axcodes, spacing=spacing)
        data = preproc({"image": REFERENCE_NIFTI})
        img = data["image"]  # [1,D,H,W] MetaTensor
        out_shape = tuple(int(x) for x in img.shape[1:])  # D,H,W
        try:
            ref_affine = np.array(img.affine, dtype=np.float32)
        except Exception:
            ref_affine = None
    else:
        out_shape = tuple(int(x) for x in SHAPE_DHW)

    # If no affine, create a simple one using voxel spacing
    if ref_affine is None:
        sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
        ref_affine = np.array(
            [
                [sx, 0,  0,  0],
                [0,  sy, 0,  0],
                [0,  0,  sz, 0],
                [0,  0,  0,  1],
            ],
            dtype=np.float32,
        )

    # Scheduler params (match training defaults unless ckpt cfg says otherwise)
    num_train_timesteps = int(cfg.get("num_train_timesteps", 1000))
    beta_start = float(cfg.get("beta_start", 0.0015))
    beta_end = float(cfg.get("beta_end", 0.0195))
    schedule = str(cfg.get("schedule", "scaled_linear_beta"))

    # Scale factor: prefer ckpt extra, then pretrained_bundle_info.json near ckpt, else 1.0
    scale_factor = 1.0
    if "scale_factor" in extra:
        scale_factor = float(extra["scale_factor"])
    else:
        if (not USE_PRETRAINED_DIFFUSION) and ckpt_path is not None:
            info_path = ckpt_path.parent / "pretrained_bundle_info.json"
            if info_path.exists():
                try:
                    info = json.loads(info_path.read_text())
                    scale_factor = float(info.get("scale_factor", 1.0))
                except Exception:
                    pass


    # Device + AMP
    if DEVICE == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[warn] CUDA not available; using CPU.")
    else:
        device = torch.device(DEVICE)

    amp_dtype = _amp_dtype_from_str(AMP)

    # Load autoencoder weights from HF bundle
    # Load bundle
    bundle_root = snapshot_bundle(hf_repo_id, hf_revision, bundle_cache_dir)

    # Load autoencoder weights from HF bundle
    load_autoencoder_weights(autoencoder, bundle_root)

    # Load diffusion weights:
    if USE_PRETRAINED_DIFFUSION:
        load_pretrained_diffusion_weights(diffusion, bundle_root)
    else:
        # already loaded from CKPT_PATH above
        pass

    # Move to device
    autoencoder = autoencoder.to(device).eval()
    diffusion = diffusion.to(device).eval()

    # Fix 2: wrap diffusion to auto-concatenate coordinate channels
    diffusion = CoordWrappedDiffusion(diffusion).to(device).eval()

    # Inferer
    scheduler = DDPMScheduler(
        schedule=schedule,
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
    )
    inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=scale_factor)

    print(f"[cfg] out_shape={out_shape} patch_size={patch_size} overlap={OVERLAP}")
    print(f"[cfg] steps={NUM_INFERENCE_STEPS} scale_factor={scale_factor:.6f} device={device} amp={AMP}")

    # Generate N_SAMPLES
    for i in range(int(N_SAMPLES)):
        sample_seed = None if BASE_SEED is None else int(BASE_SEED) + i * 10_000

        vol = generate_full_volume(
            inferer=inferer,
            autoencoder=autoencoder,
            diffusion=diffusion,
            out_shape_dhw=tuple(out_shape),
            patch_size_dhw=tuple(patch_size),
            overlap=float(OVERLAP),
            device=device,
            num_inference_steps=int(NUM_INFERENCE_STEPS),
            base_seed=sample_seed,
            amp_dtype=amp_dtype,
        )

        vol_np = vol.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)  # [D,H,W]

        # Save NIfTI :contentReference[oaicite:4]{index=4}
        out_nii = out_dir / f"sample_{i:03d}.nii.gz"
        nib.save(nib.Nifti1Image(vol_np, ref_affine), str(out_nii))
        print(f"[saved] {out_nii}")

        # Save PNG grid
        if SAVE_PNG_GRID:
            out_png = out_dir / f"sample_{i:03d}_grid.png"
            vmin, vmax = float(GRID_CLIP_RANGE[0]), float(GRID_CLIP_RANGE[1])
            save_slice_grid_png(
                vol_dhw=vol_np,
                out_png=out_png,
                axis=GRID_AXIS,
                num_slices=int(GRID_NUM_SLICES),
                ncols=int(GRID_NCOLS),
                vmin=vmin,
                vmax=vmax,
            )
            print(f"[saved] {out_png}")

    print("Done.")


if __name__ == "__main__":
    main()
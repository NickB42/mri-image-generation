#!/usr/bin/env python3
"""
infer_ldm_3d.py

Unconditional inference for a 3D latent diffusion model trained on BraTS (4 modalities).
- Loads VAE3D + GaussianDiffusionLatent3D(UNet3DModel)
- Auto-detects latent spatial size by running VAE.encode_to_latent() on a dummy patch
- Samples latents using diffusion.sample(batch_size, spatial_size)
- Decodes to image space using VAE decoder
- Saves:
    - raw tensor .pt
    - slice grid PNG
    - optional NIfTI .nii.gz per modality if nibabel installed

Run (from your repo root, so imports work):
  PYTHONPATH=. python infer_ldm_3d.py \
    --vae_ckpt /path/to/vae3d_final.pt \
    --ldm_ckpt /path/to/3d_ldm_diffusion_best.pt \
    --outdir ./inference_out \
    --num_samples 2
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import torch
import matplotlib.pyplot as plt


# -------------------------
# EDIT THESE IMPORTS
# -------------------------
# Make these match your project layout.
from your_package.vae import VAE3D
from your_package.unet import UNet3DModel
from your_package.diffusion import GaussianDiffusionLatent3D


def pick_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _unwrap_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """Support both raw state_dict and {"state_dict": ...} checkpoints."""
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        return ckpt_obj["state_dict"]
    if isinstance(ckpt_obj, dict):
        # could already be a state dict
        # (heuristic: values are tensors)
        if len(ckpt_obj) > 0 and all(torch.is_tensor(v) for v in ckpt_obj.values()):
            return ckpt_obj  # type: ignore
    raise ValueError("Checkpoint format not recognized (expected state_dict or {'state_dict': state_dict}).")


def _remap_ddp_keys_if_needed(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Fix common DDP prefixes:
    - "module." -> ""
    - "model.module." -> "model."
    This is crucial if you saved diffusion.state_dict() while diffusion.model was DDP-wrapped.
    """
    keys = list(state.keys())
    remapped = {}

    # Case 1: top-level "module."
    has_module_prefix = any(k.startswith("module.") for k in keys)

    # Case 2: diffusion keys with "model.module."
    has_model_module_prefix = any(k.startswith("model.module.") for k in keys)

    for k, v in state.items():
        nk = k
        if has_module_prefix and nk.startswith("module."):
            nk = nk[len("module.") :]
        if has_model_module_prefix and nk.startswith("model.module."):
            nk = "model." + nk[len("model.module.") :]
        remapped[nk] = v

    return remapped


def load_state_dict_safely(module: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = _unwrap_state_dict(ckpt)
    state = _remap_ddp_keys_if_needed(state)

    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys loading {ckpt_path.name} (showing up to 20): {missing[:20]}")
    if unexpected:
        print(f"[WARN] Unexpected keys loading {ckpt_path.name} (showing up to 20): {unexpected[:20]}")


@torch.no_grad()
def infer_latent_spatial_size(vae: torch.nn.Module, patch: Tuple[int, int, int], device: torch.device) -> Tuple[int, int, int]:
    """
    Run a dummy patch through encode_to_latent to infer the latent D/H/W.
    This avoids guessing whether your VAE downsamples spatially.
    """
    x = torch.zeros((1, 4, patch[0], patch[1], patch[2]), device=device, dtype=torch.float32)
    z = vae.encode_to_latent(x)
    if z.ndim != 5:
        raise RuntimeError(f"Expected latent to be 5D (B,C,D,H,W), got shape {tuple(z.shape)}")
    return (int(z.shape[-3]), int(z.shape[-2]), int(z.shape[-1]))


@torch.no_grad()
def decode_latents(vae: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    """
    Attempts likely decode APIs for your VAE3D.
    Returns (B, 4, D, H, W)
    """
    if hasattr(vae, "decode_from_latent") and callable(getattr(vae, "decode_from_latent")):
        return vae.decode_from_latent(z)
    if hasattr(vae, "decode") and callable(getattr(vae, "decode")):
        return vae.decode(z)
    if hasattr(vae, "decoder"):
        return vae.decoder(z)
    raise AttributeError(
        "Could not find a supported VAE decode method.\n"
        "Tried: decode_from_latent(z), decode(z), vae.decoder(z)."
    )


def save_slice_grid_png(
    vol_4ch: torch.Tensor,
    out_png: Path,
    title: str = "",
    clamp_percentiles: Optional[Tuple[float, float]] = (1.0, 99.0),
) -> None:
    """
    vol_4ch: (4, D, H, W) CPU tensor
    Saves a 4 (modalities) x 3 (axial/coronal/sagittal) mid-slice grid.
    """
    assert vol_4ch.ndim == 4 and vol_4ch.shape[0] == 4, f"Expected (4,D,H,W), got {tuple(vol_4ch.shape)}"

    vol = vol_4ch.numpy()
    D, H, W = vol.shape[1], vol.shape[2], vol.shape[3]
    mids = (D // 2, H // 2, W // 2)

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(10, 10))
    if title:
        fig.suptitle(title)

    for c in range(4):
        ch = vol[c]
        axial = ch[mids[0], :, :]
        coronal = ch[:, mids[1], :]
        sagittal = ch[:, :, mids[2]]
        slices = [axial, coronal, sagittal]

        for j, sl in enumerate(slices):
            ax = axes[c, j]
            if clamp_percentiles is not None:
                lo, hi = clamp_percentiles
                t = torch.tensor(sl)
                vmin = float(torch.quantile(t, lo / 100.0))
                vmax = float(torch.quantile(t, hi / 100.0))
                ax.imshow(sl, vmin=vmin, vmax=vmax, cmap="gray")
            else:
                ax.imshow(sl, cmap="gray")
            ax.axis("off")

        axes[c, 0].set_ylabel(f"mod{c}", rotation=0, labelpad=20, va="center")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def maybe_save_nifti_per_modality(vol_4ch: torch.Tensor, out_stem: Path) -> List[Path]:
    """
    Saves each modality as NIfTI if nibabel is installed.
    Uses identity affine (good for viewing, not physical spacing).
    """
    try:
        import nibabel as nib
        import numpy as np
    except Exception:
        return []

    vol = vol_4ch.numpy()  # (4,D,H,W)
    affine = np.eye(4)

    written: List[Path] = []
    for c in range(4):
        nii = nib.Nifti1Image(vol[c].astype("float32"), affine)
        out_path = out_stem.parent / f"{out_stem.name}_mod{c}.nii.gz"
        nib.save(nii, str(out_path))
        written.append(out_path)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae_ckpt", type=str, required=True)
    ap.add_argument("--ldm_ckpt", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)

    # Patch size (matches train.py defaults)
    ap.add_argument("--patch_d", type=int, default=128)
    ap.add_argument("--patch_h", type=int, default=160)
    ap.add_argument("--patch_w", type=int, default=160)

    # Model hyperparams (matches train.py defaults)
    ap.add_argument("--latent_channels", type=int, default=16)
    ap.add_argument("--vae_base_channels", type=int, default=32)
    ap.add_argument("--vae_num_down", type=int, default=3)

    ap.add_argument("--unet_base_channels", type=int, default=128)
    ap.add_argument("--unet_channel_mults", type=str, default="1,2,4")

    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[INFO] Device: {device}")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    patch = (args.patch_d, args.patch_h, args.patch_w)
    channel_mults = tuple(int(x.strip()) for x in args.unet_channel_mults.split(",") if x.strip())

    # Build models
    vae = VAE3D(
        in_channels=4,
        base_channels=args.vae_base_channels,
        num_down=args.vae_num_down,
        latent_channels=args.latent_channels,
    ).to(device)

    unet = UNet3DModel(
        in_channels=args.latent_channels,
        base_channels=args.unet_base_channels,
        channel_mults=channel_mults,
    ).to(device)

    diffusion = GaussianDiffusionLatent3D(
        model=unet,
        channels=args.latent_channels,
        timesteps=args.timesteps,
    ).to(device)

    vae.eval()
    diffusion.eval()

    # Load weights
    vae_ckpt = Path(args.vae_ckpt)
    ldm_ckpt = Path(args.ldm_ckpt)
    if not vae_ckpt.exists():
        raise FileNotFoundError(vae_ckpt)
    if not ldm_ckpt.exists():
        raise FileNotFoundError(ldm_ckpt)

    print(f"[INFO] Loading VAE from {vae_ckpt}")
    load_state_dict_safely(vae, vae_ckpt, device)

    print(f"[INFO] Loading diffusion from {ldm_ckpt}")
    load_state_dict_safely(diffusion, ldm_ckpt, device)

    # Infer latent spatial size from VAE
    latent_spatial = infer_latent_spatial_size(vae, patch, device)
    print(f"[INFO] Patch size:  {patch}")
    print(f"[INFO] Latent size: {latent_spatial} (detected via vae.encode_to_latent)")

    # Sample + decode
    for i in range(args.num_samples):
        print(f"[INFO] Sampling {i+1}/{args.num_samples} ...")

        # Your diffusion.sample signature:
        #   sample(batch_size, spatial_size, cond=None)
        z = diffusion.sample(batch_size=1, spatial_size=latent_spatial, cond=None)
        x = decode_latents(vae, z)  # (1,4,D,H,W)

        x = x.detach().float().cpu()[0]  # (4,D,H,W)

        out_stem = outdir / f"sample_{i:03d}"

        # Save tensor
        torch.save(x, out_stem.with_suffix(".pt"))
        print(f"[OK] Wrote {out_stem.with_suffix('.pt')}")

        # Save PNG
        png_path = out_stem.with_suffix(".png")
        save_slice_grid_png(x, png_path, title=f"sample_{i:03d}")
        print(f"[OK] Wrote {png_path}")

        # Save NIfTI (optional)
        niis = maybe_save_nifti_per_modality(x, out_stem)
        if niis:
            for p in niis:
                print(f"[OK] Wrote {p}")
        else:
            print("[INFO] nibabel not installed; skipping NIfTI export.")

    print("[DONE] Inference complete.")


if __name__ == "__main__":
    main()

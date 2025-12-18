"""
Unconditional inference for a 3D latent diffusion model trained on BraTS (4 modalities).
- Loads VAE3D + GaussianDiffusionLatent3D(UNet3DModel)
- Auto-detects latent spatial size by running VAE.encode_to_latent() on a dummy patch
- Samples latents using diffusion.sample(batch_size, spatial_size)
- Decodes to image space using VAE decoder
- Saves:
    - raw tensor .pt
    - slice grid PNG
    - optional NIfTI .nii.gz per modality if nibabel installed
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import math

import torch
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib


from .vae import VAE3D
from .unet import UNet3DModel
from .unet_attention import UNet3DModelWithAttention
from .diffusion import GaussianDiffusionLatent3D
from .dataset import BraTS3DVolumeDataset


def pick_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_unwrapped_model(m):
    return m.module if hasattr(m, "module") else m

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

    missing, unexpected = module.load_state_dict(state, strict=True)
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

    vol = vol_4ch.numpy()  # (4,D,H,W)
    affine = np.eye(4)

    written: List[Path] = []
    for c in range(4):
        nii = nib.Nifti1Image(vol[c].astype("float32"), affine)
        out_path = out_stem.parent / f"{out_stem.name}_mod{c}.nii.gz"
        nib.save(nii, str(out_path))
        written.append(out_path)
    return written


def setup_models(device: torch.device, vae_config: Dict[str, Any], unet_config: Dict[str, Any], diffusion_config: Dict[str, Any]) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Build and initialize VAE and diffusion models."""
    vae = VAE3D(
        in_channels=4,
        base_channels=vae_config["base_channels"],
        num_down=vae_config["num_down"],
        latent_channels=vae_config["latent_channels"],
    ).to(device)

    if unet_config.get("use_attention", False):
        unet = UNet3DModelWithAttention(
            in_channels=vae_config["latent_channels"],
            base_channels=unet_config["base_channels"],
            channel_mults=unet_config["channel_mults"],
            time_emb_dim=unet_config.get("time_emb_dim", 256),
            groups=unet_config.get("groups", 8),
            num_heads=unet_config.get("num_heads", 4),
        ).to(device)
    else:
        unet = UNet3DModel(
            in_channels=vae_config["latent_channels"],
            base_channels=unet_config["base_channels"],
            channel_mults=unet_config["channel_mults"],
        ).to(device)

    diffusion = GaussianDiffusionLatent3D(
        model=unet,
        channels=vae_config["latent_channels"],
        timesteps=diffusion_config["timesteps"],
    ).to(device)

    vae.eval()
    diffusion.eval()

    return vae, diffusion


def load_model_weights(vae: torch.nn.Module, diffusion: torch.nn.Module, vae_ckpt: Path, ldm_ckpt: Path, device: torch.device) -> None:
    """Load pretrained weights for VAE and diffusion models."""
    if not vae_ckpt.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt}")
    if not ldm_ckpt.exists():
        raise FileNotFoundError(f"LDM checkpoint not found: {ldm_ckpt}")


    def print_weight_norms(tag, m):
        w = next(p for p in m.parameters() if p.ndim >= 2)
        print(f"{tag}: first weight norm = {w.detach().float().norm().item():.4f}")

    print(f"[INFO] Loading VAE from {vae_ckpt}")
    load_state_dict_safely(vae, vae_ckpt, device)

    print(f"[INFO] Loading diffusion from {ldm_ckpt}")
    print_weight_norms("UNet before", diffusion.model)
    load_state_dict_safely(diffusion, Path(ldm_ckpt), device)
    print_weight_norms("UNet after", diffusion.model)


def save_sample(x: torch.Tensor, out_stem: Path, sample_idx: int) -> None:
    """Save a single sample as tensor, PNG, and optionally NIfTI."""
    # Save tensor
    torch.save(x, out_stem.with_suffix(".pt"))
    print(f"[OK] Wrote {out_stem.with_suffix('.pt')}")

    # Save PNG
    png_path = out_stem.with_suffix(".png")
    save_slice_grid_png(x, png_path, title=f"sample_{sample_idx:03d}")
    print(f"[OK] Wrote {png_path}")

    # Save NIfTI (optional)
    niis = maybe_save_nifti_per_modality(x, out_stem)
    if niis:
        for p in niis:
            print(f"[OK] Wrote {p}")
    else:
        print("[INFO] nibabel not installed; skipping NIfTI export.")


def generate_samples(vae: torch.nn.Module, diffusion: torch.nn.Module, latent_spatial: Tuple[int, int, int], outdir: Path, num_samples: int) -> None:
    """Generate and save multiple samples."""
    for i in range(num_samples):
        print(f"[INFO] Sampling {i+1}/{num_samples} ...")

        z = diffusion.sample(batch_size=1, spatial_size=latent_spatial, cond=None)
        x = vae.decode_from_latent(z)  # (1,4,D,H,W)
        x = x.detach().float().cpu()[0]  # (4,D,H,W)

        out_stem = outdir / f"sample_{i:03d}"
        save_sample(x, out_stem, i)


@torch.no_grad()
def vae_recon_sanity(vae, loader, device, out_png):
    vae.eval()
    x = next(iter(loader)).to(device)              # (B,4,D,H,W)
    recon, mu, logvar = vae(x)                     # forward() path
    x0 = x[0].detach().float().cpu()
    r0 = recon[0].detach().float().cpu()

    # stack as 8 channels: original 4 + recon 4
    stacked = torch.cat([x0, r0], dim=0)           # (8,D,H,W)
    save_slice_grid_png(
        stacked[:4], Path(str(out_png).replace(".png","_orig.png")),
        title="orig"
    )
    save_slice_grid_png(
        stacked[4:], Path(str(out_png).replace(".png","_recon.png")),
        title="recon"
    )

@torch.no_grad()
def latent_stats(vae, loader, device, n_batches=20):
    vae.eval()
    mus = []
    stds = []
    for i, x in enumerate(loader):
        if i >= n_batches: break
        x = x.to(device)
        z = vae.encode_to_latent(x).detach().float()
        # global stats
        mus.append(z.mean().cpu())
        stds.append(z.std().cpu())
    m = torch.stack(mus).mean().item()
    s = torch.stack(stds).mean().item()
    print(f"latent mean ~ {m:.4f}, latent std ~ {s:.4f}")
    return m, s

@torch.no_grad()
def run_roundtrip_and_save(diffusion, vae, loader, device, outdir, t_start=399):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    latent_scale = estimate_latent_scale(vae, loader, device, num_batches=100)

    x0, xrec = roundtrip_test(diffusion, vae, loader, device, t_start=t_start, latent_scale=latent_scale)
    # x0, xrec are (4,D,H,W) on CPU from roundtrip_test

    save_slice_grid_png(x0, outdir / f"roundtrip_t{t_start:04d}_orig.png", title=f"orig (t={t_start})")
    save_slice_grid_png(xrec, outdir / f"roundtrip_t{t_start:04d}_recon.png", title=f"recon (t={t_start})")

    # optional nifti per modality
    maybe_save_nifti_per_modality(x0, outdir / f"roundtrip_t{t_start:04d}_orig")
    maybe_save_nifti_per_modality(xrec, outdir / f"roundtrip_t{t_start:04d}_recon")

    print("[OK] wrote roundtrip outputs to", outdir)


@torch.no_grad()
def roundtrip_test(diffusion, vae, loader, device, t_start=399, latent_scale=1.0):
    diffusion.eval()
    vae.eval()

    x = next(iter(loader)).to(device)                 # (B,4,D,H,W)
    z0 = vae.encode_to_latent(x).float() * latent_scale

    t = torch.full((z0.size(0),), t_start, device=device, dtype=torch.long)
    noise = torch.randn_like(z0)
    zt = diffusion.q_sample(z0, t, noise=noise)

    # deterministic reverse (DDIM)
    z_rec = diffusion.sample_from_ddim(zt, start_t=t_start, cond=None)

    # invert latent scaling before decode
    x_rec = vae.decode_from_latent(z_rec / latent_scale).float()

    return x[0].cpu(), x_rec[0].cpu()


@torch.no_grad()
def eps_mse_by_t(diffusion, vae, loader, device, t_values):
    diffusion.eval(); vae.eval()
    x = next(iter(loader)).to(device)
    z0 = vae.encode_to_latent(x).float()

    for t0 in t_values:
        t = torch.full((z0.size(0),), t0, device=device, dtype=torch.long)
        noise = torch.randn_like(z0)
        zt = diffusion.q_sample(z0, t, noise=noise)
        eps = diffusion.model(zt, t)
        mse = torch.mean((eps - noise) ** 2).item()
        print(f"t={t0:4d}  eps-MSE={mse:.4f}")

@torch.no_grad()
def estimate_latent_scale(vae, loader, device, num_batches=200):
    vae.eval()
    base_vae = get_unwrapped_model(vae)

    vars_ = []
    for i, x in enumerate(loader):
        if i >= num_batches:
            break
        x = x.to(device, non_blocking=True)
        z = base_vae.encode_to_latent(x).float()  # (B,C,d,h,w)
        vars_.append(z.var(unbiased=False).item())

    v = float(np.mean(vars_)) if len(vars_) else 1.0
    return 1.0 / math.sqrt(max(v, 1e-8))

def main():
    # Configuration
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    EXPERIMENT_NAME = "ddpm_3d_ldm"
    EXPERIMENT_ROOT = PROJECT_ROOT / EXPERIMENT_NAME

    RUN_ID = "1594474"

    CHECKPOINT_PATH = (
        EXPERIMENT_ROOT
        / "models"
        / RUN_ID
    )
    VAE_CKPT = f"{CHECKPOINT_PATH}/vae3d_final.pt"
    LDM_CKPT = f"{CHECKPOINT_PATH}/3d_ldm_diffusion_best.pt"
    OUTDIR = f"{EXPERIMENT_ROOT}/samples_inference/{RUN_ID}"
    DEVICE = None  # auto-detect
    SEED = 0
    NUM_SAMPLES = 1

    patch = (128, 160, 160)  # (D, H, W)

    vae_config = {
        "base_channels": 32,
        "num_down": 3,
        "latent_channels": 16,
    }

    unet_config = {
        "base_channels": 128,
        "channel_mults": (1, 2, 4),
        "use_attention": True,
        "time_emb_dim": 256,
        "groups": 8,
        "num_heads": 4,
    }

    diffusion_config = {
        "timesteps": 400,
    }

    # Setup
    device = pick_device(DEVICE)
    print(f"[INFO] Device: {device}")

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build models
    vae, diffusion = setup_models(device, vae_config, unet_config, diffusion_config)

    # Load weights
    load_model_weights(vae, diffusion, Path(VAE_CKPT), Path(LDM_CKPT), device)

    # Infer latent spatial size
    latent_spatial = infer_latent_spatial_size(vae, patch, device)
    print(f"[INFO] Patch size:  {patch}")
    print(f"[INFO] Latent size: {latent_spatial} (detected via vae.encode_to_latent)")

    DATASET_ROOT = (PROJECT_ROOT / "../datasets/dataset").resolve()

    dataset = BraTS3DVolumeDataset(
        root_dir=DATASET_ROOT,
        patch_size=patch,
        random_crop=False,
    )

    vae_recon_sanity(
        vae,
        loader=torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
        ),
        device=device,
        out_png=outdir / "vae_recon_sanity.png"
    )

    latent_stats(
        vae,
        loader=torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
        ),
        device=device,
        n_batches=20
    )

    for t_start in [50, 100, 200, 399]:
        run_roundtrip_and_save(
            diffusion=diffusion,
            vae=vae,
            loader=torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=2,
                shuffle=False,
            ),
            device=device,
            outdir=outdir,
            t_start=t_start,
        )

    eps_mse_by_t(diffusion, vae,
        loader=torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
        ),
        device=device,
        t_values=[0, 50, 100, 200, 399]
    )


    # Generate samples
    generate_samples(vae, diffusion, latent_spatial, outdir, NUM_SAMPLES)

    print("[DONE] Inference complete.")


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Union, Optional, List

import torch
from torchvision.utils import save_image

from .unet import UNet
from .diffusion import GaussianDiffusion

# ------------------ device ------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ------------------ config ------------------
BASE_DIR = Path(__file__).resolve().parent

IMAGE_SIZE = 128

# MUST match train.py
TIMESTEPS = 1000

CENTER_MODALITIES = 4          # e.g. [FLAIR, T1, T1ce, T2] – adjust if needed
SLICE_RADIUS = 2
CONTEXT_SLICES = 2 * SLICE_RADIUS   # number of neighbouring slices used as context
IN_CHANNELS = CENTER_MODALITIES + CENTER_MODALITIES * CONTEXT_SLICES
OUT_CHANNELS = CENTER_MODALITIES

# Optional: names for nicer filenames
MODALITY_NAMES = ["flair", "t1", "t1ce", "t2"]  # adjust to your dataset order


# ------------------ helpers ------------------
def _modality_name(idx: int) -> str:
    if 0 <= idx < len(MODALITY_NAMES):
        return MODALITY_NAMES[idx]
    return f"mod{idx}"


# ------------------ sampling: 2D ------------------
@torch.no_grad()
def sample_and_save(
    diffusion: GaussianDiffusion,
    epoch: int,
    num_samples: int = 16,
    out_dir: Union[str, Path] = "samples",
    nrow: int = 4,
    which_modality: Optional[int] = None,
) -> None:
    """
    Generate 2D samples and save PNGs.

    diffusion.sample(...) is assumed to return (B, OUT_CHANNELS, H, W)
    with values in [-1, 1].

    - If which_modality is None: save one grid per modality.
    - If which_modality is int: save only that modality.
    """
    diffusion.model.eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Random 2D slices (z_pos default inside diffusion.sample, e.g. 0.5)
    samples = diffusion.sample(batch_size=num_samples).cpu()  # (B, C=OUT_CHANNELS, H, W)

    print(
        "raw sample stats:",
        "min", samples.min().item(),
        "max", samples.max().item(),
        "mean", samples.mean().item(),
        "std", samples.std().item(),
    )

    # map from [-1, 1] to [0, 1]
    samples = samples.clamp(-1, 1)
    samples = (samples + 1.0) / 2.0

    print(
        "vis sample stats:",
        "min", samples.min().item(),
        "max", samples.max().item(),
        "mean", samples.mean().item(),
        "std", samples.std().item(),
    )

    if which_modality is not None:
        # save only one modality
        m = which_modality
        if m < 0 or m >= OUT_CHANNELS:
            raise ValueError(f"which_modality={m} out of range [0, {OUT_CHANNELS - 1}]")

        mod_samples = samples[:, m:m+1, :, :]  # (B, 1, H, W)
        mod_name = _modality_name(m)
        save_path = out_dir / f"samples_epoch_{epoch:03d}_{mod_name}.png"
        save_image(mod_samples, save_path, nrow=nrow)
        print(f"Saved samples for modality '{mod_name}' to {save_path}")
    else:
        # save one grid per modality
        for m in range(OUT_CHANNELS):
            mod_samples = samples[:, m:m+1, :, :]  # (B, 1, H, W)
            mod_name = _modality_name(m)
            save_path = out_dir / f"samples_epoch_{epoch:03d}_{mod_name}.png"
            save_image(mod_samples, save_path, nrow=nrow)
            print(f"Saved samples for modality '{mod_name}' to {save_path}")


# ------------------ sampling: pseudo-3D volume ------------------
@torch.no_grad()
def generate_pseudo_3d_brain(
    diffusion: GaussianDiffusion,
    num_slices: int = 155,
    out_dir: str = "pseudo3d_brains",
    volume_name: str = "brain0",
    grid_nrow: int = 16,
    save_example_slices: bool = False,
    use_context: bool = True,
) -> torch.Tensor:
    """
    Generate a pseudo-3D brain by sampling 2D slices conditioned on z_pos
    (and optionally neighbouring slices as context).

    Returns:
        volume: (num_slices, OUT_CHANNELS, H, W) in [-1, 1]
    """
    diffusion.model.eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = diffusion.betas.device  # or next(diffusion.parameters()).device

    # z positions in [0, 1] for each slice
    z_pos_all = torch.linspace(0.1, 0.9, steps=num_slices, device=device)


    if not use_context:
        # ---- Easy case: no neighbour context, all slices in one batch ----
        volume = diffusion.sample(batch_size=num_slices, z_pos=z_pos_all)  # (S, C, H, W)
    else:
        # ---- Sequential generation using previously generated slices as context ----
        context_channels = IN_CHANNELS - OUT_CHANNELS
        expected_ctx_slices = context_channels // OUT_CHANNELS
        assert expected_ctx_slices == CONTEXT_SLICES, (
            f"Config mismatch: expected {CONTEXT_SLICES} context slices "
            f"but inferred {expected_ctx_slices} from IN_CHANNELS."
        )

        H = W = IMAGE_SIZE
        volume = torch.zeros(num_slices, OUT_CHANNELS, H, W, device=device)

        print(
            f"Generating {num_slices} slices with neighbour context "
            f"(radius={SLICE_RADIUS}, total context slices={CONTEXT_SLICES})..."
        )

        for k in range(num_slices):
            # Build context for slice k from already generated slices.
            ctx_slices: List[torch.Tensor] = []

            # Past neighbours: k - SLICE_RADIUS ... k - 1
            for offset in range(-SLICE_RADIUS, 0):
                j = k + offset
                if 0 <= j < num_slices and j < k:
                    ctx_slices.append(volume[j])  # (C, H, W)
                else:
                    ctx_slices.append(torch.zeros(OUT_CHANNELS, H, W, device=device))

            # Future neighbours: k + 1 ... k + SLICE_RADIUS
            # (we don't know them yet → use zeros)
            for offset in range(1, SLICE_RADIUS + 1):
                j = k + offset
                if 0 <= j < num_slices and j < k:
                    ctx_slices.append(volume[j])
                else:
                    ctx_slices.append(torch.zeros(OUT_CHANNELS, H, W, device=device))

            context = torch.cat(ctx_slices, dim=0)  # (context_channels, H, W)
            context = context.unsqueeze(0)          # (1, context_channels, H, W)

            current_z = z_pos_all[k:k+1]           # (1,)

            # IMPORTANT: this assumes diffusion.sample(..., context=...) exists
            slice_sample = diffusion.sample(
                batch_size=1,
                z_pos=current_z,
                context=context,
            )  # (1, OUT_CHANNELS, H, W)

            if k == 80:
                print(
                    "raw sample stats:",
                    "min", slice_sample.min().item(),
                    "max", slice_sample.max().item(),
                    "mean", slice_sample.mean().item(),
                    "std", slice_sample.std().item(),
                )

            volume[k] = slice_sample[0]

    # volume is in [-1, 1]; prepare [0, 1] for visualization
    volume_clamped = volume.clamp(-1, 1)
    volume_vis = (volume_clamped + 1.0) / 2.0  # (S, C, H, W)

    # ---- 1) Save per-modality grids over all slices ----
    for m in range(OUT_CHANNELS):
        mod_volume = volume_vis[:, m:m+1, :, :]  # (S, 1, H, W)
        mod_name = _modality_name(m)
        grid_path = out_dir / f"{volume_name}_all_slices_{mod_name}.png"
        save_image(mod_volume, grid_path, nrow=grid_nrow)
        print(f"[pseudo-3D] Saved all {num_slices} slices grid ({mod_name}) to {grid_path}")

    # ---- 2) Optionally save a few example slices individually (for each modality) ----
    if save_example_slices:
        example_indices = [0, num_slices // 4, num_slices // 2,
                           3 * num_slices // 4, num_slices - 1]
        example_indices = sorted(set(example_indices))

        for idx in example_indices:
            for m in range(OUT_CHANNELS):
                slice_img = volume_vis[idx:idx + 1, m:m+1, :, :]  # keep batch & channel dims
                mod_name = _modality_name(m)
                slice_path = out_dir / f"{volume_name}_slice_{idx:03d}_{mod_name}.png"
                save_image(slice_img, slice_path, nrow=1)
                print(f"[pseudo-3D] Saved example slice {idx} ({mod_name}) to {slice_path}")

    return volume.cpu()  # (num_slices, OUT_CHANNELS, H, W) in [-1, 1]


# ------------------ model loading & entry point ------------------
def load_best_model_and_sample(
    checkpoint_path: Union[str, Path] = BASE_DIR / "2d_central_ddpm_flair_best.pt",
    num_samples: int = 16,
    out_dir: Union[str, Path] = BASE_DIR / "samples_inference",
    nrow: int = 4,
    mode: str = "2d",              # "2d" or "pseudo3d"
    pseudo3d_num_slices: int = 155,
    pseudo3d_volume_name: str = "brain0",
    pseudo3d_grid_nrow: int = 16,
    which_modality: Optional[int] = None,  # for 2D mode
    use_context_in_pseudo3d: bool = True,
) -> None:
    """
    Reconstruct the diffusion model, load weights from a checkpoint, and
    either generate 2D samples or a pseudo-3D brain volume.
    """
    checkpoint_path = Path(checkpoint_path)
    out_dir = Path(out_dir)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Make sure training has created '2d_central_ddpm_flair_best.pt'."
        )

    # Recreate model & diffusion exactly as in training (architecture-wise)
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
    ).to(device)

    # ---- Load weights (handle DataParallel prefix) ----
    state_dict = torch.load(checkpoint_path, map_location=device)

    # If the checkpoint was saved with DataParallel, keys start with "model.module."
    if any(k.startswith("model.module.") for k in state_dict.keys()):
        print("Detected DataParallel checkpoint; stripping 'module.' prefixes.")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.module."):
                new_k = k.replace("model.module.", "model.", 1)
            else:
                new_k = k
            new_state_dict[new_k] = v
        state_dict = new_state_dict

    diffusion.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {checkpoint_path}")

    p = next(diffusion.parameters())
    print("Loaded param norm:", p.data.norm().item())


    out_dir.mkdir(parents=True, exist_ok=True)

    if mode.lower() == "2d":
        sample_and_save(
            diffusion=diffusion,
            epoch=0,
            num_samples=num_samples,
            out_dir=str(out_dir),
            nrow=nrow,
            which_modality=which_modality,
        )
    elif mode.lower() == "pseudo3d":
        generate_pseudo_3d_brain(
            diffusion=diffusion,
            num_slices=pseudo3d_num_slices,
            out_dir=str(out_dir),
            volume_name=pseudo3d_volume_name,
            grid_nrow=pseudo3d_grid_nrow,
            use_context=use_context_in_pseudo3d,
        )
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use '2d' or 'pseudo3d'.")


if __name__ == "__main__":
    path = BASE_DIR / "models" / "1591706" / "25d_ddpm_all_modalities_best.pt"
    print("Using checkpoint path:", path)

    load_best_model_and_sample(
        checkpoint_path=path,
        mode="pseudo3d",
        pseudo3d_num_slices=155,
        pseudo3d_volume_name="brain_all_modalities_25d_ddpm",
        use_context_in_pseudo3d=True,
    )

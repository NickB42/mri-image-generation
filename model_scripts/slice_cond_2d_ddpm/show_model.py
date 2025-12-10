from pathlib import Path
from typing import Union

import torch

from .unet import UNet
from .diffusion import GaussianDiffusion
from torchvision.utils import save_image

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
TIMESTEPS = 800 # 800

def sample_and_save(
    diffusion,
    epoch,
    num_samples=16,
    out_dir="samples",
    context_slices=3,
    nrow=4,
):
    """
    Generate samples and save a grid image to disk.

    - diffusion: your GaussianDiffusion instance
    - epoch: current epoch (used in filename)
    - num_samples: how many images to sample
    - out_dir: folder where PNGs will be saved
    - context_slices: if using 2.5D, number of slice-channels (to pick center one)
    - nrow: number of images per row in the grid
    """
    diffusion.model.eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # (B, C, H, W), C = channels (1 for 2D, >1 for 2.5D)
        samples = diffusion.sample(batch_size=num_samples).cpu()

    # map from [-1, 1] to [0, 1]
    samples = samples.clamp(-1, 1)
    samples = (samples + 1) / 2.0

    # If using 2.5D (multi-slice channels), pick the center slice for visualization
    if context_slices is not None and context_slices > 1:
        center_idx = context_slices // 2
        # keep center channel as 1-channel image
        samples = samples[:, center_idx:center_idx+1, :, :]  # (B, 1, H, W)

    # Build filename
    save_path = out_dir / f"samples_epoch_{epoch:03d}.png"

    # Save a grid of images
    save_image(samples, save_path, nrow=nrow)

    print(f"Saved samples to {save_path}")

# ------------------ functions ------------------
def load_best_model_and_sample(
    checkpoint_path: Union[str, Path] = BASE_DIR / "2d_central_ddpm_flair_best.pt",
    num_samples: int = 16,
    out_dir: Union[str, Path] = BASE_DIR / "samples_inference",
    nrow: int = 4,
    mode: str = "2d",              # "2d" or "pseudo3d"
    pseudo3d_num_slices: int = 155,
    pseudo3d_volume_name: str = "brain0",
    pseudo3d_grid_nrow: int = 16,
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
        img_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256,
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        image_size=IMAGE_SIZE,
        channels=1,
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

    out_dir.mkdir(parents=True, exist_ok=True)

    if mode.lower() == "2d":
        # Standard random sampling (uses default z_pos=0.5 inside diffusion.sample)
        sample_and_save(
            diffusion=diffusion,
            epoch=0,
            num_samples=num_samples,
            out_dir=str(out_dir),
            nrow=nrow,
        )
    elif mode.lower() == "pseudo3d":
        # Sweep z_pos to generate a pseudo-3D brain
        generate_pseudo_3d_brain(
            diffusion=diffusion,
            num_slices=pseudo3d_num_slices,
            out_dir=str(out_dir),
            volume_name=pseudo3d_volume_name,
            grid_nrow=pseudo3d_grid_nrow,
        )
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use '2d' or 'pseudo3d'.")


def generate_pseudo_3d_brain(
    diffusion: GaussianDiffusion,
    num_slices: int = 155,
    out_dir: str = "pseudo3d_brains",
    volume_name: str = "brain0",
    grid_nrow: int = 16,
    save_example_slices: bool = True,
) -> torch.Tensor:
    """
    Generate a pseudo-3D brain by sampling 2D slices conditioned on z_pos.

    - Uses z_pos in [0, 1] linearly spaced over `num_slices`.
    - Returns a tensor of shape (num_slices, C, H, W) in [-1, 1].
    - Saves:
        * a big grid PNG with all slices
        * a few selected slices as separate PNGs (optional)
    """
    diffusion.model.eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = diffusion.betas.device  # or next(diffusion.parameters()).device

    # z positions: 0..1 → 155 slices
    # if you only trained on central 80%, you might prefer:
    # z_pos = torch.linspace(0.1, 0.9, steps=num_slices, device=device)
    z_pos = torch.linspace(0.0, 1.0, steps=num_slices, device=device)

    # Sample all slices in one batch: shape (num_slices, C, H, W)
    samples = diffusion.sample(
        batch_size=num_slices,
        z_pos=z_pos,
    ).cpu()

    # Map from [-1, 1] to [0, 1] for saving
    samples = samples.clamp(-1, 1)
    samples_vis = (samples + 1.0) / 2.0

    # ---- 1) Save all slices in a single big grid ----
    grid_path = out_dir / f"{volume_name}_all_slices.png"
    save_image(samples_vis, grid_path, nrow=grid_nrow)
    print(f"[pseudo-3D] Saved all {num_slices} slices grid to {grid_path}")

    # ---- 2) Save a few example slices individually ----
    if save_example_slices:
        # choose some representative indices across the volume
        example_indices = [0, num_slices // 4, num_slices // 2,
                           3 * num_slices // 4, num_slices - 1]
        example_indices = sorted(set(example_indices))

        for idx in example_indices:
            slice_img = samples_vis[idx:idx + 1]  # keep batch dim
            slice_path = out_dir / f"{volume_name}_slice_{idx:03d}.png"
            save_image(slice_img, slice_path, nrow=1)
            print(f"[pseudo-3D] Saved example slice {idx} to {slice_path}")

    return samples  # shape: (num_slices, C, H, W) in [-1, 1]

if __name__ == "__main__":
    # Choose what you want to do here:
    #   mode="2d"       → normal random samples
    #   mode="pseudo3d" → sweep z_pos and build a volume
    load_best_model_and_sample(
        checkpoint_path=BASE_DIR / "models" / "1591447" / "2d_central_ddpm_flair_best.pt",
        # checkpoint_path=BASE_DIR / "2d_central_ddpm_flair_best.pt",
        mode="pseudo3d",     # change to "2d" if you want plain sampling
        pseudo3d_num_slices=155,
        pseudo3d_volume_name="brain7",
    )

"""
brain 4  -> 2d_central_ddpm_flair_best.pt
brain 5  -> 2d_central_ddpm_flair_best_new.pt
brain 6  -> 2d_central_ddpm_flair_best_new.pt neue variante
"""
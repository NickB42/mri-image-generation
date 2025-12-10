from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .dataset import BraTSSliceDataset
from .unet import UNet
from .diffusion import GaussianDiffusion

# ------------------ config (MUST match train.py) ------------------
IMAGE_SIZE = 128
TIMESTEPS = 1000

CENTER_MODALITIES = 4
SLICE_RADIUS = 2
CONTEXT_SLICES = 2 * SLICE_RADIUS      # 4 neighbours: -2, -1, +1, +2
IN_CHANNELS = CENTER_MODALITIES + CENTER_MODALITIES * CONTEXT_SLICES  # 4 + 4*4 = 20
OUT_CHANNELS = CENTER_MODALITIES       # 4

# ---- paths: adjust these to your project layout ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = (PROJECT_ROOT / "../dataset").resolve()

# POINT THIS TO YOUR NEW ALL-MODALITIES CHECKPOINT
CHECKPOINT_PATH = PROJECT_ROOT / "multimodel_25d_ddpm" / "models" / "1591706" / "25d_ddpm_all_modalities_best.pt"
print("Using checkpoint path:", CHECKPOINT_PATH)

# ------------------ device ------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


def load_diffusion_from_checkpoint(checkpoint_path: Path) -> GaussianDiffusion:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Recreate model & diffusion exactly as in training
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

    state_dict = torch.load(checkpoint_path, map_location=device)

    # Handle DataParallel ("model.module.") vs non-DataParallel ("model.")
    if any(k.startswith("model.module.") for k in state_dict.keys()):
        print("Detected DataParallel checkpoint; stripping 'model.module.' prefixes.")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.module."):
                new_k = k.replace("model.module.", "model.", 1)
            else:
                new_k = k
            new_state_dict[new_k] = v
        state_dict = new_state_dict

    diffusion.load_state_dict(state_dict)

    # Small sanity print
    p = next(diffusion.parameters())
    print("Loaded param L2 norm:", p.data.norm().item())

    diffusion.eval()
    return diffusion


@torch.no_grad()
def debug_single_sample(diffusion: GaussianDiffusion):
    """
    1) Take one (x_center, x_context, z_pos) from the dataset.
    2) Generate a sample conditioned on the same context + z_pos.
    3) Save real vs generated FLAIR slices (channel 0) to PNG.
    """
    print("Loading dataset from:", DATASET_ROOT)
    dataset = BraTSSliceDataset(
        DATASET_ROOT,
        image_size=IMAGE_SIZE,
        slice_radius=SLICE_RADIUS,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    x_center, x_context, z_pos = next(iter(loader))
    x_center = x_center.to(device)      # (1, 4, H, W)
    x_context = x_context.to(device)    # (1, 16, H, W)
    z_pos = z_pos.to(device).float()    # (1,)

    print("Shapes:")
    print("  x_center :", x_center.shape)
    print("  x_context:", x_context.shape)
    print("  z_pos    :", z_pos.shape)

    # Sample conditioned on real context + z_pos
    print("Sampling conditioned on real context + z_pos...")
    sample = diffusion.sample(
        batch_size=1,
        z_pos=z_pos,
        context=x_context,
    )  # (1, 4, H, W)

    # Stats to see if it's collapsed
    print(
        "Sample stats:",
        "min", sample.min().item(),
        "max", sample.max().item(),
        "mean", sample.mean().item(),
        "std", sample.std().item(),
    )

    # Map to [0, 1] for saving
    x_center_vis = (x_center.clamp(-1, 1) + 1) / 2.0
    sample_vis   = (sample.clamp(-1, 1) + 1) / 2.0

    out_dir = PROJECT_ROOT / "debug_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Assume channel 0 is FLAIR (change index if your ordering is different)
    flair_idx = 0

    real_path = out_dir / "debug_real_flair.png"
    gen_path  = out_dir / "debug_gen_flair.png"

    save_image(x_center_vis[:, flair_idx:flair_idx+1], real_path)
    save_image(sample_vis[:, flair_idx:flair_idx+1], gen_path)

    print(f"Saved real FLAIR slice to: {real_path}")
    print(f"Saved generated FLAIR slice to: {gen_path}")


def main():
    diffusion = load_diffusion_from_checkpoint(CHECKPOINT_PATH)
    debug_single_sample(diffusion)


if __name__ == "__main__":
    main()
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .dataset import BraTSSliceDataset
from .unet import UNet
from .diffusion import GaussianDiffusion

# ---- match train.py ----
IMAGE_SIZE = 128
TIMESTEPS = 1000

CENTER_MODALITIES = 4
SLICE_RADIUS = 2
CONTEXT_SLICES = 2 * SLICE_RADIUS      # 4 neighbours: -2, -1, +1, +2
IN_CHANNELS = CENTER_MODALITIES + CENTER_MODALITIES * CONTEXT_SLICES  # 4 + 4*4 = 20
OUT_CHANNELS = CENTER_MODALITIES       # 4

MODALITY_NAMES = ["t1", "t1ce", "t2", "flair"]  # adjust if needed

# ---- device ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def load_diffusion_from_checkpoint(checkpoint_path: Path) -> GaussianDiffusion:
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
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)

    # Handle DataParallel vs non-DataParallel
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
    p = next(diffusion.parameters())
    print("Loaded param L2 norm:", p.data.norm().item())

    diffusion.eval()
    return diffusion


def get_subject_indices(dataset: BraTSSliceDataset, subject_idx: int = 0) -> List[int]:
    """
    Group dataset entries by volume path and return all indices belonging
    to the `subject_idx`-th unique volume (sorted by slice index).
    """
    # dataset.slice_tuples is [(flair_path, z), ...]
    all_paths: List[Path] = [p for (p, _) in dataset.slice_tuples]
    unique_paths: List[Path] = sorted(set(all_paths))

    if subject_idx < 0 or subject_idx >= len(unique_paths):
        raise IndexError(
            f"subject_idx {subject_idx} out of range (have {len(unique_paths)} subjects)"
        )

    target_path = unique_paths[subject_idx]
    print(f"Using subject {subject_idx} with FLAIR volume: {target_path}")

    # collect indices with that path
    indices: List[int] = []
    for i, (p, z) in enumerate(dataset.slice_tuples):
        if p == target_path:
            indices.append(i)

    # sort by slice index z
    indices = sorted(indices, key=lambda i: dataset.slice_tuples[i][1])
    print(f"Subject has {len(indices)} usable center slices.")
    return indices


@torch.no_grad()
def generate_volume_for_subject(
    diffusion: GaussianDiffusion,
    dataset_root: Path,
    checkpoint_path: Path,
    subject_idx: int = 0,
    out_dir: Path = Path("pseudo3d_from_dataset"),
    flair_channel: int = 3,
) -> torch.Tensor:
    """
    Generate a pseudo-3D brain for a single subject from the dataset, using
    the real neighbours (x_context) + z_pos as conditioning.

    Returns:
        volume: (num_slices, OUT_CHANNELS, H, W) in [-1, 1]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = BraTSSliceDataset(
        dataset_root,
        image_size=IMAGE_SIZE,
        slice_radius=SLICE_RADIUS,
    )

    indices = get_subject_indices(dataset, subject_idx=subject_idx)
    num_slices = len(indices)

    H = W = IMAGE_SIZE
    volume = torch.zeros(num_slices, OUT_CHANNELS, H, W, device=device)

    for k, ds_idx in enumerate(indices):
        x_center, x_context, z_pos = dataset[ds_idx]

        x_center = x_center.unsqueeze(0).to(device)    # (1, 4, H, W)
        x_context = x_context.unsqueeze(0).to(device)  # (1, 16, H, W)
        z_pos = torch.tensor([z_pos], device=device)   # (1,)

        # conditional sampling
        sample = diffusion.sample(
            batch_size=1,
            z_pos=z_pos,
            context=x_context,
        )  # (1, 4, H, W)

        volume[k] = sample[0]

    # map to [0, 1] for saving
    volume_vis = (volume.clamp(-1, 1) + 1.0) / 2.0  # (S, C, H, W)

    # 1) save per-modality grids of all slices
    for c in range(OUT_CHANNELS):
        mod_name = MODALITY_NAMES[c] if c < len(MODALITY_NAMES) else f"mod{c}"
        mod_vol = volume_vis[:, c:c+1, :, :]  # (S, 1, H, W)
        grid_path = out_dir / f"subject{subject_idx}_all_slices_{mod_name}.png"
        save_image(mod_vol, grid_path, nrow=16)
        print(f"Saved all slices grid for {mod_name} to {grid_path}")

    # 2) save a few example slices for a quick look (e.g., FLAIR)
    example_indices = [0, num_slices // 4, num_slices // 2,
                       3 * num_slices // 4, num_slices - 1]
    example_indices = sorted(set(example_indices))

    for idx in example_indices:
        slice_img = volume_vis[idx:idx + 1, flair_channel:flair_channel+1, :, :]
        slice_path = out_dir / f"subject{subject_idx}_slice_{idx:03d}_flair.png"
        save_image(slice_img, slice_path)
        print(f"Saved example FLAIR slice {idx} to {slice_path}")

    return volume.cpu()  # (S, C, H, W) in [-1, 1]


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATASET_ROOT = (PROJECT_ROOT / "../dataset").resolve()

    CHECKPOINT_PATH = PROJECT_ROOT / "multimodel_25d_ddpm" / "models" / "1591706" / "25d_ddpm_all_modalities_best.pt"
    diffusion = load_diffusion_from_checkpoint(CHECKPOINT_PATH)

    generate_volume_for_subject(
        diffusion=diffusion,
        dataset_root=DATASET_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
        subject_idx=23,
        out_dir=PROJECT_ROOT / "multimodel_25d_ddpm" / "pseudo3d_from_dataset",
    )
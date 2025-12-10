from pathlib import Path
from typing import List

import torch
from torchvision.utils import save_image

from .dataset import BraTSSliceDataset
from .unet import UNet
from .diffusion import GaussianDiffusion

IMAGE_SIZE = 128
TIMESTEPS = 1000

CENTER_MODALITIES = 4
SLICE_RADIUS = 2
CONTEXT_SLICES = 2 * SLICE_RADIUS          # 4 neighbours
IN_CHANNELS = CENTER_MODALITIES + CENTER_MODALITIES * CONTEXT_SLICES  # 4 + 4*4 = 20
OUT_CHANNELS = CENTER_MODALITIES           # 4

MODALITY_NAMES = ["t1", "t1ce", "t2", "flair"]


# -------- device --------
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
    """Same idea as before: map (flair_path, z) → per-subject index list."""
    all_paths = [p for (p, _) in dataset.slice_tuples]
    unique_paths = sorted(set(all_paths))

    if subject_idx < 0 or subject_idx >= len(unique_paths):
        raise IndexError(
            f"subject_idx {subject_idx} out of range (have {len(unique_paths)} subjects)"
        )

    target_path = unique_paths[subject_idx]
    print(f"Using subject {subject_idx} with FLAIR volume: {target_path}")

    indices = [i for i, (p, _) in enumerate(dataset.slice_tuples) if p == target_path]
    indices = sorted(indices, key=lambda i: dataset.slice_tuples[i][1])  # sort by z
    print(f"Subject has {len(indices)} usable center slices.")
    return indices


@torch.no_grad()
def generate_hybrid_volume_for_subject(
    diffusion: GaussianDiffusion,
    dataset_root: Path,
    subject_idx: int = 0,
    out_dir: Path = Path("pseudo3d_hybrid_subject"),
    flair_channel: int = 3,
) -> torch.Tensor:
    """
    Generate a pseudo-3D brain for a single subject, starting from real
    neighbors as context and progressively using generated neighbors for
    past slices.

    Returns:
        volume_gen: (num_slices, OUT_CHANNELS, H, W) in [-1, 1]
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

    # Precompute real centers and z_pos for this subject
    real_centers = []
    z_positions = []
    for ds_idx in indices:
        x_center, x_context, z_pos = dataset[ds_idx]
        real_centers.append(x_center)     # (4, H, W)
        z_positions.append(float(z_pos))  # scalar in [0,1]

    H = W = IMAGE_SIZE
    volume_gen = torch.zeros(num_slices, OUT_CHANNELS, H, W, device=device)
    generated_mask = [False] * num_slices

    # Generate slices in increasing z order (0..num_slices-1)
    for k in range(num_slices):
        # Build context for slice k: 4 neighbours (-2,-1,+1,+2), all modalities.
        context_channels = []

        for dz in range(-SLICE_RADIUS, SLICE_RADIUS + 1):
            if dz == 0:
                continue
            j = k + dz
            if j < 0 or j >= num_slices:
                # This shouldn't happen because dataset already trimmed edges,
                # but just in case, use real_center[k] as a fallback.
                neighbor_slice = real_centers[k]
            else:
                # Use generated neighbor if it exists and is "behind" us.
                if generated_mask[j] and j < k:
                    neighbor_slice = volume_gen[j].detach().cpu()  # (4, H, W)
                else:
                    neighbor_slice = real_centers[j]               # (4, H, W)

            # Mimic dataset ordering: for each dz, then for each modality
            for m in range(CENTER_MODALITIES):
                context_channels.append(neighbor_slice[m:m+1, :, :])  # (1, H, W)

        # Stack to (C_context, H, W) then add batch dim
        x_context = torch.cat(context_channels, dim=0)             # (16, H, W)
        x_context = x_context.unsqueeze(0).to(device)              # (1, 16, H, W)

        # z_pos for this slice
        z_pos = torch.tensor([z_positions[k]], device=device)      # (1,)

        # Sample center slice conditioned on hybrid context + z_pos
        sample = diffusion.sample(
            batch_size=1,
            z_pos=z_pos,
            context=x_context,
        )  # (1, 4, H, W)

        volume_gen[k] = sample[0]
        generated_mask[k] = True

        if (k + 1) % 10 == 0 or (k + 1) == num_slices:
            print(f"Generated slice {k+1}/{num_slices}")

    # Map to [0,1] for saving
    volume_vis = (volume_gen.clamp(-1, 1) + 1.0) / 2.0

    # Save per-modality grids
    for c in range(OUT_CHANNELS):
        mod_name = MODALITY_NAMES[c] if c < len(MODALITY_NAMES) else f"mod{c}"
        mod_vol = volume_vis[:, c:c+1, :, :]  # (S, 1, H, W)
        grid_path = out_dir / f"subject{subject_idx}_hybrid_all_slices_{mod_name}.png"
        save_image(mod_vol, grid_path, nrow=16)
        print(f"Saved hybrid all-slices grid for {mod_name} to {grid_path}")

    # Save a few example FLAIR slices
    example_indices = [0, num_slices // 4, num_slices // 2,
                       3 * num_slices // 4, num_slices - 1]
    example_indices = sorted(set(example_indices))

    for idx in example_indices:
        slice_img = volume_vis[idx:idx + 1, flair_channel:flair_channel+1, :, :]
        slice_path = out_dir / f"subject{subject_idx}_hybrid_slice_{idx:03d}_flair.png"
        save_image(slice_img, slice_path)
        print(f"Saved hybrid FLAIR slice {idx} to {slice_path}")

    return volume_gen.cpu()


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATASET_ROOT = (PROJECT_ROOT / "../dataset").resolve()
    CHECKPOINT_PATH = PROJECT_ROOT / "multimodel_25d_ddpm" / "models" / "1591706" / "25d_ddpm_all_modalities_best.pt"
    print
    diffusion = load_diffusion_from_checkpoint(CHECKPOINT_PATH)

    volume_gen = generate_hybrid_volume_for_subject(
        diffusion=diffusion,
        dataset_root=DATASET_ROOT,
        subject_idx=0,
        out_dir=PROJECT_ROOT / "multimodel_25d_ddpm" / "pseudo3d_from_dataset",
    )

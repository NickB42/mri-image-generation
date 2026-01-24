from pathlib import Path
from typing import List, Union
import time

import numpy as np
import nibabel as nib
from copy import deepcopy

import torch
from torchvision.utils import save_image

from .dataset import BraTSSliceDataset
from .unet import UNet
from .diffusion import GaussianDiffusion

IMAGE_SIZE = 128 # 256
TIMESTEPS = 1000

CENTER_MODALITIES = 4
SLICE_RADIUS = 2
CONTEXT_SLICES = 2 * SLICE_RADIUS
IN_CHANNELS = CENTER_MODALITIES + CENTER_MODALITIES * CONTEXT_SLICES
OUT_CHANNELS = CENTER_MODALITIES

MODALITY_NAMES = ["t1", "t1ce", "t2", "flair"]

# -------- device --------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ----------------------------------------------------------------------
# NIfTI saving helpers
# ----------------------------------------------------------------------

def _get_subject_flair_path(dataset: BraTSSliceDataset, subject_idx: int) -> Path:
    all_paths = [p for (p, _) in dataset.slice_tuples]
    unique_paths = sorted(set(all_paths))
    if subject_idx < 0 or subject_idx >= len(unique_paths):
        raise IndexError(f"subject_idx {subject_idx} out of range (have {len(unique_paths)} subjects)")
    return Path(unique_paths[subject_idx])


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
            out[:, :, :, z] = vol_sc_hw[k]  # (C,H,W) into z
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
    """
    Saves BraTS-style modality volumes:
      subject{idx}_{mod}.nii.gz  (3D each)
    Optionally saves:
      subject{idx}_allmods.nii.gz (4D)

    vol_sc_hw is expected to be (S, C, H, W) in [-1,1] (your model output)
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert to desired numeric range
    if value_range == "0_1":
        vol = (vol_sc_hw.clamp(-1, 1) + 1.0) / 2.0
    elif value_range == "-1_1":
        vol = vol_sc_hw.clamp(-1, 1)
    else:
        raise ValueError("value_range must be '0_1' or '-1_1'")

    vol = vol.float()

    # Load reference nifti if provided (to copy affine/header)
    ref_img = None
    if reference_nifti_path is not None and Path(reference_nifti_path).is_file():
        ref_img = nib.load(str(reference_nifti_path))

    # Decide how to build the 3D arrays
    S, C, H, W = vol.shape

    if ref_img is not None and pad_to_reference_depth and z_indices is not None:
        # Try to pad back to reference depth (Z)
        ref_shape = ref_img.shape
        # Typically (X,Y,Z); sometimes (X,Y,Z,T) but BraTS is usually 3D
        ref_depth = int(ref_shape[2]) if len(ref_shape) >= 3 else S

        full_c_hwz = _tensor_volume_to_full_depth_numpy(vol, z_indices=z_indices, full_depth=ref_depth)  # (C,H,W,Z)
        target_depth = ref_depth
    else:
        # No padding; save only generated slices contiguously
        full_c_hwz = vol.permute(1, 2, 3, 0).detach().cpu().numpy()  # (C,H,W,S) treat S as Z
        target_depth = S

    # Build affine/header
    if ref_img is not None:
        affine = ref_img.affine
        header = deepcopy(ref_img.header)
    else:
        affine = np.eye(4, dtype=np.float32)
        header = nib.Nifti1Header()

    # Our array is (H,W,Z) but NIfTI expects (X,Y,Z) with X=H, Y=W here.
    # If your dataset uses different convention you can swap axes, but this matches your saved slice layout.
    header.set_data_dtype(np.float32)

    # Save each modality as 3D NIfTI
    for c_idx in range(C):
        name = modality_names[c_idx] if c_idx < len(modality_names) else f"mod{c_idx}"
        vol_hwz = full_c_hwz[c_idx]  # (H,W,Z)
        img = nib.Nifti1Image(vol_hwz.astype(np.float32), affine, header=header)
        out_path = out_dir / f"subject{subject_idx:03d}_{name}.nii.gz"
        nib.save(img, str(out_path))
        print(f"Saved NIfTI: {out_path}  shape={vol_hwz.shape}")

    # Optionally save a single 4D file (H,W,Z,C)
    if also_save_4d:
        vol_hwzc = np.stack([full_c_hwz[c] for c in range(C)], axis=-1)  # (H,W,Z,C)
        img4d = nib.Nifti1Image(vol_hwzc.astype(np.float32), affine, header=header)
        out_path_4d = out_dir / f"subject{subject_idx:03d}_allmods.nii.gz"
        nib.save(img4d, str(out_path_4d))
        print(f"Saved 4D NIfTI: {out_path_4d}  shape={vol_hwzc.shape}")


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
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

    # diffusion.load_state_dict(state_dict)

    missing, unexpected = diffusion.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

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


# ----------------------------------------------------------------------
# Option 1: pure dataset context (original show_model_subject behavior)
# ----------------------------------------------------------------------
@torch.no_grad()
def generate_hybrid_volume_for_subject(
    diffusion: GaussianDiffusion,
    dataset_root: Path,
    subject_idx: int = 0,
    out_dir: Path = Path("pseudo3d_hybrid_subject"),
    flair_channel: int = 3,
    save_example_slices: bool = False,
    num_samples: int = 1,
) -> torch.Tensor:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = BraTSSliceDataset(
        dataset_root,
        image_size=IMAGE_SIZE,
        slice_radius=SLICE_RADIUS,
    )

    H = W = IMAGE_SIZE
    last_volume_gen = None

    for sample_idx in range(num_samples):
        subj = subject_idx + sample_idx

        sample_out_dir = out_dir / f"sample_{sample_idx:03d}"
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        indices = get_subject_indices(dataset, subject_idx=subj)
        num_slices = len(indices)

        z_indices = [dataset.slice_tuples[i][1] for i in indices]
        ref_flair_path = _get_subject_flair_path(dataset, subj)

        # Precompute real centers and z_pos for this subject
        real_centers = []
        z_positions = []
        for ds_idx in indices:
            x_center, x_context, z_pos = dataset[ds_idx]
            real_centers.append(x_center)     # (4, H, W)
            z_positions.append(float(z_pos))  # scalar in [0,1]

        volume_gen = torch.zeros(num_slices, OUT_CHANNELS, H, W, device=device)
        generated_mask = [False] * num_slices

        for k in range(num_slices):
            context_channels = []

            for dz in range(-SLICE_RADIUS, SLICE_RADIUS + 1):
                if dz == 0:
                    continue
                j = k + dz
                if j < 0 or j >= num_slices:
                    neighbor_slice = real_centers[k]
                else:
                    if generated_mask[j] and j < k:
                        neighbor_slice = volume_gen[j].detach().cpu()  # (4, H, W)
                    else:
                        neighbor_slice = real_centers[j]

                for m in range(CENTER_MODALITIES):
                    context_channels.append(neighbor_slice[m:m+1, :, :])  # (1, H, W)

            x_context = torch.cat(context_channels, dim=0)        # (16, H, W)
            x_context = x_context.unsqueeze(0).to(device)         # (1, 16, H, W)
            z_pos = torch.tensor([z_positions[k]], device=device) # (1,)

            t0 = time.time()

            # sample = diffusion.sample( # DDPM
            #     batch_size=1,
            #     z_pos=z_pos,
            #     context=x_context,
            # )  # (1, 4, H, W)

            sample = diffusion.sample_ddim(
                batch_size=1,
                z_pos=z_pos,
                context=x_context,
                sample_timesteps=500,
                eta=0.0
            )


            print(f"[subject {subj:03d} | sample {sample_idx:03d}] slice {k+1}/{num_slices} took {time.time()-t0:.1f}s", flush=True)

            volume_gen[k] = sample[0]
            generated_mask[k] = True

            if (k + 1) % 10 == 0 or (k + 1) == num_slices:
                print(f"[subject {subj:03d} | sample {sample_idx:03d}] Generated slice {k+1}/{num_slices}")

        volume_vis = (volume_gen.clamp(-1, 1) + 1.0) / 2.0

        for c in range(OUT_CHANNELS):
            mod_name = MODALITY_NAMES[c] if c < len(MODALITY_NAMES) else f"mod{c}"
            mod_vol = volume_vis[:, c:c+1, :, :]
            grid_path = sample_out_dir / f"subject{subj:03d}_hybrid_all_slices_{mod_name}.png"
            save_image(mod_vol, grid_path, nrow=16)
            print(f"[subject {subj:03d} | sample {sample_idx:03d}] Saved grid {mod_name} -> {grid_path}")

        save_brats_like_nifti(
            vol_sc_hw=volume_gen,
            out_dir=sample_out_dir,
            subject_idx=subj,
            reference_nifti_path=ref_flair_path,
            z_indices=z_indices,
            pad_to_reference_depth=True,
            value_range="0_1",
            also_save_4d=True,
        )

        if save_example_slices:
            example_indices = sorted(set([
                0,
                num_slices // 4,
                num_slices // 2,
                3 * num_slices // 4,
                num_slices - 1,
            ]))
            for idx in example_indices:
                slice_img = volume_vis[idx:idx + 1, flair_channel:flair_channel+1, :, :]
                slice_path = sample_out_dir / f"subject{subj:03d}_hybrid_slice_{idx:03d}_flair.png"
                save_image(slice_img, slice_path)
                print(f"[subject {subj:03d} | sample {sample_idx:03d}] Saved example slice {idx} -> {slice_path}")

        last_volume_gen = volume_gen.detach().cpu()

    return last_volume_gen


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATASET_ROOT = (PROJECT_ROOT / "../datasets/train").resolve()
    EXPERIMENT_NAME = "ddpm_25d_all_modalities" 
    MODEL_ID = "1591706" # old model
    # MODEL_ID = "1602667" 
    
    CHECKPOINT_PATH = (
        PROJECT_ROOT
        / EXPERIMENT_NAME
        / "models"
        / MODEL_ID
        / "25d_ddpm_all_modalities_best.pt"
        # / "2d_central_ddpm_flair_best.pt"
    )

    subject_idx = 10

    OUT_DIR = (
        PROJECT_ROOT / EXPERIMENT_NAME / "pseudo3d_from_dataset"
    )


    diffusion = load_diffusion_from_checkpoint(CHECKPOINT_PATH)
    
    # generate_volume_for_subject(
    #     diffusion=diffusion,
    #     dataset_root=DATASET_ROOT,
    #     subject_idx=subject_idx,
    #     out_dir=OUT_DIR,
    # )

    generate_hybrid_volume_for_subject(
        diffusion=diffusion,
        dataset_root=DATASET_ROOT,
        subject_idx=subject_idx,
        out_dir=OUT_DIR,
        num_samples=5
    )
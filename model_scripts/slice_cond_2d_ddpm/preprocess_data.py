import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def preprocess_volume(volume_path: Path, output_path: Path, image_size: int, modality_suffix: str):
    """
    - Finds all *flair.nii.gz files
    - Uses central 80% slices from each volume
    - Returns normalized 2D slice as tensor in [-1, 1], shape (1, H, W)
    - Also returns normalized slice index in [0, 1]
    """
    print(f"[INFO] Processing volume: {volume_path}")

    img = nib.load(str(volume_path))
    vol = np.asanyarray(img.dataobj).astype(np.float32)  # (H, W, D)

    if vol.ndim != 3:
        print(f"[WARN] Skipping {volume_path}, not 3D (shape={vol.shape})")
        return

    H, W, D = vol.shape
    z_start = int(0.1 * D)
    z_end   = int(0.9 * D)

    slices = []
    z_pos_list = []

    for z in range(z_start, z_end):
        slice_2d = vol[:, :, z].copy()

        # --- SAME NORMALIZATION AS IN BraTSSliceDataset ---
        mask = slice_2d != 0
        if np.any(mask):
            mean = slice_2d[mask].mean()
            std = slice_2d[mask].std()
            std = std if std > 0 else 1.0
            slice_2d[mask] = (slice_2d[mask] - mean) / std

        # Clip to [-5, 5] and rescale to [0, 1]
        slice_2d = np.clip(slice_2d, -5, 5)
        slice_2d = (slice_2d + 5) / 10.0

        # To tensor and resize to (image_size, image_size)
        slice_t = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        slice_t = F.interpolate(
            slice_t,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        slice_t = slice_t.squeeze(0)  # (1, H, W)

        # Map [0, 1] -> [-1, 1]
        slice_t = slice_t * 2.0 - 1.0

        # z normalization:
        z_pos = np.float32(z / (D - 1))

        slices.append(slice_t)
        z_pos_list.append(z_pos)

    if not slices:
        print(f"[WARN] No valid slices for {volume_path}")
        return

    slices = torch.stack(slices, dim=0)  # (num_slices, 1, image_size, image_size)
    z_pos = torch.tensor(z_pos_list, dtype=torch.float32)  # (num_slices,)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"slices": slices, "z_pos": z_pos}, output_path)

    print(
        f"[OK] Saved {slices.shape[0]} slices for {volume_path} "
        f"to {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory of original BraTS data (where *_flair.nii.gz lives).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store preprocessed .pt files.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Target image size (image_size x image_size).",
    )
    parser.add_argument(
        "--modality_suffix",
        type=str,
        default="_flair.nii.gz",
        help="Suffix to match FLAIR volumes.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print(f"[INFO] Scanning for volumes under: {root_dir}")
    volume_paths = sorted(root_dir.rglob(f"*{args.modality_suffix}"))

    if not volume_paths:
        raise RuntimeError(f"No volumes matching *{args.modality_suffix} found under {root_dir}")

    print(f"[INFO] Found {len(volume_paths)} volumes.")

    for i, vol_path in enumerate(volume_paths, start=1):
        # Keep folder structure, just change root and extension:
        # e.g. root/.../xxx_flair.nii.gz -> output/.../xxx_flair.pt
        rel = vol_path.relative_to(root_dir)
        # Remove last suffix (.gz), then replace .nii with .pt
        out_rel = rel.with_suffix("")      # drop .gz
        out_rel = out_rel.with_suffix(".pt")  # replace .nii -> .pt
        out_path = output_dir / out_rel

        print(f"[{i}/{len(volume_paths)}] -> {out_path}")
        preprocess_volume(vol_path, out_path, args.image_size, args.modality_suffix)


if __name__ == "__main__":
    main()
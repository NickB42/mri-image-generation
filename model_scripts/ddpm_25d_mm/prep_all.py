"""
prep_all.py — Preprocess BraTS NIfTI volumes into numpy memmap files.

Adapted for the workspace layout:
  - Raw data lives in datasets/train/ and datasets/val/ (each containing
    BraTS2021_XXXXX/<subject>_flair.nii.gz etc.)
  - Outputs separate memmap files for train and val splits.

Usage (from project root):
  # Train set (FLAIR only, 1001 subjects)
  python -m model_scripts.ddpm_25d_mm.prep_all \
      --root_dir datasets/train \
      --output_file datasets/memmap/train_flair.npy

  # Val set (FLAIR only, 125 subjects)
  python -m model_scripts.ddpm_25d_mm.prep_all \
      --root_dir datasets/val \
      --output_file datasets/memmap/val_flair.npy

  # Debug (first 10 volumes only)
  python -m model_scripts.ddpm_25d_mm.prep_all \
      --root_dir datasets/train \
      --output_file datasets/memmap/train_flair_debug.npy \
      --debug
"""

import torchio as tio
import argparse
from pathlib import Path
import time

import numpy as np

NUM_SLICES = 155


def preprocess_volume(volume_path: Path, image_size: int, masks: bool = False):
    """
    Preprocess a single NIfTI volume:
      - Crop/pad to (168, 224, NUM_SLICES)
      - Resize to (image_size, image_size, NUM_SLICES)
      - Normalize to [-1, 1] (or binarize for masks)

    Returns tensor of shape (NUM_SLICES, 1, image_size, image_size).
    """
    print(f"[INFO] Processing volume: {volume_path}")

    img = tio.ScalarImage(volume_path)

    if masks:
        data = img.data
        data = (data > 10).float()
        img = tio.ScalarImage(tensor=data, affine=img.affine)

    crop = tio.CropOrPad((168, 224, NUM_SLICES))
    resize = tio.Resize((image_size, image_size, NUM_SLICES))

    if not masks:
        normalize = tio.RescaleIntensity((-1, 1))
        img = resize(normalize(crop(img)))
    else:
        img = resize(crop(img))

    vol = img.data  # (1, H, W, D)
    slices = vol.permute(3, 0, 1, 2)  # (D, 1, H, W)
    return slices


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BraTS volumes into a numpy memmap file."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing subject folders (e.g. datasets/train).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path for the output memmap .npy file (e.g. datasets/memmap/train_flair.npy).",
    )
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Generate binary masks instead of intensity images.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Target spatial size (image_size x image_size). Default: 128.",
    )
    parser.add_argument(
        "--modality_suffix",
        type=str,
        default="_flair.nii.gz",
        help="Suffix to match volumes. Default: _flair.nii.gz",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Only process the first 10 volumes (for quick testing).",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    output_file = Path(args.output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Scanning for volumes under: {root_dir}")
    t_start = time.time()
    volume_paths = sorted(root_dir.rglob(f"*{args.modality_suffix}"))
    t_scan = time.time()
    print(f"Scanning took {t_scan - t_start:.1f} seconds.")

    if not volume_paths:
        raise RuntimeError(
            f"No volumes matching *{args.modality_suffix} found under {root_dir}"
        )

    print(f"[INFO] Found {len(volume_paths)} volumes.")

    if args.debug:
        volume_paths = volume_paths[:10]
        print(f"[DEBUG] Truncated to {len(volume_paths)} volumes.")

    image_size = args.image_size
    N = len(volume_paths)
    D = NUM_SLICES
    H = W = image_size

    # Create memmap
    mm = np.memmap(
        output_file, mode="w+", dtype=np.float32, shape=(N, D, 1, H, W)
    )
    print(f"[INFO] Created memmap: shape={mm.shape}, file={output_file}")

    t_proc = time.time()
    for i, vol_path in enumerate(volume_paths):
        print(f"[{i + 1}/{N}] {vol_path.name}")
        slices = preprocess_volume(vol_path, image_size, args.masks)  # (D, 1, H, W)
        mm[i] = slices.numpy()
        t_now = time.time()
        print(f"  took {t_now - t_proc:.1f}s")
        t_proc = t_now

    mm.flush()
    print(f"\n[DONE] Total: {time.time() - t_start:.1f}s")
    print(f"  Output: {output_file}")
    print(f"  Shape:  {mm.shape}")


if __name__ == "__main__":
    main()

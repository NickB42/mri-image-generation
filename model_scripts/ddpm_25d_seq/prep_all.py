import torchio as tio
import argparse
from pathlib import Path
import time

import numpy as np
import torch.nn.functional as F

NUM_SLICES = 155

def preprocess_volume(volume_path: Path, image_size: int, masks: bool=False):
    """
    - Finds all *flair.nii.gz files
    - Uses central 80% slices from each volume
    - Returns normalized 2D slice as tensor in [-1, 1], shape (1, H, W)
    - Also returns normalized slice index in [0, 1]
    """
    print(f"[INFO] Processing volume: {volume_path}")

    if masks:
        img = tio.ScalarImage(volume_path).data
        img = (img > 10) * 1
        crop = tio.CropOrPad((168,224,NUM_SLICES))
        resize = tio.Resize((128, 128, NUM_SLICES))
        img = resize(crop(img))
        vol = img.data
    else:
        img = tio.ScalarImage(volume_path).data
        normalize = tio.RescaleIntensity((-1, 1))
        crop = tio.CropOrPad((168,224,NUM_SLICES))
        resize = tio.Resize((128, 128, NUM_SLICES))
        img = resize(normalize(crop(img)))
        vol = img.data

    slices = vol.permute(3, 0, 1, 2) # [1, H, W, D] -> [D, 1, H, W]

    return slices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory of original BraTS data (where *_flair.nii.gz lives).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Directory to store preprocessed .pt files.",
    )
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Whether to generate masks or slices",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    output_file = Path(args.output_file).resolve()

    print(f"[INFO] Scanning for volumes under: {root_dir}")
    t1 = time.time()
    volume_paths = sorted(root_dir.rglob(f"*{args.modality_suffix}"))
    t2 = time.time()
    print(f"Scanning took {t2 - t1} seconds.")

    if not volume_paths:
        raise RuntimeError(f"No volumes matching *{args.modality_suffix} found under {root_dir}")

    print(f"[INFO] Found {len(volume_paths)} volumes.")

    if args.debug:
        volume_paths = volume_paths[:10]

    image_size = args.image_size

    t3 = time.time()
    N, D, H, W = len(volume_paths), NUM_SLICES, image_size, image_size
    mm = np.memmap(output_file, mode="w+", dtype=np.float32, shape=(N, D, 1, H, W))
    print(f"{mm.shape=}")
    for i, vol_path in enumerate(volume_paths, start=1):
        print(f"[{i}/{len(volume_paths)}] -> {output_file}")
        slices = preprocess_volume(vol_path, args.image_size, args.masks) # [D, 1, H, W]
        slice_np = slices.numpy()
        print(f"{slice_np.shape=}")
        mm[i-1] = slices.numpy()
        t4 = time.time()
        print(f"Processing took {t4 - t3} seconds.")
        t3 = t4

    print(f"Total: {time.time() - t1} seconds.")
    mm.flush()
    print(f"{mm.shape=}")


if __name__ == "__main__":
    main()
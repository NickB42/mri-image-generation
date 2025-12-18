"""
Split a dataset of subject subfolders into train/val/test (default 80/10/10).

- Assumes `--src` contains N immediate subdirectories (one per subject).
- Creates `--out-root/train`, `--out-root/val`, `--out-root/test`
- Writes manifests to `--out-root/splits/{train,val,test}.txt`

Examples:
  # Preview
  python split_train_val_test.py --src ./dataset --out-root ./data --seed 42 --dry-run

  # Actually move
  python split_train_val_test.py --src ./dataset --out-root ./data --seed 42 --mode move

  # Use symlinks instead of moving (recommended if you want to keep raw intact)
  python split_train_val_test.py --src ./dataset --out-root ./data --seed 42 --mode symlink
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple


def list_subject_dirs(src: Path) -> List[Path]:
    dirs = [p for p in src.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return sorted(dirs, key=lambda p: p.name)  # stable ordering before shuffle


def is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def split_indices(n: int, train_frac: float, val_frac: float, test_frac: float) -> Tuple[int, int, int]:
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-9:
        raise ValueError("Fractions must sum to 1.0")

    # floor for val/test, remainder to train so sums exactly to n
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test

    # Ensure non-empty splits if possible (for small datasets)
    if n >= 3:
        if n_train == 0: n_train = 1
        if n_val == 0: n_val = 1
        if n_test == 0: n_test = 1
        # re-balance to sum to n
        total = n_train + n_val + n_test
        if total != n:
            n_train += (n - total)

    return n_train, n_val, n_test


def place(src_dir: Path, dst_dir: Path, mode: str, dry_run: bool) -> None:
    if dst_dir.exists():
        raise FileExistsError(f"Destination already exists: {dst_dir}")

    if dry_run:
        print(f"[DRY-RUN] {mode}: {src_dir} -> {dst_dir}")
        return

    if mode == "move":
        shutil.move(str(src_dir), str(dst_dir))
    elif mode == "copy":
        shutil.copytree(src_dir, dst_dir)
    elif mode == "symlink":
        os.symlink(src_dir.resolve(), dst_dir, target_is_directory=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def write_manifest(path: Path, names: List[str], dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY-RUN] Would write manifest: {path} ({len(names)} ids)")
        return
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="Folder with all subject subfolders (e.g. ./dataset)")
    ap.add_argument("--out-root", type=Path, required=True, help="Output root (creates train/val/test inside)")
    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed (reproducible split)")
    ap.add_argument("--mode", choices=["move", "copy", "symlink"], default="move",
                    help="How to create split dirs (default: move)")
    ap.add_argument("--dry-run", action="store_true", help="Print actions only, do not modify files")
    args = ap.parse_args()

    src = args.src
    out_root = args.out_root

    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"--src does not exist or is not a directory: {src}")

    # Safety: don't let out_root be inside src
    if is_subpath(out_root, src):
        raise ValueError(f"--out-root ({out_root}) must NOT be inside --src ({src}). Choose a sibling folder.")

    subjects = list_subject_dirs(src)
    n_total = len(subjects)
    if n_total == 0:
        raise RuntimeError(f"No subject folders found in {src}")

    n_train, n_val, n_test = split_indices(
        n_total, args.train_frac, args.val_frac, args.test_frac
    )

    rng = random.Random(args.seed)
    rng.shuffle(subjects)

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:n_train + n_val + n_test]

    train_dir = out_root / "train"
    val_dir = out_root / "val"
    test_dir = out_root / "test"

    if args.dry_run:
        print(f"[DRY-RUN] Would create: {train_dir}, {val_dir}, {test_dir}")
    else:
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {n_total} subject folders in {src}")
    print(f"Split: train={len(train_subjects)} val={len(val_subjects)} test={len(test_subjects)} "
          f"(seed={args.seed}, mode={args.mode})")

    splits_dir = out_root / "splits"
    if not args.dry_run:
        splits_dir.mkdir(parents=True, exist_ok=True)

    write_manifest(splits_dir / "train.txt", [p.name for p in train_subjects], args.dry_run)
    write_manifest(splits_dir / "val.txt",   [p.name for p in val_subjects],   args.dry_run)
    write_manifest(splits_dir / "test.txt",  [p.name for p in test_subjects],  args.dry_run)

    for p in train_subjects:
        place(p, train_dir / p.name, args.mode, args.dry_run)
    for p in val_subjects:
        place(p, val_dir / p.name, args.mode, args.dry_run)
    for p in test_subjects:
        place(p, test_dir / p.name, args.mode, args.dry_run)

    print("Done.")
    if not args.dry_run:
        print(f"Train: {train_dir}")
        print(f"Val  : {val_dir}")
        print(f"Test : {test_dir}")
        print(f"Manifests: {splits_dir/'train.txt'}, {splits_dir/'val.txt'}, {splits_dir/'test.txt'}")


if __name__ == "__main__":
    main()
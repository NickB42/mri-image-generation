import os
from pathlib import Path
import math
import time

import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt

from 2d_central_diffusion.model import UNet, GaussianDiffusion
from 25d_ddpm.dataset import BraTSSliceDataset

DATASET_ROOT = Path("/dataset")

IMAGE_SIZE = 128
BATCH_SIZE = 8
NUM_WORKERS = 0
NUM_EPOCHS = 20
TIMESTEPS = 1000
LEARNING_RATE = 2e-4
CONTEXT_SLICES = 5

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


#dataset loading
full_dataset = BraTSSliceDataset(DATASET_ROOT, image_size=IMAGE_SIZE)

train_size = int(0.9 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

print(f"Train slices: {len(train_dataset)}, Val slices: {len(val_dataset)}")

CONTEXT_SLICES = 3  # or 5, 7, ...

model = UNet(
    img_channels=CONTEXT_SLICES,
    base_channels=64,
    channel_mults=(1, 2, 4, 8),
    time_emb_dim=256,
)

diffusion = GaussianDiffusion(
    model=model,
    image_size=IMAGE_SIZE,
    channels=CONTEXT_SLICES,
    timesteps=TIMESTEPS,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop

def train_one_epoch(epoch):
    diffusion.train()
    running_loss = 0.0
    n_steps = 0

    for step, x in enumerate(train_loader, start=1):
        x = x.to(device)

        t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device).long()

        loss = diffusion.p_losses(x, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_steps += 1
        if step % 500 == 0:
            avg = running_loss / n_steps
            print(f"[epoch {epoch} | step {step}] avg loss: {avg:.4f}")

    avg_loss = running_loss / max(1, n_steps)
    print(f"Epoch {epoch} | Train loss: {avg_loss:.4f}")
    return avg_loss

@torch.no_grad()
def validate(epoch):
    diffusion.eval()
    running_loss = 0.0
    n_steps = 0

    for x in val_loader:
        x = x.to(device)
        t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device).long()
        loss = diffusion.p_losses(x, t)
        running_loss += loss.item()
        n_steps += 1

    avg_loss = running_loss / max(1, n_steps)
    print(f"Epoch {epoch} | Val loss:   {avg_loss:.4f}")
    return avg_loss

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


if __name__ == "__main__":
    print("Starting Training")
    best_val = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_one_epoch(epoch)
    val_loss   = validate(epoch)

    # check for improvement
    if val_loss < best_val - MIN_DELTA:
        best_val = val_loss
        epochs_without_improvement = 0
        torch.save(diffusion.state_dict(), "25d_central_ddpm_flair_best.pt")
        print(f"✅ New best val loss: {best_val:.4f}")
    else:
        epochs_without_improvement += 1
        print(f"⚠️ No improvement for {epochs_without_improvement} epoch(s)")

    if epoch % 5 == 0:
        sample_and_save(
            diffusion,
            epoch=epoch,
            num_samples=16,
            out_dir="samples",
            context_slices=CONTEXT_SLICES
        )

    # early stopping condition
    if epochs_without_improvement >= PATIENCE:
        print(f"⏹ Early stopping at epoch {epoch} "
              f"(no val improvement for {PATIENCE} epochs).")
        break
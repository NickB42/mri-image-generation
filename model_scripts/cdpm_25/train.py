# train_cdpm25d.py
from __future__ import annotations

import argparse
import os
from itertools import cycle

import torch
from torch.utils.data import DataLoader, random_split

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # fallback

from .brats_dataset import BraTSVolumeDataset
from .cdpm25d import CDPM25D, GaussianDiffusion, train_step, generate_volume_staged


def save_checkpoint(path: str, model: torch.nn.Module, opt: torch.optim.Optimizer, step: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
        path,
    )


def load_checkpoint(path: str, model: torch.nn.Module, opt: torch.optim.Optimizer | None = None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if opt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    return int(ckpt.get("step", 0))


@torch.no_grad()
def quick_sample(model, diffusion, device, out_path: str, D: int, H: int, W: int, stage_size: int):
    model.eval()
    vol = generate_volume_staged(
        model=model,
        diffusion=diffusion,
        D=D,
        H=H,
        W=W,
        stage_size=stage_size,
        device=device,
    )  # [1,D,H,W]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(vol.cpu(), out_path)
    print(f"[sample] saved torch volume to: {out_path}  (shape={tuple(vol.shape)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="/dataset")
    p.add_argument("--modality", type=str, default="t1", choices=["t1", "t1ce", "t2", "flair"])
    p.add_argument("--target_shape", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--no_resize", action="store_true")
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tau_max", type=int, default=20)
    p.add_argument("--diffusion_T", type=int, default=1000)

    p.add_argument("--save_dir", type=str, default="runs/cdpm25d")
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=100)

    p.add_argument("--resume", type=str, default="")
    p.add_argument("--amp", action="store_true")

    p.add_argument("--sample_every", type=int, default=0)
    p.add_argument("--stage_size", type=int, default=10)

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    ds = BraTSVolumeDataset(
        data_root=args.data_root,
        modality=args.modality,
        target_shape=tuple(args.target_shape),
        resize=(not args.no_resize),
    )

    # Simple split
    val_size = max(1, int(0.05 * len(ds)))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Figure out D,H,W after preprocessing
    sample_vol, _ = ds[0]
    D, H, W = sample_vol.shape

    model = CDPM25D(
        tau_max=args.tau_max,
        volume_depth=D,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        attn_heads=4,
    ).to(device)

    diffusion = GaussianDiffusion(T=args.diffusion_T).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, opt)
        print(f"[resume] loaded {args.resume} at step={start_step}")

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    train_iter = cycle(train_loader)
    running = 0.0

    for step in tqdm(range(start_step, args.steps), total=args.steps - start_step):
        model.train()
        vols, _ = next(train_iter)  # vols: [B,D,H,W]
        vols = vols.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            loss = train_step(model=model, diffusion=diffusion, volumes=vols)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        running += float(loss.detach().cpu().item())

        if (step + 1) % args.log_every == 0:
            avg = running / args.log_every
            running = 0.0
            print(f"step={step+1}  loss={avg:.6f}")

        if (step + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"ckpt_step_{step+1}.pt")
            save_checkpoint(ckpt_path, model, opt, step + 1)
            print(f"[ckpt] saved: {ckpt_path}")

        if args.sample_every and (step + 1) % args.sample_every == 0:
            out_path = os.path.join(args.save_dir, f"sample_step_{step+1}.pt")
            quick_sample(model, diffusion, device, out_path, D=D, H=H, W=W, stage_size=args.stage_size)

    final_path = os.path.join(args.save_dir, "ckpt_final.pt")
    save_checkpoint(final_path, model, opt, args.steps)
    print(f"[done] saved: {final_path}")


if __name__ == "__main__":
    main()

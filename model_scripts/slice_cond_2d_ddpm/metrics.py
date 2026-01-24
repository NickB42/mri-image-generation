"""
Evaluation script for your slice-position-conditioned DDPM on BraTS NIfTI volumes.

Computes:
  - FID, KID, PRC (precision/recall) using torch-fidelity on folders of PNGs
  - Diversity on generated samples: MS-SSIM + LPIPS (random pairs)
  - Inter-slice consistency: adjacent-slice SSIM + L1 (real vs synthetic windows)

Updates in this version:
  - No tqdm (Slurm-friendly)
  - Mixed precision (Accelerate)
  - Step 5 (ISC) uses all GPUs (distributed sharding + reduce)
  - Save results after step 4 and after step 5
"""

from __future__ import annotations

import json
import os
import random
from glob import glob
from typing import Any, Dict, List, Tuple
from pathlib import Path
from datetime import timedelta

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import nibabel as nib
import torch_fidelity

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import InitProcessGroupKwargs
from accelerate.utils import broadcast_object_list

from .unet import UNet
from .diffusion import GaussianDiffusion


# ============================================================
# CONFIG CONSTANTS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = (PROJECT_ROOT / "slice_cond_2d_ddpm").resolve()
BRATS_ROOT = (PROJECT_ROOT / "../datasets/test").resolve()
MODALITY = "flair"
CHECKPOINT = str(EXPERIMENTS_ROOT / "models" / "1591624" / "2d_central_ddpm_flair_best.pt")
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID")
OUT_DIR = f"./eval_out/slice/{SLURM_JOB_ID}"

# Model / diffusion
IMAGE_SIZE = 128
TIMESTEPS = 1000

# UNet args
IMG_CHANNELS = 1
BASE_CHANNELS = 64
CHANNEL_MULTS = (1, 2, 4, 8)
TIME_EMB_DIM = 256

# sampling distribution for z_pos
Z_MIN = 0.0
Z_MAX = 1.0

# counts
NUM_IMAGES = 4000
BATCH_SIZE = 32

# normalization percentiles
P_LO = 0.5
P_HI = 99.5

# diversity
DIV_PAIRS = 2000

# inter-slice consistency
ISC_NUM_SEQUENCES = 50
ISC_WINDOW = 16

# misc
SEED = 123
DEVICE = "auto"

# mixed precision (Accelerate): "fp16", "bf16", or "no"
MIXED_PRECISION = "fp16"

# logging (Slurm-friendly: low volume)
ISC_LOG_EVERY_LOCAL = 2  # each rank prints every N local sequences in step 5

# debug
DEBUG = False
DEBUG_ONLY = False
SMOKE_FOLDER_METRICS = True
SMOKE_N = 64
DEBUG_CAP_NUM_IMAGES = 256


# ----------------------------
# Helpers
# ----------------------------
def chunked(seq, n: int):
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def save_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def log_main(accelerator: Accelerator, msg: str) -> None:
    if accelerator.is_main_process:
        print(msg, flush=True)


# ----------------------------
# BraTS helpers
# ----------------------------
def find_subject_dirs(brats_root: str) -> List[str]:
    subs = [p for p in glob(os.path.join(brats_root, "*")) if os.path.isdir(p)]
    subs.sort()
    if not subs:
        raise FileNotFoundError(f"No subject directories under {brats_root}")
    return subs


def find_modality_file(subject_dir: str, modality: str) -> str:
    modality = modality.lower()
    patterns = [f"*_{modality}.nii*", f"*_{modality.upper()}.nii*"]
    if modality in ("t1ce", "t1c"):
        patterns += ["*_t1ce.nii*", "*_t1c.nii*", "*_T1CE.nii*", "*_T1C.nii*"]

    candidates: List[str] = []
    for pat in patterns:
        candidates += glob(os.path.join(subject_dir, pat))
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(f"No '{modality}' NIfTI found in {subject_dir}")
    return candidates[0]


def load_nifti(path: str) -> np.ndarray:
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got {data.shape} at {path}")
    return data  # H,W,Z


def percentile_lohi_nonzero(x: np.ndarray, p_lo: float, p_hi: float) -> Tuple[float, float]:
    x = x.astype(np.float32)
    nz = x[x != 0]
    if nz.size < 500:
        nz = x.reshape(-1)
    lo, hi = np.percentile(nz, [p_lo, p_hi])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def to_u8(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.clip(x.astype(np.float32), lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return (x * 255.0).round().astype(np.uint8)


def resize_u8_square(u8_hw: np.ndarray, size: int) -> np.ndarray:
    im = Image.fromarray(u8_hw)
    im = im.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def save_gray_as_rgb_png(u8_hw: np.ndarray, out_path: str) -> None:
    rgb = np.stack([u8_hw, u8_hw, u8_hw], axis=-1)
    Image.fromarray(rgb).save(out_path)


def make_grid(image_paths: List[str], nrow: int, out_path: str, pad: int = 4) -> None:
    ims = [Image.open(p).convert("RGB") for p in image_paths]
    if not ims:
        return
    w, h = ims[0].size
    n = len(ims)
    ncol = nrow
    nrow_grid = int(np.ceil(n / ncol))
    canvas = Image.new(
        "RGB",
        (ncol * w + (ncol + 1) * pad, nrow_grid * h + (nrow_grid + 1) * pad),
        color=(20, 20, 20),
    )
    for idx, im in enumerate(ims):
        r = idx // ncol
        c = idx % ncol
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        canvas.paste(im, (x, y))
    canvas.save(out_path)


# ----------------------------
# Model build + checkpoint load
# ----------------------------
def build_diffusion(
    image_size: int,
    timesteps: int,
    device: torch.device,
    img_channels: int = 1,
    base_channels: int = 64,
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
    time_emb_dim: int = 256,
) -> torch.nn.Module:
    model = UNet(
        img_channels=img_channels,
        base_channels=base_channels,
        channel_mults=channel_mults,
        time_emb_dim=time_emb_dim,
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        image_size=image_size,
        channels=img_channels,
        timesteps=timesteps,
    ).to(device)

    return diffusion


def load_checkpoint_like_yours(diffusion: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device)

    if isinstance(state_dict, dict) and any(k.startswith("model.module.") for k in state_dict.keys()):
        print("Detected DataParallel-style keys 'model.module.*' -> rewriting to 'model.*'")
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("model.module."):
                new_state[k.replace("model.module.", "model.", 1)] = v
            else:
                new_state[k] = v
        state_dict = new_state

    missing, unexpected = diffusion.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    print(f"Loaded checkpoint: {checkpoint_path}")


@torch.inference_mode()
def diffusion_sample_01(diffusion: torch.nn.Module, z_pos: torch.Tensor) -> torch.Tensor:
    """
    Returns samples in [0,1], shape [B,1,H,W].
    Your diffusion.sample returns [-1,1].
    """
    out = diffusion.sample(batch_size=int(z_pos.shape[0]), z_pos=z_pos)
    if isinstance(out, (tuple, list)):
        out = out[0]
    x = torch.as_tensor(out, device=z_pos.device)

    if x.ndim == 3:
        x = x.unsqueeze(1)
    if x.shape[1] != 1:
        x = x[:, :1]

    x = x.clamp(-1, 1)
    x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


# ----------------------------
# Metrics
# ----------------------------
def torch_fidelity_folder_metrics(
    real_dir: str,
    fake_dir: str,
    use_cuda: bool,
    kid_subset_size: int = 1000,
    kid_subsets: int = 100,
    num_workers: int = 2,
    batch_size: int = 16,
) -> Dict[str, float]:
    n1 = len(glob(os.path.join(real_dir, "*.png")))
    n2 = len(glob(os.path.join(fake_dir, "*.png")))
    n = min(n1, n2)
    kid_subset_size = min(kid_subset_size, n)
    kid_flag = (n >= 2) and (kid_subset_size >= 2)

    m = torch_fidelity.calculate_metrics(
        input1=fake_dir,
        input2=real_dir,
        cuda=bool(use_cuda),
        fid=True,
        kid=kid_flag,
        prc=True,
        verbose=False,
        kid_subset_size=kid_subset_size,
        kid_subsets=kid_subsets,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    out: Dict[str, float] = {}
    for k, v in m.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    out["kid_subset_size_used"] = int(kid_subset_size)
    out["kid_enabled"] = bool(kid_flag)
    out["n_real"] = int(n1)
    out["n_fake"] = int(n2)
    return out


@torch.inference_mode()
def diversity_metrics(fake_dir: str, device: torch.device, num_pairs: int, resize_to: int = 256) -> Dict[str, float]:
    paths = sorted(glob(os.path.join(fake_dir, "*.png")))
    if len(paths) < 2:
        return {"ms_ssim_pair_mean": float("nan"), "lpips_pair_mean": float("nan")}

    max_load = min(len(paths), max(300, int(np.sqrt(len(paths))) * 40))
    paths = random.sample(paths, k=max_load)

    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        arr = np.asarray(im).astype(np.float32) / 255.0
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
    x = torch.stack(imgs, dim=0).to(device)
    if resize_to:
        x = F.interpolate(x, size=(resize_to, resize_to), mode="bilinear", align_corners=False)

    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, sync_on_compute=False).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True, sync_on_compute=False).to(device)

    n = x.shape[0]
    pairs = [tuple(random.sample(range(n), 2)) for _ in range(num_pairs)]
    ms_vals, lp_vals = [], []
    for i, j in pairs:
        a, b = x[i : i + 1], x[j : j + 1]
        ms_vals.append(ms_ssim(a, b).item())
        lp_vals.append(lpips(a, b).item())

    return {
        "ms_ssim_pair_mean": float(np.mean(ms_vals)),
        "ms_ssim_pair_std": float(np.std(ms_vals)),
        "lpips_pair_mean": float(np.mean(lp_vals)),
        "lpips_pair_std": float(np.std(lp_vals)),
        "diversity_num_loaded": int(n),
        "diversity_num_pairs": int(num_pairs),
    }


@torch.inference_mode()
def inter_slice_consistency_distributed(
    accelerator: Accelerator,
    diffusion: torch.nn.Module,
    subject_dirs: List[str],
    modality: str,
    device: torch.device,
    image_size: int,
    num_sequences: int,
    window: int,
    p_lo: float,
    p_hi: float,
    z_min: float,
    z_max: float,
    log_every_local: int = 2,
) -> Dict[str, float]:
    """
    Distributed ISC:
      - Main process selects 'chosen' subjects and broadcasts to all ranks
      - Each rank processes chosen[rank::world_size]
      - One reduce at the end to combine sum/sumsq/count for each metric
    """
    # Pick sequences on main and broadcast so all ranks agree on the same set
    if accelerator.is_main_process:
        chosen = random.sample(subject_dirs, k=min(num_sequences, len(subject_dirs)))
        obj_list = [chosen]
    else:
        obj_list = [None]
    broadcast_object_list(obj_list)
    chosen = obj_list[0]

    local_chosen = chosen[accelerator.process_index :: accelerator.num_processes]

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, sync_on_compute=False).to(device)

    # We accumulate (sum, sumsq, count) for 4 metrics:
    # real_ssim, fake_ssim, real_l1, fake_l1
    sum_real_ssim = 0.0
    sumsq_real_ssim = 0.0
    cnt_real_ssim = 0

    sum_fake_ssim = 0.0
    sumsq_fake_ssim = 0.0
    cnt_fake_ssim = 0

    sum_real_l1 = 0.0
    sumsq_real_l1 = 0.0
    cnt_real_l1 = 0

    sum_fake_l1 = 0.0
    sumsq_fake_l1 = 0.0
    cnt_fake_l1 = 0

    local_sequences_used = 0
    local_skipped = 0

    for idx, sdir in enumerate(local_chosen, start=1):
        vol = load_nifti(find_modality_file(sdir, modality))
        _, _, Z = vol.shape
        if Z < window + 1:
            local_skipped += 1
            continue

        v_lo, v_hi = percentile_lohi_nonzero(vol, p_lo, p_hi)
        start = random.randint(0, Z - window - 1)
        idxs = list(range(start, start + window + 1))

        real = []
        for k in idxs:
            u8 = to_u8(vol[:, :, k], v_lo, v_hi)
            u8 = resize_u8_square(u8, image_size)
            t = torch.from_numpy(u8.astype(np.float32) / 255.0)[None, None, ...].to(device)
            real.append(t)

        z_pos = torch.tensor(
            [z_min + (z_max - z_min) * ((k / (Z - 1)) if Z > 1 else 0.5) for k in idxs],
            dtype=torch.float32,
            device=device,
        )

        # Mixed precision sampling
        with accelerator.autocast():
            fake = diffusion_sample_01(diffusion, z_pos=z_pos)

        # For stability in metrics, compute SSIM/L1 in float32
        fake = fake.float()

        for i in range(window):
            ra, rb = real[i].float(), real[i + 1].float()
            fa, fb = fake[i : i + 1], fake[i + 1 : i + 2]

            v = float(ssim_metric(ra, rb).item())
            sum_real_ssim += v
            sumsq_real_ssim += v * v
            cnt_real_ssim += 1

            v = float(ssim_metric(fa, fb).item())
            sum_fake_ssim += v
            sumsq_fake_ssim += v * v
            cnt_fake_ssim += 1

            v = float(torch.mean(torch.abs(ra - rb)).item())
            sum_real_l1 += v
            sumsq_real_l1 += v * v
            cnt_real_l1 += 1

            v = float(torch.mean(torch.abs(fa - fb)).item())
            sum_fake_l1 += v
            sumsq_fake_l1 += v * v
            cnt_fake_l1 += 1

        local_sequences_used += 1

        # Low-volume progress printing (per rank, throttled)
        if log_every_local > 0 and (idx % log_every_local == 0):
            print(
                f"[ISC][rank {accelerator.process_index}] done {idx}/{len(local_chosen)} "
                f"(used={local_sequences_used}, skipped={local_skipped})",
                flush=True,
            )

    # Reduce once at the end (safe collectives)
    stats = torch.tensor(
        [
            sum_real_ssim, sumsq_real_ssim, float(cnt_real_ssim),
            sum_fake_ssim, sumsq_fake_ssim, float(cnt_fake_ssim),
            sum_real_l1,   sumsq_real_l1,   float(cnt_real_l1),
            sum_fake_l1,   sumsq_fake_l1,   float(cnt_fake_l1),
            float(local_sequences_used), float(local_skipped),
        ],
        dtype=torch.float64,
        device=device,
    )
    stats = accelerator.reduce(stats, reduction="sum")
    stats_cpu = stats.detach().cpu().numpy().tolist()

    def mean_std(sumv: float, sumsqv: float, cnt: float) -> Tuple[float, float]:
        if cnt <= 0:
            return float("nan"), float("nan")
        m = sumv / cnt
        var = (sumsqv / cnt) - (m * m)
        var = max(0.0, var)
        return float(m), float(np.sqrt(var))

    (
        sr, ssr, cr,
        sf, ssf, cf,
        slr, sslr, clr,
        slf, sslf, clf,
        seq_used, seq_skipped,
    ) = stats_cpu

    real_ssim_mean, real_ssim_std = mean_std(sr, ssr, cr)
    fake_ssim_mean, fake_ssim_std = mean_std(sf, ssf, cf)
    real_l1_mean, real_l1_std     = mean_std(slr, sslr, clr)
    fake_l1_mean, fake_l1_std     = mean_std(slf, sslf, clf)

    out: Dict[str, float] = {
        "real_adj_ssim_mean": real_ssim_mean,
        "real_adj_ssim_std": real_ssim_std,
        "fake_adj_ssim_mean": fake_ssim_mean,
        "fake_adj_ssim_std": fake_ssim_std,
        "real_adj_l1_mean": real_l1_mean,
        "real_adj_l1_std": real_l1_std,
        "fake_adj_l1_mean": fake_l1_mean,
        "fake_adj_l1_std": fake_l1_std,
        "num_sequences_used": int(seq_used),
        "num_sequences_skipped": int(seq_skipped),
        "window": int(window),
    }
    return out


# ----------------------------
# Debug
# ----------------------------
@torch.inference_mode()
def run_debug(
    accelerator: Accelerator,
    diffusion: torch.nn.Module,
    subject_dirs: List[str],
    modality: str,
    out_dir: str,
    device: torch.device,
    image_size: int,
    p_lo: float,
    p_hi: float,
    z_min: float,
    z_max: float,
    smoke_n: int,
    smoke_folder_metrics: bool,
) -> None:
    dbg = os.path.join(out_dir, "debug")
    os.makedirs(dbg, exist_ok=True)

    sdir = random.choice(subject_dirs)
    fpath = find_modality_file(sdir, modality)
    vol = load_nifti(fpath)
    _, _, Z = vol.shape
    v_lo, v_hi = percentile_lohi_nonzero(vol, p_lo, p_hi)

    print("\n[DEBUG] Data")
    print(f"  subject: {sdir}")
    print(f"  modality file: {fpath}")
    print(f"  vol shape: {vol.shape}")
    print(f"  per-volume p{p_lo}/p{p_hi}: lo={v_lo:.4f}, hi={v_hi:.4f}")
    print(f"  image_size: {image_size}")
    print(f"  z_min/z_max: {z_min} / {z_max}")

    idxs = sorted(set([0, Z // 4, Z // 2, (3 * Z) // 4, Z - 1]))
    real_paths, fake_paths = [], []

    z_list = []
    for k in idxs:
        frac = (k / (Z - 1)) if Z > 1 else 0.5
        z_list.append(z_min + (z_max - z_min) * frac)
    z_pos = torch.tensor(z_list, dtype=torch.float32, device=device)

    for k, z in zip(idxs, z_list):
        u8 = to_u8(vol[:, :, k], v_lo, v_hi)
        u8 = resize_u8_square(u8, image_size)
        p = os.path.join(dbg, f"real_idx{k:03d}_z{z:.3f}.png")
        save_gray_as_rgb_png(u8, p)
        real_paths.append(p)

    with accelerator.autocast():
        fake = diffusion_sample_01(diffusion, z_pos=z_pos)

    print("\n[DEBUG] Sample output")
    print(f"  z_pos: {z_list}")
    print(
        f"  fake: shape={tuple(fake.shape)}, min={fake.min().item():.4f}, "
        f"max={fake.max().item():.4f}, mean={fake.mean().item():.4f}"
    )

    for k, z, img in zip(idxs, z_list, fake):
        u8 = (img[0].detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        p = os.path.join(dbg, f"fake_idx{k:03d}_z{z:.3f}.png")
        save_gray_as_rgb_png(u8, p)
        fake_paths.append(p)

    make_grid(real_paths, nrow=len(real_paths), out_path=os.path.join(dbg, "grid_real.png"))
    make_grid(fake_paths, nrow=len(fake_paths), out_path=os.path.join(dbg, "grid_fake.png"))

    print(f"\n[DEBUG] Wrote: {os.path.join(dbg, 'grid_real.png')}")
    print(f"[DEBUG] Wrote: {os.path.join(dbg, 'grid_fake.png')}")

    if smoke_folder_metrics:
        real_dir = os.path.join(dbg, "smoke_real")
        fake_dir = os.path.join(dbg, "smoke_fake")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)

        print(f"\n[DEBUG] Smoke test: building tiny real/fake folders (N={smoke_n})")
        for i in range(smoke_n):
            s = random.choice(subject_dirs)
            v = load_nifti(find_modality_file(s, modality))
            _, _, zz = v.shape
            k = random.randint(0, zz - 1)
            lo, hi = percentile_lohi_nonzero(v[:, :, k], p_lo, p_hi)
            u8 = to_u8(v[:, :, k], lo, hi)
            u8 = resize_u8_square(u8, image_size)
            save_gray_as_rgb_png(u8, os.path.join(real_dir, f"r_{i:04d}.png"))

            frac = (k / (zz - 1)) if zz > 1 else 0.5
            z = z_min + (z_max - z_min) * frac
            with accelerator.autocast():
                img = diffusion_sample_01(diffusion, z_pos=torch.tensor([z], device=device))[0, 0]
            u8f = (img.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
            save_gray_as_rgb_png(u8f, os.path.join(fake_dir, f"f_{i:04d}.png"))

        m = torch_fidelity_folder_metrics(
            real_dir,
            fake_dir,
            use_cuda=(device.type == "cuda"),
            kid_subset_size=min(64, smoke_n),
            kid_subsets=10,
            num_workers=2,
            batch_size=16,
        )

        print("\n[DEBUG] Smoke folder metrics ran successfully. (Values not meaningful at tiny N)")
        print(json.dumps(m, indent=2))


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    if MODALITY not in {"t1", "t1ce", "t1c", "t2", "flair"}:
        raise ValueError(f"MODALITY must be one of t1,t1ce,t1c,t2,flair (got {MODALITY})")

    pg = InitProcessGroupKwargs(timeout=timedelta(hours=24))
    accelerator = Accelerator(
        cpu=(DEVICE == "cpu"),
        mixed_precision=MIXED_PRECISION,
        kwargs_handlers=[pg],
    )
    device = accelerator.device

    if accelerator.device.type == "cuda":
        torch.cuda.set_device(accelerator.device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    set_seed(SEED, device_specific=True)

    if accelerator.is_main_process:
        os.makedirs(OUT_DIR, exist_ok=True)
        print(
            f"Accelerate: num_processes={accelerator.num_processes}, "
            f"process_index={accelerator.process_index}, device={device}, "
            f"mixed_precision={MIXED_PRECISION}",
            flush=True,
        )
    accelerator.wait_for_everyone()

    subject_dirs = find_subject_dirs(BRATS_ROOT)

    diffusion = build_diffusion(
        image_size=IMAGE_SIZE,
        timesteps=TIMESTEPS,
        device=device,
        img_channels=IMG_CHANNELS,
        base_channels=BASE_CHANNELS,
        channel_mults=CHANNEL_MULTS,
        time_emb_dim=TIME_EMB_DIM,
    ).to(device)
    diffusion.eval()

    load_checkpoint_like_yours(diffusion, CHECKPOINT, device=device)

    if DEBUG and accelerator.is_main_process:
        run_debug(
            accelerator=accelerator,
            diffusion=diffusion,
            subject_dirs=subject_dirs,
            modality=MODALITY,
            out_dir=OUT_DIR,
            device=device,
            image_size=IMAGE_SIZE,
            p_lo=P_LO,
            p_hi=P_HI,
            z_min=Z_MIN,
            z_max=Z_MAX,
            smoke_n=SMOKE_N,
            smoke_folder_metrics=SMOKE_FOLDER_METRICS,
        )
    accelerator.wait_for_everyone()
    if DEBUG_ONLY and DEBUG:
        return

    num_images = min(NUM_IMAGES, DEBUG_CAP_NUM_IMAGES) if DEBUG else NUM_IMAGES

    real_dir = os.path.join(OUT_DIR, "real_png")
    fake_dir = os.path.join(OUT_DIR, "fake_png")
    if accelerator.is_main_process:
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Shard indices across processes:
    local_indices = list(range(accelerator.process_index, num_images, accelerator.num_processes))

    log_main(accelerator, f"\n[1/5] Creating REAL folder (N={num_images}) -> {real_dir}")

    local_meta: List[Tuple[int, int, int]] = []

    for i in local_indices:
        sdir = random.choice(subject_dirs)
        vol = load_nifti(find_modality_file(sdir, MODALITY))
        _, _, Z = vol.shape
        k = random.randint(0, Z - 1)

        lo, hi = percentile_lohi_nonzero(vol[:, :, k], P_LO, P_HI)
        u8 = to_u8(vol[:, :, k], lo, hi)
        u8 = resize_u8_square(u8, IMAGE_SIZE)

        save_gray_as_rgb_png(u8, os.path.join(real_dir, f"real_{i:06d}.png"))
        local_meta.append((i, k, Z))
    accelerator.wait_for_everyone()

    log_main(accelerator, f"\n[2/5] Creating FAKE folder (N={num_images}) -> {fake_dir}")

    local_meta.sort(key=lambda t: t[0])

    for batch in chunked(local_meta, BATCH_SIZE):
        z_list = []
        out_indices = []
        for (gi, k, Z) in batch:
            frac = (k / (Z - 1)) if Z > 1 else 0.5
            z_list.append(Z_MIN + (Z_MAX - Z_MIN) * frac)
            out_indices.append(gi)

        z_pos = torch.tensor(z_list, dtype=torch.float32, device=device)

        with accelerator.autocast():
            imgs = diffusion_sample_01(diffusion, z_pos=z_pos)

        imgs_u8 = (imgs[:, 0].detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        for j, gi in enumerate(out_indices):
            save_gray_as_rgb_png(imgs_u8[j], os.path.join(fake_dir, f"fake_{gi:06d}.png"))

    accelerator.wait_for_everyone()

    # Step 3 + 4 only on main
    tf = None
    div = None
    if accelerator.is_main_process:
        print("\n[3/5] Computing FID/KID/PRC (torch-fidelity)...", flush=True)
        tf = torch_fidelity_folder_metrics(real_dir, fake_dir, use_cuda=(device.type == "cuda"))

        print("\n[4/5] Computing diversity (MS-SSIM + LPIPS) on fake set...", flush=True)
        div = diversity_metrics(fake_dir, device=device, num_pairs=DIV_PAIRS)

        config = {
            "BRATS_ROOT": str(BRATS_ROOT),
            "MODALITY": MODALITY,
            "CHECKPOINT": CHECKPOINT,
            "OUT_DIR": OUT_DIR,
            "IMAGE_SIZE": IMAGE_SIZE,
            "TIMESTEPS": TIMESTEPS,
            "IMG_CHANNELS": IMG_CHANNELS,
            "BASE_CHANNELS": BASE_CHANNELS,
            "CHANNEL_MULTS": list(CHANNEL_MULTS),
            "TIME_EMB_DIM": TIME_EMB_DIM,
            "Z_MIN": Z_MIN,
            "Z_MAX": Z_MAX,
            "NUM_IMAGES": NUM_IMAGES,
            "BATCH_SIZE": BATCH_SIZE,
            "P_LO": P_LO,
            "P_HI": P_HI,
            "DIV_PAIRS": DIV_PAIRS,
            "ISC_NUM_SEQUENCES": ISC_NUM_SEQUENCES,
            "ISC_WINDOW": ISC_WINDOW,
            "SEED": SEED,
            "DEVICE": DEVICE,
            "MIXED_PRECISION": MIXED_PRECISION,
            "DEBUG": DEBUG,
            "DEBUG_ONLY": DEBUG_ONLY,
            "SMOKE_FOLDER_METRICS": SMOKE_FOLDER_METRICS,
            "SMOKE_N": SMOKE_N,
            "DEBUG_CAP_NUM_IMAGES": DEBUG_CAP_NUM_IMAGES,
        }

        results_step4 = {
            "config": config,
            "torch_fidelity": tf,
            "diversity": div,
            "inter_slice_consistency": None,
        }

        out_json_step4 = os.path.join(OUT_DIR, "metrics_step4.json")
        save_json(out_json_step4, results_step4)
        print(f"\nSaved step-4 metrics to: {out_json_step4}", flush=True)

    accelerator.wait_for_everyone()

    # Step 5 on ALL processes (distributed)
    log_main(accelerator, "\n[5/5] Computing inter-slice consistency (distributed across GPUs)...")

    isc = inter_slice_consistency_distributed(
        accelerator=accelerator,
        diffusion=diffusion,
        subject_dirs=subject_dirs,
        modality=MODALITY,
        device=device,
        image_size=IMAGE_SIZE,
        num_sequences=ISC_NUM_SEQUENCES,
        window=ISC_WINDOW,
        p_lo=P_LO,
        p_hi=P_HI,
        z_min=Z_MIN,
        z_max=Z_MAX,
        log_every_local=ISC_LOG_EVERY_LOCAL,
    )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # tf/div exist only on main, so reuse them here
        # (they were computed above and still in scope)
        config = {
            "BRATS_ROOT": str(BRATS_ROOT),
            "MODALITY": MODALITY,
            "CHECKPOINT": CHECKPOINT,
            "OUT_DIR": OUT_DIR,
            "IMAGE_SIZE": IMAGE_SIZE,
            "TIMESTEPS": TIMESTEPS,
            "IMG_CHANNELS": IMG_CHANNELS,
            "BASE_CHANNELS": BASE_CHANNELS,
            "CHANNEL_MULTS": list(CHANNEL_MULTS),
            "TIME_EMB_DIM": TIME_EMB_DIM,
            "Z_MIN": Z_MIN,
            "Z_MAX": Z_MAX,
            "NUM_IMAGES": NUM_IMAGES,
            "BATCH_SIZE": BATCH_SIZE,
            "P_LO": P_LO,
            "P_HI": P_HI,
            "DIV_PAIRS": DIV_PAIRS,
            "ISC_NUM_SEQUENCES": ISC_NUM_SEQUENCES,
            "ISC_WINDOW": ISC_WINDOW,
            "SEED": SEED,
            "DEVICE": DEVICE,
            "MIXED_PRECISION": MIXED_PRECISION,
            "DEBUG": DEBUG,
            "DEBUG_ONLY": DEBUG_ONLY,
            "SMOKE_FOLDER_METRICS": SMOKE_FOLDER_METRICS,
            "SMOKE_N": SMOKE_N,
            "DEBUG_CAP_NUM_IMAGES": DEBUG_CAP_NUM_IMAGES,
        }

        results_final = {
            "config": config,
            "torch_fidelity": tf,
            "diversity": div,
            "inter_slice_consistency": isc,
        }

        out_json = os.path.join(OUT_DIR, "metrics.json")
        save_json(out_json, results_final)

        print("\n=== DONE ===", flush=True)
        print(json.dumps(results_final, indent=2), flush=True)
        print(f"\nSaved final metrics to: {out_json}", flush=True)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
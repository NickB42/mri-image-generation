"""
Eval script for neighbour-slice-conditioned (2.5D) DDPM on BraTS NIfTI volumes.

Metrics:
  - torch-fidelity: FID, KID, PRC per modality (folders of PNGs) + macro average
  - Diversity: MS-SSIM + LPIPS per modality + macro average
  - Inter-slice consistency: adjacent SSIM + L1 over windows (real vs fake), averaged across modalities
  - Conditional reconstruction: SSIM + L1 between real center and generated center under real context

CHANGES:
  - NO Accelerator.broadcast_object_list (compat w/ older accelerate)
    * subject_paths computed on every rank
    * ISC chosen subjects: rank0 writes JSON, other ranks read
  - Set CUDA device via LOCAL_RANK BEFORE Accelerator init (reduces NCCL mapping warnings/hang risk)
  - Use ALL GPUs for Step 5 (ISC) and Step 6 (cond recon) via sharding + gather/reduce-on-rank0
  - Mixed precision via Accelerator(mixed_precision="fp16") + accelerator.autocast (guarded)
  - Batch sampling inside ISC (sample window+1 slices in batches)
  - Save results after Step 4, after Step 5, final after Step 6
  - More progress prints
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import OrderedDict
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import timedelta
from contextlib import nullcontext

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import nibabel as nib
import torch_fidelity

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import InitProcessGroupKwargs

from .unet import UNet
from .diffusion import GaussianDiffusion


# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS_ROOT = (PROJECT_ROOT / "ddpm_25d_all_modalities").resolve()
BRATS_ROOT = (PROJECT_ROOT / "../datasets/test").resolve()

CHECKPOINT = str(EXPERIMENTS_ROOT / "models" / "1591706" / "25d_ddpm_all_modalities_best.pt")

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID")
OUT_DIR = f"./eval_out/neigh_con/{SLURM_JOB_ID}"

# Training-matched params
IMAGE_SIZE = 128
TIMESTEPS = 1000

CENTER_MODALITIES = 4
SLICE_RADIUS = 2
CONTEXT_SLICES = 2 * SLICE_RADIUS
IN_CHANNELS = CENTER_MODALITIES + CENTER_MODALITIES * CONTEXT_SLICES
OUT_CHANNELS = CENTER_MODALITIES

BASE_CHANNELS = 64
CHANNEL_MULTS = (1, 2, 4, 8)
TIME_EMB_DIM = 256

# Sampling counts
NUM_IMAGES = 4000
BATCH_SIZE = 32

# diversity
DIV_PAIRS = 2000
DIVERSITY_RESIZE_TO = 256

# inter-slice consistency
ISC_NUM_SEQUENCES = 50
ISC_WINDOW = 16
ISC_SAMPLE_BATCH = BATCH_SIZE  # batch used inside ISC sampling (window+1 slices)

# conditional reconstruction
DO_COND_RECON = True
COND_RECON_SAMPLES = 256
COND_RECON_BATCH = BATCH_SIZE

# dataset-like preprocessing
CLIP_MIN = -5.0
CLIP_MAX = 5.0

# misc
SEED = 123
DEVICE = "auto"  # "auto" | "cpu"

# mixed precision: "fp16" or "bf16" (A100 supports bf16 nicely)
DEFAULT_MIXED_PRECISION = os.environ.get("MIXED_PRECISION", "fp16")

# debug
DEBUG = False
DEBUG_ONLY = False
SMOKE_FOLDER_METRICS = True
SMOKE_N = 64
DEBUG_CAP_NUM_IMAGES = 256

# modality naming (for folders + NIfTI lookup)
MODALITIES = ["t1", "t1ce", "t2", "flair"]
MODALITY_SUFFIXES = {
    "t1": ["*_t1.nii*", "*_T1.nii*"],
    "t1ce": ["*_t1ce.nii*", "*_t1c.nii*", "*_T1CE.nii*", "*_T1C.nii*"],
    "t2": ["*_t2.nii*", "*_T2.nii*"],
    "flair": ["*_flair.nii*", "*_FLAIR.nii*"],
}


# ============================================================
# HELPERS
# ============================================================
def chunked(seq, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


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


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def save_results_json(out_dir: str, filename: str, results: Dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return out_path


def gather_tensor(accelerator: Accelerator, t: torch.Tensor) -> torch.Tensor:
    """
    Compatibility gather:
      - prefer Accelerator.gather
      - fallback to torch.distributed.all_gather
    Returns tensor stacked over world size on all ranks.
    """
    if hasattr(accelerator, "gather"):
        return accelerator.gather(t)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ws = torch.distributed.get_world_size()
        outs = [torch.zeros_like(t) for _ in range(ws)]
        torch.distributed.all_gather(outs, t)
        return torch.stack(outs, dim=0)
    return t.unsqueeze(0)


class RunningStats:
    """Population mean/std accumulator (matches numpy std ddof=0)."""
    def __init__(self) -> None:
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def add(self, x: float) -> None:
        x = float(x)
        self.n += 1
        self.sum += x
        self.sumsq += x * x

    def tensors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.sum, device=device, dtype=torch.float64),
            torch.tensor(self.sumsq, device=device, dtype=torch.float64),
            torch.tensor(self.n, device=device, dtype=torch.int64),
        )

    @staticmethod
    def finalize(sum_v: float, sumsq_v: float, n_v: int) -> Dict[str, float]:
        if n_v <= 0:
            return {"mean": float("nan"), "std": float("nan")}
        mean = sum_v / n_v
        var = (sumsq_v / n_v) - (mean * mean)
        var = max(var, 0.0)
        return {"mean": float(mean), "std": float(np.sqrt(var))}


# ============================================================
# BraTS I/O
# ============================================================
def find_subject_dirs(brats_root: str) -> List[str]:
    subs = [p for p in glob(os.path.join(brats_root, "*")) if os.path.isdir(p)]
    subs.sort()
    if not subs:
        raise FileNotFoundError(f"No subject directories under {brats_root}")
    return subs


def find_modality_file(subject_dir: str, modality: str) -> str:
    modality = modality.lower()
    patterns = MODALITY_SUFFIXES.get(modality)
    if not patterns:
        raise ValueError(f"Unknown modality: {modality}")

    candidates: List[str] = []
    for pat in patterns:
        candidates += glob(os.path.join(subject_dir, pat))
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(f"No '{modality}' NIfTI found in {subject_dir}")
    return candidates[0]


def build_subject_path_map(subject_dirs: List[str]) -> Dict[str, List[str]]:
    """Precompute NIfTI paths per subject to avoid repeated globbing."""
    out: Dict[str, List[str]] = {}
    for sdir in subject_dirs:
        out[sdir] = [find_modality_file(sdir, m) for m in MODALITIES]
    return out


class VolumeCache:
    """Small per-process LRU cache to avoid reloading the same NIfTI repeatedly."""
    def __init__(self, max_items: int = 6):
        self.max_items = max_items
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def load(self, path: str) -> np.ndarray:
        path = str(path)
        if path in self._cache:
            vol = self._cache.pop(path)
            self._cache[path] = vol
            return vol
        img = nib.load(path)
        vol = np.asanyarray(img.dataobj).astype(np.float32)
        if vol.ndim == 4 and vol.shape[-1] == 1:
            vol = vol[..., 0]
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D NIfTI, got {vol.shape} at {path}")
        self._cache[path] = vol
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return vol


def preprocess_slice_np(slice_2d: np.ndarray, image_size: int) -> torch.Tensor:
    """
    Mirrors your BraTSSliceDataset._preprocess_slice:
      - z-score on non-zero voxels per slice
      - clip to [-5,5]
      - map to [0,1]
      - resize bilinear
      - map to [-1,1]
    Returns tensor shape (1,H,W) float32 in [-1,1].
    """
    x = slice_2d.astype(np.float32).copy()
    mask = x != 0
    if np.any(mask):
        mean = float(x[mask].mean())
        std = float(x[mask].std())
        std = std if std > 0 else 1.0
        x[mask] = (x[mask] - mean) / std

    x = np.clip(x, CLIP_MIN, CLIP_MAX)
    x = (x - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)  # [-5,5] -> [0,1]

    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = F.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
    t = t.squeeze(0)  # (1,H,W)
    t = t * 2.0 - 1.0
    return t.contiguous()


def to_u8_from_minus1_1(t: torch.Tensor) -> np.ndarray:
    if t.ndim == 3 and t.shape[0] == 1:
        t = t[0]
    x01 = (t.clamp(-1, 1) + 1.0) / 2.0
    return (x01.detach().cpu().numpy() * 255.0).round().astype(np.uint8)


def build_center_tensor(vols: List[np.ndarray], z: int, image_size: int, device: torch.device) -> torch.Tensor:
    chans: List[torch.Tensor] = []
    for vol in vols:
        chans.append(preprocess_slice_np(vol[:, :, z], image_size=image_size))  # (1,H,W)
    return torch.cat(chans, dim=0).unsqueeze(0).to(device)  # (1,4,H,W) in [-1,1]


def build_context_tensor(vols: List[np.ndarray], z: int, radius: int, image_size: int, device: torch.device) -> torch.Tensor:
    """
    Channel order matches your dataset:
      for dz in [-r..r], dz!=0:
        for modality in [t1,t1ce,t2,flair]:
          append preprocess(vol[:,:,z+dz])
    Returns (1, 16, H, W) for radius=2.
    """
    D = vols[0].shape[-1]
    chans: List[torch.Tensor] = []
    for dz in range(-radius, radius + 1):
        if dz == 0:
            continue
        zz = z + dz
        zz = max(0, min(D - 1, zz))
        for vol in vols:
            chans.append(preprocess_slice_np(vol[:, :, zz], image_size=image_size))
    ctx = torch.cat(chans, dim=0).unsqueeze(0).to(device)  # (1,Cctx,H,W)
    return ctx


# ============================================================
# Model build + checkpoint load
# ============================================================
def build_diffusion(device: torch.device) -> GaussianDiffusion:
    model = UNet(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        base_channels=BASE_CHANNELS,
        channel_mults=CHANNEL_MULTS,
        time_emb_dim=TIME_EMB_DIM,
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        image_size=IMAGE_SIZE,
        channels=OUT_CHANNELS,
        timesteps=TIMESTEPS,
    ).to(device)
    return diffusion


def load_checkpoint_like_yours(diffusion: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device)

    # rewrite DataParallel-style keys
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
def diffusion_sample_center_01(diffusion: GaussianDiffusion, z_pos: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """
    z_pos:   (B,) in [0,1]
    context: (B,16,H,W) in [-1,1]
    returns: (B,4,H,W) in [0,1]
    """
    out = diffusion.sample(batch_size=int(z_pos.shape[0]), z_pos=z_pos, context=context)
    x = torch.as_tensor(out, device=context.device).clamp(-1, 1)
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


# ============================================================
# Metrics
# ============================================================
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
    x = torch.stack(imgs, dim=0).to(device)  # (N,3,H,W)
    if resize_to:
        x = F.interpolate(x, size=(resize_to, resize_to), mode="bilinear", align_corners=False)

    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

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
    diffusion: GaussianDiffusion,
    subject_dirs: List[str],
    subject_paths: Dict[str, List[str]],
    cache: VolumeCache,
    device: torch.device,
    accelerator: Accelerator,
    out_dir: str,
    num_sequences: int,
    window: int,
    sample_batch: int,
) -> Dict[str, float]:
    """
    Distributed ISC:
      - rank0 picks subjects and writes JSON in out_dir
      - all ranks read the same chosen list
      - shard chosen list across ranks
      - batch sampling inside each sequence (window+1 slices, sampled in mini-batches)
      - gather sums/sumsq/counts to rank0 and compute mean/std
    """
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # ---- choose subjects on rank0, save to file, others read
    chosen_path = os.path.join(out_dir, "isc_chosen_subjects.json")

    if accelerator.is_main_process:
        chosen = random.sample(subject_dirs, k=min(num_sequences, len(subject_dirs)))
        with open(chosen_path, "w") as f:
            json.dump(chosen, f)
        print(f"[{now_str()}] [ISC] wrote chosen subjects -> {chosen_path} (N={len(chosen)})")

    accelerator.wait_for_everyone()

    with open(chosen_path, "r") as f:
        chosen: List[str] = json.load(f)

    # ---- shard
    rank = accelerator.process_index
    world = accelerator.num_processes
    local_chosen = chosen[rank::world]

    # ---- autocast context (compat)
    autocast_ctx = accelerator.autocast if hasattr(accelerator, "autocast") else nullcontext

    if accelerator.is_local_main_process:
        print(f"[{now_str()}] [ISC] world={world} window={window} sample_batch={sample_batch}")

    print(f"[{now_str()}] [ISC] rank {rank}/{world} processing {len(local_chosen)} sequences")

    stats = {
        "real_adj_ssim": RunningStats(),
        "fake_adj_ssim": RunningStats(),
        "real_adj_l1": RunningStats(),
        "fake_adj_l1": RunningStats(),
    }
    local_sequences_used = 0
    local_pairs_used = 0

    # only show tqdm on local main to avoid log spam
    pbar = None
    if accelerator.is_local_main_process:
        pbar = tqdm(total=len(local_chosen), desc=f"ISC rank{rank}", leave=True)

    for si, sdir in enumerate(local_chosen):
        paths = subject_paths[sdir]
        vols = [cache.load(p) for p in paths]
        _, _, D = vols[0].shape

        z_start = int(0.1 * D) + SLICE_RADIUS
        z_end = int(0.9 * D) - SLICE_RADIUS
        if z_end <= z_start or (z_end - z_start) < (window + 1):
            if pbar is not None:
                pbar.update(1)
            continue

        start = random.randint(z_start, z_end - (window + 1))
        idxs = list(range(start, start + window + 1))

        # real sequence (N,4,H,W) in [0,1]
        real_seq = []
        for z in idxs:
            cen = build_center_tensor(vols, z=z, image_size=IMAGE_SIZE, device=device)  # [-1,1]
            real_seq.append((cen + 1.0) / 2.0)
        real_seq = torch.cat(real_seq, dim=0)

        # fake sequence: build all contexts, sample in batches
        ctx_all = []
        zpos_all = []
        for z in idxs:
            ctx_all.append(build_context_tensor(vols, z=z, radius=SLICE_RADIUS, image_size=IMAGE_SIZE, device=device))
            zpos_all.append(z / (D - 1))
        context_b = torch.cat(ctx_all, dim=0)  # (N,16,H,W)
        z_pos = torch.tensor(zpos_all, dtype=torch.float32, device=device)  # (N,)

        fake_parts = []
        for k in range(0, len(idxs), sample_batch):
            with autocast_ctx():
                fake_parts.append(
                    diffusion_sample_center_01(
                        diffusion,
                        z_pos=z_pos[k : k + sample_batch],
                        context=context_b[k : k + sample_batch],
                    )
                )
        fake_seq = torch.cat(fake_parts, dim=0)  # (N,4,H,W) in [0,1]

        # adjacent metrics (flatten modalities into batch-of-grayscale)
        for i in range(window):
            ra, rb = real_seq[i : i + 1], real_seq[i + 1 : i + 2]
            fa, fb = fake_seq[i : i + 1], fake_seq[i + 1 : i + 2]

            _, C, H, W = ra.shape  # C=4
            ra_g = ra.reshape(C, 1, H, W)
            rb_g = rb.reshape(C, 1, H, W)
            fa_g = fa.reshape(C, 1, H, W)
            fb_g = fb.reshape(C, 1, H, W)

            ssim_r = float(ssim(ra_g, rb_g).item())
            ssim_f = float(ssim(fa_g, fb_g).item())
            l1_r = float(torch.mean(torch.abs(ra - rb)).item())
            l1_f = float(torch.mean(torch.abs(fa - fb)).item())

            stats["real_adj_ssim"].add(ssim_r)
            stats["fake_adj_ssim"].add(ssim_f)
            stats["real_adj_l1"].add(l1_r)
            stats["fake_adj_l1"].add(l1_f)
            local_pairs_used += 1

        local_sequences_used += 1

        if pbar is not None:
            pbar.update(1)

        if (si + 1) % 5 == 0 or (si + 1) == len(local_chosen):
            print(f"[{now_str()}] [ISC] rank {rank}: {si+1}/{len(local_chosen)} seq done "
                  f"(seq_used={local_sequences_used}, pairs={local_pairs_used})")

    if pbar is not None:
        pbar.close()

    accelerator.wait_for_everyone()

    # ---- gather sums/sumsq/n to rank0
    out: Dict[str, float] = {}
    packed = {}
    for key, rs in stats.items():
        s, ss, n = rs.tensors(device=device)
        s_all = gather_tensor(accelerator, s)
        ss_all = gather_tensor(accelerator, ss)
        n_all = gather_tensor(accelerator, n)
        if accelerator.is_main_process:
            packed[key] = (
                float(s_all.sum().item()),
                float(ss_all.sum().item()),
                int(n_all.sum().item()),
            )

    seq_used_all = gather_tensor(accelerator, torch.tensor(local_sequences_used, device=device, dtype=torch.int64))
    pairs_all = gather_tensor(accelerator, torch.tensor(local_pairs_used, device=device, dtype=torch.int64))

    if accelerator.is_main_process:
        for key, (sum_v, sumsq_v, n_v) in packed.items():
            fin = RunningStats.finalize(sum_v, sumsq_v, n_v)
            out[f"{key}_mean"] = fin["mean"]
            out[f"{key}_std"] = fin["std"]

        out["num_sequences_target"] = int(len(chosen))
        out["num_sequences_used"] = int(seq_used_all.sum().item())
        out["num_adj_pairs_used"] = int(pairs_all.sum().item())
        out["window"] = int(window)
        out["slice_radius"] = int(SLICE_RADIUS)
        out["sample_batch"] = int(sample_batch)

    accelerator.wait_for_everyone()
    return out


@torch.inference_mode()
def conditional_reconstruction_distributed(
    diffusion: GaussianDiffusion,
    subject_dirs: List[str],
    subject_paths: Dict[str, List[str]],
    cache: VolumeCache,
    device: torch.device,
    accelerator: Accelerator,
    num_samples: int,
    batch_size: int,
) -> Dict[str, float]:
    """
    Distributed conditional reconstruction:
      - split num_samples across ranks
      - batch sampling
      - gather sums/sumsq/counts to rank0 for mean/std
    """
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    rank = accelerator.process_index
    world = accelerator.num_processes

    base = num_samples // world
    rem = num_samples % world
    local_target = base + (1 if rank < rem else 0)

    autocast_ctx = accelerator.autocast if hasattr(accelerator, "autocast") else nullcontext

    print(f"[{now_str()}] [COND] rank {rank}/{world} target={local_target} (global={num_samples}) batch={batch_size}")

    stats_ssim = RunningStats()
    stats_l1 = RunningStats()

    local_used = 0
    tries = 0
    max_tries = max(local_target * 8, 128)

    pbar = None
    if accelerator.is_local_main_process:
        pbar = tqdm(total=local_target, desc=f"COND rank{rank}", leave=True)

    while local_used < local_target and tries < max_tries:
        tries += 1

        cur_bs = min(batch_size, local_target - local_used)

        ctx_list, zpos_list, real_list = [], [], []
        filled = 0
        inner_tries = 0

        while filled < cur_bs and inner_tries < cur_bs * 10:
            inner_tries += 1

            sdir = random.choice(subject_dirs)
            paths = subject_paths[sdir]
            vols = [cache.load(p) for p in paths]
            _, _, D = vols[0].shape

            z_start = int(0.1 * D) + SLICE_RADIUS
            z_end = int(0.9 * D) - SLICE_RADIUS
            if z_end <= z_start:
                continue

            z = random.randint(z_start, z_end - 1)

            real = build_center_tensor(vols, z=z, image_size=IMAGE_SIZE, device=device)  # [-1,1]
            real01 = (real + 1.0) / 2.0
            ctx = build_context_tensor(vols, z=z, radius=SLICE_RADIUS, image_size=IMAGE_SIZE, device=device)
            z_pos = torch.tensor([z / (D - 1)], dtype=torch.float32, device=device)

            real_list.append(real01)
            ctx_list.append(ctx)
            zpos_list.append(z_pos)
            filled += 1

        if filled == 0:
            continue

        real_b = torch.cat(real_list, dim=0)  # (B,4,H,W)
        ctx_b = torch.cat(ctx_list, dim=0)    # (B,16,H,W)
        z_pos_b = torch.cat(zpos_list, dim=0).view(-1)  # (B,)

        with autocast_ctx():
            fake_b = diffusion_sample_center_01(diffusion, z_pos=z_pos_b, context=ctx_b)

        B, C, H, W = real_b.shape
        for b in range(B):
            fb = fake_b[b : b + 1]
            rb = real_b[b : b + 1]

            fb_g = fb.reshape(C, 1, H, W)
            rb_g = rb.reshape(C, 1, H, W)

            stats_ssim.add(float(ssim(fb_g, rb_g).item()))
            stats_l1.add(float(torch.mean(torch.abs(fb - rb)).item()))

        local_used += B
        if pbar is not None:
            pbar.update(B)

        if local_used % max(32, batch_size * 2) == 0 or local_used >= local_target:
            print(f"[{now_str()}] [COND] rank {rank}: {local_used}/{local_target} done")

    if pbar is not None:
        pbar.close()

    accelerator.wait_for_everyone()

    # gather to rank0
    out: Dict[str, float] = {}
    s1, ss1, n1 = stats_ssim.tensors(device=device)
    s2, ss2, n2 = stats_l1.tensors(device=device)

    s1_all = gather_tensor(accelerator, s1)
    ss1_all = gather_tensor(accelerator, ss1)
    n1_all = gather_tensor(accelerator, n1)

    s2_all = gather_tensor(accelerator, s2)
    ss2_all = gather_tensor(accelerator, ss2)
    n2_all = gather_tensor(accelerator, n2)

    used_all = gather_tensor(accelerator, torch.tensor(local_used, device=device, dtype=torch.int64))

    if accelerator.is_main_process:
        fin1 = RunningStats.finalize(float(s1_all.sum().item()), float(ss1_all.sum().item()), int(n1_all.sum().item()))
        fin2 = RunningStats.finalize(float(s2_all.sum().item()), float(ss2_all.sum().item()), int(n2_all.sum().item()))

        out["cond_recon_ssim_mean"] = fin1["mean"]
        out["cond_recon_ssim_std"] = fin1["std"]
        out["cond_recon_l1_mean"] = fin2["mean"]
        out["cond_recon_l1_std"] = fin2["std"]
        out["num_samples_target"] = int(num_samples)
        out["num_samples_used"] = int(used_all.sum().item())
        out["batch_size"] = int(batch_size)

    accelerator.wait_for_everyone()
    return out


# ============================================================
# Debug
# ============================================================
@torch.inference_mode()
def run_debug(
    diffusion: GaussianDiffusion,
    subject_dirs: List[str],
    subject_paths: Dict[str, List[str]],
    cache: VolumeCache,
    out_dir: str,
    device: torch.device,
) -> None:
    dbg = os.path.join(out_dir, "debug")
    os.makedirs(dbg, exist_ok=True)

    sdir = random.choice(subject_dirs)
    paths = subject_paths[sdir]
    vols = [cache.load(p) for p in paths]
    _, _, D = vols[0].shape

    z_start = int(0.1 * D) + SLICE_RADIUS
    z_end = int(0.9 * D) - SLICE_RADIUS
    z = random.randint(z_start, max(z_start, z_end - 1))

    img_paths = []
    for dz in range(-SLICE_RADIUS, SLICE_RADIUS + 1):
        if dz == 0:
            continue
        zz = max(0, min(D - 1, z + dz))
        for mi, m in enumerate(MODALITIES):
            t = preprocess_slice_np(vols[mi][:, :, zz], IMAGE_SIZE)
            u8 = to_u8_from_minus1_1(t)
            p = os.path.join(dbg, f"real_ctx_dz{dz:+d}_{m}_z{zz:03d}.png")
            save_gray_as_rgb_png(u8, p)
            img_paths.append(p)

    for mi, m in enumerate(MODALITIES):
        t = preprocess_slice_np(vols[mi][:, :, z], IMAGE_SIZE)
        u8 = to_u8_from_minus1_1(t)
        p = os.path.join(dbg, f"real_center_{m}_z{z:03d}.png")
        save_gray_as_rgb_png(u8, p)
        img_paths.append(p)

    ctx = build_context_tensor(vols, z=z, radius=SLICE_RADIUS, image_size=IMAGE_SIZE, device=device)
    z_pos = torch.tensor([z / (D - 1)], dtype=torch.float32, device=device)
    fake01 = diffusion_sample_center_01(diffusion, z_pos=z_pos, context=ctx)

    fake_paths = []
    for mi, m in enumerate(MODALITIES):
        u8 = (fake01[0, mi].detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        p = os.path.join(dbg, f"fake_center_{m}_z{z:03d}.png")
        save_gray_as_rgb_png(u8, p)
        fake_paths.append(p)

    make_grid(img_paths, nrow=min(8, len(img_paths)), out_path=os.path.join(dbg, "grid_real_ctx_plus_center.png"))
    make_grid(fake_paths, nrow=len(fake_paths), out_path=os.path.join(dbg, "grid_fake_center_modalities.png"))

    print(f"[DEBUG] subject={sdir} D={D} z={z}")
    print(f"[DEBUG] Wrote: {os.path.join(dbg, 'grid_real_ctx_plus_center.png')}")
    print(f"[DEBUG] Wrote: {os.path.join(dbg, 'grid_fake_center_modalities.png')}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    # ---- set cuda device BEFORE Accelerator init (helps NCCL mapping warnings)
    use_cuda = (DEVICE != "cpu") and torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if use_cuda:
        torch.cuda.set_device(local_rank)

    pg = InitProcessGroupKwargs(timeout=timedelta(hours=24))

    mixed_precision = DEFAULT_MIXED_PRECISION if use_cuda else "no"
    accelerator = Accelerator(cpu=not use_cuda, mixed_precision=mixed_precision, kwargs_handlers=[pg])
    device = accelerator.device

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    set_seed(SEED, device_specific=True)

    if accelerator.is_main_process:
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"[{now_str()}] Accelerate: num_processes={accelerator.num_processes}, "
              f"process_index={accelerator.process_index}, local_rank={local_rank}, "
              f"device={device}, mixed_precision={mixed_precision}")
        print(f"[{now_str()}] OUT_DIR: {OUT_DIR}")
        print(f"[{now_str()}] BRATS_ROOT: {BRATS_ROOT}")
        print(f"[{now_str()}] CHECKPOINT: {CHECKPOINT}")

    accelerator.wait_for_everyone()

    subject_dirs = find_subject_dirs(str(BRATS_ROOT))

    # Build subject_paths on EVERY rank (avoids broadcast_object_list)
    if accelerator.is_local_main_process:
        print(f"[{now_str()}] Building modality path map on each rank (Nsubjects={len(subject_dirs)}) ...")
    subject_paths = build_subject_path_map(subject_dirs)

    cache = VolumeCache(max_items=6)

    diffusion = build_diffusion(device=device)
    diffusion.eval()
    load_checkpoint_like_yours(diffusion, CHECKPOINT, device=device)

    if DEBUG and accelerator.is_main_process:
        run_debug(diffusion, subject_dirs, subject_paths, cache, OUT_DIR, device=device)
    accelerator.wait_for_everyone()
    if DEBUG_ONLY and DEBUG:
        return

    num_images = min(NUM_IMAGES, DEBUG_CAP_NUM_IMAGES) if DEBUG else NUM_IMAGES

    # folders per modality for torch-fidelity
    real_root = os.path.join(OUT_DIR, "real_png")
    fake_root = os.path.join(OUT_DIR, "fake_png")
    if accelerator.is_main_process:
        for m in MODALITIES:
            os.makedirs(os.path.join(real_root, m), exist_ok=True)
            os.makedirs(os.path.join(fake_root, m), exist_ok=True)
    accelerator.wait_for_everyone()

    # shard indices across processes
    local_indices = list(range(accelerator.process_index, num_images, accelerator.num_processes))

    if accelerator.is_main_process:
        print(f"\n[{now_str()}] [1/6] Creating REAL folders (N={num_images}) under {real_root}")

    local_meta: List[Tuple[int, str, int, int]] = []

    pbar_real = tqdm(local_indices, disable=not accelerator.is_local_main_process, desc=f"REAL rank{accelerator.process_index}")
    for i in pbar_real:
        sdir = random.choice(subject_dirs)
        paths = subject_paths[sdir]
        vols = [cache.load(p) for p in paths]
        _, _, D = vols[0].shape

        z_start = int(0.1 * D) + SLICE_RADIUS
        z_end = int(0.9 * D) - SLICE_RADIUS
        if z_end <= z_start:
            continue
        z = random.randint(z_start, z_end - 1)

        for mi, m in enumerate(MODALITIES):
            t = preprocess_slice_np(vols[mi][:, :, z], IMAGE_SIZE)  # [-1,1]
            u8 = to_u8_from_minus1_1(t)
            save_gray_as_rgb_png(u8, os.path.join(real_root, m, f"real_{i:06d}.png"))

        local_meta.append((i, sdir, z, D))

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"\n[{now_str()}] [2/6] Creating FAKE folders (N={num_images}) under {fake_root}")

    local_meta.sort(key=lambda t: t[0])
    pbar_fake = tqdm(total=len(local_meta), disable=not accelerator.is_local_main_process, desc=f"FAKE rank{accelerator.process_index}")

    autocast_ctx = accelerator.autocast if hasattr(accelerator, "autocast") else nullcontext

    for batch in chunked(local_meta, BATCH_SIZE):
        ctx_list = []
        zpos_list = []
        out_indices = []

        for (gi, sdir, z, D) in batch:
            paths = subject_paths[sdir]
            vols = [cache.load(p) for p in paths]
            ctx_list.append(build_context_tensor(vols, z=z, radius=SLICE_RADIUS, image_size=IMAGE_SIZE, device=device))
            zpos_list.append(z / (D - 1))
            out_indices.append(gi)

        context_b = torch.cat(ctx_list, dim=0)  # (B,16,H,W) [-1,1]
        z_pos = torch.tensor(zpos_list, dtype=torch.float32, device=device)  # (B,)

        with autocast_ctx():
            fake01 = diffusion_sample_center_01(diffusion, z_pos=z_pos, context=context_b)  # (B,4,H,W) [0,1]

        for bi, gi in enumerate(out_indices):
            for mi, m in enumerate(MODALITIES):
                u8 = (fake01[bi, mi].detach().cpu().numpy() * 255.0).round().astype(np.uint8)
                save_gray_as_rgb_png(u8, os.path.join(fake_root, m, f"fake_{gi:06d}.png"))

        pbar_fake.update(len(batch))

    pbar_fake.close()
    accelerator.wait_for_everyone()

    config = {
        "BRATS_ROOT": str(BRATS_ROOT),
        "CHECKPOINT": CHECKPOINT,
        "OUT_DIR": OUT_DIR,
        "IMAGE_SIZE": IMAGE_SIZE,
        "TIMESTEPS": TIMESTEPS,
        "CENTER_MODALITIES": CENTER_MODALITIES,
        "SLICE_RADIUS": SLICE_RADIUS,
        "CONTEXT_SLICES": CONTEXT_SLICES,
        "IN_CHANNELS": IN_CHANNELS,
        "OUT_CHANNELS": OUT_CHANNELS,
        "BASE_CHANNELS": BASE_CHANNELS,
        "CHANNEL_MULTS": list(CHANNEL_MULTS),
        "TIME_EMB_DIM": TIME_EMB_DIM,
        "NUM_IMAGES": NUM_IMAGES,
        "BATCH_SIZE": BATCH_SIZE,
        "DIV_PAIRS": DIV_PAIRS,
        "DIVERSITY_RESIZE_TO": DIVERSITY_RESIZE_TO,
        "ISC_NUM_SEQUENCES": ISC_NUM_SEQUENCES,
        "ISC_WINDOW": ISC_WINDOW,
        "ISC_SAMPLE_BATCH": ISC_SAMPLE_BATCH,
        "DO_COND_RECON": DO_COND_RECON,
        "COND_RECON_SAMPLES": COND_RECON_SAMPLES,
        "COND_RECON_BATCH": COND_RECON_BATCH,
        "SEED": SEED,
        "DEVICE": DEVICE,
        "MIXED_PRECISION": mixed_precision,
        "DEBUG": DEBUG,
        "DEBUG_ONLY": DEBUG_ONLY,
        "SMOKE_FOLDER_METRICS": SMOKE_FOLDER_METRICS,
        "SMOKE_N": SMOKE_N,
        "DEBUG_CAP_NUM_IMAGES": DEBUG_CAP_NUM_IMAGES,
    }

    results: Dict = {"config": config}

    # Steps 3 & 4 remain main-only (folder metrics)
    if accelerator.is_main_process:
        print(f"\n[{now_str()}] [3/6] Computing FID/KID/PRC (torch-fidelity) per modality...")
        tf_by_mod = {}
        for m in MODALITIES:
            tf_by_mod[m] = torch_fidelity_folder_metrics(
                real_dir=os.path.join(real_root, m),
                fake_dir=os.path.join(fake_root, m),
                use_cuda=(device.type == "cuda"),
                kid_subset_size=1000,
                kid_subsets=100,
                num_workers=2,
                batch_size=16,
            )

        def macro_avg(key: str) -> float:
            vals = [tf_by_mod[m].get(key) for m in MODALITIES if isinstance(tf_by_mod[m].get(key), (int, float))]
            return float(np.mean(vals)) if vals else float("nan")

        tf_macro = {
            "fid_macro": macro_avg("frechet_inception_distance"),
            "kid_mean_macro": macro_avg("kernel_inception_distance_mean"),
            "kid_std_macro": macro_avg("kernel_inception_distance_std"),
            "precision_macro": macro_avg("precision"),
            "recall_macro": macro_avg("recall"),
        }

        results["torch_fidelity_by_modality"] = tf_by_mod
        results["torch_fidelity_macro"] = tf_macro

        print(f"\n[{now_str()}] [4/6] Computing diversity (MS-SSIM + LPIPS) per modality on fake set...")
        div_by_mod = {}
        for m in MODALITIES:
            div_by_mod[m] = diversity_metrics(
                fake_dir=os.path.join(fake_root, m),
                device=device,
                num_pairs=DIV_PAIRS,
                resize_to=DIVERSITY_RESIZE_TO,
            )

        def macro_div(key: str) -> float:
            vals = [div_by_mod[m].get(key) for m in MODALITIES if isinstance(div_by_mod[m].get(key), (int, float))]
            return float(np.mean(vals)) if vals else float("nan")

        div_macro = {
            "ms_ssim_pair_mean_macro": macro_div("ms_ssim_pair_mean"),
            "lpips_pair_mean_macro": macro_div("lpips_pair_mean"),
        }

        results["diversity_by_modality"] = div_by_mod
        results["diversity_macro"] = div_macro

        step4_path = save_results_json(OUT_DIR, "metrics_after_step4.json", results)
        print(f"[{now_str()}] Saved after Step 4 -> {step4_path}")

    accelerator.wait_for_everyone()

    # Step 5 distributed (all GPUs)
    if accelerator.is_main_process:
        print(f"\n[{now_str()}] [5/6] Computing inter-slice consistency using ALL GPUs...")

    isc = inter_slice_consistency_distributed(
        diffusion=diffusion,
        subject_dirs=subject_dirs,
        subject_paths=subject_paths,
        cache=cache,
        device=device,
        accelerator=accelerator,
        out_dir=OUT_DIR,
        num_sequences=ISC_NUM_SEQUENCES,
        window=ISC_WINDOW,
        sample_batch=ISC_SAMPLE_BATCH,
    )

    if accelerator.is_main_process:
        results["inter_slice_consistency"] = isc
        step5_path = save_results_json(OUT_DIR, "metrics_after_step5.json", results)
        print(f"[{now_str()}] Saved after Step 5 -> {step5_path}")

    accelerator.wait_for_everyone()

    # Step 6 distributed (all GPUs)
    cond_recon: Dict[str, float] = {}
    if DO_COND_RECON:
        if accelerator.is_main_process:
            print(f"\n[{now_str()}] [6/6] Computing conditional reconstruction using ALL GPUs...")

        cond_recon = conditional_reconstruction_distributed(
            diffusion=diffusion,
            subject_dirs=subject_dirs,
            subject_paths=subject_paths,
            cache=cache,
            device=device,
            accelerator=accelerator,
            num_samples=COND_RECON_SAMPLES,
            batch_size=COND_RECON_BATCH,
        )

    if accelerator.is_main_process:
        results["conditional_reconstruction"] = cond_recon

        final_path = save_results_json(OUT_DIR, "metrics_after_step6_final.json", results)
        metrics_path = save_results_json(OUT_DIR, "metrics.json", results)

        print("\n=== DONE ===")
        print(json.dumps(results, indent=2))
        print(f"\nSaved final metrics to: {final_path}")
        print(f"Also saved final metrics to: {metrics_path}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

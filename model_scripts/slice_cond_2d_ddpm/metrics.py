"""
Evaluate a slice-position-conditioned 2D DDPM on BraTS slices.

Computes:
  - FID, KID (global)
  - FID, KID per z_pos bin
  - Diversity: MS-SSIM (gen1 vs gen2), LPIPS (gen1 vs gen2)
  - Optional: Improved Precision/Recall (Kynkäänniemi et al.) on ResNet18 features

Assumptions:
  - This script is placed next to: dataset.py, unet.py, diffusion.py
  - Your dataset returns: (slice_t in [-1,1] of shape [1,H,W], z_pos in [0,1])
  - Your checkpoint is diffusion.state_dict() saved from training
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .dataset import BraTSSliceDataset
from .unet import UNet
from .diffusion import GaussianDiffusion


def seed_all(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_01(x: torch.Tensor) -> torch.Tensor:
    # model/data are in [-1,1]
    x = (x + 1.0) * 0.5
    return x.clamp(0.0, 1.0)


def to_3ch(x: torch.Tensor) -> torch.Tensor:
    # x: (N,C,H,W)
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x


def fix_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Handle checkpoints saved when diffusion.model was wrapped in DataParallel.
    Typical prefix patterns:
      - "model.module.<...>"
      - sometimes "module.<...>"
    """
    out = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("model.module."):
            nk = "model." + nk[len("model.module.") :]
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        out[nk] = v
    return out


def volume_split_indices(dataset: BraTSSliceDataset, seed: int, test_frac: float) -> List[int]:
    """
    Split by *volume* so slices from the same patient/volume don't leak across sets.
    Returns indices for the test subset.
    """
    rng = random.Random(seed)
    vol_paths = list(dataset.volume_paths)
    rng.shuffle(vol_paths)

    n_test = max(1, int(round(len(vol_paths) * test_frac)))
    test_paths = set(vol_paths[:n_test])

    test_indices = [i for i, (p, _z) in enumerate(dataset.slice_tuples) if p in test_paths]
    return test_indices


@torch.no_grad()
def sample_like_real_batch(
    diffusion: GaussianDiffusion,
    z_pos: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a batch conditioned on the provided z_pos values.
    Returns tensor in [-1,1], shape (B,1,H,W)
    """
    z_pos = z_pos.to(device).float()
    return diffusion.sample(batch_size=z_pos.shape[0], z_pos=z_pos)


def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def safe_std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = safe_mean(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def build_torchmetrics(device: torch.device):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError as e:
        raise SystemExit(
            "Missing deps. Install with:\n"
            "  pip install torchmetrics torchvision torch-fidelity\n\n"
            f"Original error: {e}"
        )

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    # KID: subset_size must be <= number of samples seen; we’ll set later safely.
    kid = KernelInceptionDistance(feature=2048, subsets=50, subset_size=500, normalize=True).to(device)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    # LPIPS default normalize=False => expects [-1,1]
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)

    return fid, kid, ssim, lpips


def pr_features_resnet18(x01_3ch: torch.Tensor) -> torch.Tensor:
    """
    Extract features using a torchvision ResNet18 backbone (global average pooled output).
    x01_3ch: float in [0,1], shape (N,3,H,W)
    returns: (N,512)
    """
    from torchvision.models import resnet18, ResNet18_Weights
    import torch.nn as nn

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()

    return model


def imagenet_normalize(x01_3ch: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x01_3ch.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x01_3ch.device).view(1, 3, 1, 1)
    return (x01_3ch - mean) / std


def knn_radii(feats: torch.Tensor, k: int = 3, chunk: int = 256) -> torch.Tensor:
    """
    For each feature vector, compute distance to its k-th nearest neighbor within the same set.
    feats: (N,D)
    Returns radii: (N,)
    """
    N = feats.shape[0]
    radii = torch.empty((N,), device=feats.device, dtype=feats.dtype)

    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        x = feats[i:j]  # (B,D)
        d = torch.cdist(x, feats)  # (B,N)
        idx = torch.arange(i, j, device=feats.device)
        d[torch.arange(j - i, device=feats.device), idx] = float("inf")  # exclude self
        radii[idx] = torch.kthvalue(d, k, dim=1).values

    return radii


def membership_fraction(
    queries: torch.Tensor,
    manifold: torch.Tensor,
    manifold_radii: torch.Tensor,
    q_chunk: int = 128,
) -> float:
    """
    Fraction of queries that fall inside the union of hyperspheres around manifold points,
    where each manifold point has radius manifold_radii.
    """
    Nq = queries.shape[0]
    inside_count = 0

    for i in range(0, Nq, q_chunk):
        j = min(i + q_chunk, Nq)
        q = queries[i:j]  # (B,D)
        d = torch.cdist(q, manifold)  # (B,Nm)
        inside = (d <= manifold_radii.unsqueeze(0)).any(dim=1)
        inside_count += int(inside.sum().item())

    return inside_count / max(1, Nq)


@torch.no_grad()
def compute_improved_pr(
    real_feats: torch.Tensor,
    fake_feats: torch.Tensor,
    k: int = 3,
) -> Tuple[float, float]:
    """
    Improved precision/recall from Kynkäänniemi et al.:
      precision: fraction of fake within real manifold
      recall:    fraction of real within fake manifold
    """
    r_radii = knn_radii(real_feats, k=k)
    f_radii = knn_radii(fake_feats, k=k)

    precision = membership_fraction(fake_feats, real_feats, r_radii)
    recall = membership_fraction(real_feats, fake_feats, f_radii)
    return float(precision), float(recall)


def main() -> None:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATASET_ROOT = (PROJECT_ROOT / "../datasets/dataset").resolve()
    EXPERIMENTS_ROOT = PROJECT_ROOT / "slice_cond_2d_ddpm"
    MODEL_ROOT = EXPERIMENTS_ROOT / "models"
    
    # Configuration variables
    dataset_root = str(DATASET_ROOT)
    ckpt_path = str(MODEL_ROOT / "1591624" /"2d_central_ddpm_flair_best.pt")
    image_size = 128
    timesteps = None  # If None, inferred from checkpoint betas length
    batch_size = 16
    num_workers = 0
    num_samples = 1000  # How many real samples to compare (and how many generated)
    test_frac = 0.15  # Volume-level test split fraction
    seed = 42
    device_str = None
    
    z_bins = 8
    diversity_pairs = 64  # How many (gen1,gen2) pairs for MS-SSIM/LPIPS
    save_samples = False
    out_dir_path = "./eval_out"
    
    compute_pr = False  # Compute improved precision/recall (slower)
    pr_samples = 500  # How many samples for PR (real & fake). Keep <= 2000 for speed
    
    # Create simple namespace object to hold args
    class Args:
        pass
    
    args = Args()
    args.dataset_root = dataset_root
    args.ckpt = ckpt_path
    args.image_size = image_size
    args.timesteps = timesteps
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.num_samples = num_samples
    args.test_frac = test_frac
    args.seed = seed
    args.device = device_str
    args.z_bins = z_bins
    args.diversity_pairs = diversity_pairs
    args.save_samples = save_samples
    args.out_dir = out_dir_path
    args.compute_pr = compute_pr
    args.pr_samples = pr_samples

    seed_all(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint first to infer timesteps if needed
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise SystemExit("Checkpoint does not look like a diffusion.state_dict() dict.")
    ckpt = fix_state_dict_keys(ckpt)

    if args.timesteps is None:
        if "betas" not in ckpt:
            raise SystemExit("Could not infer timesteps: checkpoint missing 'betas' buffer.")
        args.timesteps = int(ckpt["betas"].numel())

    # Build model & diffusion
    model = UNet(
        img_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256,
    )
    diffusion = GaussianDiffusion(
        model=model,
        image_size=args.image_size,
        channels=1,
        timesteps=args.timesteps,
    )
    missing, unexpected = diffusion.load_state_dict(ckpt, strict=False)
    diffusion = diffusion.to(device).eval()

    print(f"Loaded ckpt. missing={len(missing)} unexpected={len(unexpected)} timesteps={args.timesteps}")

    # Dataset (volume-level test split)
    dataset = BraTSSliceDataset(args.dataset_root, image_size=args.image_size)
    test_indices = volume_split_indices(dataset, seed=args.seed, test_frac=args.test_frac)
    test_ds = Subset(dataset, test_indices)

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Metrics
    fid, kid, ssim, lpips = build_torchmetrics(device)

    # Per-z-bin metrics
    z_bins = int(args.z_bins)
    edges = torch.linspace(0.0, 1.0, z_bins + 1)
    fid_bins = [type(fid)(feature=2048, normalize=True).to(device) for _ in range(z_bins)]
    kid_bins = [type(kid)(feature=2048, subsets=20, subset_size=200, normalize=True).to(device) for _ in range(z_bins)]

    # Optionally save some samples
    if args.save_samples:
        try:
            from torchvision.utils import save_image
        except ImportError:
            save_image = None
            print("torchvision not available; cannot save samples.")

    # Main loop: stream updates into FID/KID
    seen = 0
    for real, z_pos in loader:
        if seen >= args.num_samples:
            break

        b = real.shape[0]
        take = min(b, args.num_samples - seen)
        real = real[:take]
        z_pos = z_pos[:take]

        # Generate matched-condition fakes
        fake = sample_like_real_batch(diffusion, z_pos=z_pos, device=device)

        # FID/KID expect 3ch RGB; normalize=True => float in [0,1]
        real_01_3 = to_3ch(to_01(real.to(device)))
        fake_01_3 = to_3ch(to_01(fake))

        fid.update(real_01_3, real=True)
        fid.update(fake_01_3, real=False)
        kid.update(real_01_3, real=True)
        kid.update(fake_01_3, real=False)

        # Per-bin
        z = z_pos.cpu()
        bin_idx = torch.bucketize(z, edges[1:-1], right=False)  # 0..z_bins-1
        for bi in range(z_bins):
            m = (bin_idx == bi)
            if m.any():
                rb = real_01_3[m.to(device)]
                fb = fake_01_3[m.to(device)]
                fid_bins[bi].update(rb, real=True)
                fid_bins[bi].update(fb, real=False)
                kid_bins[bi].update(rb, real=True)
                kid_bins[bi].update(fb, real=False)

        # Save a small grid (first batch only)
        if args.save_samples and seen == 0:
            if save_image is not None:
                save_image(to_01(real.to(device)), out_dir / "real_examples.png", nrow=8)
                save_image(to_01(fake), out_dir / "fake_examples.png", nrow=8)

        seen += take
        if seen % (args.batch_size * 10) == 0:
            print(f"Processed {seen}/{args.num_samples} samples...")

    # Compute global FID/KID
    fid_score = float(fid.compute().detach().cpu().item())
    kid_mean, kid_std = kid.compute()
    kid_mean = float(kid_mean.detach().cpu().item())
    kid_std = float(kid_std.detach().cpu().item())

    # Compute per-bin
    per_bin = {}
    for bi in range(z_bins):
        # If a bin got no samples, compute() may error; guard it.
        try:
            f = float(fid_bins[bi].compute().detach().cpu().item())
        except Exception:
            f = None
        try:
            km, ks = kid_bins[bi].compute()
            km = float(km.detach().cpu().item())
            ks = float(ks.detach().cpu().item())
        except Exception:
            km, ks = None, None

        per_bin[f"bin_{bi}"] = {
            "z_range": [float(edges[bi].item()), float(edges[bi + 1].item())],
            "fid": f,
            "kid_mean": km,
            "kid_std": ks,
        }

    # Diversity: generate (gen1, gen2) pairs at random z positions
    ssim_vals: List[float] = []
    lpips_vals: List[float] = []
    pairs_done = 0

    # sample z_pos uniformly across [0,1]
    while pairs_done < args.diversity_pairs:
        b = min(args.batch_size, args.diversity_pairs - pairs_done)
        z_pos = torch.rand((b,), device=device)

        g1 = diffusion.sample(batch_size=b, z_pos=z_pos)  # [-1,1]
        g2 = diffusion.sample(batch_size=b, z_pos=z_pos)  # [-1,1]

        g1_01 = to_01(g1)
        g2_01 = to_01(g2)

        # MS-SSIM in [0,1], provide data_range=1.0
        s = ssim(to_3ch(g1_01), to_3ch(g2_01))
        ssim_vals.append(float(s.detach().cpu().item()))

        # LPIPS expects (N,3,H,W), default normalize=False => [-1,1]
        lp = lpips(to_3ch(g1), to_3ch(g2))
        lpips_vals.append(float(lp.detach().cpu().item()))

        pairs_done += b

    # Optional improved precision/recall
    pr_precision = None
    pr_recall = None
    if args.compute_pr:
        from torchvision.models import resnet18, ResNet18_Weights
        import torch.nn as nn

        feat_net = resnet18(weights=ResNet18_Weights.DEFAULT)
        feat_net.fc = nn.Identity()
        feat_net = feat_net.to(device).eval()

        def extract_feats(stream_real: bool, n: int) -> torch.Tensor:
            feats = []
            got = 0
            for real, z_pos in loader:
                if got >= n:
                    break
                b = real.shape[0]
                take = min(b, n - got)
                real = real[:take]
                z_pos = z_pos[:take]

                if stream_real:
                    x = to_3ch(to_01(real.to(device)))
                else:
                    x = to_3ch(to_01(sample_like_real_batch(diffusion, z_pos, device)))

                # ResNet expects ~ImageNet normalization
                x = imagenet_normalize(x)
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

                f = feat_net(x)  # (B,512)
                feats.append(f.detach())
                got += take
            return torch.cat(feats, dim=0)

        npr = int(args.pr_samples)
        real_feats = extract_feats(True, npr)
        fake_feats = extract_feats(False, npr)

        pr_precision, pr_recall = compute_improved_pr(real_feats, fake_feats, k=3)

    results = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "num_samples": int(seen),
        "fid": fid_score,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "per_z_bin": per_bin,
        "diversity": {
            "ssim_mean": safe_mean(ssim_vals),
            "ssim_std": safe_std(ssim_vals),
            "lpips_mean": safe_mean(lpips_vals),
            "lpips_std": safe_std(lpips_vals),
            "pairs": int(args.diversity_pairs),
        },
        "improved_precision_recall": {
            "enabled": bool(args.compute_pr),
            "precision": pr_precision,
            "recall": pr_recall,
            "k": 3,
            "samples": int(args.pr_samples) if args.compute_pr else None,
        },
        "notes": {
            "fid_kid_inputs": "FID/KID computed on float images in [0,1] (normalize=True) and repeated to 3 channels.",
            "lpips_inputs": "LPIPS computed on images in [-1,1] (normalize=False) and repeated to 3 channels.",
        },
    }

    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    print("\n=== Results ===")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
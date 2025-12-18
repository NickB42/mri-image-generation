"""
Evaluate one or more slice-position-conditioned DDPMs on BraTS slices.

Works with:
- basic dataset: returns (real, z_pos)
- context dataset: returns (real, context, z_pos)

Writes:
- eval_out/<model_name>_metrics.json
- eval_out/summary.json
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .dataset import BraTSSliceDataset  # basic dataset
from .unet import UNet
from .diffusion import GaussianDiffusion

from ..ddpm_25d_all_modalities.dataset import BraTSSliceDataset as BraTSSliceContextDataset
from ..ddpm_25d_all_modalities.unet import UNet as UNetContext
from ..ddpm_25d_all_modalities.diffusion import GaussianDiffusion as GaussianDiffusionContext

# -----------------------------
# Helpers
# -----------------------------
def seed_all(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_variant_classes(variant: str):
    if variant == "basic":
        return BraTSSliceDataset, UNet, GaussianDiffusion
    if variant == "context":
        return BraTSSliceContextDataset, UNetContext, GaussianDiffusionContext
    raise ValueError(f"Unknown variant: {variant}")


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
    # if you accidentally pass >1 channels here, it's ambiguous for FID/KID
    raise ValueError(f"to_3ch expects 1 channel input, got C={x.shape[1]}. "
                     f"Use eval_channel to select one target channel.")


def fix_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("model.module."):
            nk = "model." + nk[len("model.module.") :]
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        out[nk] = v
    return out


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

    # Keep KID smaller to avoid memory issues on HPC.
    kid = KernelInceptionDistance(feature=2048, subsets=20, subset_size=200, normalize=True).to(device)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)

    return fid, kid, ssim, lpips


def volume_split_indices(dataset: Any, seed: int, test_frac: float) -> List[int]:
    """
    Split by *volume* so slices from the same patient/volume don't leak across sets.
    Requires dataset.volume_paths and dataset.slice_tuples.
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
    context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    z_pos = z_pos.to(device).float()
    if context is not None:
        context = context.to(device, non_blocking=True)
        return diffusion.sample(batch_size=z_pos.shape[0], z_pos=z_pos, context=context)
    else:
        return diffusion.sample(batch_size=z_pos.shape[0], z_pos=z_pos)


def build_unet_for_config(UNetCls, c_target: int, c_context: int, use_context: bool):
    in_ch = c_target + (c_context if use_context else 0)
    out_ch = c_target

    print(str(UNetCls))
    try:
        return UNetCls(
            in_channels=in_ch,
            out_channels=out_ch,
            base_channels=64,
            channel_mults=(1, 2, 4, 8),
            time_emb_dim=256,
        )
    except TypeError:
        if use_context or in_ch != out_ch:
            raise
        return UNetCls(
            img_channels=out_ch,
            base_channels=64,
            channel_mults=(1, 2, 4, 8),
            time_emb_dim=256,
        )


def select_eval_channel(x: torch.Tensor, eval_channel: int) -> torch.Tensor:
    """
    x: (B, C, H, W) -> (B, 1, H, W) selecting one channel for FID/KID/SSIM/LPIPS
    """
    if x.ndim != 4:
        raise ValueError(f"Expected (B,C,H,W), got shape {tuple(x.shape)}")
    c = x.shape[1]
    if c == 1:
        return x
    if not (0 <= eval_channel < c):
        raise ValueError(f"eval_channel={eval_channel} out of range for C={c}")
    return x[:, eval_channel : eval_channel + 1, :, :]

def imagenet_normalize(x01_3ch: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x01_3ch.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x01_3ch.device).view(1, 3, 1, 1)
    return (x01_3ch - mean) / std


def knn_radii(feats: torch.Tensor, k: int = 3, chunk: int = 256) -> torch.Tensor:
    N = feats.shape[0]
    radii = torch.empty((N,), device=feats.device, dtype=feats.dtype)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        x = feats[i:j]
        d = torch.cdist(x, feats)
        idx = torch.arange(i, j, device=feats.device)
        d[torch.arange(j - i, device=feats.device), idx] = float("inf")
        radii[idx] = torch.kthvalue(d, k, dim=1).values
    return radii


def membership_fraction(
    queries: torch.Tensor,
    manifold: torch.Tensor,
    manifold_radii: torch.Tensor,
    q_chunk: int = 128,
) -> float:
    Nq = queries.shape[0]
    inside_count = 0
    for i in range(0, Nq, q_chunk):
        j = min(i + q_chunk, Nq)
        q = queries[i:j]
        d = torch.cdist(q, manifold)
        inside = (d <= manifold_radii.unsqueeze(0)).any(dim=1)
        inside_count += int(inside.sum().item())
    return inside_count / max(1, Nq)


@torch.no_grad()
def compute_improved_pr(
    real_feats: torch.Tensor,
    fake_feats: torch.Tensor,
    k: int = 3,
) -> Tuple[float, float]:
    r_radii = knn_radii(real_feats, k=k)
    f_radii = knn_radii(fake_feats, k=k)
    precision = membership_fraction(fake_feats, real_feats, r_radii)
    recall = membership_fraction(real_feats, fake_feats, f_radii)
    return float(precision), float(recall)



# -----------------------------
# Core evaluation
# -----------------------------
def evaluate_one_model(
    *,
    variant: str,
    model_name: str,
    ckpt_path: str,
    dataset: Any,
    image_size: int,
    c_target: int,
    c_context: int,
    use_context: bool,
    eval_channel: int,
    timesteps: Optional[int],
    batch_size: int,
    num_workers: int,
    num_samples: int,
    test_frac: float,
    seed: int,
    device: torch.device,
    z_bins: int,
    diversity_pairs: int,
    save_samples: bool,
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # breadcrumbs
    (out_dir / f"{model_name}_RUNNING.txt").write_text("started\n")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise SystemExit(f"{model_name}: checkpoint does not look like a state_dict dict.")
    ckpt = fix_state_dict_keys(ckpt)

    if timesteps is None:
        if "betas" not in ckpt:
            raise SystemExit(f"{model_name}: could not infer timesteps (missing betas).")
        timesteps = int(ckpt["betas"].numel())

    DatasetCls, UNetCls, DiffusionCls = get_variant_classes(variant)
    
    model = build_unet_for_config(UNetCls, c_target=c_target, c_context=c_context, use_context=use_context)

    diffusion = DiffusionCls(
        model=model,
        image_size=image_size,
        channels=c_target,
        timesteps=timesteps,
    )
    missing, unexpected = diffusion.load_state_dict(ckpt, strict=False)
    diffusion = diffusion.to(device).eval()

    print(f"[{model_name}] Loaded ckpt. missing={len(missing)} unexpected={len(unexpected)} timesteps={timesteps}")
    if len(missing) > 0:
        print(f"[{model_name}] Missing keys (first 20): {missing[:20]}")

    test_indices = volume_split_indices(dataset, seed=seed, test_frac=test_frac)
    test_ds = Subset(dataset, test_indices)

    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    fid, kid, ssim, lpips = build_torchmetrics(device)

    edges = torch.linspace(0.0, 1.0, z_bins + 1)
    fid_bins = [type(fid)(feature=2048, normalize=True).to(device) for _ in range(z_bins)]
    kid_bins = [type(kid)(feature=2048, subsets=10, subset_size=50, normalize=True).to(device) for _ in range(z_bins)]

    if save_samples:
        try:
            from torchvision.utils import save_image
        except ImportError:
            save_image = None
            print("torchvision not available; cannot save samples.")
    else:
        save_image = None

    seen = 0
    for batch in loader:
        if seen >= num_samples:
            break

        # Support both dataset formats:
        #   (real, z_pos)
        #   (real, context, z_pos)
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            real, z_pos = batch
            context = None
        elif isinstance(batch, (tuple, list)) and len(batch) == 3:
            real, context, z_pos = batch
        else:
            raise ValueError(
                f"{model_name}: Expected batch of len 2 or 3, got type={type(batch)} len={len(batch) if hasattr(batch,'__len__') else '??'}"
            )

        b = real.shape[0]
        take = min(b, num_samples - seen)
        real = real[:take]
        z_pos = z_pos[:take]
        if context is not None:
            context = context[:take]

        # Generate matched-condition fakes
        fake = sample_like_real_batch(diffusion, z_pos=z_pos, device=device, context=context)

        # Select a single channel for evaluation (important if c_target > 1)
        real_eval = select_eval_channel(real.to(device), eval_channel)
        fake_eval = select_eval_channel(fake, eval_channel)

        real_01_3 = to_3ch(to_01(real_eval))
        fake_01_3 = to_3ch(to_01(fake_eval))

        fid.update(real_01_3, real=True)
        fid.update(fake_01_3, real=False)
        kid.update(real_01_3, real=True)
        kid.update(fake_01_3, real=False)

        # Per-bin
        z = z_pos.cpu()
        bin_idx = torch.bucketize(z, edges[1:-1], right=False)
        for bi in range(z_bins):
            m = (bin_idx == bi)
            if m.any():
                rb = real_01_3[m.to(device)]
                fb = fake_01_3[m.to(device)]
                fid_bins[bi].update(rb, real=True)
                fid_bins[bi].update(fb, real=False)
                try:
                    kid_bins[bi].update(rb, real=True)
                    kid_bins[bi].update(fb, real=False)
                except Exception:
                    pass

        if save_image is not None and seen == 0:
            save_image(to_01(real_eval), out_dir / f"{model_name}_real_examples.png", nrow=8)
            save_image(to_01(fake_eval), out_dir / f"{model_name}_fake_examples.png", nrow=8)

        seen += take
        if seen % max(1, (batch_size * 10)) == 0:
            print(f"[{model_name}] Processed {seen}/{num_samples} samples...")
            (out_dir / f"{model_name}_progress.txt").write_text(f"seen={seen}\n")

    fid_score = float(fid.compute().detach().cpu().item())
    kid_mean, kid_std = kid.compute()
    kid_mean = float(kid_mean.detach().cpu().item())
    kid_std = float(kid_std.detach().cpu().item())

    per_bin = {}
    for bi in range(z_bins):
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

    # Diversity: generate pairs at random z positions.
    # For context models, we *need* a context tensor. Easiest: reuse real batches to provide context.
    ssim_vals: List[float] = []
    lpips_vals: List[float] = []
    pairs_done = 0

    # Make a small iterator to grab context when needed
    loader_iter = iter(loader)

    while pairs_done < diversity_pairs:
        b = min(batch_size, diversity_pairs - pairs_done)

        if use_context:
            # pull one batch from loader to get context + matching z_pos
            try:
                batch2 = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch2 = next(loader_iter)

            if len(batch2) != 3:
                raise ValueError(f"{model_name}: use_context=True but dataset did not return (real, context, z_pos).")

            _real2, ctx2, z2 = batch2
            ctx2 = ctx2[:b].to(device, non_blocking=True)
            z_ctx = z2[:b].to(device).float()

            g1 = diffusion.sample(batch_size=b, z_pos=z_ctx, context=ctx2)
            g2 = diffusion.sample(batch_size=b, z_pos=z_ctx, context=ctx2)

        else:
            # unconditional on context: just random z positions
            z_rand = torch.rand((b,), device=device)
            g1 = diffusion.sample(batch_size=b, z_pos=z_rand)
            g2 = diffusion.sample(batch_size=b, z_pos=z_rand)

        g1_eval = select_eval_channel(g1, eval_channel)
        g2_eval = select_eval_channel(g2, eval_channel)

        g1_01 = to_01(g1_eval)
        g2_01 = to_01(g2_eval)

        s = ssim(to_3ch(g1_01), to_3ch(g2_01))
        ssim_vals.append(float(s.detach().cpu().item()))

        lp = lpips(to_3ch(g1_eval), to_3ch(g2_eval))
        lpips_vals.append(float(lp.detach().cpu().item()))

        pairs_done += b

    results = {
        "model_name": model_name,
        "ckpt": str(Path(ckpt_path).resolve()),
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
            "pairs": int(diversity_pairs),
        },
        "config": {
            "use_context": bool(use_context),
            "c_target": int(c_target),
            "c_context": int(c_context),
            "eval_channel": int(eval_channel),
            "image_size": int(image_size),
            "timesteps": int(timesteps),
            "batch_size": int(batch_size),
        },
        "notes": {
            "fid_kid_inputs": "FID/KID computed on float images in [0,1] (normalize=True) and repeated to 3 channels.",
            "lpips_inputs": "LPIPS computed on images in [-1,1] (normalize=False) and repeated to 3 channels.",
        },
    }

    out_path = out_dir / f"{model_name}_metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    (out_dir / f"{model_name}_RUNNING.txt").write_text("finished\n")

    print(f"\n[{model_name}] Saved to: {out_path}")
    return results, diffusion, loader


@torch.no_grad()
def compute_pr_for_model(
    *,
    model_name: str,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    device: torch.device,
    eval_channel: int,
    pr_samples: int,
    use_context: bool,
) -> Dict[str, Any]:
    from torchvision.models import resnet18, ResNet18_Weights
    import torch.nn as nn

    feat_net = resnet18(weights=ResNet18_Weights.DEFAULT)
    feat_net.fc = nn.Identity()
    feat_net = feat_net.to(device).eval()

    def extract_feats(stream_real: bool) -> torch.Tensor:
        feats = []
        got = 0

        for batch in loader:
            if got >= pr_samples:
                break

            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                real, z_pos = batch
                context = None
            elif isinstance(batch, (tuple, list)) and len(batch) == 3:
                real, context, z_pos = batch
            else:
                raise ValueError(f"{model_name}: batch must have len 2 or 3")

            b = real.shape[0]
            take = min(b, pr_samples - got)

            real = real[:take].to(device)
            z_pos = z_pos[:take].to(device).float()
            if context is not None:
                context = context[:take].to(device, non_blocking=True)

            if stream_real:
                x = select_eval_channel(real, eval_channel)         # (B,1,H,W)
            else:
                fake = diffusion.sample(batch_size=take, z_pos=z_pos, context=context)
                x = select_eval_channel(fake, eval_channel)         # (B,1,H,W)

            # To 3ch [0,1] and ImageNet normalize for ResNet
            x01_3 = to_3ch(to_01(x))
            x01_3 = imagenet_normalize(x01_3)
            x01_3 = F.interpolate(x01_3, size=(224, 224), mode="bilinear", align_corners=False)

            f = feat_net(x01_3)  # (B,512)
            feats.append(f.detach())
            got += take

        return torch.cat(feats, dim=0)

    real_feats = extract_feats(stream_real=True)
    fake_feats = extract_feats(stream_real=False)

    precision, recall = compute_improved_pr(real_feats, fake_feats, k=3)
    return {
        "model_name": model_name,
        "enabled": True,
        "k": 3,
        "samples": int(pr_samples),
        "precision": precision,
        "recall": recall,
    }


def main() -> None:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATASET_ROOT = (PROJECT_ROOT / "../datasets/dataset").resolve()
    EXPERIMENTS_ROOT = PROJECT_ROOT / "slice_cond_2d_ddpm"
    MODEL_ROOT = EXPERIMENTS_ROOT / "models"

    MODEL_CONFIGS = [
        # {
        #     "model_name": "basic_flair",
        #     "variant": "basic",
        #     "ckpt_path": str(MODEL_ROOT / "1591624" / "2d_central_ddpm_flair_best.pt"),
        #     "dataset_kwargs": {"image_size": 128},
        #     "use_context": False,
        #     "c_target": 1,
        #     "c_context": 0,
        #     "eval_channel": 0,
        # },
        {
            "model_name": "context_4mod",
            "variant": "context",
            "ckpt_path": str(PROJECT_ROOT / "ddpm_25d_all_modalities" / "models" / "1591706" / "25d_ddpm_all_modalities_best.pt"),
            "dataset_kwargs": {"image_size": 128, "slice_radius": 2},
            "use_context": True,
            "c_target": 4,
            "c_context": 16,
            "eval_channel": 3,
        },
    ]


    # Common settings
    image_size = 128
    timesteps = None
    batch_size = 16
    num_workers = 0
    num_samples = 1000
    test_frac = 0.15
    seed = 42
    device_str = None

    z_bins = 8
    diversity_pairs = 64
    save_samples = False
    out_dir = Path("./eval_out")

    compute_pr = True
    pr_samples = 300

    seed_all(seed)
    device = pick_device(device_str)

    # Build datasets once per config (so context model can use its own dataset)
    all_results = []
    handles_for_pr = []  # store (cfg, diffusion, loader)

    for cfg in MODEL_CONFIGS:
        DatasetCls, _, _ = get_variant_classes(cfg["variant"])
        print(f"Building dataset for model: {cfg['model_name']}")
        print(f"  variant: {cfg['variant']}")
        ds = DatasetCls(str(DATASET_ROOT), **cfg.get("dataset_kwargs", {}))


        res, diffusion, loader = evaluate_one_model(
            variant=cfg["variant"],
            model_name=cfg["model_name"],
            ckpt_path=cfg["ckpt_path"],
            dataset=ds,
            image_size=image_size,
            c_target=cfg["c_target"],
            c_context=cfg["c_context"],
            use_context=cfg["use_context"],
            eval_channel=cfg["eval_channel"],
            timesteps=timesteps,
            batch_size=batch_size,
            num_workers=num_workers,
            num_samples=num_samples,
            test_frac=test_frac,
            seed=seed,
            device=device,
            z_bins=z_bins,
            diversity_pairs=diversity_pairs,
            save_samples=save_samples,
            out_dir=out_dir,
        )
        all_results.append(res)
        handles_for_pr.append((cfg, diffusion, loader))

    # ---- Write summary FIRST ----
    (out_dir / "summary.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved summary to: {out_dir / 'summary.json'}")

    if compute_pr:
        pr_results = []
        for cfg, diffusion, loader in handles_for_pr:
            pr = compute_pr_for_model(
                model_name=cfg["model_name"],
                diffusion=diffusion,
                loader=loader,
                device=device,
                eval_channel=cfg["eval_channel"],
                pr_samples=pr_samples,
                use_context=cfg["use_context"],
            )
            pr_results.append(pr)

        (out_dir / "precision_recall.json").write_text(json.dumps(pr_results, indent=2))
        print(f"Saved precision/recall to: {out_dir / 'precision_recall.json'}")

if __name__ == "__main__":
    main()

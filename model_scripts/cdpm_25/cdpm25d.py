# cdpm25d_model.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Helpers
# ---------------------------

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract a[t] for batch indices t and reshape for broadcasting to x."""
    B = t.shape[0]
    out = a.gather(0, t)  # [B]
    return out.view(B, *([1] * (len(x_shape) - 1)))


# ---------------------------
# Beta schedules
# ---------------------------

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


# ---------------------------
# Gaussian Diffusion (epsilon prediction)
# ---------------------------

class GaussianDiffusion(nn.Module):
    def __init__(self, T: int = 1000):
        super().__init__()
        self.T = int(T)

        betas = linear_beta_schedule(self.T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))

        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor):
        x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred).clamp(0.0, 1.0)
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        mean, _, log_var = self.p_mean_variance(x_t, t, eps_pred)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(t.shape[0], *([1] * (len(x_t.shape) - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample_targets(
        self,
        model: nn.Module,
        *,
        cond_slices: torch.Tensor,   # [B, Cn, H, W] can be empty
        cond_idx: torch.Tensor,      # [B, Cn]
        target_idx: torch.Tensor,    # [B, P]
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        B, P = target_idx.shape
        x = torch.randn(B, P, H, W, device=device)

        for ti in reversed(range(self.T)):
            t = torch.full((B,), ti, device=device, dtype=torch.long)
            eps_pred = model(
                x_target_t=x,
                x_cond=cond_slices,
                idx_target=target_idx,
                idx_cond=cond_idx,
                t=t,
            )
            x = self.p_sample(x, t, eps_pred)
        return x


# ---------------------------
# Embeddings
# ---------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dtype != torch.float32:
            t = t.float()
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ---------------------------
# U-Net blocks
# ---------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class AxialSliceAttention(nn.Module):
    """Self-attention across slice axis S at each (h,w)."""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, B: int, S: int) -> torch.Tensor:
        BS, C, H, W = x.shape
        x_seq = x.view(B, S, C, H, W).permute(0, 3, 4, 1, 2).reshape(B * H * W, S, C)
        residual = x_seq
        x_seq = self.norm(x_seq)
        x_attn, _ = self.attn(x_seq, x_seq, x_seq, need_weights=False)
        x_seq = residual + x_attn
        x_out = x_seq.reshape(B, H, W, S, C).permute(0, 3, 4, 1, 2).reshape(B * S, C, H, W)
        return x_out


class UNet2DWithSliceConditioning(nn.Module):
    def __init__(
        self,
        *,
        max_depth: int,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        attn_heads: int = 4,
    ):
        super().__init__()
        self.max_depth = max_depth

        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # slice index + slice type (cond vs target)
        self.slice_idx_embed = nn.Embedding(max_depth, time_emb_dim)
        self.slice_type_embed = nn.Embedding(2, time_emb_dim)  # 0=cond, 1=target

        self.init_conv = nn.Conv2d(1, base_channels, 3, padding=1)

        # Down
        self.down_blocks = nn.ModuleList()
        self.skip_channels: List[int] = []
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, time_emb_dim))
                ch = out_ch
                self.skip_channels.append(ch)
            if i != len(channel_mults) - 1:
                self.down_blocks.append(Downsample(ch))
                self.skip_channels.append(ch)

        # Mid
        self.mid1 = ResBlock(ch, ch, time_emb_dim)
        self.mid_attn = AxialSliceAttention(ch, num_heads=attn_heads)
        self.mid2 = ResBlock(ch, ch, time_emb_dim)

        # Up
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                skip_ch = self.skip_channels.pop()
                self.up_blocks.append(ResBlock(ch + skip_ch, out_ch, time_emb_dim))
                ch = out_ch
            if i != 0:
                skip_ch = self.skip_channels.pop()
                self.up_blocks.append(ResBlock(ch + skip_ch, ch, time_emb_dim))
                self.up_blocks.append(Upsample(ch))

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(
        self,
        *,
        x_all: torch.Tensor,     # [B, S, H, W]
        idx_all: torch.Tensor,   # [B, S] in [0, D-1]
        type_all: torch.Tensor,  # [B, S] 0=cond 1=target
        t: torch.Tensor,         # [B]
    ) -> torch.Tensor:
        B, S, H, W = x_all.shape
        x = x_all.reshape(B * S, 1, H, W)
        x = self.init_conv(x)

        temb = self.time_mlp(self.time_embed(t))             # [B, E]
        temb = temb.repeat_interleave(S, dim=0)              # [B*S, E]
        emb = temb + self.slice_idx_embed(idx_all.reshape(-1)) + self.slice_type_embed(type_all.reshape(-1))

        skips: List[torch.Tensor] = []
        h = x
        for blk in self.down_blocks:
            if isinstance(blk, ResBlock):
                h = blk(h, emb)
                skips.append(h)
            else:
                h = blk(h)
                skips.append(h)

        h = self.mid1(h, emb)
        h = self.mid_attn(h, B=B, S=S)
        h = self.mid2(h, emb)

        for blk in self.up_blocks:
            if isinstance(blk, ResBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                h = blk(h, emb)
            else:
                h = blk(h)

        out = self.out_conv(F.silu(self.out_norm(h)))
        return out.reshape(B, S, H, W)


class CDPM25D(nn.Module):
    def __init__(self, *, tau_max: int, volume_depth: int, **unet_kwargs):
        super().__init__()
        self.tau_max = int(tau_max)
        self.volume_depth = int(volume_depth)
        self.unet = UNet2DWithSliceConditioning(max_depth=self.volume_depth, **unet_kwargs)

    def forward(
        self,
        *,
        x_target_t: torch.Tensor,  # [B, P, H, W]
        x_cond: torch.Tensor,      # [B, Cn, H, W] (can be empty)
        idx_target: torch.Tensor,  # [B, P]
        idx_cond: torch.Tensor,    # [B, Cn]
        t: torch.Tensor,           # [B]
    ) -> torch.Tensor:
        B, P, H, W = x_target_t.shape
        Cn = x_cond.shape[1]
        if Cn + P > self.tau_max:
            raise ValueError(f"|C|+|P| must be <= tau_max ({self.tau_max}), got {Cn}+{P}")

        x_all = torch.cat([x_cond, x_target_t], dim=1) if Cn > 0 else x_target_t
        idx_all = torch.cat([idx_cond, idx_target], dim=1) if Cn > 0 else idx_target
        type_all = torch.cat(
            [
                torch.zeros((B, Cn), device=x_all.device, dtype=torch.long),
                torch.ones((B, P), device=x_all.device, dtype=torch.long),
            ],
            dim=1,
        ) if Cn > 0 else torch.ones((B, P), device=x_all.device, dtype=torch.long)

        eps_all = self.unet(x_all=x_all, idx_all=idx_all, type_all=type_all, t=t)
        return eps_all[:, Cn:, :, :] if Cn > 0 else eps_all


# ---------------------------
# Training slice-set sampling
# ---------------------------

def sample_condition_target_indices(D: int, tau_max: int, cond_max: int = 19, tgt_min: int = 1) -> Tuple[List[int], List[int]]:
    # Pick P first (>=1), then C within remaining budget. Mirrors paper constraint |C|+|P|<=tau_max. :contentReference[oaicite:3]{index=3}
    P = random.randint(tgt_min, tau_max)
    C = random.randint(0, min(cond_max, tau_max - P))
    all_idx = list(range(D))
    random.shuffle(all_idx)
    cond_idx = sorted(all_idx[:C])
    tgt_idx = sorted(all_idx[C:C + P])
    return cond_idx, tgt_idx


@dataclass
class SliceBatch:
    x_cond: torch.Tensor
    x_target0: torch.Tensor
    idx_cond: torch.Tensor
    idx_target: torch.Tensor


def make_slice_batch(volumes: torch.Tensor, tau_max: int) -> SliceBatch:
    """
    volumes: [B, D, H, W] in [0,1]
    returns fixed-size (Cn,P) across batch for easy collation.
    """
    B, D, H, W = volumes.shape
    cond_idx, tgt_idx = sample_condition_target_indices(D=D, tau_max=tau_max)
    Cn, P = len(cond_idx), len(tgt_idx)

    idx_cond = torch.tensor(cond_idx, device=volumes.device, dtype=torch.long).unsqueeze(0).repeat(B, 1) if Cn > 0 \
        else torch.empty(B, 0, device=volumes.device, dtype=torch.long)
    idx_tgt = torch.tensor(tgt_idx, device=volumes.device, dtype=torch.long).unsqueeze(0).repeat(B, 1)

    x_cond = volumes[:, cond_idx] if Cn > 0 else torch.empty(B, 0, H, W, device=volumes.device, dtype=volumes.dtype)
    x_tgt0 = volumes[:, tgt_idx]
    return SliceBatch(x_cond=x_cond, x_target0=x_tgt0, idx_cond=idx_cond, idx_target=idx_tgt)


def train_step(
    *,
    model: CDPM25D,
    diffusion: GaussianDiffusion,
    volumes: torch.Tensor,  # [B,D,H,W] in [0,1]
) -> torch.Tensor:
    B = volumes.shape[0]
    batch = make_slice_batch(volumes, tau_max=model.tau_max)
    noise = torch.randn_like(batch.x_target0)
    t = torch.randint(0, diffusion.T, (B,), device=volumes.device, dtype=torch.long)
    x_target_t = diffusion.q_sample(batch.x_target0, t, noise=noise)

    eps_pred = model(
        x_target_t=x_target_t,
        x_cond=batch.x_cond,
        idx_target=batch.idx_target,
        idx_cond=batch.idx_cond,
        t=t,
    )
    return F.mse_loss(eps_pred, noise)


@torch.no_grad()
def generate_volume_staged(
    *,
    model: CDPM25D,
    diffusion: GaussianDiffusion,
    D: int,
    H: int,
    W: int,
    stage_size: int = 10,
    device: torch.device,
) -> torch.Tensor:
    """
    Stage-wise generation (paper uses 10-slice stages to build 128 slices). :contentReference[oaicite:4]{index=4}
    Returns [1, D, H, W] in ~[0,1].
    """
    model.eval()
    vol = torch.zeros(1, D, H, W, device=device)

    start = 0
    cond_slices = torch.empty(1, 0, H, W, device=device)
    cond_idx = torch.empty(1, 0, dtype=torch.long, device=device)

    while start < D:
        end = min(start + stage_size, D)
        tgt_list = list(range(start, end))
        target_idx = torch.tensor(tgt_list, device=device, dtype=torch.long).unsqueeze(0)

        x0_targets = diffusion.sample_targets(
            model,
            cond_slices=cond_slices,
            cond_idx=cond_idx,
            target_idx=target_idx,
            H=H,
            W=W,
            device=device,
        )
        vol[:, tgt_list] = x0_targets

        cond_slices = x0_targets
        cond_idx = target_idx
        start = end

    return vol

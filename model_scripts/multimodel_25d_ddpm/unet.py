import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) integer timesteps
        returns: (B, dim) sinusoidal embedding
        """
        device = t.device
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Linear(t_dim, out_ch)

        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act = nn.SiLU()

        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # time embedding
        t = self.time_mlp(t_emb)
        t = self.act(t)
        h = h + t[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.res_conv(x)


class DownBlock(nn.Module):
    """
    One U-Net down block:
      in_ch -> out_ch (same spatial size)
      then downsample to half spatial size
    Returns (x_down, skip)
    """
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.res1 = ResidualBlock(in_ch, out_ch, t_dim)
        self.res2 = ResidualBlock(out_ch, out_ch, t_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    """
    One U-Net up block:
      upsample x (in_ch -> out_ch, double spatial size),
      concat with skip (skip_ch),
      then two residual blocks -> out_ch
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res1 = ResidualBlock(out_ch + skip_ch, out_ch, t_dim)
        self.res2 = ResidualBlock(out_ch, out_ch, t_dim)

        print(f"UpBlock: in_ch={in_ch}, skip_ch={skip_ch}, out_ch={out_ch}")

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)

        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x


class UNet(nn.Module):
    """
    Time-conditioned U-Net used inside the DDPM.
    Works for 1-channel 2D MRI slices (e.g. 128x128).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim: int = 256,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # ----- time embedding -----
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # ----- slice position embedding -----
        self.slice_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # channel sizes at each resolution
        self.chs = [base_channels * m for m in channel_mults]

        # initial conv: (img_channels -> chs[0])
        self.init_conv = nn.Conv2d(in_channels, self.chs[0], 3, padding=1)

        # ----- down path -----
        downs = []
        for in_ch, out_ch in zip(self.chs[:-1], self.chs[1:]):
            downs.append(DownBlock(in_ch, out_ch, time_emb_dim))
        self.downs = nn.ModuleList(downs)

        # bottleneck
        self.mid_block1 = ResidualBlock(self.chs[-1], self.chs[-1], time_emb_dim)
        self.mid_block2 = ResidualBlock(self.chs[-1], self.chs[-1], time_emb_dim)

        # ----- up path -----
        ups = []
        # skips have channels [chs[1], chs[2], ..., chs[-1]]
        skip_chs = self.chs[1:]
        in_ch = self.chs[-1]
        # reverse to go from coarsest to finest
        for skip_ch, out_ch in zip(reversed(skip_chs), reversed(self.chs[:-1])):
            ups.append(UpBlock(in_ch, skip_ch, out_ch, time_emb_dim))
            in_ch = out_ch
        self.ups = nn.ModuleList(ups)

        # final conv
        self.out_norm = nn.GroupNorm(8, self.chs[0])
        self.out_conv = nn.Conv2d(self.chs[0], out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z_pos: torch.Tensor,
        context: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        x:       (B, C_target, H, W)  - noisy center-slice modalities
        context: (B, C_context, H, W) - clean neighboring slices (optional)
        """
        t = t.to(x.device)
        z_pos = z_pos.to(x.device).float()

        # time embedding
        t_emb = self.time_mlp(t)

        # slice position embedding
        z_emb = self.slice_mlp(z_pos.unsqueeze(-1))

        # combine
        cond_emb = t_emb + z_emb

        # concatenate context channels if provided
        if context is not None:
            x = torch.cat([x, context], dim=1)

        x = self.init_conv(x)

        skips = []
        for down in self.downs:
            x, skip = down(x, cond_emb)
            skips.append(skip)

        x = self.mid_block1(x, cond_emb)
        x = self.mid_block2(x, cond_emb)

        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip, cond_emb)

        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)
        return x
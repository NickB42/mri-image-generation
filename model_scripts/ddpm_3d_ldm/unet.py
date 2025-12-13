import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        """
        timesteps: (B,) int64
        returns: (B, dim)
        """
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=8):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)

        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t=None):
        h = self.conv1(self.act1(self.norm1(x)))
        if t is not None:
            time_emb = self.time_mlp(t)
            h = h + time_emb[:, :, None, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class UNet3DModel(nn.Module):
    def __init__(self, in_channels, base_channels=64, channel_mults=(1, 2, 4), time_emb_dim=256, groups=8):
        super().__init__()
        self.in_channels = in_channels
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.num_levels = len(channel_mults)
        chs = [base_channels * m for m in channel_mults]
        self.chs = chs

        # Input convolution
        self.in_conv = nn.Conv3d(in_channels, chs[0], 3, padding=1)

        # Downsampling path
        downs = []
        for i in range(self.num_levels):
            ch = chs[i]
            res1 = ResidualBlock3D(ch, ch, time_emb_dim, groups)
            res2 = ResidualBlock3D(ch, ch, time_emb_dim, groups)
            if i != self.num_levels - 1:
                down = nn.Conv3d(ch, chs[i + 1], 4, stride=2, padding=1)
            else:
                down = nn.Identity()
            downs.append(
                nn.ModuleDict({"res1": res1, "res2": res2, "down": down})
            )
        self.downs = nn.ModuleList(downs)

        # Bottleneck
        self.mid1 = ResidualBlock3D(chs[-1], chs[-1], time_emb_dim, groups)
        self.mid2 = ResidualBlock3D(chs[-1], chs[-1], time_emb_dim, groups)

        # Upsampling path
        ups = []
        cur_ch = chs[-1]
        for i in reversed(range(self.num_levels)):
            ch = chs[i]
            if i != self.num_levels - 1:
                up = nn.ConvTranspose3d(cur_ch, ch, 4, stride=2, padding=1)
            else:
                up = nn.Identity()
            res1 = ResidualBlock3D(ch * 2, ch, time_emb_dim, groups)
            res2 = ResidualBlock3D(ch, ch, time_emb_dim, groups)
            ups.append(
                nn.ModuleDict({"up": up, "res1": res1, "res2": res2})
            )
            cur_ch = ch

        self.ups = nn.ModuleList(ups)
        self.out_norm = nn.GroupNorm(groups, chs[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv3d(chs[0], in_channels, 3, padding=1)

    def forward(self, x, t):
        """
        x: (B, C, D, H, W)
        t: (B,) int64
        """
        t_emb = self.time_mlp(t)
        h = self.in_conv(x)

        skips = []

        # Down
        for block in self.downs:
            h = block["res1"](h, t_emb)
            h = block["res2"](h, t_emb)
            skips.append(h)
            h = block["down"](h)

        # Bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # Up
        for block in self.ups:
            h = block["up"](h)
            skip = skips.pop()

            # crop skip if shapes mismatch
            if h.shape[-3:] != skip.shape[-3:]:
                dz = (skip.shape[-3] - h.shape[-3]) // 2
                dy = (skip.shape[-2] - h.shape[-2]) // 2
                dx = (skip.shape[-1] - h.shape[-1]) // 2
                skip = skip[
                    ...,
                    dz: dz + h.shape[-3],
                    dy: dy + h.shape[-2],
                    dx: dx + h.shape[-1],
                ]

            h = torch.cat([h, skip], dim=1)
            h = block["res1"](h, t_emb)
            h = block["res2"](h, t_emb)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h

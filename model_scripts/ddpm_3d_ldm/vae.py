import torch
import torch.nn as nn


class ResidualBlock3DNoTime(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class Encoder3D(nn.Module):
    def __init__(self, in_channels=4, base_channels=32, num_down=3, latent_channels=8, groups=8):
        """ Downsamples spatial dims by factor 2**(num_down-1). """
        super().__init__()
        self.num_down = num_down
        self.in_conv = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        downs = []
        cur_ch = base_channels
        for i in range(num_down):
            # residual at current channel size
            downs.append(ResidualBlock3DNoTime(cur_ch, cur_ch, groups))
            if i != num_down - 1:
                # change channels + downsample
                downs.append(ResidualBlock3DNoTime(cur_ch, cur_ch * 2, groups))
                downs.append(
                    nn.Conv3d(cur_ch * 2, cur_ch * 2, 4, stride=2, padding=1)
                )
                cur_ch *= 2

        self.downs = nn.ModuleList(downs)
        self.out_channels = cur_ch
        self.to_mu_logvar = nn.Conv3d(cur_ch, 2 * latent_channels, 3, padding=1)

    def forward(self, x):
        h = self.in_conv(x)
        for layer in self.downs:
            h = layer(h)
        stats = self.to_mu_logvar(h)
        mu, logvar = torch.chunk(stats, 2, dim=1)
        return mu, logvar


class Decoder3D(nn.Module):
    def __init__(self, out_channels=4, base_channels=32, num_down=3, latent_channels=8, enc_out_channels=None, groups=8):
        super().__init__()
        if enc_out_channels is None:
            enc_out_channels = base_channels * (2 ** (num_down - 1))
        cur_ch = enc_out_channels

        self.from_latent = nn.Conv3d(latent_channels, cur_ch, 3, padding=1)

        ups = []
        for i in reversed(range(num_down)):
            ups.append(ResidualBlock3DNoTime(cur_ch, cur_ch, groups))
            if i != 0:
                ups.append(ResidualBlock3DNoTime(cur_ch, cur_ch // 2, groups))
                ups.append(
                    nn.ConvTranspose3d(
                        cur_ch // 2, cur_ch // 2, 4, stride=2, padding=1
                    )
                )
                cur_ch //= 2

        self.ups = nn.ModuleList(ups)
        self.out_conv = nn.Conv3d(cur_ch, out_channels, 3, padding=1)

    def forward(self, z):
        h = self.from_latent(z)
        for layer in self.ups:
            h = layer(h)
        x_recon = self.out_conv(h)
        return x_recon


class VAE3D(nn.Module):
    def __init__(self, in_channels=4, base_channels=32, num_down=3, latent_channels=8, groups=8):
        super().__init__()
        self.encoder = Encoder3D(
            in_channels, base_channels, num_down, latent_channels, groups
        )
        self.decoder = Decoder3D(
            in_channels, base_channels, num_down, latent_channels,
            enc_out_channels=self.encoder.out_channels, groups=groups
        )

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
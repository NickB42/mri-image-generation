import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDiffusionLatent3D(nn.Module):
    """
    Generic DDPM-style Gaussian diffusion for 3D *latents* (B, C, D, H, W).
    - model is typically a 3D UNet that takes (x_t, t, cond) or (x_t, t).
    - channels is the number of latent channels.
    """
    def __init__(
        self,
        model,
        channels,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
    ):
        super().__init__()
        self.model = model
        self.channels = channels
        self.timesteps = timesteps
        print(f"Setting up Gaussian Diffusion (3D latent) with {timesteps} timesteps.")

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )

    def _extract(self, a, t, x_shape):
        """
        Extract values for batch of indices t, shape (B,) from a tensor a of shape (T,)
        and reshape to (B, 1, 1, 1, ..., 1) for broadcasting to x_shape.
        """
        B = t.shape[0]
        out = a.gather(-1, t)  # (B,)
        view_shape = (B,) + (1,) * (len(x_shape) - 1)
        return out.view(*view_shape)

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from q(x_t | x_0).
        x_start: (B, C, D, H, W)
        t: (B,)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, cond=None, noise=None):
        """
        Loss for a given batch. Predicts the noise and uses MSE between true noise and predicted noise.
        x_start: (B, C, D, H, W)
        t: (B,)
        cond: optional conditioning (ignored if model doesn't use it)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if cond is None:
            predicted_noise = self.model(x_noisy, t)
        else:
            predicted_noise = self.model(x_noisy, t, cond)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, x, t, cond=None):
        """ One reverse diffusion step: p(x_{t-1} | x_t, cond). """
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)

        if cond is None:
            eps_theta = self.model(x, t)
        else:
            eps_theta = self.model(x, t, cond)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t / sqrt_one_minus_alphas_cumprod_t * eps_theta
        )

        noise = torch.randn_like(x)
        mask_shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        nonzero_mask = (t != 0).float().view(*mask_shape)

        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond=None):
        """
        Generate a full sample starting from x_T ~ N(0, I)
        shape: (B, C, D, H, W)
        cond: optional conditioning
        """
        device = self.betas.device
        B = shape[0]
        img = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, cond)
        return img

    @torch.no_grad()
    def sample(self, batch_size, spatial_size, cond=None):
        """
        Sample a batch of latent volumes.
        spatial_size: (D, H, W) or int (then D=H=W)
        """
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size,) * 3
        shape = (batch_size, self.channels, *spatial_size)
        return self.p_sample_loop(shape, cond=cond)

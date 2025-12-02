import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        image_size,
        channels=1,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
    ):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps

        print(f"Setting up Gaussian Diffusion with {timesteps} timesteps.")

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

        # Posterior variance (Eq. 7 in DDPM paper)
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
        and reshape to (B, 1, 1, 1) for broadcasting.
        """
        B = t.shape[0]
        out = a.gather(-1, t)
        return out.view(B, 1, 1, 1).expand(x_shape)

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from q(x_t | x_0).
        x_start: (B, C, H, W) in [-1, 1]
        t: (B,) integer timesteps
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

    def p_losses(self, x_start, t, z_pos, noise=None):
        """
        Loss for a given batch.
        Predicts the noise and uses MSE between true noise and predicted noise.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        predicted_noise = self.model(x_noisy, t, z_pos)

        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, x, t, z_pos):
        """
        One reverse diffusion step: p(x_{t-1} | x_t, z_pos).
        """
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)

        # Predict noise with conditioning
        eps_theta = self.model(x, t, z_pos)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t / sqrt_one_minus_alphas_cumprod_t * eps_theta
        )

        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)  # 0 at t=0, 1 otherwise

        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, z_pos):
        """
        Generate a full sample starting from x_T ~ N(0, I)
        conditioned on slice position z_pos.
        
        z_pos: (B,) or scalar (float)
        """
        device = self.betas.device
        B = shape[0]
        img = torch.randn(shape, device=device)

        if not torch.is_tensor(z_pos):
            z_pos = torch.full((B,), float(z_pos), device=device)
        else:
            z_pos = z_pos.to(device).float()

        for i in reversed(range(self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, z_pos)

        return img

    @torch.no_grad()
    def sample(self, batch_size=16, z_pos=0.5):
        """
        Sample a batch of images conditioned on slice position z_pos.
        z_pos: float in [0,1] or tensor of shape (B,)
        """
        return self.p_sample_loop(
            (batch_size, self.channels, self.image_size, self.image_size),
            z_pos=z_pos,
        )
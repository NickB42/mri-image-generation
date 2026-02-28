import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta: float = 0.999, s: float = 0.008):
    """
    Cosine schedule from Improved DDPM (Nichol & Dhariwal).
    """
    def alpha_bar(t):
        return math.cos((t + s) / (1 + s) * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        beta = min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta)
        betas.append(beta)
    return torch.tensor(betas, dtype=torch.float32)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        image_size,
        channels=1,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        schedule="cosine",
    ):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        elif schedule == "cosine":
            betas = betas_for_alpha_bar(timesteps)
        else:
            raise ValueError("schedule must be 'linear' or 'cosine'")

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
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1.0),
        )

    def _extract(self, a, t, x_shape):
        B = t.shape[0]
        out = a.gather(-1, t)
        return out.view(B, 1, 1, 1).expand(x_shape)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, z_pos, fg_frac=None, context=None, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Context augmentation (same as ddpm_25d_seq)
        if context is not None:
            # Context dropout
            p_drop = 0.1
            if torch.rand((), device=context.device) < p_drop:
                context = torch.zeros_like(context)

            # Context diffusion-noise
            p_ctx_noise = 0.5
            if torch.rand((), device=context.device) < p_ctx_noise:
                ctx_noise = torch.randn_like(context)
                context = self.q_sample(x_start=context, t=t, noise=ctx_noise)

        predicted_noise = self.model(x_noisy, t, z_pos, fg_frac=fg_frac, context=context)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, x, t, z_pos, fg_frac=None, context=None):
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)

        eps_theta = self.model(x, t, z_pos, fg_frac=fg_frac, context=context)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t / sqrt_one_minus_alphas_cumprod_t * eps_theta
        )

        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, z_pos, fg_frac=None, context=None):
        device = self.betas.device
        B = shape[0]
        img = torch.randn(shape, device=device)

        if not torch.is_tensor(z_pos):
            z_pos = torch.full((B,), float(z_pos), device=device)
        else:
            z_pos = z_pos.to(device).float()

        if fg_frac is not None:
            fg_frac = fg_frac.to(device).float()
        if context is not None:
            context = context.to(device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, z_pos, fg_frac=fg_frac, context=context)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, z_pos=0.5, fg_frac=None, context=None):
        return self.p_sample_loop(
            (batch_size, self.channels, self.image_size, self.image_size),
            z_pos=z_pos,
            fg_frac=fg_frac,
            context=context,
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    @torch.no_grad()
    def ddim_sample_loop(self, shape, z_pos, fg_frac=None, context=None, sample_timesteps: int = 50, eta: float = 0.0):
        device = self.betas.device
        B = shape[0]
        img = torch.randn(shape, device=device)

        if not torch.is_tensor(z_pos):
            z_pos = torch.full((B,), float(z_pos), device=device)
        else:
            z_pos = z_pos.to(device).float()

        if fg_frac is not None:
            fg_frac = fg_frac.to(device).float()
        if context is not None:
            context = context.to(device)

        T = max(1, min(int(sample_timesteps), self.timesteps))
        steps = torch.linspace(self.timesteps - 1, 0, T, device=device).long()

        for idx in range(len(steps)):
            t_int = int(steps[idx].item())
            t = torch.full((B,), t_int, device=device, dtype=torch.long)

            t_prev_int = 0 if idx == len(steps) - 1 else int(steps[idx + 1].item())
            t_prev = torch.full((B,), t_prev_int, device=device, dtype=torch.long)

            eps = self.model(img, t, z_pos, fg_frac=fg_frac, context=context)
            x0 = self.predict_start_from_noise(img, t, eps).clamp(-1.0, 1.0)

            a_t = self._extract(self.alphas_cumprod, t, img.shape)
            a_prev = self._extract(self.alphas_cumprod, t_prev, img.shape)

            sigma = eta * torch.sqrt((1.0 - a_prev) / (1.0 - a_t)) * torch.sqrt(1.0 - (a_t / a_prev))
            noise = torch.randn_like(img) if eta > 0 else torch.zeros_like(img)
            dir_xt = torch.sqrt(torch.clamp(1.0 - a_prev - sigma**2, min=0.0)) * eps
            img = torch.sqrt(a_prev) * x0 + dir_xt + sigma * noise

        return img

    @torch.no_grad()
    def sample_ddim(self, batch_size=16, z_pos=0.5, fg_frac=None, context=None, sample_timesteps: int = 50, eta: float = 0.0):
        return self.ddim_sample_loop(
            (batch_size, self.channels, self.image_size, self.image_size),
            z_pos=z_pos,
            fg_frac=fg_frac,
            context=context,
            sample_timesteps=sample_timesteps,
            eta=eta,
        )

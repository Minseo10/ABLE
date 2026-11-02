"""Generative warm-start models: features -> a distribution over torque plans.

Each model has `fit(x, y)`, `predict(x)` (its single most likely plan) and
`sample(x, k)` (k plans per row, `(n, k, dim)`); the mixture, flow and diffusion models
also have `log_prob(x, y)`, the log density of each row's plan. Each trains for `steps` Adam steps and keeps
the weights with the lowest held-out loss, holding out a sixth of the segments as
`DropoutMLP` does.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from able.api.learner import FlowMatching, Generative, mlp, ode_log_prob, time_embedding
from able.tasks.limb.warmstart import BASE_FEATURES

__all__ = ["FlowMatching", "Generative", "KINDS", "limb_segments"]


def limb_segments(features) -> np.ndarray:
    """One group per planned segment: a new one starts at step 0."""
    return np.cumsum(np.asarray(features)[:, BASE_FEATURES - 1] == 0)


class MixtureDensity(Generative):
    """`components` weighted Gaussians with diagonal covariance; a draw picks one by weight."""

    components: int = 5

    def build(self, cond, dim):
        self.net = mlp(cond, self.components * (1 + 2 * dim), self.width, self.depth)

    def _split(self, c):
        o = self.net(c)
        k, d = self.components, self.dim
        logits = o[:, :k]
        mean = o[:, k:k + k * d].reshape(-1, k, d)
        log_scale = o[:, k + k * d:].reshape(-1, k, d).clamp(-5, 2)
        return logits, mean, log_scale

    def loss(self, c, y, g):
        logits, mean, log_scale = self._split(c)
        z = (y[:, None] - mean) / log_scale.exp()
        log_prob = (-0.5 * z ** 2 - log_scale - 0.5 * math.log(2 * math.pi)).sum(-1)
        return -torch.logsumexp(torch.log_softmax(logits, -1) + log_prob, -1).mean() / self.dim

    def log_density(self, c, y):
        logits, mean, log_scale = self._split(c)
        z = (y[:, None] - mean) / log_scale.exp()
        log_prob = (-0.5 * z ** 2 - log_scale - 0.5 * math.log(2 * math.pi)).sum(-1)
        return torch.logsumexp(torch.log_softmax(logits, -1) + log_prob, -1)

    def draw(self, c, z):
        logits, mean, log_scale = self._split(c)
        if torch.all(z == 0):
            pick = logits.argmax(-1)
            return mean[torch.arange(len(c)), pick]
        u = torch.rand(len(c), generator=None)
        cdf = torch.softmax(logits, -1).cumsum(-1)
        pick = (u[:, None] > cdf).sum(-1).clamp(max=self.components - 1)
        idx = torch.arange(len(c))
        return mean[idx, pick] + log_scale[idx, pick].exp() * z



class Diffusion(Generative):
    """DDPM with a cosine schedule (100 levels); the network predicts the clean plan, and a
    draw runs deterministic DDIM in 10 steps from noise. Its density is the DDIM
    (probability-flow) ODE's from the first noise level to the last, not the 10-step
    sampler's; it leans on the network's Jacobian at the lowest noise levels, which that
    sampler never visits."""

    timesteps: int = 100
    sample_steps: int = 10
    density_steps: int = 50

    def build(self, cond, dim):
        self.net = mlp(dim + cond + 64, dim, self.width, self.depth)
        s = torch.linspace(0, 1, self.timesteps + 1)
        alpha_bar = torch.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
        self.alpha_bar = (alpha_bar / alpha_bar[0]).clamp(1e-4, 1.0)

    def _x0(self, x, t, c):
        return self.net(torch.cat([x, c, time_embedding(t.float() / self.timesteps, 64)], -1))

    def loss(self, c, y, g):
        t = torch.randint(1, self.timesteps + 1, (len(y),), generator=g)
        noise = torch.randn(y.shape, generator=g)
        a = self.alpha_bar[t][:, None]
        xt = a.sqrt() * y + (1 - a).sqrt() * noise
        return ((self._x0(xt, t, c) - y) ** 2).mean()

    def draw(self, c, z):
        ts = torch.linspace(self.timesteps, 0, self.sample_steps + 1).long()
        x = z
        for t, t_next in zip(ts[:-1], ts[1:]):
            a, a_next = self.alpha_bar[t], self.alpha_bar[t_next]
            x0 = self._x0(x, torch.full((len(c),), int(t)), c)
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
            x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
        return x

    def log_density(self, c, y):
        def field(x, t):
            # alpha_bar is linear between levels, so its slope is the level difference.
            low = t.floor().long().clamp(max=self.timesteps - 1)
            slope = self.alpha_bar[low + 1] - self.alpha_bar[low]
            alpha_bar = self.alpha_bar[low] + (t - low) * slope
            a, s = alpha_bar.sqrt(), (1 - alpha_bar).sqrt()
            x0 = self.net(torch.cat([x, c, time_embedding(t.expand(len(c)) / self.timesteps, 64)], -1))
            return slope / (2 * a) * x0 - slope / (2 * s * s) * (x - a * x0)

        # Geometric steps: the field changes fastest at the low-noise end.
        times = torch.logspace(0.0, math.log10(self.timesteps), self.density_steps + 1)
        return ode_log_prob(field, y, times)


class ConditionalVAE(Generative):
    """A latent z (8-d) per plan; the decoder maps (features, z) to a plan; z ~ N(0, I) to draw."""

    z_dim: int = 8
    beta: float = 0.05

    def build(self, cond, dim):
        self.encoder = mlp(cond + dim, 2 * self.z_dim, self.width, self.depth)
        self.decoder = mlp(cond + self.z_dim, dim, self.width, self.depth)

    def loss(self, c, y, g):
        mu, log_var = self.encoder(torch.cat([c, y], -1)).chunk(2, -1)
        log_var = log_var.clamp(-8, 4)
        z = mu + (0.5 * log_var).exp() * torch.randn(mu.shape, generator=g)
        recon = ((self.decoder(torch.cat([c, z], -1)) - y) ** 2).mean()
        kl = 0.5 * (mu ** 2 + log_var.exp() - 1 - log_var).sum(-1).mean() / self.dim
        return recon + self.beta * kl

    def draw(self, c, z):
        return self.decoder(torch.cat([c, z], -1))

    @property
    def latent(self):
        return self.z_dim


class EnergyBased(Generative):
    """Implicit model: an energy E(features, plan), trained with InfoNCE against shuffled and
    perturbed plans; a draw is a Langevin chain from noise, the most likely plan the
    lowest-energy of 8 chains."""

    negatives: int = 31
    langevin_steps: int = 40
    needs_grad = True

    def build(self, cond, dim):
        self.net = mlp(cond + dim, 1, self.width, self.depth)

    def energy(self, c, y):
        return self.net(torch.cat([c, y], -1)).squeeze(-1)

    def loss(self, c, y, g):
        n, k = len(y), self.negatives
        shuffled = y[torch.randint(n, (n, k // 2), generator=g)]
        scales = torch.tensor([0.1, 0.3, 1.0])[torch.randint(3, (n, k - k // 2, 1), generator=g)]
        perturbed = y[:, None] + scales * torch.randn((n, k - k // 2, y.shape[1]), generator=g)
        candidates = torch.cat([y[:, None], shuffled, perturbed], 1)
        e = self.energy(c[:, None].expand(-1, k + 1, -1), candidates)
        return nn.functional.cross_entropy(-e, torch.zeros(n, dtype=torch.long))

    def draw(self, c, z):
        y = z.clone().requires_grad_(True)
        for i in range(self.langevin_steps):
            step = 0.1 * (0.02 / 0.1) ** (i / max(1, self.langevin_steps - 1))
            grad, = torch.autograd.grad(self.energy(c, y).sum(), y)
            noise = torch.randn_like(y) if i < self.langevin_steps - 1 else 0.0
            y = (y - step * grad + math.sqrt(2 * step) * 0.1 * noise).detach().requires_grad_(True)
        return y.detach()

    def predict(self, features):
        features = np.asarray(features)
        plans = self.sample(features, 8, seed=0)
        c = self._tensor((np.repeat(self._x(features), 8, axis=0) - self.x_mean) / self.x_std)
        ys = self._tensor(((plans.reshape(-1, plans.shape[-1])) - self.y_mean) / self.y_std)
        with torch.no_grad():
            e = self.energy(c, ys).reshape(len(features), 8).numpy()
        return plans[np.arange(len(features)), e.argmin(1)]


KINDS = {"mdn": MixtureDensity, "flow": FlowMatching, "ddpm": Diffusion,
         "cvae": ConditionalVAE, "ebm": EnergyBased}

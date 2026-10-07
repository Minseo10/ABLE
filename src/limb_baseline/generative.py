"""Generative warm-start models: features -> a distribution over torque plans.

Each model has `fit(x, y)`, `predict(x)` (its single most likely plan) and
`sample(x, k)` (k plans per row, `(n, k, dim)`), trains for `steps` Adam steps and keeps
the weights with the lowest held-out loss, holding out a sixth of the segments as
`DropoutMLP` does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from limb_baseline.flow import time_embedding
from limb_baseline.warmstart import BASE_FEATURES


def mlp(inp: int, out: int, width: int, depth: int) -> nn.Sequential:
    layers, d = [], inp
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, out))


@dataclass
class Generative:
    width: int = 256
    depth: int = 3
    steps: int = 4000
    batch_size: int = 256
    lr: float = 1e-3
    min_segments: int = 6
    seed: int = 0
    columns: int | None = None

    # ------------------------------------------------------------- subclass API

    def build(self, cond: int, dim: int) -> None:
        raise NotImplementedError

    def loss(self, c: torch.Tensor, y: torch.Tensor, g: torch.Generator | None) -> torch.Tensor:
        raise NotImplementedError

    def draw(self, c: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Plans (standardized) for conditions `c` from base noise `z`."""
        raise NotImplementedError

    # ---------------------------------------------------------------- shared

    def _x(self, features) -> np.ndarray:
        x = np.asarray(features, np.float64)
        return x[:, : self.columns] if self.columns else x

    def _tensor(self, a) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a), dtype=torch.float32)

    def fit(self, features, labels):
        torch.manual_seed(self.seed)
        segment = np.cumsum(np.asarray(features)[:, BASE_FEATURES - 1] == 0)
        x, y = self._x(features), np.asarray(labels, np.float64)
        self.n_features_in_ = np.asarray(features).shape[1]
        self.x_mean, self.x_std = x.mean(0), np.where(x.std(0) < 1e-8, 1.0, x.std(0))
        self.y_mean, self.y_std = y.mean(0), np.where(y.std(0) < 1e-8, 1.0, y.std(0))
        held = np.zeros(len(x), bool)
        if segment[-1] >= self.min_segments:
            rng = np.random.default_rng(self.seed)
            chosen = rng.choice(np.arange(1, segment[-1] + 1), size=segment[-1] // 6, replace=False)
            held = np.isin(segment, chosen)
        xs, ys = self._tensor((x - self.x_mean) / self.x_std), self._tensor((y - self.y_mean) / self.y_std)
        train = torch.as_tensor(np.where(~held)[0])
        self.dim = y.shape[1]
        self.build(x.shape[1], self.dim)
        params = [p for m in self.modules() for p in m.parameters()]
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.lr, total_steps=self.steps,
                                                       pct_start=max(0.05, 3 / self.steps))
        generator = torch.Generator().manual_seed(self.seed)
        best, state, self.best_step = float("inf"), None, self.steps
        for step in range(1, self.steps + 1):
            self.train_mode(True)
            batch = train[torch.randint(len(train), (self.batch_size,), generator=generator)]
            loss = self.loss(xs[batch], ys[batch], None)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            if held.any() and (step % 100 == 0 or step == self.steps):
                self.train_mode(False)
                with torch.no_grad():
                    val = float(self.loss(xs[held], ys[held], torch.Generator().manual_seed(7)))
                if val < best:
                    best, self.best_step = val, step
                    state = [{k: v.clone() for k, v in m.state_dict().items()} for m in self.modules()]
        if state is not None:
            for m, s in zip(self.modules(), state):
                m.load_state_dict(s)
        self.train_mode(False)
        return self

    def modules(self) -> list[nn.Module]:
        return [v for v in vars(self).values() if isinstance(v, nn.Module)]

    def train_mode(self, on: bool) -> None:
        for m in self.modules():
            m.train(on)

    def _draw(self, features, z) -> np.ndarray:
        c = self._tensor((self._x(features) - self.x_mean) / self.x_std)
        with torch.no_grad() if not getattr(self, "needs_grad", False) else torch.enable_grad():
            out = self.draw(c, z)
        return out.detach().numpy() * self.y_std + self.y_mean

    def sample(self, features, k: int = 8, seed: int | None = None) -> np.ndarray:
        features = np.asarray(features)
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        rows = np.repeat(features, k, axis=0)
        z = torch.randn(len(rows), self.latent, generator=g)
        return self._draw(rows, z).reshape(len(features), k, -1)

    def predict(self, features) -> np.ndarray:
        features = np.asarray(features)
        return self._draw(features, torch.zeros(len(features), self.latent))

    @property
    def latent(self) -> int:
        return self.dim


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


class FlowMatching(Generative):
    """Conditional flow matching (rectified flow); 10 Euler steps from noise to a plan."""

    integration_steps: int = 10

    def build(self, cond, dim):
        self.net = mlp(dim + cond + 64, dim, self.width, self.depth)

    def _v(self, x, t, c):
        return self.net(torch.cat([x, c, time_embedding(t, 64)], -1))

    def loss(self, c, y, g):
        x0 = torch.randn(y.shape, generator=g)
        t = torch.rand(len(y), generator=g)
        xt = (1 - t[:, None]) * x0 + t[:, None] * y
        return ((self._v(xt, t, c) - (y - x0)) ** 2).mean()

    def draw(self, c, z):
        x, dt = z, 1.0 / self.integration_steps
        for i in range(self.integration_steps):
            x = x + dt * self._v(x, torch.full((len(c),), i * dt), c)
        return x


class Diffusion(Generative):
    """DDPM with a cosine schedule (100 levels); the network predicts the clean plan, and a
    draw runs deterministic DDIM in 10 steps from noise."""

    timesteps: int = 100
    sample_steps: int = 10

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

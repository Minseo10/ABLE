"""Conditional generative learners for psi: features -> distribution over labels.

Training keeps the weights with the lowest loss on a held-out sixth of the groups.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn


@runtime_checkable
class Learner(Protocol):
    def fit(self, x, y, groups=None) -> Learner: ...

    def predict(self, x) -> np.ndarray: ...

    def sample(self, x, k: int = 8, seed: int | None = None) -> np.ndarray: ...


def mlp(inp: int, out: int, width: int, depth: int) -> nn.Sequential:
    layers, d = [], inp
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, out))


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    freqs = torch.exp(-math.log(1000.0) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    angles = t[:, None] * freqs[None] * 1000.0
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def ode_log_prob(field, y: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """Log density of `y` under the flow `field(x, t)` (N(0, I) at `times[-1]`), RK4 with exact divergence."""

    def slope(x, t):
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            v = field(x, t)
            div = sum(torch.autograd.grad(v[:, i].sum(), x, retain_graph=True)[0][:, i]
                      for i in range(x.shape[1]))
        return v.detach(), div.detach()

    x, log_det = y, torch.zeros(len(y))
    for t, t_next in zip(times[:-1], times[1:]):
        h = t_next - t
        v1, d1 = slope(x, t)
        v2, d2 = slope(x + h / 2 * v1, t + h / 2)
        v3, d3 = slope(x + h / 2 * v2, t + h / 2)
        v4, d4 = slope(x + h * v3, t_next)
        x = x + h / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
        log_det = log_det + h / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
    return (-0.5 * x ** 2 - 0.5 * math.log(2 * math.pi)).sum(-1) + log_det


@dataclass
class Generative:
    width: int = 256
    depth: int = 3
    steps: int = 4000
    batch_size: int = 256
    lr: float = 1e-3
    min_segments: int = 6
    seed: int = 0
    columns: int | None = None  # use only the first `columns` features
    group_fn: Callable[[np.ndarray], np.ndarray] | None = None  # groups from x when fit gets none

    def build(self, cond: int, dim: int) -> None:
        raise NotImplementedError

    def loss(self, c: torch.Tensor, y: torch.Tensor, g: torch.Generator | None) -> torch.Tensor:
        raise NotImplementedError

    def draw(self, c: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Standardized labels for conditions `c` from base noise `z`."""
        raise NotImplementedError

    def log_density(self, c: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _x(self, features) -> np.ndarray:
        x = np.asarray(features, np.float64)
        return x[:, : self.columns] if self.columns else x

    def _tensor(self, a) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a), dtype=torch.float32)

    def fit(self, features, labels, groups=None):
        torch.manual_seed(self.seed)
        features = np.asarray(features)
        if groups is None:
            groups = self.group_fn(features) if self.group_fn else np.arange(len(features))
        groups = np.asarray(groups)
        x, y = self._x(features), np.asarray(labels, np.float64)
        self.n_features_in_ = features.shape[1]
        self.x_mean, self.x_std = x.mean(0), np.where(x.std(0) < 1e-8, 1.0, x.std(0))
        self.y_mean, self.y_std = y.mean(0), np.where(y.std(0) < 1e-8, 1.0, y.std(0))
        ids = np.unique(groups)
        held = np.zeros(len(x), bool)
        if len(ids) >= self.min_segments:
            chosen = np.random.default_rng(self.seed).choice(ids, size=len(ids) // 6, replace=False)
            held = np.isin(groups, chosen)
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
        # Reported only; history rows are (step, train loss, held-out loss).
        self.history, self.best_val, self.val_nmse = [], float("nan"), float("nan")
        watched = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(1))[:2048]]
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
                    fit = float(self.loss(xs[watched], ys[watched], torch.Generator().manual_seed(7)))
                self.history.append((step, fit, val))
                if val < best:
                    best, self.best_step, self.best_val = val, step, val
                    state = [{k: v.clone() for k, v in m.state_dict().items()} for m in self.modules()]
        if state is not None:
            for m, s in zip(self.modules(), state):
                m.load_state_dict(s)
        self.train_mode(False)
        if held.any():
            with torch.random.fork_rng():  # some draws use the global RNG
                pred = self.predict(features[held])
            self.val_nmse = float((((pred - y[held]) / self.y_std) ** 2).mean())
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

    def log_prob(self, features, labels) -> np.ndarray:
        """Log density of each row's label, in the labels' own units."""
        c = self._tensor((self._x(features) - self.x_mean) / self.x_std)
        y = self._tensor((np.asarray(labels, np.float64) - self.y_mean) / self.y_std)
        with torch.no_grad():
            log_p = self.log_density(c, y)
        return log_p.numpy().astype(np.float64) - np.log(self.y_std).sum()

    @property
    def latent(self) -> int:
        return self.dim


class FlowMatching(Generative):
    """Conditional rectified flow; 10 Euler steps from noise. Density is the continuous flow's."""

    integration_steps: int = 10
    density_steps: int = 20

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

    def log_density(self, c, y):
        return ode_log_prob(lambda x, t: self._v(x, t.expand(len(c)), c), y,
                            torch.linspace(1.0, 0.0, self.density_steps + 1))


class Regression(Generative):
    """MSE regression (the MLP ablation); every sample is the prediction."""

    def build(self, cond, dim):
        self.net = mlp(cond, dim, self.width, self.depth)

    def loss(self, c, y, g):
        return ((self.net(c) - y) ** 2).mean()

    def draw(self, c, z):
        return self.net(c)


LEARNERS = {"flow": FlowMatching, "mlp": Regression}


def make_learner(kind: str = "flow", **options) -> Generative:
    """`flow` (default) or `mlp`."""
    return LEARNERS[kind](**options)

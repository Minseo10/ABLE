"""A conditional flow-matching policy: features -> a distribution over torque plans."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


class Standardizer:
    def fit(self, a: np.ndarray) -> "Standardizer":
        self.mean = a.mean(0)
        self.std = np.where(a.std(0) < 1e-8, 1.0, a.std(0))
        return self

    def __call__(self, a: np.ndarray) -> np.ndarray:
        return (a - self.mean) / self.std

    def inverse(self, a: np.ndarray) -> np.ndarray:
        return a * self.std + self.mean


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    freqs = torch.exp(-math.log(1000.0) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    angles = t[:, None] * freqs[None] * 1000.0
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(width, width),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.net(h)


class VelocityField(nn.Module):
    def __init__(self, action_dim: int, cond_dim: int, width: int, depth: int,
                 dropout: float, time_dim: int = 64):
        super().__init__()
        self.time_dim = time_dim
        self.inp = nn.Linear(action_dim + cond_dim + time_dim, width)
        self.blocks = nn.Sequential(*[ResBlock(width, dropout) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, action_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.inp(torch.cat([x, cond, time_embedding(t, self.time_dim)], dim=-1))
        return self.out(self.blocks(h))


@dataclass
class FlowPolicy:
    width: int = 512
    depth: int = 4
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    batch_size: int = 512
    steps: int = 20
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    history: list = field(default_factory=list)

    def _tensor(self, a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.float32, device=self.device)

    def loss(self, x: np.ndarray, y: np.ndarray, seed: int = 0, batch: int = 8192) -> float:
        """Flow-matching loss with fixed noise and times, so values are comparable across epochs."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        self.net.eval()
        total = 0.0
        with torch.no_grad():
            for i in range(0, len(x), batch):
                c, x1 = self._tensor(self.xs(x[i:i + batch])), self._tensor(self.ys(y[i:i + batch]))
                x0 = torch.randn(x1.shape, generator=g, device=self.device)
                t = torch.rand(len(x1), generator=g, device=self.device)
                xt = (1 - t[:, None]) * x0 + t[:, None] * x1
                total += float(((self.net(xt, t, c) - (x1 - x0)) ** 2).sum())
        return total / (len(x) * y.shape[1])

    def fit(self, x: np.ndarray, y: np.ndarray, monitor: dict | None = None,
            every: int = 5) -> "FlowPolicy":
        """`monitor` maps a name to `(x, y)`; its losses are recorded in `history` every `every`
        epochs, and the weights with the lowest `monitor["val"]` loss are kept."""
        torch.manual_seed(self.seed)
        self.xs, self.ys = Standardizer().fit(x), Standardizer().fit(y)
        self.net = VelocityField(y.shape[1], x.shape[1], self.width, self.depth,
                                 self.dropout).to(self.device)
        c_all, x1_all = self._tensor(self.xs(x)), self._tensor(self.ys(y))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        per_epoch = max(1, math.ceil(len(x) / self.batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, total_steps=self.epochs * per_epoch, pct_start=0.05)
        best, self.best_epoch = float("inf"), self.epochs
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            perm = torch.randperm(len(x), device=self.device)
            for i in range(per_epoch):
                idx = perm[i * self.batch_size:(i + 1) * self.batch_size]
                c, x1 = c_all[idx], x1_all[idx]
                x0 = torch.randn_like(x1)
                t = torch.rand(len(x1), device=self.device)
                xt = (1 - t[:, None]) * x0 + t[:, None] * x1
                loss = ((self.net(xt, t, c) - (x1 - x0)) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            if monitor and (epoch % every == 0 or epoch == self.epochs):
                self.history.append({"epoch": epoch, **{k: self.loss(*v) for k, v in monitor.items()}})
                if "val" in monitor and self.history[-1]["val"] < best:
                    best, self.best_epoch = self.history[-1]["val"], epoch
                    state = {k: v.clone() for k, v in self.net.state_dict().items()}
        if monitor and "val" in monitor:
            self.net.load_state_dict(state)
        self.net.eval()
        return self

    @torch.no_grad()
    def sample(self, x: np.ndarray, k: int = 1, noise_scale: float = 1.0,
               seed: int | None = None, batch: int = 16384) -> np.ndarray:
        """`(len(x), k, action_dim)` plans; `noise_scale=0` gives the deterministic plan."""
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        cond = np.repeat(self.xs(x), k, axis=0)
        out = []
        for i in range(0, len(cond), batch):
            c = self._tensor(cond[i:i + batch])
            z = noise_scale * torch.randn(len(c), len(self.ys.mean), generator=g, device=self.device)
            dt = 1.0 / self.steps
            for s in range(self.steps):
                t = torch.full((len(c),), s * dt, device=self.device)
                z = z + dt * self.net(z, t, c)
            out.append(z.cpu().numpy())
        return self.ys.inverse(np.concatenate(out)).reshape(len(x), k, -1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.sample(np.asarray(x), k=1, noise_scale=0.0)[:, 0]

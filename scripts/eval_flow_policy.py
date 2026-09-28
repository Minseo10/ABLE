"""Offline evaluation of the flow policy against regression on the dream runs' learner rows.

Rows come from the learned-condition checkpoints; duplicate segments are dropped and the
remaining segments split 70/15/15 into train/val/test.

    python scripts/eval_flow_policy.py search    # flow configs, chosen on val energy score
    python scripts/eval_flow_policy.py scaling   # test metrics vs number of train segments
    python scripts/make_flow_figures.py
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn

from limb_baseline import dream_pipeline
from limb_baseline.flow import FlowPolicy, Standardizer

H, J = 12, 7
OUT = Path("data/dream/flow")
FIG = Path("figures/dream/flow")
FLOW_CONFIGS = [
    dict(width=256, depth=3, dropout=0.0, epochs=200),
    dict(width=512, depth=4, dropout=0.0, epochs=200),
    dict(width=512, depth=4, dropout=0.1, epochs=200),
    dict(width=512, depth=4, dropout=0.1, epochs=400),
    dict(width=512, depth=6, dropout=0.1, epochs=300),
    dict(width=1024, depth=4, dropout=0.1, epochs=200),
]
FRACTIONS = (0.03, 0.1, 0.3, 1.0)
KS = (1, 2, 4, 8, 16)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = dream_pipeline.__name__
        return super().find_class(module, name)


def load_rows(pattern: str):
    xs, ys, segs, runs = [], [], [], []
    offset = 0
    for run, path in enumerate(sorted(glob.glob(pattern))):
        if "cache" in path or "plain" in path:
            continue
        with open(path, "rb") as handle:
            learner = _Unpickler(handle).load()["learner"]
        x, y = np.asarray(learner.features), np.asarray(learner.labels)
        seg = np.cumsum(np.r_[1, np.diff(x[:, -1]) < 0]) - 1 + offset
        offset = seg.max() + 1
        xs.append(x), ys.append(y), segs.append(seg), runs.append(np.full(len(x), run))
    x, y, seg = np.vstack(xs), np.vstack(ys), np.concatenate(segs)
    keep, seen = np.zeros(len(x), bool), set()
    for s in np.unique(seg):
        m = seg == s
        key = hashlib.md5(np.round(np.hstack([x[m], y[m]]), 5).tobytes()).hexdigest()
        if key not in seen:
            seen.add(key)
            keep |= m
    return add_history(x[keep], y[keep], seg[keep]), y[keep], seg[keep]


def add_history(x, y, seg):
    first = np.r_[True, seg[1:] != seg[:-1]]
    pos = x[:, :13]
    prev_pos = np.vstack([pos[:1], pos[:-1]])
    prev_pos[first] = pos[first]
    prev_torque = np.vstack([y[:1, :J], y[:-1, :J]])
    prev_torque[first] = 0.0
    return np.hstack([x, pos - prev_pos, prev_torque])


def split(seg, seed=1234):
    u = np.random.default_rng(seed).permutation(np.unique(seg))
    n = len(u)
    val, test = set(u[:int(0.15 * n)]), set(u[int(0.15 * n):int(0.3 * n)])
    part = np.array(["test" if s in test else "val" if s in val else "train" for s in seg])
    return part


class Regression:
    """The best regressor from the sweep: MLP 256x2, dropout 0.3, early-stopped on val."""

    def __init__(self, epochs=150, seed=0, device="cuda"):
        self.epochs, self.seed, self.device = epochs, seed, device
        self.history = []

    def fit(self, x, y, monitor):
        torch.manual_seed(self.seed)
        self.xs, self.ys = Standardizer().fit(x), Standardizer().fit(y)
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)
        X, Y = t(self.xs(x)), t(self.ys(y))
        mon = {k: (t(self.xs(a)), t(self.ys(b))) for k, (a, b) in monitor.items()}
        self.net = nn.Sequential(nn.Linear(x.shape[1], 256), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(256, y.shape[1])).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3)
        per_epoch = max(1, math.ceil(len(x) / 256))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=self.epochs * per_epoch,
                                                    pct_start=0.05)
        best = float("inf")
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            perm = torch.randperm(len(X), device=self.device)
            for i in range(per_epoch):
                idx = perm[i * 256:(i + 1) * 256]
                loss = ((self.net(X[idx]) - Y[idx]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            if epoch % max(1, self.epochs // 40) == 0 or epoch == self.epochs:
                self.net.eval()
                with torch.no_grad():
                    row = {k: float(((self.net(a) - b) ** 2).mean()) for k, (a, b) in mon.items()}
                self.history.append({"epoch": epoch, **row})
                if row["val"] < best:
                    best, self.best_epoch = row["val"], epoch
                    state = {k: v.clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(state)
        self.net.eval()
        return self

    @torch.no_grad()
    def predict(self, x):
        out = self.net(torch.as_tensor(self.xs(x), dtype=torch.float32, device=self.device))
        return self.ys.inverse(out.cpu().numpy())


def nmse(pred, y, var):
    return float((((pred - y) ** 2) / var).mean())


def energy_score(samples, y, std):
    s, t = samples / std, y[:, None] / std
    d = np.sqrt(H * J)
    fit = np.linalg.norm(s - t, axis=-1).mean(1)
    spread = np.linalg.norm(s[:, :, None] - s[:, None], axis=-1).mean((1, 2)) / 2
    return float((fit - spread).mean() / d)


def best_of_k(samples, y, var):
    err = (((samples - y[:, None]) ** 2) / var).mean(-1)
    return {k: float(err[:, :k].min(1).mean()) for k in KS if k <= samples.shape[1]}


def evaluate_flow(model, x, y, var, std, seed=0):
    samples = model.sample(x, k=max(KS), seed=seed)
    return {"nmse_det": nmse(model.predict(x), y, var), "nmse_sample": nmse(samples[:, 0], y, var),
            "best_of_k": best_of_k(samples, y, var), "energy": energy_score(samples, y, std)}


def evaluate_regression(model, x, y, var, std):
    pred = model.predict(x)
    return {"nmse_det": nmse(pred, y, var), "energy": energy_score(pred[:, None], y, std)}


def data():
    x, y, seg = load_rows("data/dream/ep15/*_checkpoint.pkl")
    part = split(seg)
    sets = {p: (x[part == p], y[part == p]) for p in ("train", "val", "test")}
    return sets, seg, part


def search(args):
    sets, seg, part = data()
    (xtr, ytr), (xva, yva) = sets["train"], sets["val"]
    var, std = ytr.var(0), ytr.std(0)
    results = []
    for cfg in FLOW_CONFIGS:
        model = FlowPolicy(**cfg).fit(xtr, ytr, monitor={k: sets[k] for k in sets}, every=5)
        row = {"config": cfg, "best_epoch": model.best_epoch, "history": model.history,
               "val": evaluate_flow(model, xva, yva, var, std)}
        results.append(row)
        print(cfg, row["best_epoch"], {k: row["val"][k] for k in ("nmse_det", "energy")},
              row["val"]["best_of_k"], flush=True)
    reg = Regression().fit(xtr, ytr, monitor=sets)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"flow": results, "regression": {"history": reg.history, "best_epoch": reg.best_epoch,
               "val": evaluate_regression(reg, xva, yva, var, std)},
               "sizes": {k: {"rows": int((part == k).sum()), "segments": len(np.unique(seg[part == k]))}
                         for k in sets}},
              open(OUT / "search.json", "w"))


def best_config():
    rows = json.load(open(OUT / "search.json"))["flow"]
    return min(rows, key=lambda r: r["val"]["energy"])["config"]


def scaling(args):
    sets, seg, part = data()
    cfg = best_config()
    (xva, yva), (xte, yte) = sets["val"], sets["test"]
    train_segs = np.unique(seg[part == "train"])
    results = []
    for frac in FRACTIONS:
        for seed in range(args.seeds):
            keep = np.random.default_rng(seed).choice(
                train_segs, size=max(2, int(frac * len(train_segs))), replace=False)
            m = np.isin(seg, keep) & (part == "train")
            xtr, ytr = sets["train"][0][np.isin(seg[part == "train"], keep)], \
                sets["train"][1][np.isin(seg[part == "train"], keep)]
            var, std = ytr.var(0), ytr.std(0)
            var[var < 1e-12], std[std < 1e-6] = 1.0, 1.0
            monitor = {"train": (xtr, ytr), "val": (xva, yva), "test": (xte, yte)}
            scale = 1 / math.sqrt(frac)
            flow = FlowPolicy(**{**cfg, "epochs": int(cfg["epochs"] * scale), "seed": seed}).fit(
                xtr, ytr, monitor=monitor, every=max(1, int(5 * scale)))
            reg = Regression(epochs=int(150 * scale), seed=seed).fit(xtr, ytr, monitor)
            row = {"fraction": frac, "seed": seed, "segments": len(keep), "rows": int(m.sum()),
                   "flow": {"test": evaluate_flow(flow, xte, yte, var, std, seed),
                            "train_nmse_det": nmse(flow.predict(xtr), ytr, var),
                            "history": flow.history, "best_epoch": flow.best_epoch},
                   "regression": {"test": evaluate_regression(reg, xte, yte, var, std),
                                  "train_nmse_det": nmse(reg.predict(xtr), ytr, var),
                                  "history": reg.history, "best_epoch": reg.best_epoch}}
            results.append(row)
            print(frac, seed, len(keep), "flow", row["flow"]["test"]["nmse_det"],
                  row["flow"]["test"]["energy"], "reg", row["regression"]["test"]["nmse_det"],
                  row["regression"]["test"]["energy"], flush=True)
            json.dump(results, open(OUT / "scaling.json", "w"))
            if frac == 1.0 and seed == 0:
                torch.save({"flow": flow, "regression": reg}, OUT / "models_full.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["search", "scaling"])
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    {"search": search, "scaling": scaling}[args.stage](args)


if __name__ == "__main__":
    main()

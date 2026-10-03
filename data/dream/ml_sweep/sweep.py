"""Pure-ML fitting sweep for the dream pipeline's warm-start model.

Rows: features (26) -> next 12 executed torques x 7 joints (84), from the ep15 checkpoints.
Splits are by segment (a segment's rows are one torque sequence, so rows leak otherwise).
Metric: nMSE = mean over output dims of MSE / train variance (= 1 - mean R^2).

    python sweep.py <grid-name> [--workers 8]
Appends one JSON line per (dataset, model, seed) to results/<grid-name>.jsonl, skipping
jobs already there, so a killed sweep resumes.
"""

from __future__ import annotations

import argparse, itertools, json, os, sys, time, traceback, warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = np.load(HERE / "data.npz")
X_ALL, Y_ALL, RUN, SEG, NAMES = (DATA[k] for k in ("X", "Y", "run", "seg", "names"))
NAMES = [str(n) for n in NAMES]
H, J = 12, 7


def _dedupe(within_run: bool):
    """Keep only the first copy of each segment (runs share their early episodes), across
    all runs for pools, or within each run for per-run datasets."""
    import hashlib
    seen, keep = set(), np.zeros(len(SEG), bool)
    for s in np.unique(SEG):
        m = SEG == s
        k = hashlib.md5(np.round(np.hstack([X_ALL[m], Y_ALL[m]]), 5).tobytes()).hexdigest()
        if within_run:
            k = (k, int(RUN[m][0]))
        if k not in seen:
            seen.add(k)
            keep |= m
    return keep


KEEP_POOL, KEEP_RUN = _dedupe(False), _dedupe(True)

# Representative per-run datasets: 4-dream runs (45-75 segments) and 0-dream (9-15).
RUNS_4 = ["mlp_4_p23", "mlp_4_p25", "mlp_4_p29", "ridge_4_p24", "ridge_4_step_p26"]
RUNS_0 = ["mlp_0_p25", "ridge_0_p29", "mlp_0_step_p26"]


# ------------------------------------------------------------------ features


def add_features(x: np.ndarray, seg: np.ndarray, y: np.ndarray, kind: str) -> np.ndarray:
    """Optional structure the stored rows imply but the pipeline does not feed today."""
    parts = [x]
    first = np.r_[True, seg[1:] != seg[:-1]]
    if "vel" in kind:  # finite-difference of limb (0:6) and robot (6:13) positions
        pos = x[:, :13]
        prev = np.vstack([pos[:1], pos[:-1]])
        prev[first] = pos[first]
        parts.append(pos - prev)
    if "prev" in kind:  # torque executed at the previous control step (0 at segment start)
        prev = np.vstack([y[:1, :J], y[:-1, :J]])
        prev[first] = 0.0
        parts.append(prev)
    if "delta" in kind:  # goal minus limb position
        parts.append(x[:, 13:19] - x[:, 0:6])
    return np.hstack(parts)


def dataset(spec: str, feats: str, seed: int = 0):
    """`run:<name>` (chronological 80/20 by segment), `pool:<substr>` (random 80/20 segments
    over matching runs), or `poolfrac:<substr>:<f>` (pool, training on a fraction f)."""
    kind, *rest = spec.split(":")
    if kind == "run":
        m = (RUN == NAMES.index(rest[0])) & KEEP_RUN
    else:
        ids = [i for i, n in enumerate(NAMES) if rest[0] in n or rest[0] == "all"]
        m = np.isin(RUN, ids) & KEEP_POOL
    x, y, s = X_ALL[m], Y_ALL[m], SEG[m]
    x = add_features(x, s, y, feats)
    u = np.unique(s)
    if kind == "run":
        cut = u[int(0.8 * len(u))]
        tr = s < cut
    else:
        rng = np.random.default_rng(1234)  # fixed split across seeds
        val_segs = rng.choice(u, size=int(0.2 * len(u)), replace=False)
        tr = ~np.isin(s, val_segs)
        if kind == "poolfrac":
            frac = float(rest[1])
            tr_segs = np.unique(s[tr])
            keep = np.random.default_rng(seed).choice(
                tr_segs, size=max(2, int(frac * len(tr_segs))), replace=False)
            va = ~tr
            tr = np.isin(s, keep)
            return x[tr], y[tr], x[va], y[va], len(keep)
    return x[tr], y[tr], x[~tr], y[~tr], len(np.unique(s[tr]))


def nmse(y, p, var):
    return float((((y - p) ** 2).mean(0) / var).mean())


# -------------------------------------------------------------------- models


def cp_basis(k: int) -> np.ndarray:
    src, tgt = np.linspace(0, 1, k), np.linspace(0, 1, H)
    return np.stack([np.interp(tgt, src, e) for e in np.eye(k)], axis=1)  # (H, k)


def fit_torch(xtr, ytr, xva, yva, cfg, seed, device):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    xm, xs = xtr.mean(0), xtr.std(0)
    xs[xs < 1e-8] = 1.0
    ym, ys = ytr.mean(0), ytr.std(0)
    ys[ys < 1e-8] = 1.0
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    Xtr, Ytr = t((xtr - xm) / xs), t((ytr - ym) / ys)
    Xva, Yva = t((xva - xm) / xs), t((yva - ym) / ys)

    width, depth = cfg["width"], cfg["depth"]
    act = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}[cfg.get("act", "relu")]
    norm, drop = cfg.get("norm", False), cfg.get("dropout", 0.0)
    out_dim = cfg.get("cp", 0) * J or H * J
    K, mix = cfg.get("K", 1), cfg.get("mix", "wta")
    # K > 1: K candidate plans plus K mode logits (and, for an MDN, K log-scales).
    head_dim = out_dim if K == 1 else K * out_dim + K + (K if mix == "mdn" else 0)

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.f = nn.Sequential(*( [nn.LayerNorm(d)] if norm else []), nn.Linear(d, d), act(),
                                   *([nn.Dropout(drop)] if drop else []), nn.Linear(d, d))
        def forward(self, h):
            return h + self.f(h)

    if cfg.get("arch", "mlp") == "resmlp":
        layers = [nn.Linear(xtr.shape[1], width)] + [Block(width) for _ in range(depth)]
        layers += [*( [nn.LayerNorm(width)] if norm else []), act(), nn.Linear(width, head_dim)]
    else:
        layers, d = [], xtr.shape[1]
        for _ in range(depth):
            layers += [nn.Linear(d, width)] + ([nn.LayerNorm(width)] if norm else []) + [act()]
            layers += [nn.Dropout(drop)] if drop else []
            d = width
        layers.append(nn.Linear(d, head_dim))
    net = nn.Sequential(*layers).to(device)
    if cfg.get("cp"):
        # The net emits k control points per joint in per-joint standardized units; they are
        # expanded to the horizon in raw torque units, then standardized like the labels.
        B = t(np.kron(cp_basis(cfg["cp"]), np.eye(J)))  # (H*J, k*J), row-major (h, j)
        y3 = ytr.reshape(-1, H, J)
        jm, js = t(np.tile(y3.mean((0, 1)), cfg["cp"])), t(np.tile(y3.std((0, 1)) + 1e-8, cfg["cp"]))
        Ym, Ys = t(ym), t(ys)
        expand = lambda o: ((o * js + jm) @ B.T - Ym) / Ys
    else:
        expand = lambda o: o

    D = H * J

    def modes(x):
        """(B, K, D) candidate plans in standardized label units, (B, K) logits, log-scales."""
        o = net(x)
        if K == 1:
            return expand(o)[:, None], torch.zeros(len(o), 1, device=o.device), None
        mu = expand(o[:, :K * out_dim].reshape(-1, out_dim)).reshape(len(o), K, D)
        logits = o[:, K * out_dim:K * out_dim + K]
        logs = o[:, K * out_dim + K:].clamp(-5, 3) if mix == "mdn" else None
        return mu, logits, logs

    def loss_fn(x, y):
        mu, logits, logs = modes(x)
        err = ((mu - y[:, None]) ** 2).mean(-1)  # (B, K)
        if K == 1:
            return err[:, 0].mean()
        if mix == "mdn":
            var = torch.exp(2 * logs)
            ll = torch.log_softmax(logits, -1) - 0.5 * D * (err / var + 2 * logs)
            return -torch.logsumexp(ll, -1).mean() / D
        win = err.argmin(-1)
        # Relaxed winner-takes-all keeps losing heads from dying; the classifier learns the winner.
        return (err.gather(1, win[:, None]).mean() + cfg.get("eps", 0.05) * err.mean()
                + torch.nn.functional.cross_entropy(logits, win))

    def scores(x, y):
        """(top-1 error, oracle best-of-K error) in standardized units."""
        mu, logits, _ = modes(x)
        err = ((mu - y[:, None]) ** 2).mean(-1)
        return float(err.gather(1, logits.argmax(-1)[:, None]).mean()), float(err.min(-1)[0].mean())

    def top1(x):
        mu, logits, _ = modes(x)
        return mu[torch.arange(len(mu)), logits.argmax(-1)]

    epochs, bs = cfg.get("epochs", 2000), cfg.get("bs", 256)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("wd", 0.0))
    steps_per = max(1, int(np.ceil(len(Xtr) / bs)))
    total = epochs * steps_per
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 1e-3), total_steps=total,
                                                pct_start=0.05)
    noise = cfg.get("noise", 0.0)
    best, best_ep, curve = (1e9, None), 0, []
    eval_every = max(1, epochs // 40)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(steps_per):
            idx = perm[i * bs:(i + 1) * bs]
            xb = Xtr[idx]
            if noise:
                xb = xb + noise * torch.randn_like(xb)
            loss = loss_fn(xb, Ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                tr_l, _ = scores(Xtr, Ytr)
                va_l, _ = scores(Xva, Yva)
            curve.append((ep + 1, round(tr_l, 5), round(va_l, 5)))
            if va_l < best[0]:
                best, best_ep = (va_l, tr_l), ep + 1
    net.eval()
    with torch.no_grad():
        ptr = top1(Xtr).cpu().numpy() * ys + ym
        pva = top1(Xva).cpu().numpy() * ys + ym
        tr_or, va_or = scores(Xtr, Ytr)[1], scores(Xva, Yva)[1]
    nparams = sum(p.numel() for p in net.parameters())
    return ptr, pva, {"best_val_std": best[0], "best_ep": best_ep, "curve": curve, "params": nparams,
                      "train_oracle_std": tr_or, "val_oracle_std": va_or}


def fit_sklearn(xtr, ytr, xva, cfg, seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import RidgeCV
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

    kind = cfg["model"]
    if kind == "ridge":
        steps = [StandardScaler()]
        if cfg.get("poly"):
            steps += [PolynomialFeatures(cfg["poly"], include_bias=False), StandardScaler()]
        est = make_pipeline(*steps, RidgeCV(alphas=np.logspace(-4, 5, 30)))
    elif kind == "knn":
        est = make_pipeline(StandardScaler(), KNeighborsRegressor(cfg["k"], weights=cfg.get("w", "distance")))
    elif kind == "et":
        est = ExtraTreesRegressor(cfg.get("n", 300), min_samples_leaf=cfg.get("leaf", 1),
                                  max_features=cfg.get("mf", 1.0), n_jobs=2, random_state=seed)
    elif kind == "rf":
        est = RandomForestRegressor(cfg.get("n", 300), min_samples_leaf=cfg.get("leaf", 1),
                                    max_features=cfg.get("mf", 1.0), n_jobs=2, random_state=seed)
    elif kind == "sk_mlp":  # the pipeline's current model, for reference
        sys.path.insert(0, "/home/minseo/research/learning-forceful-manipulation/src")
        from limb_baseline.warmstart import make_regressor
        est = make_regressor(cfg["spec"])
    est.fit(xtr, ytr)
    return est.predict(xtr), est.predict(xva), {}


# ---------------------------------------------------------------------- grids


def mlp(width, depth, **kw):
    return {"model": "torch", "arch": "mlp", "width": width, "depth": depth, **kw}


def grids():
    per_run = [f"run:{r}" for r in RUNS_4 + RUNS_0]
    pools = ["pool:all", "pool:_4_"]
    g = {}
    # 1) Capacity: no regularization, long training. Does train loss go to ~0?
    g["capacity"] = [(d, "base", m) for d in per_run + pools for m in (
        [{"model": "ridge"}, {"model": "ridge", "poly": 2}, {"model": "knn", "k": 1},
         {"model": "knn", "k": 5}, {"model": "et", "leaf": 1}]
        + [mlp(w, dep, epochs=1500 if d.startswith("run") else 400)
           for w in (64, 256, 1024) for dep in (1, 2, 4)]
        + [{"model": "torch", "arch": "resmlp", "width": 256, "depth": 4, "norm": True,
            "epochs": 1500 if d.startswith("run") else 400}]
    )]
    g["current"] = [(d, "base", {"model": "sk_mlp", "spec": s}) for d in per_run
                    for s in ("mlp:128x128:a0.001", "ridge")]
    # 2) Structure: which inputs and output parameterization carry signal?
    feat_sets = ["base", "base+vel", "base+prev", "base+vel+prev", "base+vel+prev+delta"]
    run4 = [f"run:{r}" for r in RUNS_4]
    g["structure"] = [(d, f, m) for d in run4 + ["pool:all"] for f in feat_sets for m in (
        [{"model": "ridge"}, {"model": "et", "leaf": 5, "mf": 0.5}]
        + [mlp(256, 2, epochs=300 if d.startswith("run") else 100, **kw)
           for kw in ({}, {"cp": 4}, {"cp": 2})]
    )]
    # 3) Regularization, per run, where data is scarce.
    g["regularize"] = [(d, f, mlp(w, 2, epochs=300, **kw)) for d in run4
                       for f in ("base", "base+vel+prev") for w in (64, 256)
                       for kw in ({"wd": 1e-2}, {"wd": 1e-1}, {"wd": 1.0}, {"dropout": 0.1},
                                  {"dropout": 0.3}, {"noise": 0.1}, {"noise": 0.3},
                                  {"dropout": 0.3, "noise": 0.3, "wd": 0.1})]
    # 4) Data scaling: val loss vs number of training segments (pooled over people).
    g["scaling"] = [(f"poolfrac:all:{fr}", f, m) for fr in (0.03, 0.1, 0.3, 1.0)
                    for f in ("base", "base+vel+prev") for m in (
                        {"model": "ridge"}, mlp(256, 3, epochs=150), mlp(1024, 3, epochs=150))]
    # 5) Multimodal heads: do distinct plan modes exist, and can the input pick one?
    g["multimodal"] = [(d, f, mlp(256, 3, epochs=100 if d.startswith("pool") else 300, **kw))
                       for d in ["pool:all"] + run4 for f in ("base", "base+vel+prev")
                       for kw in ([{"K": 1}] + [{"K": k, "mix": m} for k in (2, 4, 8, 16)
                                                for m in ("wta", "mdn")])]
    return g


# ---------------------------------------------------------------------- runner


def key(ds, feats, cfg, seed):
    return json.dumps([ds, feats, cfg, seed], sort_keys=True)


def run_job(args):
    ds, feats, cfg, seed, device = args
    warnings.filterwarnings("ignore")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    try:
        import torch
        torch.set_num_threads(2)
    except ImportError:
        pass
    t0 = time.time()
    try:
        xtr, ytr, xva, yva, nseg = dataset(ds, feats, seed)
        var = ytr.var(0)
        var[var < 1e-12] = 1.0
        if cfg["model"] == "torch":
            ptr, pva, extra = fit_torch(xtr, ytr, xva, yva, cfg, seed, device)
        else:
            ptr, pva, extra = fit_sklearn(xtr, ytr, xva, cfg, seed)
        res = {"train": nmse(ytr, ptr, var), "val": nmse(yva, pva, var),
               "val_first": nmse(yva[:, :J], pva[:, :J], var[:J]),  # first torque only
               "n_train": len(xtr), "n_val": len(xva), "n_seg": nseg, **extra}
    except Exception:
        res = {"error": traceback.format_exc()[-800:]}
    return {"ds": ds, "feats": feats, "cfg": cfg, "seed": seed, "sec": round(time.time() - t0, 1), **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    out = HERE / "results" / f"{a.grid}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            r = json.loads(line)
            if "error" not in r:
                done.add(key(r["ds"], r["feats"], r["cfg"], r["seed"]))
    jobs = [(d, f, c, s, "cuda") for (d, f, c) in grids()[a.grid] for s in range(a.seeds)
            if key(d, f, c, s) not in done]
    # Slow jobs (pools, big nets) first.
    jobs.sort(key=lambda j: -(len(j[0]) * 0 + (j[0].startswith("pool")) * 10
                              + j[2].get("width", 0) * j[2].get("depth", 0) / 1000))
    print(f"{len(jobs)} jobs ({len(done)} done)", flush=True)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool, out.open("a") as fh:
        for i, r in enumerate(pool.imap_unordered(run_job, jobs)):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            tag = r.get("error", "").splitlines()[-1:] if "error" in r else ""
            print(f"[{i+1}/{len(jobs)}] {r['ds']} {r['feats']} {r['cfg']} "
                  f"train={r.get('train', float('nan')):.4f} val={r.get('val', float('nan')):.4f} "
                  f"{r['sec']}s {tag}", flush=True)


if __name__ == "__main__":
    main()

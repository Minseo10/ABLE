"""Does learning theta matter? Count reachable placement slots under the prior vs. the true theta.

A slot counts as usable under belief b if P_b(m(x, theta) >= 0) >= 1 - delta.
Usage: python scripts/bedside/slot_analysis.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from layout import PRIOR_HI, PRIOR_LO, REGIONS, all_slots, margin  # noqa: E402

DELTA = 0.1
N_PARTICLES = 20000
N_USERS = 2000

rng = np.random.default_rng(0)
names, pts = all_slots()
prior = rng.uniform(PRIOR_LO, PRIOR_HI, size=(N_PARTICLES, 3))
users = rng.uniform(PRIOR_LO, PRIOR_HI, size=(N_USERS, 3))

p_reach = (margin(pts, prior) >= 0).mean(0)
safe_prior = p_reach >= 1 - DELTA
reach_true = margin(pts, users) >= 0  # (users, slots)
n_true = reach_true.sum(1)

print(f"total slots: {len(pts)}")
print(f"{'region':<20}{'slots':>6}{'prior-safe':>12}{'true (median)':>15}{'P(reach) range':>18}")
for r in REGIONS:
    k = names == r
    print(f"{r:<20}{k.sum():>6}{safe_prior[k].sum():>12}{int(np.median(reach_true[:, k].sum(1))):>15}"
          f"{p_reach[k].min():>9.2f}-{p_reach[k].max():.2f}")
print(f"\nprior-safe slots (delta={DELTA}): {safe_prior.sum()}")
q = np.percentile(n_true, [5, 25, 50, 75, 95]).astype(int)
print(f"true-theta reachable slots, percentiles 5/25/50/75/95: {q.tolist()}")
for n_req in (6, 8, 10):
    print(f"n_req={n_req}: P(prior-only feasible)={(safe_prior.sum() >= n_req):d}, "
          f"P(feasible with true theta)={(n_true >= n_req).mean():.2f}")

# How fast does the usable set grow as observations arrive? Random binary queries at slot locations.
EPS, SIGMA = 0.02, 0.05
from scipy.stats import norm  # noqa: E402

print("\nqueries -> prior-safe slots under posterior (median over 50 users, 5/95 pct)")
m_prior = margin(pts, prior)  # (particles, slots)
test_users = users[:50]
for k in (0, 3, 6, 10, 20, 40):
    counts = []
    for u in test_users:
        w = np.ones(N_PARTICLES)
        idx = rng.choice(len(pts), size=k)
        y = margin(pts[idx], u[None])[0] >= 0
        for j, yj in zip(idx, y):
            ps = EPS + (1 - 2 * EPS) * norm.cdf(m_prior[:, j] / SIGMA)
            w *= ps if yj else 1 - ps
        w /= w.sum()
        safe = (w @ (m_prior >= 0)) >= 1 - DELTA
        counts.append(safe.sum())
    c = np.array(counts)
    print(f"k={k:>2}: median {int(np.median(c)):>2}  [{int(np.percentile(c, 5))}, {int(np.percentile(c, 95))}]")

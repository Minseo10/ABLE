"""Show one belief update: prior, one query, posterior, with the noise measured.

`python scripts/make_update_figure.py --out figures`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from limb_baseline.belief import Belief  # noqa: E402
from limb_baseline.human import (  # noqa: E402
    DEFAULT_PRIOR,
    active_motions,
    default_magnitudes,
)
from limb_baseline.planner import QueryPolicy, gather_information  # noqa: E402
from limb_baseline.queries import DEFAULT_QUERY_MODEL  # noqa: E402

KEEP = "#008D63"
INK = "#141C1B"
FAINT = "#6E7A78"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": FAINT,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": FAINT,
        "ytick.color": FAINT,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def normal(x, mean, stddev):
    return np.exp(-0.5 * ((x - mean) / stddev) ** 2) / (stddev * np.sqrt(2 * np.pi))


def panel_densities(ax, prior, posterior, index, truth) -> None:
    """The belief over theta before and after the query."""
    x = np.linspace(0.62, 1.30, 800)
    before = normal(x, prior.mean[index], prior.stddev[index])
    after = normal(x, posterior.mean[index], posterior.stddev[index])

    ax.fill_between(x, before, color=FAINT, alpha=0.18, lw=0)
    ax.plot(x, before, color=FAINT, lw=1.8)
    ax.fill_between(x, after, color=KEEP, alpha=0.20, lw=0)
    ax.plot(x, after, color=KEEP, lw=2.0)
    ax.axvline(truth, color=INK, lw=1.2, ls=":")

    top = after.max()
    ax.text(prior.mean[index] - 0.13, before.max() + 1.4,
            "prior\n$\\mathcal{N}(0.9,\\,0.15^2)$", color=FAINT, fontsize=9, ha="center")
    ax.text(posterior.mean[index] - 0.045, top * 0.55, "posterior", color=KEEP,
            fontsize=9, ha="right")
    ax.text(truth, top * 1.06, r"true $\theta$", color=INK, fontsize=9, ha="center")
    ax.set_xlabel(r"$\theta$   (share of the textbook range)")
    ax.set_ylabel("density")
    ax.set_ylim(0, top * 1.18)
    ax.set_yticks([])


def panel_math(ax, prior, posterior, index, tau, degrees, truth) -> None:
    """Bayes' rule, the noise estimator, and the closed form they give."""
    ax.axis("off")
    rows = [
        (0.95, "Bayes' rule",
         r"$p(\theta \mid y) \;\propto\; p(y \mid \theta)\;p(\theta)$"),
        (0.72, "the answer, corrected for the short fall",
         r"$y \;=\; R\,\theta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,\,\tau^2)$"),
        (0.49, "the noise, measured from repeat answers",
         r"$\hat{\tau}^{2} \;=\; \dfrac{1}{n}\sum_{m,\,i} \left(a_{m i} - \bar{a}_{m}\right)^{2}$"),
        (0.26, "certainties add",
         r"$\dfrac{1}{\sigma_{+}^{2}} \;=\; \dfrac{1}{\sigma_{-}^{2}} \;+\; \dfrac{R^{2}}{\hat{\tau}^{2}}$"),
        (0.03, "the mean is their weighted average",
         r"$\mu_{+} \;=\; \sigma_{+}^{2}\left(\dfrac{\mu_{-}}{\sigma_{-}^{2}} + \dfrac{R\,y}{\hat{\tau}^{2}}\right)$"),
    ]
    for row_y, caption, equation in rows:
        ax.text(0.0, row_y + 0.085, caption, fontsize=9, color=FAINT, va="center")
        ax.text(0.0, row_y, equation, fontsize=13, color=INK, va="center")
    ax.text(0.0, -0.10,
            f"$\\hat{{\\tau}}={tau:.3f}$ rad from $n={degrees}$ repeats"
            f"    $\\mu_-={prior.mean[index]:.2f},\\ \\sigma_-={prior.stddev[index]:.2f}$"
            f"   $\\rightarrow$   $\\mu_+={posterior.mean[index]:.3f},\\ "
            f"\\sigma_+={posterior.stddev[index]:.3f}$"
            f"    (true $\\theta={truth:.3f}$)",
            fontsize=9, color=FAINT, va="center")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--motion", type=str, default="shoulder_abduction")
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()

    model = DEFAULT_QUERY_MODEL
    index = active_motions().index(args.motion)
    magnitude = default_magnitudes()[index]

    # The same session the videos come from, so the numbers agree across figures.
    rng = np.random.default_rng(args.seed)
    theta = DEFAULT_PRIOR.sample(rng)
    truth = float(theta[index])
    gathered = gather_information(
        theta, rng, QueryPolicy(num_queries=args.num_queries), estimate_tau=True
    )
    counts = np.zeros(len(active_motions()), dtype=int)
    for outcome in gathered.outcomes:
        if outcome.settled and not outcome.reached_target:
            counts[outcome.motion_index] += 1
    degrees = int(np.sum(np.maximum(counts - 1, 0)))
    answer = next(
        o.answer for o in gathered.outcomes
        if o.motion_index == index and o.settled and not o.reached_target
    )

    prior = Belief.from_prior(DEFAULT_PRIOR)
    posterior = prior.updated(index, answer, gathered.tau, model.short_fall)
    print(f"tau_hat={gathered.tau:.4f} from n={degrees}   answer={answer:.3f}   "
          f"y={answer + model.short_fall:.3f}  y/R={(answer + model.short_fall) / magnitude:.4f}  "
          f"true={truth:.4f}   mean {prior.mean[index]:.3f}->{posterior.mean[index]:.4f}  "
          f"stddev {prior.stddev[index]:.3f}->{posterior.stddev[index]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2),
                             gridspec_kw={"width_ratios": [1.0, 1.05]})
    panel_densities(axes[0], prior, posterior, index, truth)
    panel_math(axes[1], prior, posterior, index, gathered.tau, degrees, truth)
    for side in ("top", "right", "left"):
        axes[0].spines[side].set_visible(False)
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = args.out / f"update_model.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

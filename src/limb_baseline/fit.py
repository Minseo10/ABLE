"""Stage 3: fit `g`, the model that turns a task and a belief into a warm start.

Held out by person, not by task: `theta` and the belief are six of the twenty-five
features, so a model that had seen a person's other tasks could look them up.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from kinder_pddlstream_planning.limbrepositioning3d.stream import MPCConfig
from numpy.typing import NDArray

from limb_baseline.dataset import load_dataset, train_test_split_by_person
from limb_baseline.warmstart import WarmStartModel, fit_warm_start, make_regressors


@dataclass
class FitReport:
    """How one regressor did on the test people."""

    name: str
    train_mse: float
    test_mse: float
    zero_baseline_mse: float  # predicts all zero: plain pddlstream without learning
    zero_baseline_train_mse: float = 0.0

    @property
    def improvement(self) -> float:
        """Fraction of the zero-warm-start error the model removes.

        The MPC starts from zeros, so below 0.0 is worse than no warm start at all.
        """
        if self.zero_baseline_mse == 0.0:
            return 0.0
        return 1.0 - self.test_mse / self.zero_baseline_mse


def _mse(predicted: NDArray[np.float64], actual: NDArray[np.float64]) -> float:
    return float(np.mean((predicted - actual) ** 2))


def fit_all(
    dataset_path: str | Path,
    test_fraction: float = 0.2,
    seed: int = 0,
    mpc: MPCConfig | None = None,
) -> tuple[dict[str, FitReport], dict[str, WarmStartModel]]:
    """Fit every regressor and score it against doing nothing."""
    mpc = mpc or MPCConfig()
    features, labels, metadata = load_dataset(dataset_path)
    train_x, train_y, test_x, test_y = train_test_split_by_person(
        features, labels, metadata, test_fraction, seed
    )
    zero_baseline = _mse(np.zeros_like(test_y), test_y)
    zero_baseline_train = _mse(np.zeros_like(train_y), train_y)
    reports: dict[str, FitReport] = {}
    models: dict[str, WarmStartModel] = {}
    for name in make_regressors():
        model = fit_warm_start(name, train_x, train_y, mpc)
        predict = model.estimator.predict
        reports[name] = FitReport(
            name=name,
            train_mse=_mse(predict(train_x), train_y),
            test_mse=_mse(predict(test_x), test_y),
            zero_baseline_mse=zero_baseline,
            zero_baseline_train_mse=zero_baseline_train,
        )
        models[name] = model
    return reports, models


def best_model(
    reports: dict[str, FitReport], models: dict[str, WarmStartModel]
) -> WarmStartModel:
    """The regressor with the lowest error on the test people."""
    name = min(reports, key=lambda n: reports[n].test_mse)
    return models[name]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="A collect run directory, e.g. data/babble/isolated_left_arm.")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report_path = args.run_dir / "fit.json"
    model_path = args.run_dir / "warm_start.pkl"

    reports, models = fit_all(args.run_dir / "demos.csv", args.test_fraction, args.seed)
    width = max(len(n) for n in reports)
    print(f"{'model':{width}s}  {'train mse':>10s}  {'test mse':>10s}  {'vs zeros':>9s}")
    for name, report in sorted(reports.items(), key=lambda kv: kv[1].test_mse):
        print(
            f"{name:{width}s}  {report.train_mse:10.5f}  "
            f"{report.test_mse:10.5f}  {report.improvement:+8.1%}"
        )
    report_path.write_text(
        json.dumps({n: vars(r) for n, r in reports.items()}, indent=2), encoding="utf-8"
    )
    chosen = best_model(reports, models)
    with open(model_path, "wb") as handle:
        pickle.dump(chosen, handle)
    for name, model in models.items():
        with open(args.run_dir / f"warm_start_{name}.pkl", "wb") as handle:
            pickle.dump(model, handle)
    print(f"\nbest model: {chosen.name}")
    print(f"report -> {report_path.resolve()}")
    print(f"model  -> {model_path.resolve()}")


if __name__ == "__main__":
    main()

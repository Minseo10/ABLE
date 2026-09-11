"""`psi`: the MPC's initial nominal control points, and the model that predicts them.

A demonstration is much longer than one horizon, so the label is its first horizon's
worth projected back onto the MPC's control-point basis.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from kinder.envs.dynamic3d.limb_utils import NUM_ROBOT_JOINTS
from kinder_pddlstream_planning.limbrepositioning3d.stream import MPCConfig
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def control_basis(mpc: MPCConfig) -> NDArray[np.float64]:
    """The `(horizon, num_control_points)` map the MPC's `_expand` applies."""
    source = np.linspace(0.0, 1.0, mpc.num_control_points)
    target = np.linspace(0.0, 1.0, mpc.horizon)
    return np.stack(
        [np.interp(target, source, e) for e in np.eye(mpc.num_control_points)], axis=1
    )


def project_torques(
    robot_torques: list[list[float]], mpc: MPCConfig
) -> NDArray[np.float64]:
    """Fit control points to the first horizon of an executed torque sequence.

    `robot_torques` is per simulator step, so it is decimated back to control steps.
    """
    torques = np.asarray(robot_torques, dtype=np.float64)
    assert torques.ndim == 2 and torques.shape[1] == NUM_ROBOT_JOINTS
    control_steps = torques[:: mpc.action_repeat]
    window = control_steps[: mpc.horizon]
    if len(window) < mpc.horizon:
        padding = np.repeat(window[-1:], mpc.horizon - len(window), axis=0)
        window = np.vstack([window, padding])
    basis = control_basis(mpc)
    solution, *_ = np.linalg.lstsq(basis, window, rcond=None)
    return solution


def make_regressors() -> dict[str, object]:
    """The off-the-shelf multi-output regressors the baseline compares."""
    return {
        # Cross-validated on the training rows, never on the test people.
        "ridge": make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 25))
        ),
        # lbfgs beats adam on a few hundred rows, and `alpha` does the regularizing
        # early stopping was doing badly. Chosen on grouped CV within training.
        "mlp": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 64),
                solver="lbfgs",
                alpha=10.0,
                max_iter=2000,
                random_state=0,
            ),
        ),
        # l2 shrinks each leaf by n/(n+l2), damping the small noisy ones; the feature
        # and leaf caps decorrelate the trees. Chosen on grouped CV within training.
        "gradient_boosting": MultiOutputRegressor(
            __import__(
                "sklearn.ensemble", fromlist=["HistGradientBoostingRegressor"]
            ).HistGradientBoostingRegressor(
                random_state=0,
                l2_regularization=100.0,
                max_features=0.5,
                max_leaf_nodes=15,
            )
        ),
    }


@dataclass
class ScaledMLP:
    """An MLP over standardized features, trainable in rounds from its current weights.

    `warm_start` is what makes personalization possible: each `train` runs more epochs
    from the weights already there rather than starting over.
    """

    scaler: StandardScaler
    net: MLPRegressor
    # The variant it was fit on: theta means something different under another limb.
    variant: str = ""

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.net.predict(self.scaler.transform(features))

    def train(self, features, labels) -> ScaledMLP:
        """One round of training.

        The scaler is fit once and then frozen: refitting it every round would move
        what a feature value means out from under the weights already learned.
        """
        if not hasattr(self.scaler, "mean_"):
            self.scaler.fit(features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.net.fit(self.scaler.transform(features), labels)
        return self


def make_scaled_mlp(
    epochs: int,
    learning_rate: float,
    hidden: tuple[int, ...] = (256, 256),
    seed: int = 0,
    variant: str = "",
) -> ScaledMLP:
    """A fresh fine-tunable MLP, `epochs` passes per `train` call."""
    return ScaledMLP(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden,
            warm_start=True,
            max_iter=epochs,
            learning_rate_init=learning_rate,
            # No loss-based early stop: every round must spend the same budget, or the
            # two versions are not training under the same rule.
            tol=0.0,
            n_iter_no_change=epochs,
            random_state=seed,
        ),
        variant,
    )


def segment_features(
    limb_positions,
    robot_positions,
    goal: NDArray[np.float64],
    belief_features: NDArray[np.float64],
    step: float = 0.0,
) -> NDArray[np.float64]:
    """The feature vector for one control step of a `move_limb`.

    Where the limb and robot stand, where the limb is aimed, what is believed about the
    person, and how far into the motion this is: the opening push and a late correction
    need different torques from the same pose. Shared by training and prediction.
    """
    return np.concatenate(
        [
            np.asarray(limb_positions, dtype=np.float64),
            np.asarray(robot_positions, dtype=np.float64)[:NUM_ROBOT_JOINTS],
            np.asarray(goal, dtype=np.float64),
            np.asarray(belief_features, dtype=np.float64),
            [float(step)],
        ]
    )


@dataclass
class WarmStartModel:
    """A fitted `g`, mapping task and belief features to `psi`."""

    name: str
    estimator: object
    mpc: MPCConfig

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """The control points for one feature vector."""
        flat = self.estimator.predict(np.asarray(features).reshape(1, -1))[0]
        return flat.reshape(self.mpc.num_control_points, NUM_ROBOT_JOINTS)

    def as_hook(self, belief_features: NDArray[np.float64]):
        """The callable `LimbStreamContext.warm_start` expects.

        Per segment: each `move_limb` builds its own MPC from its own starting state.
        """

        def hook(state, goal: NDArray[np.float64], step: float = 0.0):
            return self.predict(
                segment_features(
                    state.limb_positions, state.robot_positions, goal,
                    belief_features, step,
                )
            )

        return hook


def fit_warm_start(
    name: str,
    features: NDArray[np.float64],
    labels: NDArray[np.float64],
    mpc: MPCConfig,
) -> WarmStartModel:
    """Fit one regressor on `(features, flattened control points)`."""
    estimator = make_regressors()[name]
    estimator.fit(features, labels)
    return WarmStartModel(name=name, estimator=estimator, mpc=mpc)

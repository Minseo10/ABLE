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

# Limb (6) and robot (7) positions, goal (6), belief (6) and step: what every model reads.
BASE_FEATURES = 26


def control_basis(mpc: MPCConfig) -> NDArray[np.float64]:
    """The `(horizon, num_control_points)` map the MPC's `_expand` applies."""
    source = np.linspace(0.0, 1.0, mpc.num_control_points)
    target = np.linspace(0.0, 1.0, mpc.horizon)
    return np.stack(
        [np.interp(target, source, e) for e in np.eye(mpc.num_control_points)], axis=1
    )


def horizon_window(
    robot_torques: list[list[float]], mpc: MPCConfig
) -> NDArray[np.float64]:
    """The first horizon of an executed torque sequence, one row per control step.

    `robot_torques` is per simulator step, so it is decimated back to control steps and
    padded with its last entry when the sequence ends inside the horizon.
    """
    torques = np.asarray(robot_torques, dtype=np.float64)
    assert torques.ndim == 2 and torques.shape[1] == NUM_ROBOT_JOINTS
    window = torques[:: mpc.action_repeat][: mpc.horizon]
    if len(window) < mpc.horizon:
        window = np.vstack([window, np.repeat(window[-1:], mpc.horizon - len(window), axis=0)])
    return window


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
        # The same net with little regularization, for fitting one person's or one
        # task's own rows online, where fitting them closely is the point.
        "mlp_light": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 64),
                solver="lbfgs",
                alpha=1e-2,
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


def base_columns(features: NDArray[np.float64], n: int = BASE_FEATURES) -> NDArray[np.float64]:
    return np.asarray(features)[:, :n]


def make_regressor(spec: str):
    """A regressor by name, or by a spec string for tuning:

    - `ridge` / `mlp` / ...: the named entries of `make_regressors`;
    - `ridge:a<lo>,<hi>[:poly2]`: RidgeCV over alphas 10**lo..10**hi, optionally on
      degree-2 polynomial features;
    - `mlp:<h1>x<h2>...:a<alpha>`: an lbfgs MLP with those hidden sizes and L2 alpha;
    - `net:<h1>x<h2>...:d<dropout>[:s<steps>]`: a `DropoutMLP`.

    Each reads only the first `BASE_FEATURES` columns unless the spec ends in `:hist`.
    """
    if spec.endswith(":hist"):
        return _make_regressor(spec[: -len(":hist")])
    from sklearn.preprocessing import FunctionTransformer

    return make_pipeline(FunctionTransformer(base_columns), _make_regressor(spec))


def _make_regressor(spec: str):
    named = make_regressors()
    if spec in named:
        return named[spec]
    kind, *parts = spec.split(":")
    if kind == "ridge":
        lo, hi = -1.0, 5.0
        poly = False
        for part in parts:
            if part.startswith("a"):
                lo, hi = (float(v) for v in part[1:].split(","))
            elif part == "poly2":
                poly = True
            else:
                raise ValueError(f"unknown ridge option {part!r} in {spec!r}")
        steps = [StandardScaler()]
        if poly:
            from sklearn.preprocessing import PolynomialFeatures

            steps += [PolynomialFeatures(2, include_bias=False), StandardScaler()]
        return make_pipeline(*steps, RidgeCV(alphas=np.logspace(lo, hi, 25)))
    if kind == "mlp":
        hidden, alpha = (64, 64), 1e-2
        for part in parts:
            if part.startswith("a"):
                alpha = float(part[1:])
            else:
                hidden = tuple(int(v) for v in part.split("x"))
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=hidden, solver="lbfgs", alpha=alpha,
                         max_iter=2000, random_state=0),
        )
    if kind == "net":
        options: dict = {}
        for part in parts:
            if part.startswith("d"):
                options["dropout"] = float(part[1:])
            elif part.startswith("s"):
                options["steps"] = int(part[1:])
            else:
                options["hidden"] = tuple(int(v) for v in part.split("x"))
        return DropoutMLP(**options)
    raise ValueError(f"unknown model spec {spec!r}")


@dataclass
class DropoutMLP:
    """An MLP with dropout, trained by Adam and kept at its best held-out loss.

    A sixth of the segments (rows restart at step 0) is held out to pick the weights, when
    there are at least `min_segments`; otherwise all rows train for the full `steps`.
    """

    hidden: tuple[int, ...] = (256, 256)
    dropout: float = 0.3
    steps: int = 4000
    batch_size: int = 256
    lr: float = 1e-3
    min_segments: int = 6
    seed: int = 0

    def fit(self, features, labels) -> DropoutMLP:
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        x, y = np.asarray(features, np.float64), np.asarray(labels, np.float64)
        self.n_features_in_ = x.shape[1]
        self.x_mean, self.x_std = x.mean(0), np.where(x.std(0) < 1e-8, 1.0, x.std(0))
        self.y_mean, self.y_std = y.mean(0), np.where(y.std(0) < 1e-8, 1.0, y.std(0))
        segment = np.cumsum(x[:, BASE_FEATURES - 1] == 0)
        held = np.zeros(len(x), bool)
        if segment[-1] >= self.min_segments:
            rng = np.random.default_rng(self.seed)
            chosen = rng.choice(np.arange(1, segment[-1] + 1), size=segment[-1] // 6, replace=False)
            held = np.isin(segment, chosen)
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float32)
        xs, ys = tensor((x - self.x_mean) / self.x_std), tensor((y - self.y_mean) / self.y_std)
        train = torch.as_tensor(np.where(~held)[0])
        layers, width = [], x.shape[1]
        for size in self.hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(self.dropout)]
            width = size
        self.net = nn.Sequential(*layers, nn.Linear(width, y.shape[1]))
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, self.lr, total_steps=self.steps, pct_start=0.05)
        best, state = float("inf"), None
        generator = torch.Generator().manual_seed(self.seed)
        for step in range(1, self.steps + 1):
            self.net.train()
            batch = train[torch.randint(len(train), (self.batch_size,), generator=generator)]
            loss = ((self.net(xs[batch]) - ys[batch]) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            if held.any() and (step % 100 == 0 or step == self.steps):
                self.net.eval()
                with torch.no_grad():
                    val = float(((self.net(xs[held]) - ys[held]) ** 2).mean())
                if val < best:
                    best, state = val, {k: v.clone() for k, v in self.net.state_dict().items()}
        if state is not None:
            self.net.load_state_dict(state)
        self.net.eval()
        return self

    def predict(self, features) -> NDArray[np.float64]:
        import torch

        x = (np.asarray(features, np.float64) - self.x_mean) / self.x_std
        with torch.no_grad():
            out = self.net(torch.as_tensor(x, dtype=torch.float32)).numpy()
        return out * self.y_std + self.y_mean


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
    limb_velocities=None,
    robot_velocities=None,
    previous_torque=None,
) -> NDArray[np.float64]:
    """The feature vector for one control step of a `move_limb`.

    Where the limb and robot stand, where the limb is aimed, what is believed about the
    person, and how far into the motion this is: the opening push and a late correction
    need different torques from the same pose. Shared by training and prediction.

    Given velocities, the limb and robot velocities and the torque applied at the previous
    control step (zero at a segment's start) follow the first `BASE_FEATURES`; only a
    `:hist` model reads them.
    """
    parts = [
        np.asarray(limb_positions, dtype=np.float64),
        np.asarray(robot_positions, dtype=np.float64)[:NUM_ROBOT_JOINTS],
        np.asarray(goal, dtype=np.float64),
        np.asarray(belief_features, dtype=np.float64),
        [float(step)],
    ]
    if limb_velocities is not None:
        previous = np.zeros(NUM_ROBOT_JOINTS) if previous_torque is None else previous_torque
        parts += [
            np.asarray(limb_velocities, dtype=np.float64),
            np.asarray(robot_velocities, dtype=np.float64)[:NUM_ROBOT_JOINTS],
            np.asarray(previous, dtype=np.float64)[:NUM_ROBOT_JOINTS],
        ]
    return np.concatenate(parts)


def state_features(state, goal, belief_features, step, previous_torque=None):
    """`segment_features` for a captured simulator state, as a warm-start hook sees it."""
    return segment_features(
        state.limb_positions, state.robot_positions, goal, belief_features, step,
        state.limb_velocities, state.robot_velocities, previous_torque,
    )


@dataclass
class WarmStartModel:
    """A fitted `g`, mapping task and belief features to `psi`."""

    name: str
    estimator: object
    mpc: MPCConfig

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """The control points for one feature vector."""
        width = getattr(self.estimator, "n_features_in_", len(features))
        flat = self.estimator.predict(np.asarray(features)[:width].reshape(1, -1))[0]
        return flat.reshape(self.mpc.num_control_points, NUM_ROBOT_JOINTS)

    def as_hook(self, belief_features: NDArray[np.float64]):
        """The callable `LimbStreamContext.warm_start` expects.

        Per segment: each `move_limb` builds its own MPC from its own starting state.
        """

        def hook(state, goal: NDArray[np.float64], step: float = 0.0):
            if step == 0:
                hook.previous = None
            return self.predict(
                state_features(state, goal, belief_features, step, hook.previous)
            )

        hook.previous = None
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

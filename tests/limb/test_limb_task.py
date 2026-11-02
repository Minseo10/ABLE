"""The MPC-only limb task: fixed setups, psi's hook, and one real episode."""

import numpy as np
import pytest

from able.api import Planner, PsiSpec, Task, make_learner
from able.tasks.limb.planning.setup import (
    SETUP_PATH,
    FixedSetup,
    load_setup,
    load_setups,
    save_setup,
)
from able.tasks.limb.planning.stream import MPCConfig
from able.tasks.limb.psi import LimbPsi, state_features


def test_setups_round_trip(tmp_path):
    path = tmp_path / "setup.json"
    setup = FixedSetup("human-left-arm", 0.03, np.pi, (0.1, -0.4, 0.0), {"reach": 0.5})
    save_setup(setup, path)
    save_setup(FixedSetup("bed-left-arm", 0.0, 0.0, (1.0, 0.0, 1.0)), path)
    assert load_setup("human-left-arm", path) == setup and len(load_setups(path)) == 2
    with pytest.raises(KeyError):
        load_setup("wheelchair-left-leg", path)


def test_psi_hook_predicts_a_horizon_and_draws_samples():
    from types import SimpleNamespace

    mpc = MPCConfig()
    psi = LimbPsi(mpc)
    rng = np.random.default_rng(0)
    state = SimpleNamespace(limb_positions=np.zeros(6), robot_positions=np.zeros(13),
                            limb_velocities=np.zeros(6), robot_velocities=np.zeros(13))
    width = len(state_features(state, np.zeros(6), np.zeros(6), 0))
    x, y = rng.normal(size=(60, width)), rng.normal(size=(60, mpc.horizon * 7))
    learner = make_learner("flow", width=16, depth=1, steps=20).fit(x, y)
    belief = SimpleNamespace(features=lambda: np.zeros(6))
    hook = psi.proposal(learner, None, belief)
    assert hook(state, np.zeros(6), 0).shape == (mpc.horizon, 7)
    assert hook.samples(state, np.zeros(6), 0, 3).shape == (3, mpc.horizon, 7)
    assert isinstance(psi, PsiSpec)


@pytest.mark.slow
@pytest.mark.skipif(not SETUP_PATH.exists(), reason="no fixed_setup.json")
def test_one_mpc_episode_from_the_fixed_setup(tmp_path, monkeypatch):
    from able.api import Budget, run_episode
    from able.tasks.limb.task import LimbTask

    monkeypatch.chdir(tmp_path)
    task = LimbTask("human-left-arm")
    assert isinstance(task, Task) and isinstance(task.make_planner(), Planner)
    rng = np.random.default_rng(0)
    episode = run_episode(task, task.make_planner(), task.sample_problem(rng), np.full(3, 1.0),
                          task.prior.belief(rng), rng, budget=Budget(400))
    assert episode.effort > 0
    if episode.success:
        rows = task.psi.examples(episode, episode.belief)
        assert len(rows) > 0 and rows.y.shape[1] == MPCConfig().horizon * 7

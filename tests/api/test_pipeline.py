"""The shared pipeline on the toy reach task from test_api."""

import pickle

import numpy as np
import pandas as pd
import pytest
from test_api import ReachTask

from able.pipeline import Memory, PipelineConfig, factory_rows, fit_factory, run_person

MODEL = {"model": "mlp", "model_options": {"width": 16, "depth": 1, "steps": 50}}


@pytest.mark.parametrize("condition", ["plain", "learned"])
def test_run_person_writes_one_row_per_episode(tmp_path, condition):
    config = PipelineConfig(condition=condition, episodes=3, dreams=2, queries=1, **MODEL)
    records = run_person(ReachTask(), config, person=1, out=tmp_path / f"{condition}_p1")
    frame = pd.read_csv(tmp_path / f"{condition}_p1.csv")
    assert len(records) == len(frame) == 3 and set(frame.condition) == {condition}
    assert (frame.queries == 1).all()
    if condition == "learned":
        assert frame.dreams_solved.sum() > 0 and frame.rows.notna().any()


def test_resume_continues_where_it_stopped(tmp_path):
    config = PipelineConfig(condition="learned", episodes=2, dreams=1, **MODEL)
    out = tmp_path / "learned_p0"
    first = run_person(ReachTask(), config, 0, out)
    config.episodes = 4
    resumed = run_person(ReachTask(), config, 0, out, resume=True)
    key = [(r["episode"], r["effort"], r["belief"]) for r in first]
    assert [(r["episode"], r["effort"], r["belief"]) for r in resumed[:2]] == key
    assert len(pd.read_csv(out.with_suffix(".csv"))) == 4


def test_factory_shards_fit_one_frozen_learner(tmp_path):
    config = PipelineConfig(**MODEL)
    shards = [factory_rows(ReachTask(), config, dreams=3, seed=s) for s in range(2)]
    learner = fit_factory(shards, config)
    path = tmp_path / "factory.pkl"
    path.write_bytes(pickle.dumps(learner))
    records = run_person(ReachTask(), PipelineConfig(condition="prior_only", episodes=2, **MODEL), 0,
                         tmp_path / "prior_only_p0", prior_model=path)
    assert len(records) == 2 and "dreams_solved" not in records[0]


def test_memory_keeps_groups_of_different_episodes_apart():
    from able.api import Examples

    memory = Memory()
    rows = Examples(np.zeros((3, 2)), np.zeros((3, 1)), np.array([0, 0, 1]))
    memory.add(rows)
    memory.add(rows)
    assert memory.groups == [0, 0, 1, 2, 2, 3]


def test_no_info_gathering_skips_queries_but_still_learns_from_execution(tmp_path):
    config = PipelineConfig(condition="plain", episodes=3, info_gathering=False, **MODEL)
    records = run_person(ReachTask(), config, 2, tmp_path / "plain_p2")
    assert all(r["queries"] == 0 and r["info_gathering"] is False for r in records)


def test_no_belief_update_keeps_the_prior(tmp_path):
    config = PipelineConfig(condition="learned", episodes=3, dreams=1, belief_update=False, **MODEL)
    records = run_person(ReachTask(), config, 2, tmp_path / "learned_p2")
    assert len({r["belief"] for r in records}) == 1 and all(r["queries"] == 0 for r in records)


def test_live_logger_encodes_video_and_leaves_nothing_behind(tmp_path, monkeypatch):
    from pathlib import Path

    from able.wandb_log import Logger

    monkeypatch.setenv("WANDB_MODE", "disabled")
    logger = Logger("reach", "test", tmp_path / "learned_p0", PipelineConfig())
    frames = [np.full((16, 16, 3), i * 20, np.uint8) for i in range(6)]
    logger.log({"episode": 0, "success": True, "plan_seconds": 1.0}, frames, fps=5)
    logger.finish()
    assert not Path(logger.dir).exists() and not list(tmp_path.iterdir())


def test_videos_need_wandb(tmp_path):
    with pytest.raises(ValueError):
        run_person(ReachTask(), PipelineConfig(episodes=1, video_every=1, **MODEL), 0, tmp_path / "x")

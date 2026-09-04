# Learning Forceful Manipulation

A robot repositions a person's limb to a goal configuration without knowing their range of
motion. It asks the person a few questions, updates a belief about that range, and uses the
belief to warm-start the MPC inside a PDDLStream planner so planning gets faster.

Full design: [`doc/limb-baseline.md`](doc/limb-baseline.md).

## Install

Needs Python 3.10-3.12 and access to the two private sibling repos, which are pinned to
branches in `pyproject.toml`.

```bash
conda create -n kinder python=3.11 && conda activate kinder
pip install -e ".[develop]"
```

To work on the environment or the planner too, clone them beside this repo and install
editable instead of letting pip fetch them:

```bash
pip install -e ../kindergarden -e ../kinder-baselines/kinder-pddlstream-planning
```

## Pipeline

Each run lives in `data/<method>/<tag>/` and holds `demos.csv`, `attempts.csv`, `fit.json`,
`warm_start.pkl` and `eval.csv`.

```bash
# 1-2. Collect demonstrations. --method is pddlstream (plan to a goal) or babble
#      (random torques, relabelled). Queries and belief updates happen per episode.
python -m limb_baseline.collect --method babble --variant isolated-left-arm \
    --tag isolated_left_arm --num-people 50 --tasks-per-person 5

# 3. Fit psi, the MPC warm start, held out by person.
python -m limb_baseline.fit --run-dir data/babble/isolated_left_arm

# 4. Evaluate on freshly drawn people.
python -m limb_baseline.evaluate --run-dir data/babble/isolated_left_arm \
    --variant isolated-left-arm --num-tasks 50

# Figures (written to figures/<method>/<tag>).
python scripts/make_figures.py --run-dir data/babble/isolated_left_arm \
    --compare-dir data/pddlstream/isolated_left_arm
```

## Layout

| Path | What |
| --- | --- |
| `src/limb_baseline/` | The package: human model, queries, belief, tasks, both collectors, fit, evaluate |
| `scripts/` | Figure and video rendering |
| `data/` | Published runs, one directory per collection method |
| `doc/` | Design document |
| `tests/` | `pytest -m "not slow"` skips the ones that build a simulation |

Committed runs are `babble/isolated_left_arm` and `pddlstream/isolated_left_arm`. Earlier
exploratory runs stay in the gitignored `data/archive/`.

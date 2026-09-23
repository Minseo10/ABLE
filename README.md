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

## Personalization

Two ways to reach a warm start for one specific person, compared round by round. Both
loop the same four steps -- ask queries, update the belief, train `psi` on that person's
demonstrations, plan and execute held-out tasks -- and differ only in where the network's
weights start.

| Version | Round 0 weights | Scaler | Data per round |
| --- | --- | --- | --- |
| 1, factory pretrained | fit on `m` people x `n` tasks | from the factory data, frozen | small |
| 2, from scratch | random | fit on the person's first batch, then frozen | larger |

```bash
# Factory learning: one MLP over an existing collect run.
python -m limb_baseline.personalize pretrain --run-dir data/pddlstream/isolated_left_arm

# Version 1: personalize from the factory model.
python -m limb_baseline.personalize run --version v1 --variant isolated-left-arm \
    --pretrained data/pddlstream/isolated_left_arm/pretrained.pkl \
    --out-dir data/personalize/compare --num-people 5 --rounds 5 --tasks-per-round 5

# Version 2, matched and larger. Both write into the same rounds.csv.
python -m limb_baseline.personalize run --version v2_matched \
    --variant isolated-left-arm --out-dir data/personalize/compare --num-people 5 --rounds 5 --tasks-per-round 5
python -m limb_baseline.personalize run --version v2_large \
    --variant isolated-left-arm --out-dir data/personalize/compare --num-people 5 --rounds 5 --tasks-per-round 15

python scripts/make_personalize_figures.py --run-dir data/personalize/compare
```

Run the matched arm as well as the larger one: without it, a tie between v1 and v2 cannot
distinguish "pretraining bought nothing" from "the extra data made up for it".

Demonstrations are planned *without* a warm start in both versions, so the label is what
the slow planner does rather than what the model already predicts. Evaluation uses the
same held-out tasks every round, so the curve moves only when the belief or `psi` does.
`--replay` mixes factory rows back into each fine-tune (as a multiple of the person's own
rows) if the handful of personal rows turns out to destabilise the net; it is off by
default.

## Planning from a plan cache or a learned model first

The simplest "learning to plan", after the cache in `notebooks/dreaming_rom_planner.ipynb`:
skip the queries and the belief (Stages 1-2), hand the planner the person's true range of
motion, and try a seed before searching. The seed is rolled out closed-loop from the
grasped state with no search, one torque per control step, under the planner's own limit
checks and stopping tolerance. If it reaches the goal it is the plan, about a second's
work; if it violates a limit or stalls it is discarded and the stock MPC plans from the
original state, exactly as `plain` does, so the worst case is plain plus that second.

- `warm`: a cache of solved plans keyed on the goal rounded to 0.1 rad, the best four per
  key. A hit rolls out the stored torques from the stored state nearest the current one;
  a miss plans from scratch. The belief would be part of the key too, but it is constant.
- `learned`: a model refit after every solved episode on the MPC's own rows, the map from
  the current state (limb and robot joints, goal, step) to the next horizon's torques, the
  same rows `fit` trains on. Queried at every control step of the rollout, so it corrects
  for a goal it has not seen. Iteration 0 has no model and runs plain.
- `plain`: the stock planner.

Two task settings: one task repeated, and each goal drawn within `V` radians of the first.
The grasp, base pose and IK seeds are held fixed across iterations so the post-grasp state
repeats; only the MPC's noise is reseeded per iteration. `--seed-mode step` instead seeds
every MPC step (batch 0 with `--seed-rollouts` noisy copies, accepted within
`--seed-tolerance` of the MPC's goal-directed candidates) and falls back per step; the
runs under `data/memo/directed_bar/`, `nominal_bar/` and `batch8/` are earlier per-step
variants, and `data/memo/replay/` replayed the stored plan open-loop.

```bash
scripts/run_memo_experiments.sh 20 0.3 isolated-left-arm   # N, V, variant
# or one sequence at a time:
python -m limb_baseline.memo --condition learned --model ridge --variation 0.3 \
    --num-tasks 20 --variant isolated-left-arm --out data/memo/varied_task.csv
python scripts/make_memo_figures.py --csv data/memo/same_task.csv \
    --csv data/memo/varied_task.csv --plain-csv data/memo/replay/same_task.csv \
    --plain-csv data/memo/replay/varied_task.csv --conditions plain,warm \
    --out figures/memo/memo_plan_time.png
```

The package runs against kinder-baselines `main`; `src/limb_baseline/utils.py` supplies
the few planner names it was written against that `main` renamed or dropped.

## Layout

| Path | What |
| --- | --- |
| `src/limb_baseline/` | The package: human model, queries, belief, tasks, both collectors, fit, evaluate, personalize |
| `scripts/` | Figure and video rendering |
| `data/` | Published runs, one directory per collection method |
| `doc/` | Design document |
| `tests/` | `pytest -m "not slow"` skips the ones that build a simulation |

Committed runs are `babble/isolated_left_arm` and `pddlstream/isolated_left_arm`. Earlier
exploratory runs stay in the gitignored `data/archive/`.

# Learning Forceful Manipulation

A robot repositions a person's limb to a goal configuration without knowing their range of
motion. It asks the person a few questions, updates a belief about that range, and uses the
belief to warm-start the MPC inside a PDDLStream planner so planning gets faster.

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
python -m limb_baseline.personalize pretrain --run-dir data/archive/pddlstream/isolated_left_arm

# Version 1: personalize from the factory model.
python -m limb_baseline.personalize run --version v1 --variant isolated-left-arm \
    --pretrained data/archive/pddlstream/isolated_left_arm/pretrained.pkl \
    --out-dir data/personalize/compare --num-people 5 --rounds 5 --tasks-per-round 5

# Version 2, matched and larger. Both write into the same rounds.csv.
python -m limb_baseline.personalize run --version v2_matched \
    --variant isolated-left-arm --out-dir data/personalize/compare --num-people 5 --rounds 5 --tasks-per-round 5
python -m limb_baseline.personalize run --version v2_large \
    --variant isolated-left-arm --out-dir data/personalize/compare --num-people 5 --rounds 5 --tasks-per-round 15
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
runs under `data/archive/memo/directed_bar/`, `nominal_bar/` and `batch8/` are earlier
per-step variants, and `data/archive/memo/replay/` replayed the stored plan open-loop.

```bash
python -m limb_baseline.memo --condition learned --model ridge --variation 0.3 \
    --num-tasks 20 --variant isolated-left-arm --out data/memo/varied_task.csv
```

The package runs against kinder-baselines `main`; `src/limb_baseline/utils.py` supplies
the few planner names it was written against that `main` renamed or dropped.

## Dreaming between episodes

`limb_baseline.dream_pipeline` runs the notebook's loop on the arm: one episode is (1) information
gathering, (2) human model learning, (3) `n_dreams` dreams, (4) one planned and executed
task. Stages 1-2, the queries, are commented out for now; the belief starts at the
population prior and, as in the notebook, learns only from execution: a limb that stayed
inside the person raises a floor the limit lies beyond, one that crossed the limit says
where it is. The planner plans under the belief's mean. The task set is `K` goals within
`V` rad of one nominal task, drawn once inside the person's true range as in the
notebook; episodes and dreams pick from it. A dream draws
a hypothetical person from the belief and one goal, plans under the current belief exactly
as a real episode would, imagines executing on that person, updates a copy of the belief
from what that revealed, and memorizes plans under that imagined next belief, if they
reached the goal (`--all-goals` solves the whole task set under it, the notebook's step
3). Two memories:

- `cache`: memo.py's plan cache, keyed on (goal, belief); a hit is rolled out first.
- `learned`: memo.py's online model, refit after every memorized episode.
- `plain`: no memory, no dreams.

By default the planner uses 32 belief particles at confidence 0.5, about the typical
person. Planning gets 600 s per real episode and per dream (`--max-time`,
`--dream-max-time`), enforced exactly: the planner is stopped mid-stream at the deadline,
since PDDLStream only checks its clock between iterations. `--particles 0` plans under the belief's mean instead. With
particles, `P` people are drawn from the belief (redrawn when it changes) and a limb
configuration is believed inside when at least `--confidence` of them allow it. The range
only enters the planner as a check on rolled-out positions, so this costs no extra
rollouts. A goal beyond that range is retargeted into the matching quantile of the
particles, and a dream's hypothetical person is one of them.

A goal beyond the belief is not refused: as the notebook's planner stops at the nearest
point it believes reachable, the plan is retargeted 0.05 rad inside the believed edge
(`goal_beyond_belief`). Stopped-short plans that stay inside the limits are memorized,
and reaching the edge raises the floor, so the belief grows until it covers the goal.
Quality is `goal_distance`, the final distance to the original goal, as the notebook's
error.

Only real planning is charged; dream time goes to `dream_time` and to `<out>_dreams.csv`.
Effort is counted in MPC rollouts (steps searched x 64, plus policy rollouts).

```bash
# All lines of one experiment, 5 people, then its figures (episodes, dreams):
scripts/run_dream.sh warmstart 20 4    # 6 repeated goals -> data/dream/warmstart_compare
scripts/run_dream.sh fresh 20 4        # a new goal every episode -> data/dream/fresh_goals
MODELS="gen:flow:hist gen:ddpm:hist" scripts/run_dream.sh generative 20 4
                                       # 8 sampled seeds, varied geometry -> data/dream/generative_models
# or one condition at a time:
python -m limb_baseline.dream_pipeline --condition cache --n-dreams 4 --num-episodes 20 \
    --variation 0.3 --out data/dream/varied_0.3/cache_4.csv
python scripts/dream_figures.py summary --csv data/dream/varied_0.3/plain_0.csv \
    --csv data/dream/varied_0.3/cache_4.csv --out figures/dream/varied_0.3/effort_and_time.png
```

`scripts/dream_figures.py` also draws the learned models' learning curves (`curves`), the
step-mode comparison (`step`) and the generative models' draws (`samples`).
`scripts/dream_eval.py` holds the offline studies on recorded runs (`belief-features`,
`backfill`, `screen`, `sampling-steps`), and `scripts/log_to_wandb.py` uploads results.

## Layout

| Path | What |
| --- | --- |
| `src/limb_baseline/` | The package: human model, queries, belief, tasks, both collectors, fit, evaluate, personalize |
| `scripts/` | The dream experiments' runner, figures, offline studies and W&B upload; the range-of-motion figure |
| `data/dream/` | Current dream-pipeline runs, one directory per experiment |
| `data/archive/`, `figures/archive/` | Earlier runs and their figures (babble, pddlstream, memo, older dream runs) |
| `doc/poster/` | NERC posters |
| `tests/` | `pytest -m "not slow"` skips the ones that build a simulation |

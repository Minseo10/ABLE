# Where an Object Can Go, and How It Should Get There

**One line.** Learn what the object allows by interacting with it, learn how it should move
by planning with it, and let each one guide the other.

Companion to [limb-baseline.md](limb-baseline.md), which learns only the MPC warm start.

---

# 1. Motivation

## 1.1 Planners search in the robot's order; constraints live on the object

A typical PDDLStream stream plan for limb repositioning:

```
sample-grasp:              (limb)                     -> (#g0)
sample-base-pose:          (limb, #g0, qL400, qL448)  -> (#q0)
plan-base-motion:          (SE2Pose(start), #q0)      -> (#b0)
plan-grasp-motion:         (limb, #g0, #q0, qL400)    -> (#a0, #s0)
plan-limb-motion:          (#s0, qL448)               -> (#t1, #s2)
check-human-joint-limits:  (#s0, qL448, #t1, #s2)     -> ()
check-human-torque-limits: (#s0, qL448, #t1, #s2)     -> ()
check-robot-torque-limits: (#s0, qL448, #t1, #s2)     -> ()
```

Robot-side choices come first; the limb's motion appears only as a by-product of MPC's
torques, and object-side constraints are checked last. Other TAMP domains share this order:

| Domain | Object-side constraint | Robot-side constraint |
| --- | --- | --- |
| limb repositioning | range of motion, joint torque, contact with the bed | reach, robot torque |
| carrying a tray or a liquid | tilt, acceleration | reach, payload |
| pouring | spill, flow rate | reach, wrist limits |
| door or drawer opening | the hinge's path, required force | reach, applicable wrench |
| insertion into a shelf | clearance along the path | reach, arm collision |

When an object-side check fails, the planner backtracks into grasps and base poses, which
were often not the cause.

## 1.2 Why object first

Let `xi` be the object's trajectory and `r = (g, q, tau)` the robot's choices.

1. An object-side constraint depends on `r` only through the motion it produces:
   `C(r) = C(xi(r))`.
2. So it can be checked on `xi` alone, with no robot model, and far more cheaply.
3. Given `xi`, the robot only has to *realize* it; object-side and robot-side constraints
   become conditionally independent.
4. A failure is attributed to the right side: "no acceptable object motion" versus "this
   robot cannot produce it".

**Evidence it works.** Kinematic TAMP already does this for poses: placement samplers pick
where an object goes before any grasp or IK is sampled. We extend the idea from **poses to
trajectories with dynamics**, where object-side constraints are velocities, accelerations,
and forces, and where planners still search robot first.

**When it does not help.** If most failures are robot-side (reach, arm collision), fixing
`xi` first saves little. This is measurable by classifying failed stream calls.

## 1.3 Uncertainty lives on the object too

The object's unknown properties (a person's range of motion, a box's mass, a hinge's
stiffness) enter object-side constraints, not robot-side ones. With `xi` explicit, the
belief about them acts in one place, on `xi`, instead of inside every robot computation.

---

# 2. Problem

Given a task (the limb's start and goal, and the robot's start) and the robot's current
belief about the person's range of motion, find a plan quickly. Two things are learned in
alternation. Queries to the person teach **where the limb can go**: its range of motion.
Planner runs teach **how it should get there**: a prior over limb paths that the robot
realizes quickly. A sharper belief shifts the prior, and the paths the prior prefers
decide which bound is worth querying next.

---

# 3. Approach

## 3.1 Object-first stream plan

```
sample-limb-path:   (limb, qL_init, qL_goal)   -> (#xi)       learned prior; human checks
sample-grasp:       (limb)                     -> (#g)        unchanged
sample-base-pose:   (limb, #g, #xi)            -> (#q)        reach and robot torque along the path
plan-base-motion:   (start, #q)                -> (#b)        unchanged
plan-grasp-motion:  (limb, #g, #q, qL_init)    -> (#a, #s)    unchanged
plan-limb-motion:   (#s, #xi)                  -> (#t, #s')   MPC follows xi
check-*:            (#s, #xi, #t, #s')         -> ()          re-check the realized path
```

The limb path `xi` is chosen once, first. Every later stream answers a narrower question:
can this robot, from this grasp and this base, reproduce that path? If no grasp or base
can, the search goes back for a new path instead of cycling through grasps.

## 3.2 Learning the limb path

The hand-written sampler already proposes many paths, and the rest of the planner reveals
which of them it realizes and how fast. The goal is a sampler that proposes the fast,
realizable ones directly, so the planner stops paying MPC time to discover the others.

**How a path is represented.** A path is the straight line from start to goal plus a
smooth correction, stretched over a duration:

```
xi(s) = (1 - s) · q_init + s · q_goal + correction(s),   correction(0) = correction(1) = 0
```

The correction is a few spline control points per joint. The model learns only these
control points and the duration, so every sampled path starts and ends in the right place,
and different tasks can share the same kinds of corrections.

**Collecting data.** Run PDDLStream many times. Each run samples a task, a simulated
person, and the belief the robot would hold after a few queries. At first, paths come from
the hand-written sampler: a straight line with a random correction and duration. The
stream log then gives one training row for every proposed path that reached
`plan-limb-motion`.

- **Features** are the task (limb start and goal, robot start) and the belief (mean and
  width per joint bound).
- **The label** is the path MPC actually produced, fitted back to control points and
  duration. Using the realized path rather than the proposed one keeps the labels to paths
  this robot is known to be able to produce.
- **The outcome** of that row: success; an object-side failure (range of motion, human
  torque, contact); a robot-side failure (reach, robot torque); or a search failure (MPC
  diverged or ran out of time).
- **Successes are weighted by speed**, so paths that planned quickly count more.

Paths that failed before reaching `plan-limb-motion` are left out; they say nothing about
the path.

**The model.** A small conditional diffusion model over the control points and duration,
conditioned on the features and on the outcome. It is trained on every row, failures
included. At planning time it is asked for successful paths, and classifier-free guidance
pushes samples away from what failure rows looked like. This uses failures without a
separate critic network. A conditional VAE or a mixture density network trained on the
same rows is a cheaper first model.

**Rounds.** The first round uses only the hand-written sampler. Later rounds draw most
paths from the learned model and a small fraction from the hand-written one, then retrain.
The model thus sees the failures of its own proposals, and the hand-written fraction keeps
the planner complete.

## 3.3 Using the path to produce the other outputs

The limb is passive, so every torque its motion needs must come through the grasp. Once
the path is fixed, where the robot's hand must go and what force it must apply are fixed
too. The grasp check, the base pose, and the MPC warm start then follow from kinematics
and inverse dynamics, with no simulation.

**Human checks, on the path alone.** Inverse dynamics of the limb along the path gives the
torque each of the person's joints bears. `sample-limb-path` rejects a path that leaves
the range of motion, exceeds either human torque bound, or runs the limb into the bed, all
before any robot computation.

**Grasp.** A grasp point on the forearm moves with the limb along the path, which gives
the path the hand must follow. The hand must also apply the force that produces the limb's
joint torques:

```
J_g(xi)ᵀ · F = limb joint torques along xi
```

If no force satisfies this, the grasp cannot drive some joint and is rejected. On the
forearm every grasp drives all arm joints, so `sample-grasp` stays unchanged and
independent of the path.

**Base pose.** For each candidate base near the hand path, solve IK along the whole path,
seeding each step from the last, and compute the robot torque needed to move the arm and
push with that force:

```
robot torque = arm inverse dynamics along the IK path + J_rᵀ · F
```

A base is kept if IK succeeds everywhere and the torque stays within the robot's limits,
and bases are tried from least to most saturated. Today the base is checked with IK at
three waypoints and the torque to hold the limb still; this checks the motion itself.

**Grasp motion.** Unchanged: IK at the limb's start and an RRT to it.

**Limb motion.** MPC receives the path as a reference, adds a cost for deviating from it,
and starts from the robot torque computed for the base as its warm start. It follows a
known path instead of searching for any path to the goal.

**Final checks.** Unchanged, applied to the path MPC actually produced, since it follows
the reference only approximately.

---

# 4. Research questions

- **RQ1. Does planning the object first help?** Swap the order: pick the limb path first,
  then the grasp and base, with no learning at all. Is planning faster, with fewer failed
  MPC calls, than today's robot-first planner?
- **RQ2. Does learning the path help further?** Keep the object-first order and change
  only where paths come from: plain interpolation, a model trained on successful paths,
  or a model trained on successes and failures. Does learning beat interpolation, and do
  failures add anything?
- **RQ3. Does it carry over to other tasks?** Do the answers to RQ1 and RQ2 hold in a
  second domain, such as carrying a box of unknown mass?

# 5. Evaluation

- **Baselines:** the current planner; the current planner with the learned warm start from
  limb-baseline.md; object first with interpolated paths; object first with a prior
  trained on successes only; ours, with the prior trained on successes and failures; and
  an oracle that knows the true range of motion.
- **Metrics:** success under a time cap, planning time, failed MPC calls, and true
  range-of-motion violations.

# 6. Preliminary evidence

- **Old planner, isolated arm** (1,044 failed episodes): MPC search 46%, timeout 36%,
  object-side 16%, robot torque 0%.
- **New planner, per-call census** (right arm, goal scale 0.5/1.0, 4 seeds, 600 s cap;
  base-furniture collisions off for bed and wheelchair):

  | Variant | Reached | Where planning time went |
  | --- | --- | --- |
  | isolated | 5/8 | 99.9% in 8 successful MPC calls; 3 runs timed out inside one call (732-1073 s) |
  | bed | 0/8 | 98% in 118 failed MPC calls: 112 human torque at control step 3, 6 range of motion |
  | wheelchair | 0/8 | first MPC call unfinished after 30 min in every run |

- No failure was robot-side, and none changed with the 42 grasps and 88 bases tried: the
  bed needs a different `xi`, and isolated and wheelchair need MPC to converge faster.

# 7. Novelty and risks

- **Novelty:** object-first order on dynamic trajectories; grasp, base, and warm start
  derived from the object path; a loop that pairs learning what the object allows with learning how
  to move it. Closest work (STAP, GSC, Diffusion-CCSP) chains skills robot first and
  assumes a known object.
- **Risks:** a single grasp may not realize every `xi`; MPC needs a tracking cost; inverse
  dynamics breaks under furniture contact; little gain where failures are robot-side.

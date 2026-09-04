# Simplest Baseline for Limb Manipulation

A robot repositions a person's limb to a goal configuration. It does not know the person's
range of motion, so it asks a few questions (Robot to Human), then uses the resulting
belief to learn a planner parameter that makes planning faster (Human to Robot).

**Code base**

- LimbRepositioning3D env: kindergarden, branch `limb/04-limb-joint-limits`
- PDDLStream planner: kinder-baselines, branch `pddlstream/limb-gravity`
- Use conda env 'kinder'

**Fixed environment**

- variant `isolated-right-arm`, joint-limits model `box`, muscle tone `spring`, gravity `(0,0,-9.81)`

**Variables**

| Symbol | Meaning | Representation | Learned by |
| --- | --- | --- | --- |
| `theta` | person's passive range of motion (RoM) | n scale factors on `RangeOfMotion.scaled` | querying the person |
| `b(theta)` | belief about `theta` | Gaussians (bound + width) | Bayesian update on answers |
| `psi` | MPC warm start | initial nominal control points | offline regression |
| `P(task)` | task distribution | fixed variant, sampled `(q_init, q_goal)` | given |

**Assumptions**

- Human parameters (RoM) are partially observable, and we maintain a belief about them.
- Robot parameters and state are fully observable.

---

# Human to Robot

## Task

- `(q_init_human, q_init_robot, q_goal_human)`
- Define a `Task` record and a sampler for it in a new `tasks.py`.

## Task distribution `P(task)`

- Sample all three for the single variant `isolated-right-arm`.
- The robot init should stay inside the robot's joint limits; the human init and goal stay also should
  inside the person's ground-truth RoM.

## Human model

### Parameterization

- The three motions the `isolated-right-arm` nominal motion actually loads: hip flexion, hip
  abduction, knee flexion.
- `theta` is one scale factor per motion, applied to the default range of motion. Every
  other motion keeps its default.
- Draw each motion's scale independently from the prior Gaussian truncated to the plausible range, and the three draws become `theta`. Redraw any motion that lands outside the plausible range.


### Belief `b(theta)`

One Gaussian per active motion, something like `b(s_i) = N(mu_i, sigma_i^2)`?

- The belief holds a mean and a std per motion, three each, starting at the prior.

## Planner

`planner(task, theta; psi) -> plan`

- `psi`: the MPC's initial distribution. The MPC's nominal control points can be `psi`.
- To learn `psi`, make a dict of off-the-shelf scikit-learn regressors: nearest neighbors, logistic regression, a small MLP, etc.

---

# Robot to Human

## Query and response

- Query: The robot asks the person to move one joint to a target angle and watches.
  - Ask for a boundary value of one joint's RoM.
- Response: the person pushes against their own
  RoM boundary and stops where they can't move anymore.
- Observation: the limb's joint angles.

### How to implement in simulation

Write a function that push a human joint toward the target, report where the person stopped.

- **End-of-range torque:** a restoring torque on the queried joint. Let `hi` be the passive bound (theta), `w` be the width of the end-range zone, `K` be the soft-tissue stiffness in zone (`hi-w` to `hi`), and `K_wall` be the hard stop past `hi`.
  - `0`,  if q <= hi - w
  - `-K * (q - (hi - w))`, else if hi - w < q < hi
  - `-K * w - K_wall * (q - hi)`, else if q >= hi (K_wall >> K)

- **Human effort:** A constant torque pushing toward the target, on the queried joint
only, drawn fresh each trial from a Gaussian.
  - Keep the effort below `K * w`, so the person
    stops a little bit short of `hi`. Active RoM is typically narrower than passive RoM.
  - How far short varies trial to trial, because the human effort does in real world.
  - Not a PD controller. PD would reach the target regardless of resistance.



---

# Pipeline

- `b(theta)` plus `K` queries gives `b^(theta)`. Sample N tasks from
`P(task)`, run the slow planner on each `(task, theta ~ b)` ("sleep"), collect
`(features, control points)`, fit `psi` with sklearn. Evaluate
`plan(task, b(theta), psi)` on held-out tasks.

## Stage 1: information gathering

Ask queries `Q = (q_1, q_2, ..., q_k)` to the user and collect response `R = (r_1, r_2, ..., r_k)` 

- Start with `b(theta)` as the prior.
  - Each round: ask a query `q_i` from the current
  belief, run the attempt, update the belief (Stage 2).
  -  After `k` rounds, hand the final posterior `b_K(theta)` to Stage 3.
- **Picking the query:** 
  - Ask for an randomly sampled angle near the believed bound.

## Stage 2: human model learning

`(q_i, r_i), b(theta) -> b'(theta)`


```
b'(theta)  =  p(r_i | q_i, theta) * b(theta) / Z             
```

- **Prior:** `b(theta) ~ N(mu, sigma^2)`, the belief from the previous round.
- **Likelihood:** `r_i = theta * R + noise, noise ~ N(0, tau^2)` 
    - R is default RoM that theta scales
    - tau comes from std of human effort / K


## Stage 3: Learn to Plan better

Learn a model `g` to
update `b(psi)` →`b(~psi)` based on `P(tasks)` and `b'(theta)`

- **Input of learned model**: the limb's
  current joint configuration, the robot's current joint configuration, the goal limb configuration, and
  the belief of theta.
- **Output of learned model**: the MPC's initial nominal control points, `num_control_points x
  NUM_ROBOT_JOINTS`. This is `psi`.
- **Data**:
  - Sample people and tasks
  - Run the PDDLStream planner for demonstrations
  - Record the
  features and the fitted control points from every successful episode. The label is the
  executed torque sequence projected back onto control points.
  - Keep the dataset as an append-only table on disk, one row per successful episode
- **Model design**: any multi-output regressor in scikit-learn library (nearest neighbors, logistic regression, MLP, etc.)
- **Cost function**:
  - Simplest idea: Mean squared error on the control points.
  But different torque sequences can reach the same goal so it's not ideal.
  - We could also include the downstream metric: simulator steps per stream call, length of the trajectory, etc.

## Stage 4: plan to solve better

`task ~ P(tasks) → planner (task, ~theta, ~psi)`

- Given new task, plan with the learned belief and the learned warm start.


We can repeat Stage 1 from Stage 4.


## Evaluation

### Baselines

| Planner | Human Model (theta) Learning | Planner Paramter (psi) Learning | 
| --- | --- | --- |
| PDDLStream | no | none | the planner as it stands today |
| + belief | yes, from `K` queries | none | querying alone |
| + warm start | no | sklearn | warm starting alone |
| baseline (this doc, PDDLStream + belief + warm start) | yes | sklearn | both, with the simplest models |
| ours (learned diffusion policy) | yes | diffusion | the method |
| oracle (reference) | ground truth `theta` | diffusion | the ceiling, unachievable |


### Metrics

- **Success rate:** Fraction of tasks where a plan was found and the goal reached.
- **Planning time:** Report it as simulator steps and wall-clock seconds.
- **RoM violations at execution:** executed paths that left the ground-truth box. This is
  where human model learning shows.
- **Simulator steps per successful MPC call:** this is where planner parameter learning
  shows. A well-learned `psi` starts the MPC near a working torque sequence, so it converges
  in fewer steps. 

## Note: Code fixes needed

- **The planner must not read ground-truth human parameters.** Every RoM check in
  `stream.py` calls `ctx.sim.limb.check_joint_limits(...)`, which reads the true RoM. Switch
  to a belief.

- Keep the true check reachable as `ctx.truly_in_limits`; evaluation needs it to score
  safety violations.
- The MPC must respect robot joint limits and torque limits when scoring rollouts. Verify
  both in `PredictiveSamplingMPC._score`.



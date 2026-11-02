"""Render BedsideStocking3D: initial scene, planned goal scene (final plan poses, no execution), slot map.

Usage: python scripts/bedside/render_env.py [out_dir] [num_requested] [num_clutter] [seed]
"""

import sys

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont
from pybullet_helpers.geometry import set_pose

from able.tasks.bedside.env import (
    ObjectCentricBedsideStocking3DEnv,
    reach_margin,
)
from able.tasks.bedside.planner.run import plan_bedside
from able.tasks.bedside.planner.stream import requested_candidate_slots

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/bedside"
R, C, SEED = (int(a) for a in (sys.argv[2:5] if len(sys.argv) > 4 else (4, 4, 0)))
W, H = 1400, 950
DELTA = 0.1
EYE, TARGET = [-1.0, 2.55, 2.25], [1.05, 0.25, 0.45]


def visual(shape, rgba, pos, **kw):
    cid = SIM.physics_client_id
    vs = p.createVisualShape(shape, rgbaColor=rgba, physicsClientId=cid, **kw)
    return p.createMultiBody(baseVisualShapeIndex=vs, basePosition=pos, physicsClientId=cid)


def ellipsoid_dots(theta):
    ts = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    pts = []
    for zf in (-0.5, 0.0, 0.5):
        s = np.sqrt(1 - zf**2)
        pts += [[theta[0] * s * np.cos(t), theta[1] * s * np.sin(t), theta[2] * zf] for t in ts]
    pts += [[theta[0] * np.cos(t), 0, theta[2] * np.sin(t)] for t in ts]
    pts += [[0, theta[1] * np.cos(t), theta[2] * np.sin(t)] for t in ts]
    ids = []
    for d in pts:
        q = SIM.shoulder + np.array(d)
        if q[2] > 0:
            ids.append(visual(p.GEOM_SPHERE, (0.95, 0.2, 0.6, 1), q, radius=0.008))
    return ids


def slot_markers():
    """Green prior-safe, yellow this user only, grey out of reach, red excluded from candidates."""
    cfg = SIM.config
    rng = np.random.default_rng(0)
    prior = rng.uniform(cfg.reach_theta_lb, cfg.reach_theta_ub, size=(20000, 3))
    n_safe = n_mine = n_all = n_excl = 0
    ids = []
    cand = {(r, tuple(np.round(sl, 4))) for r, sl in requested_candidate_slots(SIM)}
    for region in cfg.bedside_regions:
        for s in cfg.region_slots(region):
            top = s + [0.0, 0.0, 0.06]
            safe = np.mean(reach_margin(top[None], prior, SIM.shoulder) >= 0) >= 1 - DELTA
            mine = reach_margin(top, SIM.reach_theta, SIM.shoulder) >= 0
            excluded = (region, tuple(np.round(s, 4))) not in cand
            col = ((0.80, 0.15, 0.15, 1) if excluded else (0.1, 0.75, 0.25, 1) if safe
                   else (0.98, 0.8, 0.1, 1) if mine else (0.55, 0.55, 0.58, 1))
            n_excl += excluded
            ids.append(visual(p.GEOM_CYLINDER, col, s + [0, 0, 0.006], radius=0.035, length=0.012))
            n_safe += safe
            n_mine += mine
            n_all += 1
    return ids, n_safe, n_mine, n_all, n_excl


def snap():
    cid = SIM.physics_client_id
    view = p.computeViewMatrix(EYE, TARGET, [0, 0, 1], physicsClientId=cid)
    proj = p.computeProjectionMatrixFOV(46, W / H, 0.05, 15, physicsClientId=cid)
    img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER, shadow=1,
                           lightDirection=[1.0, 2.0, 4.0], lightAmbientCoeff=0.55,
                           lightDiffuseCoeff=0.55, physicsClientId=cid)
    return Image.fromarray(np.reshape(img[2], (H, W, 4))[:, :, :3].astype(np.uint8))


def label(im, text):
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except OSError:
        f = ImageFont.load_default()
    d.rectangle([0, 0, W, 50], fill=(255, 255, 255))
    d.text((18, 10), text, fill=(20, 20, 20), font=f)
    return im


def remove(ids):
    for i in ids:
        p.removeBody(i, physicsClientId=SIM.physics_client_id)


if __name__ == "__main__":
    SIM = ObjectCentricBedsideStocking3DEnv(num_requested=R, num_clutter=C, allow_state_access=True)
    state, _ = SIM.reset(seed=SEED)
    floor = p.createVisualShape(p.GEOM_BOX, halfExtents=[2.2, 1.9, 0.005], rgbaColor=(0.80, 0.72, 0.60, 1), physicsClientId=SIM.physics_client_id)
    p.createMultiBody(baseVisualShapeIndex=floor, basePosition=[1.0, 0.25, -0.006], physicsClientId=SIM.physics_client_id)
    theta = tuple(round(float(t), 2) for t in SIM.reach_theta)
    a = label(snap(), f"Initial: {R} requested (colored) on cart/dresser, {C} clutter (grey)")

    plan = plan_bedside(SIM, state, max_time=300, seed=SEED)
    assert plan is not None, "no plan found"
    SIM.reset(seed=SEED)
    for name, args in plan:
        if name == "place":
            set_pose(SIM._object_name_to_pybullet_id(args[0]), args[1].pose, SIM.physics_client_id)  # pylint: disable=protected-access
        elif name == "move_base":
            SIM.robot.set_base(args[1].pose)
    dots = ellipsoid_dots(SIM.reach_theta)
    b = label(snap(), f"Planned goal, theta = {theta} (pink = reach)")

    SIM.reset(seed=SEED)
    for name in state.requested_objects + state.clutter_objects:
        set_pose(SIM._object_name_to_pybullet_id(name), SIM.resting_pose(name, (10, 10), "dresser"), SIM.physics_client_id)  # pylint: disable=protected-access
    marks, n_safe, n_mine, n_all, n_excl = slot_markers()
    c = label(snap(), f"Slots: prior-safe {n_safe}, this user only {n_mine - n_safe}, "
                      f"out of reach {n_all - n_mine - n_excl}, excluded {n_excl} (red)")
    remove(marks + dots)

    for im, name in ((a, "env_initial"), (b, "env_goal"), (c, "env_slots")):
        im.save(f"{OUT}/{name}.png")
    combo = Image.new("RGB", (3 * W + 20, H), (255, 255, 255))
    for i, im in enumerate((a, b, c)):
        combo.paste(im, (i * (W + 10), 0))
    combo.save(f"{OUT}/env_initial_goal_slots.png")
    print("saved", f"{OUT}/env_initial_goal_slots.png", "| plan length", len(plan))

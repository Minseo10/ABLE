"""Render a PyBullet-style sketch of the Bedside Stocking domain.

Panels: initial scene, goal scene for one sampled user, and the reachable-slot map
(prior-safe vs. reachable only with this user's theta vs. unreachable).
Visual-only bodies, rendered with TinyRenderer in DIRECT mode.
Usage: python scripts/bedside/render_scene.py [out_dir]
"""
import os
import sys

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from layout import (  # noqa: E402
    PRIOR_HI,
    PRIOR_LO,
    REGIONS,
    SHOULDER,
    all_slots,
    margin,
)

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/bedside"
W, H = 1400, 950
THETA = np.array([0.80, 0.62, 0.72])  # the sampled user shown in the goal scene
DELTA = 0.1

# Furniture derived from the placement regions in layout.py.
(NSX, NSY, NS_TOP) = REGIONS["nightstand_top"]
NS = dict(c=((NSX[0] + NSX[1]) / 2, (NSY[0] + NSY[1]) / 2), sx=NSX[1] - NSX[0] + 0.04, sy=NSY[1] - NSY[0] + 0.04,
          h=NS_TOP, shelves=(REGIONS["nightstand_bottom"][2], REGIONS["nightstand_shelf"][2]))  # open front faces -y
WS = REGIONS["wall_shelf"]
FC = REGIONS["foot_chest"]
BS = dict(x=2.42, y=-0.55, sx=0.32, sy=0.95, h=1.80, n=5)  # bookshelf on the right wall
DR = dict(x=2.30, y=-1.65, sx=0.50, sy=0.95, h=0.80)  # dresser

WOOD = (0.62, 0.45, 0.30, 1)
WOOD_D = (0.45, 0.32, 0.22, 1)
GREY = (0.62, 0.62, 0.64, 1)
CLUTTER = [(0.55, 0.55, 0.58, 1), (0.70, 0.66, 0.60, 1), (0.48, 0.50, 0.54, 1), (0.66, 0.62, 0.66, 1)]
PLANT = (0.30, 0.55, 0.30, 1)
REQ_BOOK = [(0.85, 0.15, 0.15, 1), (0.15, 0.35, 0.85, 1), (0.10, 0.65, 0.25, 1), (0.95, 0.55, 0.05, 1)]
CUP, PHONE, REMOTE, PILLS, GLASSES = ((0.95, 0.85, 0.10, 1), (0.10, 0.10, 0.12, 1), (0.20, 0.20, 0.25, 1),
                                      (0.95, 0.40, 0.10, 1), (0.55, 0.15, 0.65, 1))


def box(pos, half, rgba, yaw=0.0):
    vs = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
    return p.createMultiBody(baseVisualShapeIndex=vs, basePosition=pos,
                             baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]))


def cyl(pos, r, h, rgba):
    vs = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=rgba)
    return p.createMultiBody(baseVisualShapeIndex=vs, basePosition=pos)


def sphere(pos, r, rgba):
    vs = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=rgba)
    return p.createMultiBody(baseVisualShapeIndex=vs, basePosition=pos)


def capsule(a, b, r, rgba):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = np.linalg.norm(d)
    ax = np.cross([0, 0, 1.0], d / L)
    s = np.linalg.norm(ax)
    q = [0, 0, 0, 1] if s < 1e-8 else p.getQuaternionFromAxisAngle(ax / s, np.arccos(np.clip(d[2] / L, -1, 1)))
    vs = p.createVisualShape(p.GEOM_CAPSULE, radius=r, length=L, rgbaColor=rgba)
    return p.createMultiBody(baseVisualShapeIndex=vs, basePosition=(a + b) / 2, baseOrientation=q)


# ---------------------------------------------------------------- static scene
def room():
    box([0.8, -0.45, -0.01], [2.0, 1.6, 0.01], (0.80, 0.72, 0.60, 1))  # floor
    box([0.8, 1.16, 1.2], [2.0, 0.01, 1.2], (0.90, 0.90, 0.88, 1))  # back wall
    box([2.80, -0.45, 1.2], [0.01, 1.6, 1.2], (0.86, 0.88, 0.90, 1))  # right wall


def bed_and_user():
    box([0, 0.0, 0.20], [0.52, 1.05, 0.20], WOOD_D)  # frame
    box([0, 0.0, 0.48], [0.50, 1.02, 0.08], (0.92, 0.93, 0.96, 1))  # mattress
    box([0, 1.08, 0.55], [0.52, 0.03, 0.35], WOOD_D)  # headboard
    box([0, 0.90, 0.85], [0.48, 0.05, 0.32], (0.92, 0.93, 0.96, 1))  # raised backrest
    box([0, 0.83, 0.95], [0.28, 0.04, 0.16], (1, 1, 1, 1))  # pillow behind back
    skin, shirt, pants, blanket = (0.93, 0.76, 0.62, 1), (0.35, 0.55, 0.75, 1), (0.30, 0.30, 0.40, 1), \
        (0.55, 0.70, 0.60, 1)
    capsule([0, 0.62, 0.70], [0, 0.70, 1.00], 0.15, shirt)  # torso
    capsule([-0.18, 0.68, 1.08], [0.18, 0.68, 1.08], 0.07, shirt)  # shoulders
    sphere([0, 0.70, 1.30], 0.10, skin)  # head
    capsule([-0.10, 0.55, 0.64], [-0.10, -0.35, 0.64], 0.08, pants)  # legs
    capsule([0.10, 0.55, 0.64], [0.10, -0.35, 0.64], 0.08, pants)
    box([0, -0.40, 0.69], [0.42, 0.55, 0.07], blanket)  # blanket
    capsule([-0.21, 0.66, 1.08], [-0.25, 0.55, 0.84], 0.045, shirt)
    capsule([-0.25, 0.55, 0.84], [-0.14, 0.32, 0.74], 0.04, skin)
    capsule(SHOULDER, [0.25, 0.55, 0.84], 0.045, shirt)
    capsule([0.25, 0.55, 0.84], [0.14, 0.32, 0.74], 0.04, skin)
    sphere(SHOULDER, 0.04, (0.95, 0.20, 0.60, 1))  # c_sh


def nightstand():
    (cx, cy), sx, sy, h = NS["c"], NS["sx"], NS["sy"], NS["h"]
    t = 0.02
    box([cx, cy, h - t], [sx / 2, sy / 2, t], WOOD)  # top
    box([cx - sx / 2 + t, cy, h / 2], [t, sy / 2, h / 2], WOOD)
    box([cx + sx / 2 - t, cy, h / 2], [t, sy / 2, h / 2], WOOD)
    box([cx, cy + sy / 2 - t, h / 2], [sx / 2, t, h / 2], WOOD)  # back; front (-y) open
    for z in NS["shelves"]:
        box([cx, cy, z - t / 2], [sx / 2 - t, sy / 2 - t, t / 2], WOOD)


def wall_shelf():
    (x0, x1), (y0, y1), z = WS
    box([(x0 + x1) / 2, (y0 + y1) / 2, z - 0.015], [(x1 - x0) / 2, (y1 - y0) / 2, 0.015], WOOD)
    for x in (x0 + 0.1, x1 - 0.1):  # brackets
        box([x, y1 - 0.02, z - 0.10], [0.01, 0.02, 0.08], GREY)


def foot_chest():
    (x0, x1), (y0, y1), z = FC
    box([(x0 + x1) / 2, (y0 + y1) / 2, z / 2], [(x1 - x0) / 2, (y1 - y0) / 2, z / 2], WOOD_D)
    box([x0 - 0.005, (y0 + y1) / 2, z * 0.6], [0.005, 0.10, 0.012], GREY)


def bookshelf(gaps=()):
    x, y, sx, sy, h, n = BS["x"], BS["y"], BS["sx"], BS["sy"], BS["h"], BS["n"]
    t = 0.02
    box([x + sx / 2 - t, y, h / 2], [t, sy / 2, h / 2], WOOD_D)
    box([x, y - sy / 2, h / 2], [sx / 2, t, h / 2], WOOD_D)
    box([x, y + sy / 2, h / 2], [sx / 2, t, h / 2], WOOD_D)
    rng = np.random.default_rng(3)
    req_slots = {(1, 4): 0, (2, 9): 1, (3, 2): 2, (1, 13): 3}
    for k in range(n + 1):
        box([x, y, k * h / n + t / 2], [sx / 2, sy / 2, t / 2], WOOD_D)
    for k in range(n):
        z0 = k * h / n + t
        yy = y - sy / 2 + 2 * t
        i = 0
        while yy < y + sy / 2 - 2 * t - 0.05:
            w, bh, bd = rng.uniform(0.03, 0.05), rng.uniform(0.20, 0.30), rng.uniform(0.18, 0.24)
            rid = req_slots.get((k, i))
            if rid is None or rid not in gaps:
                col = REQ_BOOK[rid] if rid is not None else tuple(rng.uniform(0.35, 0.65, 3)) + (1,)
                box([x - sx / 2 + bd / 2 + 0.03, yy + w / 2, z0 + bh / 2], [bd / 2, w / 2, bh / 2], col)
            yy += w + 0.004
            i += 1


def dresser():
    x, y, sx, sy, h = DR["x"], DR["y"], DR["sx"], DR["sy"], DR["h"]
    box([x, y, h / 2], [sx / 2, sy / 2, h / 2], WOOD)
    for k in range(3):
        box([x - sx / 2 - 0.005, y, 0.13 + k * 0.25], [0.005, sy / 2 - 0.03, 0.10], WOOD_D)
        box([x - sx / 2 - 0.02, y, 0.13 + k * 0.25], [0.015, 0.08, 0.012], GREY)


# ---------------------------------------------------------------- movable objects
def small_items(pts):
    """Requested small items (cup, phone, remote, pills, glasses) at grasp points pts[i] (surface + 0.06)."""
    shapes = [lambda q: cyl([q[0], q[1], q[2]], 0.04, 0.12, CUP),
              lambda q: box([q[0], q[1], q[2] - 0.054], [0.04, 0.07, 0.006], PHONE),
              lambda q: box([q[0], q[1], q[2] - 0.048], [0.025, 0.05, 0.012], REMOTE),
              lambda q: cyl([q[0], q[1], q[2] - 0.01], 0.025, 0.10, PILLS),
              lambda q: box([q[0], q[1], q[2] - 0.048], [0.05, 0.025, 0.012], GLASSES)]
    for f, q in zip(shapes, pts):
        f(q)


def books_upright(x0, y, z, dx=0.045):
    for i, col in enumerate(REQ_BOOK):
        bh = 0.24 - 0.02 * (i % 2)
        box([x0 + i * dx, y, z + bh / 2], [0.02, 0.07, bh / 2], col)
    box([x0 + 4 * dx, y, z + 0.08], [0.006, 0.06, 0.08], GREY)  # bookend


def clutter_initial():
    (cx, cy), top = NS["c"], NS["h"]
    box([cx - 0.10, cy - 0.12, top + 0.06], [0.10, 0.06, 0.06], CLUTTER[0])  # tissue box
    cyl([cx + 0.12, cy - 0.12, top + 0.05], 0.04, 0.10, CLUTTER[1])  # old mug
    cyl([cx + 0.10, cy + 0.12, top + 0.11], 0.06, 0.22, PLANT)  # plant
    cyl([cx - 0.10, cy + 0.10, top + 0.11], 0.035, 0.22, CLUTTER[2])  # bottle
    box([cx, cy - 0.02, NS["shelves"][1] + 0.04], [0.16, 0.14, 0.04], CLUTTER[2])  # magazines (shelf)
    (x0, x1), (y0, y1), z = WS
    box([0.05, (y0 + y1) / 2, z + 0.08], [0.09, 0.06, 0.08], CLUTTER[3])  # photo frame / box on wall shelf
    cyl([0.40, (y0 + y1) / 2, z + 0.09], 0.05, 0.18, CLUTTER[0])  # vase
    box([0.65, (y0 + y1) / 2, z + 0.05], [0.07, 0.05, 0.05], CLUTTER[1])


def clutter_goal():
    """Clutter moved out of reach: nightstand bottom, foot chest, dresser top."""
    (cx, cy) = NS["c"]
    zb = NS["shelves"][0]
    box([cx, cy - 0.02, zb + 0.04], [0.16, 0.14, 0.04], CLUTTER[2])  # magazines -> bottom shelf
    (x0, x1), (y0, y1), z = FC
    box([(x0 + x1) / 2 - 0.08, y0 + 0.12, z + 0.06], [0.10, 0.06, 0.06], CLUTTER[0])  # tissue -> chest
    cyl([(x0 + x1) / 2 + 0.10, y0 + 0.15, z + 0.05], 0.04, 0.10, CLUTTER[1])
    cyl([(x0 + x1) / 2, y1 - 0.12, z + 0.11], 0.06, 0.22, PLANT)
    x, y, h = DR["x"] - 0.05, DR["y"], DR["h"]
    cyl([x, y - 0.25, h + 0.11], 0.035, 0.22, CLUTTER[2])
    box([x, y + 0.05, h + 0.08], [0.09, 0.06, 0.08], CLUTTER[3])
    cyl([x + 0.05, y + 0.30, h + 0.09], 0.05, 0.18, CLUTTER[0])
    box([x - 0.05, y + 0.42, h + 0.05], [0.07, 0.05, 0.05], CLUTTER[1])


def robot(base_xy, yaw, reach_target=None):
    bx, by = base_xy
    box([bx, by, 0.17], [0.27, 0.27, 0.15], (0.25, 0.25, 0.28, 1), yaw)
    box([bx, by, 0.55], [0.06, 0.06, 0.25], (0.20, 0.20, 0.22, 1), yaw)
    sh = np.array([bx, by, 0.82])
    c = (0.80, 0.82, 0.86, 1)
    if reach_target is None:
        elbow = sh + [0.05 * np.cos(yaw), 0.05 * np.sin(yaw), 0.30]
        wrist = elbow + [0.18 * np.cos(yaw), 0.18 * np.sin(yaw), -0.12]
    else:
        tgt = np.asarray(reach_target, float)
        elbow, wrist = (sh + tgt) / 2 + [0, 0, 0.22], tgt + [0, 0, 0.10]
    capsule(sh, elbow, 0.045, c)
    capsule(elbow, wrist, 0.04, c)
    sphere(wrist, 0.045, (0.15, 0.15, 0.18, 1))


def ellipsoid_wire(center, radii, rgba=(0.95, 0.20, 0.60, 1)):
    ts = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    pts = []
    for zf in (-0.5, 0.0, 0.5):
        s = np.sqrt(1 - zf ** 2)
        pts += [[radii[0] * s * np.cos(t), radii[1] * s * np.sin(t), radii[2] * zf] for t in ts]
    pts += [[radii[0] * np.cos(t), 0, radii[2] * np.sin(t)] for t in ts]
    pts += [[0, radii[1] * np.cos(t), radii[2] * np.sin(t)] for t in ts]
    for d in pts:
        q = center + np.array(d)
        if q[2] > 0.0:
            sphere(q, 0.007, rgba)


def slot_markers():
    """Green: prior-safe. Yellow: reachable only with this user's theta. Grey: unreachable for this user."""
    names, pts = all_slots()
    rng = np.random.default_rng(0)
    prior = rng.uniform(PRIOR_LO, PRIOR_HI, size=(20000, 3))
    safe = (margin(pts, prior) >= 0).mean(0) >= 1 - DELTA
    mine = margin(pts, THETA[None])[0] >= 0
    for q, s, m in zip(pts, safe, mine):
        col = (0.10, 0.75, 0.25, 1) if s else (0.98, 0.80, 0.10, 1) if m else (0.55, 0.55, 0.58, 1)
        cyl([q[0], q[1], q[2] - 0.05], 0.035, 0.012, col)
    return safe.sum(), mine.sum(), len(pts)


# ---------------------------------------------------------------- scenes
def render(scene):
    p.resetSimulation()
    room()
    bed_and_user()
    nightstand()
    wall_shelf()
    foot_chest()
    dresser()
    info = None
    if scene == "initial":
        bookshelf()
        clutter_initial()
        x, y, h = DR["x"] - 0.05, DR["y"], DR["h"]
        small_items([[x, y - 0.32 + 0.14 * i, h + 0.06] for i in range(5)])
        robot((1.70, -1.55), np.pi / 2)
    elif scene == "goal":
        bookshelf(gaps=(0, 1, 2, 3))
        clutter_goal()
        (cx, cy), top = NS["c"], NS["h"]
        # small items on the nightstand top (prior-safe and user-reachable front-left area)
        small_items([[cx - 0.13, cy - 0.15, top + 0.06], [cx - 0.13, cy - 0.02, top + 0.06],
                     [cx - 0.01, cy - 0.15, top + 0.06], [cx - 0.13, cy + 0.12, top + 0.06],
                     [cx + 0.11, cy - 0.15, top + 0.06]])
        # books upright on the wall shelf, inside this user's reach but outside the prior-safe set
        (x0, x1), (y0, y1), z = WS
        books_upright(0.30, (y0 + y1) / 2, z)
        robot((1.40, 0.45), np.pi, reach_target=[1.0, 0.55, 0.95])
        ellipsoid_wire(SHOULDER, THETA)
    else:  # slot map
        info = slot_markers()
        ellipsoid_wire(SHOULDER, THETA)
    view = p.computeViewMatrix(cameraEyePosition=[0.1, -2.6, 3.2], cameraTargetPosition=[1.05, -0.05, 0.45],
                               cameraUpVector=[0, 0, 1])
    proj = p.computeProjectionMatrixFOV(fov=55, aspect=W / H, nearVal=0.05, farVal=12)
    img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER, shadow=1,
                           lightDirection=[-1.0, -2.0, 4.0], lightAmbientCoeff=0.55, lightDiffuseCoeff=0.55)
    return Image.fromarray(np.reshape(img[2], (H, W, 4))[:, :, :3].astype(np.uint8)), info


def label(im, text):
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    except OSError:
        f = ImageFont.load_default()
    d.rectangle([0, 0, W, 52], fill=(255, 255, 255))
    d.text((20, 10), text, fill=(20, 20, 20), font=f)
    return im


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    p.connect(p.DIRECT)
    a, _ = render("initial")
    b, _ = render("goal")
    c, (n_safe, n_mine, n_all) = render("slots")
    a = label(a, "Initial scene s0 (clutter on nightstand + wall shelf)")
    b = label(b, f"Goal scene, user theta = {tuple(THETA)} (pink = reach ellipsoid)")
    c = label(c, f"Slots: green = prior-safe ({n_safe}), yellow = this user only ({n_mine - n_safe}), "
                 f"grey = out of reach")
    for im, name in ((a, "initial"), (b, "goal"), (c, "slots")):
        im.save(os.path.join(OUT, f"{name}.png"))
    combo = Image.new("RGB", (3 * W + 20, H), (255, 255, 255))
    for i, im in enumerate((a, b, c)):
        combo.paste(im, (i * (W + 10), 0))
    combo.save(os.path.join(OUT, "initial_vs_goal.png"))
    print("saved to", OUT, "| slots:", n_safe, "prior-safe,", n_mine, "reachable for this user, of", n_all)

"""Shared layout and reach model for the Bedside Stocking domain sketch."""
import numpy as np

# User sits up in bed against a raised backrest; right shoulder faces the nightstand (+x).
SHOULDER = np.array([0.21, 0.66, 1.08])

# Population prior over theta = (r_x, r_y, r_z), uniform box (metres).
PRIOR_LO = np.array([0.45, 0.35, 0.40])
PRIOR_HI = np.array([0.95, 0.80, 0.90])

GRASP_DZ = 0.06  # grasp point height above the support surface
SLOT = 0.11  # slot spacing (typical item footprint)

# Placement regions near the bed. The bed surface itself is not a placement region.
# name: (x range, y range, surface z)
REGIONS = {
    "nightstand_top": ((0.56, 0.96), (0.46, 0.90), 0.76),
    "nightstand_shelf": ((0.56, 0.96), (0.46, 0.90), 0.48),
    "nightstand_bottom": ((0.56, 0.96), (0.46, 0.90), 0.14),
    "wall_shelf": ((-0.30, 0.80), (0.97, 1.13), 1.35),
    "foot_chest": ((0.58, 0.98), (-0.25, 0.25), 0.55),
}


def slots(region):
    (x0, x1), (y0, y1), z = REGIONS[region]
    xs = np.arange(x0 + SLOT / 2, x1 - SLOT / 2 + 1e-9, SLOT)
    ys = np.arange(y0 + SLOT / 2, y1 - SLOT / 2 + 1e-9, SLOT)
    return np.array([[x, y, z + GRASP_DZ] for x in xs for y in ys])


def all_slots():
    names, pts = [], []
    for r in REGIONS:
        s = slots(r)
        names += [r] * len(s)
        pts.append(s)
    return np.array(names), np.concatenate(pts)


def margin(x, theta):
    """m(x, theta) = 1 - ||(x - c_sh) / theta||; x: (N,3), theta: (K,3) -> (K,N)."""
    d = (x[None, :, :] - SHOULDER) / theta[:, None, :]
    return 1.0 - np.linalg.norm(d, axis=-1)

"""The process-wide `scene-side-limb` variant (16 shipped); set via env var so workers inherit it.

The scene only changes the furniture; the limb decides which three motions `theta` scales.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from kinder.envs.dynamic3d.limbrepositioning3d import create_variant_config

ENV_VAR = "LIMB_BASELINE_VARIANT"
DEFAULT_VARIANT = "isolated-right-arm"
SCENES: tuple[str, ...] = ("isolated", "human", "bed", "wheelchair")
LIMBS: tuple[str, ...] = ("left-arm", "right-arm", "left-leg", "right-leg")
VARIANTS: tuple[str, ...] = tuple(f"{s}-{l}" for s in SCENES for l in LIMBS)


@dataclass(frozen=True)
class LimbVariant:
    """One variant, and the three motions `theta` scales for its limb."""

    name: str
    limb_name: str
    active_motions: tuple[str, ...]
    active_joints: tuple[int, ...]
    active_signs: tuple[int, ...]

    @property
    def inactive_joints(self) -> tuple[int, ...]:
        """Limb joints no active motion bounds (held at nominal)."""
        return tuple(j for j in range(6) if j not in self.active_joints)


# Each limb's two proximal bounds plus its hinge, with the sign each motion sets.
# Read off `_joint_ranges_deg`; the sign flips between sides.
_LIMB_MOTIONS: dict[str, tuple[tuple[str, int, int], ...]] = {
    "human-right-arm": (
        ("shoulder_flexion", 0, -1),
        ("shoulder_abduction", 1, +1),
        ("elbow_flexion", 3, -1),
    ),
    "human-left-arm": (
        ("shoulder_flexion", 0, +1),
        ("shoulder_abduction", 1, +1),
        ("elbow_flexion", 3, +1),
    ),
    "human-right-leg": (
        ("hip_flexion", 0, -1),
        ("hip_abduction", 1, +1),
        ("knee_flexion", 3, +1),
    ),
    "human-left-leg": (
        ("hip_flexion", 0, -1),
        ("hip_abduction", 1, -1),
        ("knee_flexion", 3, +1),
    ),
}


def _build(name: str) -> LimbVariant:
    """Build the variant `name`; reject names the environment does not ship."""
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant {name!r}; expected one of {list(VARIANTS)}")
    limb_name = f"human-{name.split('-', 1)[1]}"
    motions = _LIMB_MOTIONS[limb_name]
    return LimbVariant(
        name=name,
        limb_name=limb_name,
        active_motions=tuple(m for m, _, _ in motions),
        active_joints=tuple(j for _, j, _ in motions),
        active_signs=tuple(s for _, _, s in motions),
    )


def select(name: str) -> LimbVariant:
    """Select `name` for this process and the workers it spawns."""
    variant = _build(name)
    os.environ[ENV_VAR] = name
    return variant


def current() -> LimbVariant:
    """The selected variant (default: isolated right arm)."""
    return _build(os.environ.get(ENV_VAR, DEFAULT_VARIANT))


def scene_config():
    """Env config for the selected variant."""
    return create_variant_config(current().name)

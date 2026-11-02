"""Show which range of motion `theta` scales, one panel per active motion.

`python scripts/make_rom_figure.py --variant human-left-arm`.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib
import numpy as np
import pybullet as p
from kinder.envs.dynamic3d.limb_scenes import HumanScene
from kinder.envs.dynamic3d.limbrepositioning3d import create_variant_config
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from able.tasks.limb import variant as variant_module
from able.tasks.limb.human import (
    DEFAULT_PRIOR,
    active_joints,
    active_motions,
    active_signs,
    default_magnitudes,
)

KEEP = "#008D63"
INK = "#141C1B"
FAINT = "#6E7A78"
SKIN = "#C79A6B"

SIZE = (760, 900)
FOV = 38.0
# The tip link, whose swept path draws the arc.
TIP_LINK = 7

plt.rcParams.update({"font.size": 10, "text.color": INK, "savefig.facecolor": "white"})


def _matrices(client: int, eye, target, aspect):
    """The view and projection matrices for one camera, and their product."""
    view = p.computeViewMatrix(list(eye), list(target), [0.0, 0.0, 1.0],
                               physicsClientId=client)
    projection = p.computeProjectionMatrixFOV(FOV, aspect, 0.05, 8.0,
                                              physicsClientId=client)
    combined = (np.array(projection).reshape(4, 4, order="F")
                @ np.array(view).reshape(4, 4, order="F"))
    return view, projection, combined


def _to_pixels(combined: NDArray[np.float64], points: NDArray[np.float64]):
    """Project world points to pixel coordinates."""
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    clip = homogeneous @ combined.T
    ndc = clip[:, :3] / clip[:, 3:4]
    width, height = SIZE
    return np.stack([(ndc[:, 0] * 0.5 + 0.5) * width,
                     (0.5 - ndc[:, 1] * 0.5) * height], axis=1)


class Scene:
    """A whole human whose active limb can be posed and photographed."""

    def __init__(self, variant: str) -> None:
        self.client = p.connect(p.DIRECT)
        config = create_variant_config(variant).scene
        config = dataclasses.replace(config, show_goal=False,
                                     limb_init_joint_positions=(0.0,) * 6)
        self.scene = HumanScene(config, self.client)
        self.limb = self.scene.passive_limb
        self.torso = np.array(config.torso_base_pose.position)

    def pose(self, joint: int, angle: float) -> None:
        """Hold every joint at zero but one."""
        for index, pybullet_joint in enumerate(self.limb.arm_joints):
            p.resetJointState(self.limb.robot_id, pybullet_joint,
                              angle if index == joint else 0.0,
                              physicsClientId=self.client)

    def tip(self) -> NDArray[np.float64]:
        """Where the hand is now, in world coordinates."""
        state = p.getLinkState(self.limb.robot_id, TIP_LINK,
                               physicsClientId=self.client)
        return np.array(state[4])

    def sweep(self, joint: int, angles: NDArray[np.float64]) -> NDArray[np.float64]:
        """The path the hand traces as one joint runs through `angles`."""
        path = []
        for angle in angles:
            self.pose(joint, float(angle))
            path.append(self.tip())
        return np.array(path)

    def shoot(self, view, projection):
        """One rendered frame, with the pixels the limb covers and those the body does."""
        width, height = SIZE
        _, _, rgb, _, segmentation = p.getCameraImage(
            width, height, view, projection, renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client,
        )
        image = np.asarray(rgb, np.uint8).reshape(height, width, 4)[:, :, :3]
        seen = np.asarray(segmentation).reshape(height, width)
        return image.astype(np.float64) / 255.0, seen == self.limb.robot_id, seen >= 0


def _camera(path: NDArray[np.float64], torso: NDArray[np.float64]):
    """Look along the normal of the plane the motion sweeps, from the limb's side."""
    centred = path - path.mean(axis=0)
    normal = np.linalg.svd(centred)[2][2]
    if normal @ (path[0] - torso) < 0.0:
        normal = -normal
    return normal


def _content_box(mask: NDArray[np.bool_], pixels: NDArray[np.float64], pad: int = 30):
    """The centre and size of everything drawn: the body pixels and the arc."""
    rows, columns = np.nonzero(mask)
    low = np.minimum([columns.min(), rows.min()], pixels.min(axis=0)) - pad
    high = np.maximum([columns.max(), rows.max()], pixels.max(axis=0)) + pad
    return (*(0.5 * (low + high)), *(high - low))


def _blend(base: NDArray[np.float64], image, mask, colour: str, alpha: float):
    """Tint the limb pixels of a ghost frame into the base frame."""
    tint = 0.45 * image[mask] + 0.55 * np.array(matplotlib.colors.to_rgb(colour))
    base[mask] = (1.0 - alpha) * base[mask] + alpha * tint


def figure_rom(variant: str, theta: float, out: Path) -> None:
    """One panel per motion `theta` scales, each showing the swept range."""
    variant_module.select(variant.replace("human-", "isolated-"))
    motions, joints, signs = active_motions(), active_joints(), active_signs()
    magnitudes = default_magnitudes()
    scene = Scene(variant)
    width, height = SIZE

    panels = []
    for motion, joint, sign, magnitude in zip(
        motions, joints, signs, magnitudes, strict=True
    ):
        angles = np.linspace(0.0, sign * magnitude, 60)
        path = scene.sweep(joint, angles)
        eye = path.mean(axis=0) + 2.5 * _camera(path, scene.torso)
        target = 0.5 * (path.mean(axis=0) + scene.torso + np.array([0.0, 0.0, 0.35]))
        view, projection, combined = _matrices(scene.client, eye, target,
                                               width / height)

        scene.pose(joint, 0.0)
        base, _, covered = scene.shoot(view, projection)
        for limit, colour in ((magnitude, FAINT), (magnitude * theta, KEEP)):
            scene.pose(joint, float(sign * limit))
            image, mask, body = scene.shoot(view, projection)
            covered |= body
            _blend(base, image, mask, colour, 0.8)
        scene.pose(joint, 0.0)

        pixels = _to_pixels(combined, path)
        panels.append({
            "motion": motion, "magnitude": magnitude, "image": base,
            "pixels": pixels, "inside": round(theta * (len(path) - 1)),
            "box": _content_box(covered, pixels),
        })

    span = np.array([max(p["box"][2] for p in panels),
                     max(p["box"][3] for p in panels)])
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 4.6))
    for ax, panel in zip(axes, panels, strict=True):
        pixels, inside = panel["pixels"], panel["inside"]
        ax.imshow(panel["image"])
        ax.plot(*pixels[inside:].T, color=FAINT, lw=2.0, zorder=3)
        ax.plot(*pixels[: inside + 1].T, color=KEEP, lw=2.4, zorder=4)
        ax.scatter(*pixels[inside], s=26, color=KEEP, zorder=5)
        ax.scatter(*pixels[-1], s=26, color=FAINT, zorder=5)
        ax.set_title(f"{panel['motion'].replace('_', ' ')}\n"
                     f"$R$ = {np.rad2deg(panel['magnitude']):.0f}°", fontsize=11)
        centre = panel["box"][:2]
        ax.set_xlim(centre[0] - span[0] / 2, centre[0] + span[0] / 2)
        ax.set_ylim(centre[1] + span[1] / 2, centre[1] - span[1] / 2)
        ax.axis("off")

    handles = [
        plt.Line2D([], [], color=SKIN, marker="o", ls="", ms=8, label="0°"),
        plt.Line2D([], [], color=KEEP, marker="o", ls="", ms=8,
                   label=fr"$\theta R$, person ($\theta$={theta:g})"),
        plt.Line2D([], [], color=FAINT, marker="o", ls="", ms=8,
                   label="$R$, default"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center",
               fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"$\\theta$ scales {len(motions)} motions of the "
                 f"{variant.replace('human-', '').replace('-', ' ')}",
                 fontsize=12, y=0.99)
    fig.subplots_adjust(top=0.83, bottom=0.09, wspace=0.02)

    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"rom_definition.{suffix}", dpi=300)
        print(f"wrote {out / f'rom_definition.{suffix}'}")
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="human-left-arm",
                        help="A whole-human variant, e.g. human-left-arm.")
    parser.add_argument("--theta", type=float, default=DEFAULT_PRIOR.mean,
                        help="The scale factor to draw as the person's limit.")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    figure_rom(args.variant, args.theta, args.out)


if __name__ == "__main__":
    main()

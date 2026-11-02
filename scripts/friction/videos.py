"""Videos of complexity-sweep episodes, replayed deterministically from (env, level, sims, seed).

    # One episode:
    python scripts/friction/videos.py one --env 2d --level L2 --sims 1000 --seed 0
    # Pick a success and a failure per (env, level) from the sweep and list them:
    python scripts/friction/videos.py select
    # Add the floor-friction legend to videos rendered before it existed:
    python scripts/friction/videos.py legend
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from able.planners.pomcpow import POMCPOWConfig
from able.tasks.friction import FrictionTask
from able.tasks.friction.env.puckpushing_utils import SURFACE_NAMES, SURFACE_RGB
from able.tasks.friction.run import run

DATA = Path("data/friction/complexity")
OUT = Path("videos/friction/complexity")
# Display-only arm lift/lower time (s).
ARM_SECONDS = 0.6
FPS = 20


try:
    LEGEND_FONT = ImageFont.truetype("DejaVuSans.ttf", 10)
except OSError:
    LEGEND_FONT = ImageFont.load_default()


def friction_legend(frame: np.ndarray, frictions) -> np.ndarray:
    """Draw each floor's true friction in a top-right box."""
    image = Image.fromarray(frame[:, :, :3]).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    ranks = np.argsort(np.argsort(frictions))
    words = ("slippery", "medium", "sticky")
    lines = [
        f"{name} \u03bc={mu:.3f} {words[min(int(r), 2)]}"
        for name, mu, r in zip(SURFACE_NAMES, frictions, ranks)
    ]
    row, swatch, pad = 13, 9, 4
    width = max(int(draw.textlength(line, font=LEGEND_FONT)) for line in lines)
    box_w, box_h = width + swatch + 3 * pad, row * len(lines) + 2 * pad
    x0, y0 = image.width - box_w - 4, 22
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(255, 255, 255, 210))
    for k, (line, rgb) in enumerate(zip(lines, SURFACE_RGB)):
        y = y0 + pad + k * row
        color = tuple(int(255 * c) for c in rgb) + (255,)
        draw.rectangle([x0 + pad, y + 2, x0 + pad + swatch, y + 2 + swatch], fill=color, outline=(90, 90, 90, 255))
        draw.text((x0 + 2 * pad + swatch, y), line, fill=(30, 30, 30, 255), font=LEGEND_FONT)
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame[:, :, :3])
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, 18], fill=(255, 255, 255))
    draw.text((6, 3), text, fill=(30, 30, 30))
    return np.asarray(image)


class Recorder:
    """Captures frames from a 2D or 3D env during an episode."""

    def __init__(self, env, env_name: str, header: str) -> None:
        self.env = env
        self.env_name = env_name
        self.header = header
        self.frames: list[np.ndarray] = []
        self.push = 0
        self.status = ""
        dt = 1.0 / env.config.sim_hz if env_name == "2d" else env.config.physics_dt
        self.every = max(int(round(1.0 / (FPS * dt))), 1)
        self._ticks = 0

    def snap(self) -> None:
        if self.env_name == "2d":
            self.env._read_state_from_space()  # pylint: disable=protected-access
        text = f"{self.header} | push {self.push} {self.status}"
        frame = caption(self.env.render(), text)
        self.frames.append(friction_legend(frame, self.env.frictions))

    def tick(self) -> None:
        """Per physics step; captures at the video frame rate."""
        self._ticks += 1
        if self._ticks % self.every == 0:
            self.snap()

    def robot_xy_heading(self) -> tuple[np.ndarray, float]:
        if self.env_name == "2d":
            pose = self.env._robot_pose  # pylint: disable=protected-access
            return np.array(pose[:2]), float(pose[2])
        base = self.env.robot.get_base()
        return np.array([base.x, base.y]), float(base.rot)

    def set_base(self, xy: np.ndarray, heading: float) -> None:
        """Move the base for display; the paddle stays in the gripper."""
        env = self.env
        if self.env_name == "2d":
            env._set_robot_pose(np.array([xy[0], xy[1], heading]))  # pylint: disable=protected-access
            env._stow_paddle()  # pylint: disable=protected-access
        else:
            from pybullet_helpers.geometry import (
                SE2Pose,  # pylint: disable=import-outside-toplevel
            )

            env.robot.set_base(SE2Pose(float(xy[0]), float(xy[1]), heading))
            env._attach_paddle()  # pylint: disable=protected-access

    def move_paddle(self, goal, seconds: float = ARM_SECONDS) -> None:
        """Animate the paddle along a straight line to pose `goal` (3D only), IK per frame."""
        env = self.env
        if self.env_name == "2d":
            return
        from pybullet_helpers.geometry import (  # pylint: disable=import-outside-toplevel
            Pose,
            get_pose,
        )
        from scipy.spatial.transform import (  # pylint: disable=import-outside-toplevel
            Rotation,
            Slerp,
        )

        start = get_pose(env.paddle_id, env.physics_client_id)
        p0, p1 = np.asarray(start.position), np.asarray(goal.position)
        turn = Slerp([0.0, 1.0], Rotation.from_quat([start.orientation, goal.orientation]))
        for t in np.linspace(0.0, 1.0, max(int(seconds * FPS), 1) + 1)[1:]:
            pose = Pose(tuple(p0 + t * (p1 - p0)), tuple(turn(t).as_quat()))
            env._hold_paddle_at(pose)  # pylint: disable=protected-access
            self.snap()

    def at_height(self, pose, height: float):
        """`pose` moved straight up or down to `height`."""
        from pybullet_helpers.geometry import (
            Pose,  # pylint: disable=import-outside-toplevel
        )

        x, y, _ = pose.position
        return Pose((x, y, height), pose.orientation)

    def save_robot(self):
        """Arm joints and paddle pose, to restore after an animation."""
        if self.env_name == "2d":
            return (self.env._paddle_stowed,)  # pylint: disable=protected-access
        from pybullet_helpers.geometry import (
            get_pose,  # pylint: disable=import-outside-toplevel
        )

        env = self.env
        return (
            list(env.robot.arm.get_joint_positions()),
            get_pose(env.paddle_id, env.physics_client_id),
        )

    def restore_robot(self, saved) -> None:
        """Undo `save_robot`'s state changes made by an animation."""
        env = self.env
        if self.env_name == "2d":
            env._paddle_stowed = saved[0]  # pylint: disable=protected-access
            return
        from pybullet_helpers.geometry import (
            set_pose,  # pylint: disable=import-outside-toplevel
        )

        env.robot.arm.set_joints(saved[0])
        set_pose(env.paddle_id, saved[1], env.physics_client_id)

    def animate_drive(
        self, puck: np.ndarray, angle: float, stance: float | None = None
    ) -> None:
        """Animate lift, drive to the push station and lower; then restore the robot for the env."""
        env, world = self.env, self.env.world
        pose = world.push_pose(puck, angle, stance)
        if pose is None:
            return
        start, heading0 = self.robot_xy_heading()
        station, heading = pose
        # Restored later: the env's IK starts from the current arm pose.
        saved = self.save_robot()
        if self.env_name == "3d":
            from pybullet_helpers.geometry import (
                get_pose,  # pylint: disable=import-outside-toplevel
            )

            # Lift straight up, then pull in: the arm cannot reach the diagonal.
            now = get_pose(env.paddle_id, env.physics_client_id)
            carry = env._carry_pose()  # pylint: disable=protected-access
            self.move_paddle(self.at_height(now, carry.position[2]))
            self.move_paddle(carry)
        else:
            env._stow_paddle()  # pylint: disable=protected-access
        path = world.drive_path(start, station, puck)
        if len(path) >= 2:
            points = np.asarray(path)
            lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            speed = 1.0  # m/s, for display only
            num = max(int(lengths[-1] / speed * FPS), 2)
            for s in np.linspace(0, lengths[-1], num):
                k = min(int(np.searchsorted(lengths, s, side="right")) - 1, len(points) - 2)
                frac = (s - lengths[k]) / max(lengths[k + 1] - lengths[k], 1e-9)
                xy = points[k] + frac * (points[k + 1] - points[k])
                h = heading0 + (heading - heading0) * (s / lengths[-1])
                self.set_base(xy, math.atan2(math.sin(h), math.cos(h)))
                self.snap()
        self.set_base(station, heading)
        direction = np.array([math.cos(angle), math.sin(angle)])
        face = puck - (world.physics.puck_radius + 0.005) * direction
        if self.env_name == "3d":
            # Reach out at carry height, then lower straight down behind the puck.
            strike = env._paddle_pose(face, angle)  # pylint: disable=protected-access
            height = env._carry_pose().position[2]  # pylint: disable=protected-access
            self.move_paddle(self.at_height(strike, height))
            self.move_paddle(strike)
        # The env drives (and charges for) the move itself from the start pose.
        self.set_base(start, heading0)
        self.restore_robot(saved)


def install_hooks(recorder: Recorder) -> None:
    """Hook frame capture into every physics step and env.step."""
    env = recorder.env
    if recorder.env_name == "2d":
        import pymunk  # pylint: disable=import-outside-toplevel

        original = pymunk.Space.step

        def step(space, dt):
            original(space, dt)
            if space is env.pymunk_space:
                recorder.tick()

        pymunk.Space.step = step  # type: ignore[method-assign]
    else:
        import pybullet  # pylint: disable=import-outside-toplevel
        from kinder.envs.dynamic3d import (
            puckpushing3d,  # pylint: disable=import-outside-toplevel
        )

        class _Proxy:
            def __getattr__(self, name):
                return getattr(pybullet, name)

            @staticmethod
            def stepSimulation(*args, **kwargs):  # pylint: disable=invalid-name
                pybullet.stepSimulation(*args, **kwargs)
                recorder.tick()

        puckpushing3d.p = _Proxy()  # type: ignore[assignment]

    original_step = env.step

    def step(action):
        recorder.push += 1
        recorder.status = ""
        puck = recorder.env.world.nearest_free_puck_position(_puck_position(recorder))
        stance = float(action[2]) if len(action) > 2 else None
        recorder.animate_drive(puck, float(action[0]), stance)
        result = original_step(action)
        info = result[4]
        if info.get("fell_in_hole"):
            recorder.status = "- fell in a hole"
        elif info.get("goal_reached"):
            recorder.status = "- goal!"
        recorder.snap()
        return result

    env.step = step


def _puck_position(recorder: Recorder) -> np.ndarray:
    env = recorder.env
    if recorder.env_name == "2d":
        body = env._puck_body  # pylint: disable=protected-access
        return np.array([body.position.x, body.position.y])
    return env._puck_position()  # pylint: disable=protected-access


def record(env_name: str, level: str, sims: int, seed: int, out: Path) -> str:
    """Replay one episode while recording; returns its outcome."""
    task = FrictionTask(env=env_name, level=level, config=POMCPOWConfig(num_sims=sims, max_depth=8))
    env = task.environment
    if env_name == "2d":
        env.render_mode = "rgb_array"
    header = f"{env_name.upper()} {level} | POMCPOW {sims} sims | seed {seed}"
    recorder = Recorder(env, env_name, header)
    install_hooks(recorder)
    log = run(task, seed)
    recorder.status = f"- {log.outcome.upper()} after {len(log.steps)} pushes"
    recorder.snap()
    frames = recorder.frames + [recorder.frames[-1]] * FPS * 2
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=FPS, macro_block_size=8)
    return log.outcome


def select() -> list[tuple[str, str, int, int, str]]:
    """One success and one failure per (env, level), preferring 1000 sims."""
    runs = collections.defaultdict(list)
    for path in glob.glob(str(DATA / "*_seed*.json")):
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        runs[(run["env"], run["level"])].append(run)
    chosen = []
    order = {1000: 0, 300: 1, 3000: 2}
    for (env_name, level), items in sorted(runs.items()):
        items.sort(key=lambda r: (order[r["sims"]], r["seed"]))
        for kind, test in (("success", lambda r: r["outcome"] == "goal"),
                           ("failure", lambda r: r["outcome"] != "goal")):
            match = next((r for r in items if test(r)), None)
            if match is not None:
                chosen.append((env_name, level, match["sims"], match["seed"], kind))
    return chosen


def add_legends() -> None:
    """Draw the friction legend onto existing videos in OUT from their logs."""
    for path in sorted(OUT.glob("*.mp4")):
        env_name, level, _, sims, seed = path.stem.split("_")
        log = DATA / f"{env_name}_{level}_{sims}_{seed}.json"
        with open(log, encoding="utf-8") as f:
            frictions = json.load(f)["true_frictions"]
        frames = [friction_legend(frame, frictions) for frame in imageio.mimread(path, memtest=False)]
        imageio.mimwrite(path, frames, fps=FPS, macro_block_size=8)
        print(f"{path}: {np.round(frictions, 3).tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("one")
    one.add_argument("--env", required=True)
    one.add_argument("--level", required=True)
    one.add_argument("--sims", type=int, required=True)
    one.add_argument("--seed", type=int, required=True)
    one.add_argument("--kind", default="episode")
    sub.add_parser("select")
    sub.add_parser("legend")
    args = parser.parse_args()
    if args.cmd == "select":
        for item in select():
            print(*item)
        return
    if args.cmd == "legend":
        add_legends()
        return
    out = OUT / f"{args.env}_{args.level}_{args.kind}_s{args.sims}_seed{args.seed}.mp4"
    outcome = record(args.env, args.level, args.sims, args.seed, out)
    print(f"{out}: {outcome}")


if __name__ == "__main__":
    main()

"""Frame capture for friction-task pushes: drive to the push station, strike, slide."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from able.tasks.friction.env.puckpushing_utils import SURFACE_NAMES, SURFACE_RGB

FPS = 10  # frames per second of sim time
ARM_SECONDS = 0.6  # display-only arm lift/lower time
DRIVE_SPEED = 1.0  # m/s, display-only base speed

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
    lines = [f"{name} μ={mu:.3f} {words[min(int(r), 2)]}"
             for name, mu, r in zip(SURFACE_NAMES, frictions, ranks)]
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
    """Draw `text` in a white strip along the top."""
    image = Image.fromarray(frame[:, :, :3])
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, 18], fill=(255, 255, 255))
    draw.text((6, 3), text, fill=(30, 30, 30))
    return np.asarray(image)


class Recorder:
    """Captures a push's frames from a 2D or 3D env; physics hooks live only during `record`."""

    def __init__(self, env: Any, env_name: str, header: str = "", fps: float = FPS) -> None:
        self.env = env
        self.env_name = env_name
        self.header = header
        self.fps = fps
        self.frames: list | None = None  # capture target while recording
        self.push = 0
        self.status = ""
        self._ticks = 0
        if env_name == "2d":
            env.render_mode = "rgb_array"

    @property
    def every(self) -> int:
        """Physics steps per captured frame."""
        dt = 1.0 / self.env.config.sim_hz if self.env_name == "2d" else self.env.config.physics_dt
        return max(int(round(1.0 / (self.fps * dt))), 1)

    def snap(self, frames: list | None = None) -> None:
        """Append the current captioned scene to `frames` (default: the active target)."""
        target = self.frames if frames is None else frames
        if target is None:
            return
        if self.env_name == "2d":
            self.env._read_state_from_space()  # pylint: disable=protected-access
        frame = caption(np.asarray(self.env.render(), np.uint8), f"{self.header} | push {self.push} {self.status}")
        target.append(friction_legend(frame, self.env.frictions))

    def tick(self) -> None:
        """Per physics step; captures at `fps`."""
        self._ticks += 1
        if self._ticks % self.every == 0:
            self.snap()

    def record(self, frames: list, action: np.ndarray, push: int) -> tuple:
        """`env.step(action)`, appending the drive, strike, slide and a final frame to `frames`."""
        self.frames, self.push, self.status, self._ticks = frames, push, "", 0
        try:
            stance = float(action[2]) if len(action) > 2 else None
            puck = self.env.world.nearest_free_puck_position(self.puck_position())
            self.animate_drive(puck, float(action[0]), stance)
            with self._physics_hook():
                result = self.env.step(action)
            info = result[4]
            self.status = "- fell in a hole" if info.get("fell_in_hole") else "- goal!" if info.get(
                "goal_reached") else ""
            self.snap()
        finally:
            self.frames = None
        return result

    @contextmanager
    def _physics_hook(self) -> Iterator[None]:
        """Tick on every physics step of this env, for the duration of the block."""
        if self.env_name == "2d":
            space = self.env.pymunk_space
            original = space.step

            def step(dt):
                original(dt)
                self.tick()

            space.step = step
            try:
                yield
            finally:
                vars(space).pop("step", None)
            return
        import pybullet

        from able.tasks.friction.env import puckpushing3d

        recorder = self

        class _Proxy:
            def __getattr__(self, name):
                return getattr(pybullet, name)

            @staticmethod
            def stepSimulation(*args, **kwargs):  # pylint: disable=invalid-name
                pybullet.stepSimulation(*args, **kwargs)
                recorder.tick()

        original_p = puckpushing3d.p
        puckpushing3d.p = _Proxy()  # type: ignore[assignment]
        try:
            yield
        finally:
            puckpushing3d.p = original_p

    def puck_position(self) -> np.ndarray:
        env = self.env
        if self.env_name == "2d":
            body = env._puck_body  # pylint: disable=protected-access
            return np.array([body.position.x, body.position.y])
        return env._puck_position()  # pylint: disable=protected-access

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
            from pybullet_helpers.geometry import SE2Pose

            env.robot.set_base(SE2Pose(float(xy[0]), float(xy[1]), heading))
            env._attach_paddle()  # pylint: disable=protected-access

    def move_paddle(self, goal, seconds: float = ARM_SECONDS) -> None:
        """Animate the paddle in a straight line to pose `goal` (3D only), IK per frame."""
        env = self.env
        if self.env_name == "2d":
            return
        from pybullet_helpers.geometry import Pose, get_pose
        from scipy.spatial.transform import Rotation, Slerp

        start = get_pose(env.paddle_id, env.physics_client_id)
        p0, p1 = np.asarray(start.position), np.asarray(goal.position)
        turn = Slerp([0.0, 1.0], Rotation.from_quat([start.orientation, goal.orientation]))
        for t in np.linspace(0.0, 1.0, max(int(seconds * self.fps), 1) + 1)[1:]:
            env._hold_paddle_at(Pose(tuple(p0 + t * (p1 - p0)), tuple(turn(t).as_quat())))  # pylint: disable=protected-access
            self.snap()

    @staticmethod
    def at_height(pose, height: float):
        """`pose` moved straight up or down to `height`."""
        from pybullet_helpers.geometry import Pose

        x, y, _ = pose.position
        return Pose((x, y, height), pose.orientation)

    def save_robot(self):
        """Arm and paddle state, to restore after an animation."""
        env = self.env
        if self.env_name == "2d":
            body = env._paddle_body  # pylint: disable=protected-access
            return env._paddle_stowed, tuple(body.position), body.angle  # pylint: disable=protected-access
        from pybullet_helpers.geometry import get_pose

        return list(env.robot.arm.get_joint_positions()), get_pose(env.paddle_id, env.physics_client_id)

    def restore_robot(self, saved) -> None:
        """Undo an animation's arm and paddle changes."""
        env = self.env
        if self.env_name == "2d":
            env._paddle_stowed = saved[0]  # pylint: disable=protected-access
            body = env._paddle_body  # pylint: disable=protected-access
            body.position, body.angle, body.velocity = saved[1], saved[2], (0.0, 0.0)
            env.pymunk_space.reindex_shapes_for_body(body)
            return
        from pybullet_helpers.geometry import set_pose

        env.robot.arm.set_joints(saved[0])
        set_pose(env.paddle_id, saved[1], env.physics_client_id)

    def animate_drive(self, puck: np.ndarray, angle: float, stance: float | None = None) -> None:
        """Animate lift, drive to the push station and lower; then restore the robot for the env."""
        env, world = self.env, self.env.world
        pose = world.push_pose(puck, angle, stance)
        if pose is None:
            return
        start, heading0 = self.robot_xy_heading()
        station, heading = pose
        saved = self.save_robot()  # the env's IK starts from the current arm pose
        if self.env_name == "3d":
            from pybullet_helpers.geometry import get_pose

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
            num = max(int(lengths[-1] / DRIVE_SPEED * self.fps), 2)
            for s in np.linspace(0, lengths[-1], num):
                k = min(int(np.searchsorted(lengths, s, side="right")) - 1, len(points) - 2)
                frac = (s - lengths[k]) / max(lengths[k + 1] - lengths[k], 1e-9)
                xy = points[k] + frac * (points[k + 1] - points[k])
                h = heading0 + (heading - heading0) * (s / max(lengths[-1], 1e-9))
                self.set_base(xy, math.atan2(math.sin(h), math.cos(h)))
                self.snap()
        self.set_base(station, heading)
        if self.env_name == "3d":
            # Reach out at carry height, then lower straight down behind the puck.
            face = puck - (world.physics.puck_radius + 0.005) * np.array([math.cos(angle), math.sin(angle)])
            strike = env._paddle_pose(face, angle)  # pylint: disable=protected-access
            self.move_paddle(self.at_height(strike, env._carry_pose().position[2]))  # pylint: disable=protected-access
            self.move_paddle(strike)
        # The env drives (and charges for) the move itself from the start pose.
        self.set_base(start, heading0)
        self.restore_robot(saved)

"""Frame capture shared by every task: RGB frames from a sim, captions, and a strided recorder."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    """`frame` with `text` in a white strip along the top (unchanged if `text` is empty)."""
    if not text:
        return frame
    image = Image.fromarray(np.ascontiguousarray(frame[:, :, :3]))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, 18], fill=(255, 255, 255))
    draw.text((6, 3), text, fill=(30, 30, 30))
    return np.asarray(image)


def render(sim: Any, text: str = "") -> np.ndarray:
    """The sim's current RGB uint8 (H, W, 3) frame, captioned."""
    return caption(np.ascontiguousarray(np.asarray(sim.render(), dtype=np.uint8)[..., :3]), text)


class FrameRecorder:
    """Appends frames of `sim` to `frames` (None = recording off); `tick` keeps every `every`-th call."""

    def __init__(self, sim: Any, frames: list | None, every: int = 1, text: str = "") -> None:
        self.sim, self.frames, self.every, self.text = sim, frames, max(int(every), 1), text
        self._ticks = 0

    @property
    def on(self) -> bool:
        return self.frames is not None

    def snap(self) -> None:
        if self.frames is not None:
            self.frames.append(render(self.sim, self.text))

    def tick(self) -> None:
        if self.frames is not None:
            self._ticks += 1
            if self._ticks % self.every == 0:
                self.snap()

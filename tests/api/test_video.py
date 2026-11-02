"""Shared frame capture."""

import numpy as np

from able.video import FrameRecorder, caption, render


class FakeSim:
    def render(self):
        return np.full((40, 60, 4), 200, np.uint8)  # RGBA, as some envs return


def test_render_is_rgb_uint8_and_captions_only_when_asked():
    plain, captioned = render(FakeSim()), render(FakeSim(), "hi")
    assert plain.shape == captioned.shape == (40, 60, 3) and plain.dtype == np.uint8
    assert (plain == 200).all() and not (captioned == 200).all()
    assert caption(plain, "") is plain


def test_recorder_strides_and_turns_off_without_a_list():
    frames = []
    rec = FrameRecorder(FakeSim(), frames, every=3)
    for _ in range(9):
        rec.tick()
    rec.snap()
    assert len(frames) == 4
    off = FrameRecorder(FakeSim(), None, every=1)
    off.tick(), off.snap()
    assert not off.on

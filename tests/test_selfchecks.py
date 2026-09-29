"""Fast checks (no media, no Resolve): every module's own self-check.

  .venv/bin/pip install pytest && .venv/bin/python -m pytest tests/test_selfchecks.py
"""
import runpy

import pytest


@pytest.mark.parametrize("mod", ["stylekit.words", "stylekit.cut", "stylekit.grade", "stylekit.faces", "stylekit.recipe"])
def test_module_selfcheck(mod):
    runpy.run_module(mod, run_name="__main__")


def test_caption_rows_avoid_face():
    from stylekit import overlays
    props = {"captions": {"style": {"y": 0.45, "size": 0.05}, "rows": [{"start": 0.0, "end": 1.0, "words": []}]}, "texts": []}
    assert overlays.avoid_faces(props, [(0.0, 5.0, 0.3, 0.55)]) == 1
    assert props["captions"]["rows"][0]["y"] > 0.55


def test_find_phrase():
    from stylekit import overlays
    ws = [{"word": w, "start": i * 0.5, "end": i * 0.5 + 0.4} for i, w in enumerate("I have Three rules, okay".split())]
    assert overlays.find_phrase(ws, "three rules") == 1.0
    assert overlays.find_phrase(ws, "four rules") is None

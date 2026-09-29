"""Where the speaker is, so a reframe (16:9 footage into a 9:16 edit) and a punch-in keep the face in frame."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import media, paths


def face_centres(file: Path, times: list[float]) -> list[tuple[float, float, float] | None]:
    """Largest face (centre x, centre y, height; 0-1 fractions of the frame) at each time, via Apple Vision."""
    tool = paths.TOOLS / "ocr"
    if not tool.exists():
        return [None] * len(times)
    with tempfile.TemporaryDirectory() as d:
        files = []
        for i, t in enumerate(times):
            fr = media.grab_frame(file, t, 640)
            p = Path(d) / f"{i:04d}.png"
            if fr is not None:
                Image.fromarray(fr).save(p)
                files.append(p)
        if not files:
            return [None] * len(times)
        r = subprocess.run([str(tool), "faces", *map(str, files)], capture_output=True, text=True)
        found = {}
        for line in r.stdout.splitlines():
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            fs = [f for f in j.get("faces", []) if f.get("conf", 0) > 0.5]
            if fs:
                x, y, w, h = max(fs, key=lambda f: f["box"][2] * f["box"][3])["box"]
                found[int(Path(j["file"]).stem)] = (x + w / 2, y + h / 2, h)
    return [found.get(i) for i in range(len(times))]


def face_band(centre: tuple, face_h: float | None, win: dict, out_h: int) -> tuple[float, float] | None:
    """The face's top and bottom as fractions of the OUTPUT frame, after scaling and cropping."""
    if not face_h:
        return None
    cy = centre[1] * win["h"] - win["top"]
    half = face_h * win["h"] * 0.62          # the box covers eyes-to-chin; hair and neck take a little more
    return ((cy - half) / out_h, (cy + half) / out_h)


def window(src_w: int, src_h: int, out_w: int, out_h: int, centre: tuple[float, float], zoom: float = 1.0) -> dict:
    """Scale-to-fill (times `zoom`) and the crop window that keeps `centre` in frame.
    Returns scale, the scaled size, the crop's top-left in scaled pixels, and the offset of the window's centre from
    the scaled image's centre (what Resolve's Pan/Tilt move)."""
    s = max(out_w / src_w, out_h / src_h) * zoom
    sw, sh = src_w * s, src_h * s
    cx, cy = centre
    left = min(max(0.0, cx * sw - out_w / 2), sw - out_w)
    # a face sits a little above the middle of a well-framed shot: aim its centre at 42 % of the height
    top = min(max(0.0, cy * sh - out_h * 0.42), sh - out_h) if sh > out_h + 1 else (sh - out_h) / 2
    return {"scale": s, "w": sw, "h": sh, "left": left, "top": top,
            "pan": (sw / 2) - (left + out_w / 2), "tilt": (top + out_h / 2) - (sh / 2)}


if __name__ == "__main__":
    # ponytail: the one runnable check for the framing arithmetic.
    w = window(1920, 1080, 1080, 1920, (0.5, 0.5))
    assert abs(w["scale"] - 1920 / 1080) < 1e-9 and abs(w["left"] - (w["w"] - 1080) / 2) < 1e-6 and abs(w["pan"]) < 1e-6
    w = window(1920, 1080, 1080, 1920, (0.9, 0.5))
    assert abs(w["left"] - (w["w"] - 1080)) < 1e-6          # clamped at the right edge, never past it
    w = window(1080, 1920, 1080, 1920, (0.5, 0.3), zoom=1.2)
    assert w["w"] > 1080 and 0 <= w["top"] <= w["h"] - 1920
    print("faces.py self-check ok")

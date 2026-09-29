"""stylekit calibrate <style>: the style's captions (and hook/title) drawn by the real renderer on top of a reference
frame that already shows the reference's own text. Left: the reference. Right: the same frame with ours on top.
When the recipe is right, ours sits exactly on theirs (same size, place, weight, colour); any doubling is the error.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import media, overlays, paths, recipe as R


def _still(props: dict, frame: int, out: Path) -> Path:
    pj = out.with_suffix(".props.json")
    pj.write_text(json.dumps(props))
    r = subprocess.run([shutil.which("npx") or "npx", "remotion", "still", "src/index.ts", "Overlay", str(out.resolve()),
                        f"--props={pj.resolve()}", f"--frame={frame}", "--image-format=png", "--log=error"],
                       cwd=paths.REMOTION, capture_output=True, text=True)
    if r.returncode or not out.exists():
        raise SystemExit(f"Remotion still failed:\n{(r.stderr or r.stdout)[-1200:]}")
    return out


def _pick(sdir: Path, kind: str):
    """(video, deep element, deep) for the longest-lived caption (kind='caption') or early title (kind='hook')."""
    refs = json.loads((sdir / "refs.json").read_text())
    best = None
    for row in refs:
        dp = sdir / "scan" / row["id"] / "deep.json"
        if not dp.exists() or not Path(row["local"]).exists():
            continue
        d = json.loads(dp.read_text())
        caps, other = R.split_text_elements(d)
        pool = caps if kind == "caption" else [e for e in other if e["life"]["t_on"] < 1.0]
        for e in pool:
            life = e["life"]["t_off"] - e["life"]["t_on"]
            if best is None or life > best[0]:
                best = (life, Path(row["local"]), e, d)
    return best[1:] if best else None


def run(style: str) -> Path:
    rec = R.load(style)
    sdir = paths.style_dir(style)
    W, H = R.ASPECTS[rec["canvas"]["aspect"]]
    fps = 30
    panels = []
    for kind in ("caption", "hook"):
        got = _pick(sdir, kind)
        if not got:
            continue
        video, el, deep = got
        t = el["life"]["t_on"] + 0.6 * (el["life"]["t_off"] - el["life"]["t_on"])
        fr = media.grab_frame(video, t)
        if fr is None:
            continue
        bg = Image.fromarray(fr).resize((W, H))
        text = (el.get("text_events") or [{"text": el["text"]}])[-1]["text"]
        block = rec["captions"] if kind == "caption" else rec["hook"]
        box = el["box_rest"]
        style = {**overlays.text_style(block), "enter": {"kind": "cut"}, "exit": {"kind": "cut"}, "align": "left",
                 "max_width": 0.98 - box["x"] / 100}
        # drawn exactly where THEIR text is, so only font / size / weight / colour / outline can differ
        props = {"width": W, "height": H, "fps": fps, "durationInFrames": 60, "captions": None,
                 "texts": [{"text": text, "start": 0.0, "end": 2.0, "y": (box["y"] + box["h"] / 2) / 100,
                            "x": max(0.0, box["x"] / 100), "style": style}]}
        png = _still(props, 45, sdir / "scan" / f"calibrate_{kind}.png")
        ov = Image.open(png).convert("RGBA")
        comp = bg.convert("RGBA")
        comp.alpha_composite(ov)
        pair = Image.new("RGB", (W * 2 + 20, H), (20, 20, 20))
        pair.paste(bg, (0, 0))
        pair.paste(comp.convert("RGB"), (W + 20, 0))
        d = ImageDraw.Draw(pair)
        from .sheets import _font
        f = _font(max(18, W // 28))
        d.text((10, 10), f"THEIRS ({kind})", fill=(255, 255, 0), font=f, stroke_width=2, stroke_fill=(0, 0, 0))
        d.text((W + 30, 10), f"THEIRS + OURS ON TOP ({kind}); recipe y={block.get('y')}", fill=(255, 255, 0), font=f,
               stroke_width=2, stroke_fill=(0, 0, 0))
        panels.append(pair.resize((pair.width // 2, pair.height // 2)))
    if not panels:
        raise SystemExit("no caption or hook text was found in the references to calibrate against")
    out = sdir / "scan" / "calibrate.jpg"
    sheet = Image.new("RGB", (max(p.width for p in panels), sum(p.height for p in panels)), (0, 0, 0))
    y = 0
    for p in panels:
        sheet.paste(p, (0, y))
        y += p.height
    sheet.save(out, quality=90)
    print(f"calibration sheet: {out}")
    return out

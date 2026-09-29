"""Contact sheets: the frames Claude LOOKS at to name what the pixels measured (fonts, colours, layout, motion)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import media


def _font(size: int = 18):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                     # Pillow < 10.1
        return ImageFont.load_default()


def _tile(img: np.ndarray | None, w: int, label: str, box: dict | None = None) -> Image.Image:
    if img is None:
        im = Image.new("RGB", (w, int(w * 16 / 9)), (40, 40, 40))
    else:
        im = Image.fromarray(img)
        im = im.resize((w, max(1, int(im.height * w / im.width))))
    d = ImageDraw.Draw(im)
    if box:
        W, H = im.size
        d.rectangle([box["x"] / 100 * W, box["y"] / 100 * H, (box["x"] + box["w"]) / 100 * W, (box["y"] + box["h"]) / 100 * H],
                    outline=(255, 0, 160), width=2)
    f = _font(max(12, w // 16))
    tw = d.textlength(label, font=f)
    d.rectangle([0, 0, tw + 10, f.size + 8], fill=(0, 0, 0))
    d.text((5, 3), label, fill=(255, 255, 255), font=f)
    return im


def grid(tiles: list[Image.Image], cols: int, out: Path, title: str = "") -> Path:
    if not tiles:
        tiles = [Image.new("RGB", (320, 60), (30, 30, 30))]
    w = max(t.width for t in tiles)
    h = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    top = 36 if title else 0
    sheet = Image.new("RGB", (cols * w + (cols + 1) * 6, rows * h + (rows + 1) * 6 + top), (18, 18, 18))
    if title:
        ImageDraw.Draw(sheet).text((8, 8), title, fill=(230, 230, 230), font=_font(20))
    for i, t in enumerate(tiles):
        sheet.paste(t, (6 + (i % cols) * (w + 6), top + 6 + (i // cols) * (h + 6)))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return out


def crop(img: np.ndarray, box: dict, pad: float = 0.35) -> np.ndarray:
    H, W = img.shape[:2]
    x0, y0 = box["x"] / 100 * W, box["y"] / 100 * H
    bw, bh = box["w"] / 100 * W, box["h"] / 100 * H
    vy = max(bh * 0.9, 0.04 * H)                  # enough rows above/below to see the whole line and its outline
    a, b = int(max(0, x0 - bw * pad)), int(max(0, y0 - vy))
    c, d = int(min(W, x0 + bw * (1 + pad))), int(min(H, y0 + bh + vy))
    return img[b:max(b + 2, d), a:max(a + 2, c)]


def make_all(video: Path, deep: dict, captions: list[dict], outdir: Path) -> dict[str, Path]:
    """overview (one frame per shot), hook (first 3 s), text (every caption/title crop), motion (entries frame by frame)."""
    outdir.mkdir(parents=True, exist_ok=True)
    dur = deep["metadata"]["total_duration_seconds"]
    portrait = deep["metadata"]["height"] > deep["metadata"]["width"]
    tw = 220 if portrait else 360
    out = {}

    shots = deep.get("shots") or []
    step = max(1, len(shots) // 40)
    tiles = [_tile(media.grab_frame(video, min(dur - 0.05, s["start_time"] + 0.12), 480), tw,
                   f"{s['id']} {s['start_time']:.1f}s") for s in shots[::step]]
    out["overview"] = grid(tiles, 6 if portrait else 4, outdir / "overview.jpg",
                           f"one frame per shot ({len(shots)} shots, {dur:.0f}s)")

    tiles = [_tile(media.grab_frame(video, t, 480), tw, f"{t:.2f}s") for t in np.arange(0, min(3.0, dur), 0.25)]
    out["hook"] = grid(tiles, 6 if portrait else 4, outdir / "hook.jpg", "the first 3 seconds, every 0.25 s")

    cap_ids = {e["id"] for e in captions}
    texts = [e for e in deep.get("elements", []) if e.get("text")]
    texts = sorted(texts, key=lambda e: (e["id"] not in cap_ids, e["life"]["t_on"]))[:30]
    tiles = []
    for e in texts:
        t = (e["life"]["t_on"] + e["life"]["t_off"]) / 2
        fr = media.grab_frame(video, t, 1080)
        if fr is None:
            continue
        c = crop(fr, e["box_rest"])
        kind = "CAPTION" if e["id"] in cap_ids else e.get("kind", "text").upper()
        tiles.append(_tile(c, 420, f"{e['id']} {kind} {t:.1f}s in:{e['motion']['in'].get('kind')}"))
    if tiles:
        h = max(t.height for t in tiles)
        tiles = [t if t.height == h else Image.fromarray(np.pad(np.asarray(t), ((0, h - t.height), (0, 0), (0, 0)))) for t in tiles]
        out["text"] = grid(tiles, 3, outdir / "text.jpg", "every on-screen text element (crop at mid-life); CAPTION = matches the spoken words")

    moving = [e for e in deep.get("elements", []) if e["motion"]["in"].get("kind") not in (None, "cut", "unmeasured_text")]
    moving = sorted(moving, key=lambda e: -(e["box_rest"]["w"] * e["box_rest"]["h"]))[:6]
    tiles = []
    for e in sorted(moving, key=lambda e: e["life"]["t_on"]):
        d_in = max(0.2, float(e["motion"]["in"].get("duration_s") or 0.2))
        for k, t in enumerate(np.linspace(e["life"]["t_on"] - 0.04, e["life"]["t_on"] + d_in + 0.1, 6)):
            tiles.append(_tile(media.grab_frame(video, max(0.0, float(t)), 480), 180 if portrait else 300,
                               f"{e['id']} {t:.2f}s" if k else f"{e['id']} {e['motion']['in'].get('kind')} {d_in:.2f}s",
                               e["box_rest"]))
    if tiles:
        out["motion"] = grid(tiles, 6, outdir / "motion.jpg", "entry animations, one row per element, left to right in time")
    return out

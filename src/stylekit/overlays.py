"""Text on screen (captions, hook, titles) as ONE transparent ProRes 4444 clip rendered by Remotion.

The same clip goes on the top track in Resolve and over the picture in the headless render, so text looks identical
in both, and every number comes from the style's recipe.json.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from . import paths, words as W

STYLE_KEYS = ("font", "weight", "size", "case", "color", "stroke_color", "stroke_width", "shadow", "box",
              "letter_spacing", "line_height", "max_width", "enter", "exit")


def text_style(block: dict) -> dict:
    return {k: block[k] for k in STYLE_KEYS if k in block}


def _norm(s: str) -> list[str]:
    return [t for t in re.sub(r"[^a-z0-9' ]", " ", s.lower()).split() if t]


def find_phrase(edit_words: list[dict], phrase: str, after: float = 0.0) -> float | None:
    """Edit-time second where `phrase` is spoken (first match at or after `after`), matched on normalised words."""
    want = _norm(phrase)
    toks = [W.bare(w["word"]).strip("'") for w in edit_words]
    for i in range(len(toks) - len(want) + 1):
        if edit_words[i]["start"] >= after and toks[i:i + len(want)] == want:
            return edit_words[i]["start"]
    return None


def resolve_titles(items: list[dict], edit_words: list[dict], length: float, default_hold: float) -> tuple[list[dict], list[str]]:
    """titles.json rows -> timed rows. A row names WHEN by `at_phrase` (the words it lands on) or `at` (edit seconds)."""
    out, warn = [], []
    for i, it in enumerate(items):
        if "at_phrase" in it:
            t = find_phrase(edit_words, it["at_phrase"])
            if t is None:
                warn.append(f"title {i} ({it.get('text', '')[:30]!r}): phrase {it['at_phrase']!r} is not in the cut; skipped")
                continue
            t += float(it.get("offset", 0.0))
        elif "at" in it:
            t = float(it["at"])
        else:
            warn.append(f"title {i}: needs 'at_phrase' or 'at'; skipped")
            continue
        t = max(0.0, min(t, length - 0.2))
        out.append({**it, "start": round(t, 3), "end": round(min(length, t + float(it.get("hold", default_hold))), 3)})
    return out, warn


def build_props(recipe: dict, width: int, height: int, fps: int, n_frames: int, edit_words: list[dict],
                hook: str | None, titles: list[dict], captions: bool = True) -> dict:
    length = n_frames / fps
    texts = []
    hk = recipe["hook"]
    if hook:
        texts.append({"text": hook, "start": 0.0, "end": min(length, float(hk.get("hold_s", 3.0))), "y": float(hk["y"]),
                      "style": text_style(hk)})
    tt = recipe["titles"]
    for t in titles:
        st = {**text_style(tt), **{k: v for k, v in (t.get("style") or {}).items() if k in STYLE_KEYS}}
        texts.append({"text": t["text"], "start": t["start"], "end": t["end"], "y": float(t.get("y", tt["y"])), "style": st})
    cap = recipe["captions"]
    cap_props = None
    if captions and cap.get("enabled", True):
        rows = W.caption_rows(edit_words, int(cap.get("words_per_row", 3)))
        if hook:                                   # captions never fight the hook title for the first seconds
            y_gap = abs(float(cap["y"]) - float(hk["y"]))
            if y_gap < 0.12:
                rows = [r for r in rows if r["start"] >= texts[0]["end"]]
        cap_props = {"style": {**text_style(cap), "y": float(cap["y"]), "mode": cap.get("mode", "row"),
                               "highlight": cap.get("highlight")}, "rows": rows}
    return {"width": width, "height": height, "fps": fps, "durationInFrames": n_frames, "captions": cap_props, "texts": texts,
            "hook_first": bool(hook)}


def has_text(props: dict) -> bool:
    return bool(props["texts"]) or bool(props["captions"] and props["captions"]["rows"])


def render(props: dict, out: Path) -> Path:
    """Remotion -> ProRes 4444 with alpha. Needs `npm ci` in remotion/ (install.sh does it) and internet for fonts."""
    if not (paths.REMOTION / "node_modules" / "remotion").exists():
        raise SystemExit("Remotion is not installed: run ./install.sh (or `npm ci` inside remotion/)")
    out.parent.mkdir(parents=True, exist_ok=True)
    pj = out.with_suffix(".props.json")
    pj.write_text(json.dumps(props))
    npx = shutil.which("npx") or "npx"
    cmd = [npx, "remotion", "render", "src/index.ts", "Overlay", str(out.resolve()), f"--props={pj.resolve()}",
           "--codec=prores", "--prores-profile=4444", "--pixel-format=yuva444p10le", "--image-format=png",
           "--log=error"]
    print(f"  overlay: rendering {props['durationInFrames']} frames of text with Remotion")
    r = subprocess.run(cmd, cwd=paths.REMOTION, capture_output=True, text=True)
    if r.returncode or not out.exists():
        raise SystemExit(f"Remotion render failed:\n{(r.stderr or r.stdout)[-1500:]}")
    return out


def avoid_faces(props: dict, bands: list[tuple[float, float, float, float]]) -> int:
    """A caption line never covers the face: a row whose band would overlap the face (in the piece it starts in) is
    moved just below the chin, or above the head when there is no room below. Returns how many rows moved."""
    if not bands:
        return 0
    n = 0

    def place(y: float, size: float, start: float) -> float | None:
        half = size * 0.75                               # half the line's height, plus a little air
        b = next((b for b in bands if b[0] <= start < b[1]), None)
        if not b or y + half < b[2] or y - half > b[3]:
            return None
        below, above = b[3] + half + 0.02, b[2] - half - 0.02
        return round(below if below <= 0.9 else max(0.08, above), 4)

    cap = props.get("captions")
    if cap:
        for row in cap["rows"]:
            y = place(float(cap["style"]["y"]), float(cap["style"]["size"]), row["start"])
            if y is not None:
                row["y"] = y
                n += 1
    for t in props.get("texts", [])[1 if props.get("hook_first") else 0:]:
        y = place(float(t["y"]), float(t["style"]["size"]) * max(1, len(t["text"]) // 18 + 1), t["start"])
        if y is not None:
            t["y"] = y
            n += 1
    return n

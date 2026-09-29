"""A style's numbers (recipe.json): what the editor does, measured from the references wherever it can be.

Every value the scan sets is listed in `provenance` with where it came from, so Claude (and the user) can tell a
measured number from a default. The style's writer step (Claude looking at the contact sheets) fixes what pixels
cannot tell: the font family, exact colours, anything the scan got wrong. The user's own words outrank both.
"""
from __future__ import annotations

import copy
import json
import statistics
from collections import Counter
from pathlib import Path

from . import paths

FONTS = ["Montserrat", "Inter", "Poppins", "Anton", "Bebas Neue", "Oswald", "Roboto", "Archivo Black", "League Spartan",
         "DM Sans", "Playfair Display", "Lora", "Permanent Marker", "Barlow Condensed", "Figtree", "Manrope",
         "Space Grotesk", "Instrument Serif", "Kanit", "Rubik", "Plus Jakarta Sans", "Outfit", "Bangers", "Lato",
         "Open Sans", "Raleway", "Work Sans", "Nunito", "Merriweather"]
ASPECTS = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080), "4:5": (1080, 1350)}
ENTER_KINDS = ["cut", "fade", "pop", "scale", "slide_up", "slide_down", "slide_left", "slide_right", "wipe", "typewriter"]
EASINGS = ["linear", "ease_in", "ease_out", "ease_in_out", "ease_out_expo", "spring", "smooth"]

TEXT = {"font": "Montserrat", "weight": 800, "size": 0.05, "case": "as-is", "color": "#FFFFFF", "stroke_color": "#000000",
        "stroke_width": 0.0, "shadow": 0.45, "box": None, "letter_spacing": 0.0, "line_height": 1.12, "max_width": 0.86,
        "enter": {"kind": "cut", "duration_s": 0.0, "easing": "ease_out"}, "exit": {"kind": "cut", "duration_s": 0.0}}

DEFAULT = {
    "canvas": {"aspect": "9:16", "fps": None},                     # fps None = follow the footage (59.94 -> 60)
    "cut": {"pause_cap_s": 0.45, "breath_s": 0.25, "remove_fillers": True, "j_cut_s": 0.1, "tail_hold_s": 0.3,
            "min_shot_s": 0.2},
    "punch_in": {"every": 0, "zoom": 1.1},                          # every N-th shot pushed in; 0 = off
    "captions": {**copy.deepcopy(TEXT), "enabled": True, "y": 0.7, "words_per_row": 3, "mode": "row",
                 "highlight": None, "avoid_face": True, "lines": 1},
    "hook": {**copy.deepcopy(TEXT), "enabled": False, "y": 0.22, "size": 0.055, "hold_s": 3.0,
             "box": {"color": "#FFFFFF", "opacity": 1.0, "padding": 0.22, "radius": 0.2}, "color": "#111111",
             "shadow": 0.0},
    "titles": {**copy.deepcopy(TEXT), "y": 0.3, "size": 0.06, "hold_s": 2.0},
    "grade": {"mode": "match", "strength": 0.7, "reference": None},
    "audio": {"voice_lufs": -16.0, "final_lufs": -14.0, "music": False, "music_under_voice_db": 18.0,
              "music_fade_in_s": 1.0, "music_fade_out_s": 2.0},
    "pacing": {},                                                   # measured, for the planner: shots/min, words/min, text/min
}


# ---------------------------------------------------------------------------------------------------------------------
def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load(style: str) -> dict:
    p = paths.style_dir(style) / "recipe.json"
    if not p.exists():
        have = sorted(d.name for d in paths.STYLES.glob("*") if (d / "recipe.json").exists())
        raise SystemExit(f"no style {style!r}. Styles: {', '.join(have) or 'none yet (run: stylekit scan <video> --name ...)'}")
    r = _deep_merge(DEFAULT, json.loads(p.read_text()))
    problems = validate(r)
    if problems:
        raise SystemExit(f"{p} has problems:\n  - " + "\n  - ".join(problems))
    return r


def validate(r: dict) -> list[str]:
    """Readable problems with a recipe, so a hand edit that breaks it fails with a sentence, not a stack trace."""
    bad = []
    if r["canvas"]["aspect"] not in ASPECTS:
        bad.append(f"canvas.aspect must be one of {list(ASPECTS)}")
    for part in ("captions", "hook", "titles"):
        t = r[part]
        if not (0.005 <= float(t["size"]) <= 0.4):
            bad.append(f"{part}.size is a fraction of the frame height (0.005-0.4), got {t['size']}")
        if not (0.0 <= float(t.get("y", 0.5)) <= 1.0):
            bad.append(f"{part}.y is a fraction of the frame height (0-1), got {t.get('y')}")
        for side in ("enter", "exit"):
            k = (t.get(side) or {}).get("kind", "cut")
            if k not in ENTER_KINDS + ["none"]:
                bad.append(f"{part}.{side}.kind {k!r} is not one of {ENTER_KINDS}")
        for c in ("color", "stroke_color"):
            v = t.get(c)
            if v and not (isinstance(v, str) and v.startswith("#") and len(v) in (4, 7)):
                bad.append(f"{part}.{c} must be a #RRGGBB colour, got {v!r}")
    if r["grade"]["mode"] not in ("match", "none"):
        bad.append("grade.mode must be 'match' or 'none'")
    if r["grade"]["mode"] == "match" and not r["grade"].get("reference"):
        bad.append("grade.mode is 'match' but grade.reference is empty (re-run the scan, or set mode to 'none')")
    return bad


# ---------------------------------------------------------------------------------------------------------------------
# derive: measured numbers -> recipe values
# ---------------------------------------------------------------------------------------------------------------------
def _med(xs, default=None, nd=3):
    xs = [float(x) for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), nd) if xs else default


def _pct(xs, q):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


_EASE = {"spring_overshoot": "spring", "ease_out_expo": "ease_out_expo", "ease_in": "ease_in", "ease_out": "ease_out",
         "ease_in_out": "ease_in_out", "linear": "linear"}


def anim_from(m: dict) -> dict:
    """One measured vidkit motion (in) -> the renderer's enter animation."""
    k = (m or {}).get("kind") or "cut"
    d = float(m.get("duration_s") or 0.0)
    ease = _EASE.get(m.get("easing") or "", "ease_out")
    if k == "slide":
        dr = (m.get("direction") or "").replace("from_", "")
        kind = {"bottom": "slide_up", "top": "slide_down", "left": "slide_right", "right": "slide_left"}.get(dr, "slide_up")
        dist = abs(float(m.get("dy_pct") or 0)) if kind in ("slide_up", "slide_down") else abs(float(m.get("dx_pct") or 0))
        return {"kind": kind, "duration_s": round(_clip(d, 0.05, 1.0), 3), "easing": ease, "distance": round(_clip(dist / 100, 0.01, 0.3), 3)}
    if k in ("scale", "pop"):
        return {"kind": k, "duration_s": round(_clip(d, 0.05, 1.0), 3), "easing": ease,
                "from_scale": round(_clip(float(m.get("scale0") or 0.7), 0.2, 0.98), 3)}
    if k == "fade":
        return {"kind": "fade", "duration_s": round(_clip(d, 0.05, 1.0), 3), "easing": ease}
    if k in ("grow", "wipe", "wipe_soft", "custom_mask"):
        return {"kind": "wipe", "duration_s": round(_clip(d, 0.05, 1.0), 3), "easing": ease}
    if k == "typewriter":
        return {"kind": "typewriter", "duration_s": round(_clip(d, 0.1, 2.0), 3), "easing": "linear"}
    return {"kind": "cut", "duration_s": 0.0}


def exit_from(m: dict) -> dict:
    lv = (m or {}).get("leaves_via") or "cut"
    d = round(_clip(float(m.get("duration_s") or 0.15), 0.05, 1.0), 3)
    if lv == "fade":
        return {"kind": "fade", "duration_s": d}
    if lv == "shrink":
        return {"kind": "scale", "duration_s": d, "from_scale": 0.8}
    for tail, kind in (("top", "slide_down"), ("bottom", "slide_up"), ("left", "slide_right"), ("right", "slide_left")):
        if lv.endswith(tail) and (lv.startswith("slide_") or lv.startswith("edge_")):
            return {"kind": kind, "duration_s": d, "distance": 0.05}
    return {"kind": "cut", "duration_s": 0.0}


def modal_anim(els: list[dict], side: str = "in") -> dict:
    """The animation most of these elements use: a kind only wins over a plain cut when at least 35 % animate."""
    conv = anim_from if side == "in" else exit_from
    anims = [conv(e["motion"][side]) for e in els if e.get("motion")]
    if not anims:
        return {"kind": "cut", "duration_s": 0.0}
    moving = [a for a in anims if a["kind"] != "cut"]
    if len(moving) < 0.35 * len(anims):
        return {"kind": "cut", "duration_s": 0.0}
    kind = Counter(a["kind"] for a in moving).most_common(1)[0][0]
    same = [a for a in moving if a["kind"] == kind]
    out = {"kind": kind, "duration_s": _med([a["duration_s"] for a in same], 0.2)}
    if any("easing" in a for a in same):
        out["easing"] = Counter(a.get("easing", "ease_out") for a in same).most_common(1)[0][0]
    for key in ("distance", "from_scale"):
        v = _med([a[key] for a in same if key in a])
        if v is not None:
            out[key] = v
    return out


def _norm(w: str) -> str:
    return "".join(c for c in w.lower() if c.isalnum())


def split_text_elements(deep: dict) -> tuple[list[dict], list[dict]]:
    """(captions, other text): a text element is a caption when most of its words were being SPOKEN while it showed."""
    spoken = [(w["start_time"], w["end_time"], _norm(w["word"])) for s in deep.get("transcript") or [] for w in s.get("word_times", [])]
    caps, other = [], []
    for e in deep.get("elements", []):
        if not e.get("text") or e.get("kind") not in ("caption", "title", "counter"):
            continue
        t0, t1 = e["life"]["t_on"] - 0.7, e["life"]["t_off"] + 0.7
        near = {w for a, b, w in spoken if b >= t0 and a <= t1}
        toks = [_norm(t) for ev in e.get("text_events", [{"text": e["text"]}]) for t in ev["text"].split()]
        toks = [t for t in toks if t]
        if not toks:
            continue
        hit = sum(t in near for t in toks) / len(toks)
        (caps if hit >= 0.5 else other).append(e)
    return caps, other


def _text_block(els: list[dict]) -> dict:
    """Position, size, case and motion shared by a group of text elements."""
    ys = [(e["box_rest"]["y"] + e["box_rest"]["h"] / 2) / 100 for e in els]
    hs = [e["box_rest"]["h"] / 100 for e in els]
    cases = Counter((e.get("typography") or {}).get("case", "mixed") for e in els)
    case = cases.most_common(1)[0][0]
    case = {"upper": "upper", "lower": "lower", "title": "title"}.get(case, "as-is")
    size = _med(hs, 0.05)
    size = round(size * (1.25 if case == "upper" else 0.95), 4)       # OCR box = cap height for CAPS, x-height+ascenders otherwise
    return {"y": _med(ys, 0.7), "size": round(_clip(size, 0.015, 0.2), 4), "case": case,
            "enter": modal_anim(els, "in"), "exit": modal_anim(els, "out")}


def derive(deeps: list[dict], colour: dict | None, audio: list[dict]) -> tuple[dict, dict]:
    """(recipe values, provenance) from every reference's deep measurement + colour + audio."""
    r, prov = {}, {}
    n = len(deeps)
    meta = [d["metadata"] for d in deeps]
    r["canvas"] = {"aspect": Counter(m["aspect_ratio"] for m in meta).most_common(1)[0][0], "fps": None}
    sizes = ", ".join(f"{m['width']}x{m['height']}" for m in meta)
    prov["canvas.aspect"] = f"measured: reference frame size ({sizes})"

    # pacing of the finished edit: gaps between spoken words are what the editor left in
    gaps, wpm = [], []
    for d in deeps:
        ws = [w for s in d.get("transcript") or [] for w in s.get("word_times", [])]
        gaps += [b["start_time"] - a["end_time"] for a, b in zip(ws, ws[1:]) if 0.0 <= b["start_time"] - a["end_time"] < 1.5]
        if ws:
            wpm.append(len(ws) / max(1e-3, d["metadata"]["total_duration_seconds"]) * 60)
    cut = {}
    if len(gaps) >= 8:
        cut["pause_cap_s"] = round(_clip(_pct(gaps, 0.9), 0.2, 0.9), 3)
        cut["breath_s"] = round(_clip(_med([g for g in gaps if g > 0.08], 0.2), 0.08, cut["pause_cap_s"]), 3)
        prov["cut.pause_cap_s"] = f"measured: 90th percentile of {len(gaps)} gaps between spoken words in the references"
        prov["cut.breath_s"] = "measured: median real pause (>80 ms) left between words"
    r["cut"] = cut

    # punch-ins: a cut between two shots of the same person where the head size jumps
    ratios, cuts = [], 0
    for d in deeps:
        sh = d.get("shots") or []
        for a, b in zip(sh, sh[1:]):
            cuts += 1
            ha, hb = (a.get("subject") or {}).get("head_height_pct"), (b.get("subject") or {}).get("head_height_pct")
            if (a.get("subject") or {}).get("present") and (b.get("subject") or {}).get("present") and ha and hb:
                q = max(ha, hb) / max(1e-3, min(ha, hb))
                if 1.06 <= q <= 1.8:
                    ratios.append(q)
    share = len(ratios) / cuts if cuts else 0
    r["punch_in"] = {"every": 2 if share >= 0.25 else 0, "zoom": round(_clip(_med(ratios, 1.1), 1.05, 1.4), 3)}
    prov["punch_in"] = f"measured: {len(ratios)} of {cuts} cuts were a push in/out on the same person ({share:.0%})"

    caps, other = [], []
    for d in deeps:
        c, o = split_text_elements(d)
        caps += c
        other += o
    spoken_s = sum(max(0.0, (s["end_time"] - s["start_time"])) for d in deeps for s in d.get("transcript") or [])
    cap_s = 0.0                                   # union of caption time (elements overlap while words build up)
    for d in deeps:
        c, _ = split_text_elements(d)
        end = -1.0
        for a, b in sorted((e["life"]["t_on"], e["life"]["t_off"]) for e in c):
            cap_s += max(0.0, b - max(a, end))
            end = max(end, b)
    capd = {"enabled": bool(caps) and cap_s >= 0.3 * max(spoken_s, 1e-3)}
    if caps:
        capd.update(_text_block(caps))
        wc = [max(len(ev["text"].split()) for ev in e.get("text_events", [{"text": e["text"]}])) for e in caps]
        capd["words_per_row"] = int(_clip(round(_med(wc, 3)), 1, 8))
        grow = sum(1 for e in caps if len(e.get("text_events", [])) > 1 and
                   all(b["text"].startswith(a["text"].rstrip()) for a, b in zip(e["text_events"], e["text_events"][1:])))
        capd["mode"] = "reveal" if grow >= 0.3 * len(caps) else "row"
        stacked = []                    # caption lines on screen together, one above the other = lines per page
        for e in caps:
            mid = (e["life"]["t_on"] + e["life"]["t_off"]) / 2
            h = e["box_rest"]["h"]
            stacked.append(1 + sum(1 for o in caps if o is not e and o["life"]["t_on"] <= mid <= o["life"]["t_off"]
                                   and 0.5 * h < abs(o["box_rest"]["y"] - e["box_rest"]["y"]) < 2.6 * h))
        capd["lines"] = int(_clip(round(_med(stacked, 1)), 1, 3))
        if capd["lines"] > 1:
            capd["y"] = round(_med([(e["box_rest"]["y"] + e["box_rest"]["h"] / 2) / 100 for e in caps]), 3)
        prov["captions"] = (f"measured: {len(caps)} caption elements (text that matched the spoken words), on screen "
                            f"{cap_s:.0f}s of {spoken_s:.0f}s speech; position/size/case/motion are their medians; "
                            f"words_per_row = median words on one line; lines = lines shown together; mode {capd['mode']!r}")
    else:
        prov["captions"] = "measured: no burned-in captions found in the references (enabled=false)"
    r["captions"] = capd

    early = [e for e in other if e["life"]["t_on"] < 1.0 and e["life"]["t_off"] - e["life"]["t_on"] >= 0.8]
    hook = {"enabled": bool(early)}
    if early:
        hook.update(_text_block(early))
        hook["hold_s"] = round(_clip(_med([e["life"]["t_off"] for e in early], 3.0), 1.0, 8.0), 2)
        prov["hook"] = f"measured: {len(early)} non-caption text element(s) on screen in the first second (a hook title)"
    else:
        prov["hook"] = "measured: no hook title in the first second of the references"
    r["hook"] = hook

    later = [e for e in other if e not in early]
    titles = {}
    if later:
        titles = _text_block(later)
        titles["hold_s"] = round(_clip(_med([e["life"]["t_off"] - e["life"]["t_on"] for e in later], 2.0), 0.6, 6.0), 2)
        prov["titles"] = f"measured: {len(later)} other on-screen text elements (medians)"
    r["titles"] = titles

    if colour:
        r["grade"] = {"mode": "match", "reference": colour}
        prov["grade"] = "measured: black/mid/white per channel + saturation over frames of every reference"

    au = [a for a in audio if a]
    if au:
        music = sum(a["music"] for a in au) >= max(1, len(au) / 2)
        r["audio"] = {"music": music}
        if music:
            r["audio"]["music_under_voice_db"] = round(_clip(_med([a["under_db"] for a in au if a["music"]], 18.0), 6.0, 30.0), 1)
        prov["audio"] = "measured: level in the pauses between words vs the level under speech (" + \
                        "; ".join(f"{a['speech_db']:.0f} dB speech / {a['gap_db']:.0f} dB pauses" for a in au) + ")"

    dens = [d.get("density") or {} for d in deeps]
    r["pacing"] = {
        "shots_per_min": _med([x.get("cuts_per_min") for x in dens]), "avg_shot_s": _med([x.get("avg_shot_s") for x in dens]),
        "words_per_min": _med(wpm, None, 0), "text_elements_per_min": _med([x.get("elements_per_min") for x in dens]),
        "overlay_coverage_pct": _med([x.get("overlay_coverage_pct") for x in dens]),
        "reference_seconds": [d["metadata"]["total_duration_seconds"] for d in deeps], "references": n,
        "transitions": sorted({k for x in dens for k in x.get("transition_kinds", [])}),
    }
    prov["pacing"] = "measured: vidkit deep (shots, elements, words) per reference, medians"
    return r, prov


def write_measured(sdir: Path, values: dict, prov: dict) -> tuple[Path, bool]:
    """recipe.measured.json always; recipe.json only when missing (a hand-tuned recipe is never overwritten)."""
    full = _deep_merge(DEFAULT, values)
    (sdir / "recipe.measured.json").write_text(json.dumps({**full, "provenance": prov}, indent=1))
    rp = sdir / "recipe.json"
    fresh = not rp.exists()
    if fresh:
        rp.write_text(json.dumps({**full, "provenance": prov}, indent=1))
    return rp, fresh


if __name__ == "__main__":
    # ponytail: one runnable check: motion mapping + validation of the default recipe.
    assert anim_from({"kind": "slide", "direction": "from_bottom", "dy_pct": 4, "duration_s": 0.3})["kind"] == "slide_up"
    assert exit_from({"leaves_via": "slide_to_top", "duration_s": 0.2})["kind"] == "slide_down"
    d = _deep_merge(DEFAULT, {"grade": {"mode": "none"}})
    assert validate(d) == [], validate(d)
    assert "canvas.aspect" in validate(_deep_merge(d, {"canvas": {"aspect": "3:2"}}))[0]
    print("recipe.py self-check ok")

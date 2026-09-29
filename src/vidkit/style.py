"""Style profiles: aggregate N deep breakdowns into one measured editing profile.

A style lives in $VIDKIT_HOME/styles/<name>/:
    style.json        refs (source -> analysis path) + user meta
    profile.json      the aggregate: consensus pacing, overlay/animation vocabulary, beats, grammar, sourcing
Zero API cost: everything here reads analyses already on disk.
"""
from __future__ import annotations

import json

import numpy as np
import re

import statistics
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import config

HANDLE_SECONDS = 0.08   # padding kept each side of a silence cut


def slugify(name: str) -> str:
    """Any human name -> safe file stem: 'Yellow box!' -> 'yellow-box'."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip().lower()).strip("-.")
    if not s:
        raise ValueError("name is empty")
    return s


def silence_threshold(gaps: list[float]) -> Optional[float]:
    """25th percentile of observed gaps (not the min, so one outlier can't set the rule)."""
    gaps = sorted(g for g in gaps if g is not None and g > 0)
    if not gaps:
        return None
    if len(gaps) == 1:
        return round(gaps[0], 3)
    return round(statistics.quantiles(gaps, n=4, method="inclusive")[0], 3)

SCHEMA_VERSION = 1

# Where Claude is allowed to pull motion graphics and imagery from, most-preferred first.
# Written into every new profile so it can be edited per style.
DEFAULT_SOURCING = {
    "order": ["local_library", "vidkit_template", "generate", "online"],
    "local_library": {"path": "assets/", "note": "reuse before downloading; drop your own PNG/SVG/Lottie here"},
    "vidkit_template": {"path": "styles/<style>/scan/", "note": "frames and measurements from the reference scan"},
    "generate": {"note": "charts, pies, progress bars and plain shapes are rendered by vidkit.visuals, not downloaded"},
    "online": [
        {"kind": "lottie", "where": "lottiefiles.com/free-animations", "license": "check per-file; prefer CC0/free tier",
         "query": "{concept} lottie free animation transparent"},
        {"kind": "svg_icon", "where": "heroicons.com, lucide.dev, tabler-icons.io", "license": "MIT",
         "query": "{concept} icon svg"},
        {"kind": "brand_logo", "where": "simpleicons.org", "license": "brand marks, respect trademark use",
         "query": "{brand} simple icons svg"},
        {"kind": "stock_photo", "where": "unsplash.com, pexels.com", "license": "free to use, no attribution required",
         "query": "{concept} {mood} vertical"},
        {"kind": "stock_video", "where": "pexels.com/videos, coverr.co", "license": "free to use",
         "query": "{concept} b-roll vertical 4k"},
        {"kind": "emoji", "where": "openmoji.org", "license": "CC BY-SA 4.0", "query": "{concept} openmoji png"},
    ],
    "rules": [
        "Match the reference art style in asset_inventory.art_style, not just the subject.",
        "Transparent background required for anything that sits over footage.",
        "Vertical assets or square with room to crop; never letterbox an asset to fit.",
        "Save every download to assets/<style>/ so the next edit reuses it instead of re-fetching.",
        "Record the source URL and license next to the file in sources.json.",
        "Never download a paid or watermarked asset. If the only match is paid, render a substitute instead.",
    ],
}

# ---------------- paths ----------------

def styles_dir() -> Path:
    d = config.home() / "styles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dir_for(name: str) -> Path:
    return styles_dir() / slugify(name)


def create(name: str) -> Path:
    d = dir_for(name)
    d.mkdir(parents=True, exist_ok=True)
    sp = d / "style.json"
    if not sp.exists():
        sp.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "name": slugify(name),
                                  "created_at": _now(), "refs": []}, indent=2))
    return d


def load_style(name: str) -> dict:
    p = dir_for(name) / "style.json"
    if not p.exists():
        raise FileNotFoundError(f"No style named {name!r}. Create it with: vidkit style scan ... --name {name}")
    return json.loads(p.read_text())


def save_style(name: str, data: dict) -> Path:
    p = dir_for(name) / "style.json"
    p.write_text(json.dumps(data, indent=2))
    return p


_refs_lock = threading.Lock()


def add_ref(name: str, source: str, analysis_path: Path, extra: dict | None = None) -> dict:
    with _refs_lock:
        return _add_ref(name, source, analysis_path, extra)


def _add_ref(name: str, source: str, analysis_path: Path, extra: dict | None = None) -> dict:
    st = load_style(name)
    ref = {"source": source, "analysis": str(analysis_path), "added_at": _now(), **(extra or {})}
    st["refs"] = [r for r in st["refs"] if r.get("analysis") != ref["analysis"]] + [ref]
    save_style(name, st)
    return st


def list_styles() -> list[dict]:
    out = []
    for d in sorted(styles_dir().iterdir()):
        p = d / "style.json"
        if not p.exists():
            continue
        st = json.loads(p.read_text())
        prof = d / "profile.json"
        out.append({"name": st.get("name", d.name), "refs": len(st.get("refs", [])),
                    "built": prof.exists()})
    return out


def profile_path(name: str) -> Path:
    return dir_for(name) / "profile.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------- stats helpers ----------------

def _nums(xs) -> list[float]:
    return [float(x) for x in xs if isinstance(x, (int, float))]


def _spread(xs) -> Optional[dict]:
    v = sorted(_nums(xs))
    if not v:
        return None
    if len(v) < 4:
        return {"median": round(statistics.median(v), 3), "p25": round(v[0], 3), "p75": round(v[-1], 3), "n": len(v)}
    q = statistics.quantiles(v, n=4, method="inclusive")
    return {"median": round(q[1], 3), "p25": round(q[0], 3), "p75": round(q[2], 3), "n": len(v)}


def _mode(xs) -> Optional[Any]:
    c = Counter(x for x in xs if x not in (None, ""))
    return c.most_common(1)[0][0] if c else None


def _freq(xs, top: int = 8) -> list[dict]:
    c = Counter(x for x in xs if x not in (None, ""))
    total = sum(c.values()) or 1
    return [{"value": k, "n": n, "share": round(n / total, 2)} for k, n in c.most_common(top)]


def _get(d: Any, *path, default=None):
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return d if d is not None else default


def _norm_rule(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


# ---------------- the aggregate ----------------

def build(name: str) -> dict:
    """Aggregate every ref's analysis into one profile. Pure disk work, no API calls."""
    st = load_style(name)
    refs = st.get("refs", [])
    if not refs:
        raise ValueError(f"style {name!r} has no reference videos yet")
    docs, missing = [], []
    for r in refs:
        p = Path(r["analysis"])
        if not p.exists():
            missing.append(str(p))
            continue
        docs.append((r, json.loads(p.read_text())))
    if not docs:
        raise ValueError(f"none of the {len(refs)} analysis files for {name!r} exist any more")

    prof: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "name": st["name"], "built_at": _now(),
        "refs": [{"source": r.get("source"), "analysis": r["analysis"],
                  "duration": _get(d, "metadata", "total_duration_seconds"),
                  "cuts": len(d.get("scene_cuts") or []),
                  "overlays": len(d.get("graphic_overlays") or []),
                  "deep": bool(d.get("beat_map") or d.get("transcript")),
                  "complete": (d.get("_completeness") or {}).get("complete"),
                  "gaps": (d.get("_completeness") or {}).get("problems") or []} for r, d in docs],
        "missing_analyses": missing,
        "consensus": _consensus(docs),
        "overlay_vocabulary": _overlay_vocab(docs),
        "animation_vocabulary": _animation_vocab(docs),
        "motion_graphics": _motion_graphics(docs),
        "asset_patterns": _asset_patterns(docs),
        "beat_rules": _beat_rules(docs),
        "shot_grammar": _shot_grammar(docs),
        "broll_grammar": _broll_grammar(docs),
        "structure_template": _structure_template(docs),
        "grammar": _grammar(docs),
        "hook": _hook(docs),
        "sourcing": DEFAULT_SOURCING,
    }
    prof["variants"] = _variants(docs, prof)
    prof["clusters"] = _clusters(prof["variants"])
    prof["divergences"] = _divergences(prof, len(docs))
    prof["signature"] = _signature(prof)
    p = profile_path(name)
    p.write_text(json.dumps(prof, indent=2))
    return prof


def _consensus(docs) -> dict:
    shots, cpm, wpm, curves, ars, fps, gaps, durs = [], [], [], [], [], [], [], []
    for _, d in docs:
        pm = d.get("pacing_metrics") or {}
        shots += _nums(pm.get("shot_lengths") or [])
        if pm.get("average_shot_length_seconds") and not pm.get("shot_lengths"):
            shots.append(pm["average_shot_length_seconds"])
        cpm.append(pm.get("cuts_per_minute"))
        curves.append(pm.get("curve_shape"))
        md = d.get("metadata") or {}
        ars.append(md.get("aspect_ratio"))
        fps.append(md.get("fps"))
        durs.append(md.get("total_duration_seconds"))
        wpm.append(_get(d, "speech_analysis", "words_per_minute"))
        gaps += [s.get("gap_seconds") for s in (d.get("silence_cuts") or []) if s.get("kept_or_cut") == "cut"]
    return {
        "aspect_ratio": _mode(ars), "fps": _mode(fps),
        "duration_seconds": _spread(durs),
        "shot_length_seconds": _spread(shots),
        "cuts_per_minute": _spread(cpm),
        "words_per_minute": _spread(wpm),
        "pacing_curve": _mode(curves),
        "cut_silence_gaps_over_seconds": silence_threshold([g for g in gaps if g]),
        "handle_seconds": HANDLE_SECONDS,
    }


def _overlay_vocab(docs) -> list[dict]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    seen_in: dict[str, set] = defaultdict(set)
    for i, (_, d) in enumerate(docs):
        for o in d.get("graphic_overlays") or []:
            t = o.get("type") or "other"
            by_type[t].append(o)
            seen_in[t].add(i)
    n_docs = len(docs)
    out = []
    for t, items in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        box = {k: _spread([_get(o, "spatial_box", k) for o in items]) for k in ("x", "y", "w", "h")}
        typo = [o.get("typography") or {} for o in items]
        out.append({
            "type": t, "count": len(items), "in_videos": len(seen_in[t]),
            "share_of_videos": round(len(seen_in[t]) / n_docs, 2),
            "box": {k: (v or {}).get("median") for k, v in box.items()},
            "box_spread": box,
            "on_screen_seconds": _spread([(o.get("end_time") or 0) - (o.get("start_time") or 0) for o in items]),
            "animation_in": _freq([o.get("animation_in") or _get(o, "motion", "in", "kind") for o in items]),
            "animation_out": _freq([o.get("animation_out") or _get(o, "motion", "out", "kind") for o in items]),
            "motion_in_seconds": _spread([_get(o, "motion", "in", "duration_seconds") for o in items]),
            "motion_out_seconds": _spread([_get(o, "motion", "out", "duration_seconds") for o in items]),
            "easing_in": _freq([_get(o, "motion", "in", "easing") for o in items]),
            "hold": _freq([_get(o, "motion", "hold", "kind") for o in items]),
            "typography": {
                "weight": _mode([x.get("weight") for x in typo]),
                "case": _mode([x.get("case") for x in typo]),
                "size_pct_of_height": _spread([x.get("size_pct_of_height") for x in typo]),
                "color": _mode([x.get("color") for x in typo]),
                "stroke_or_shadow": _mode([x.get("stroke_or_shadow") for x in typo]),
                "highlight_style": _mode([x.get("highlight_style") for x in typo]),
                "font_family_guess": _mode([x.get("font_family_guess") for x in typo]),
            },
            "colors": _freq([c for o in items for c in (o.get("colors") or [])]),
            "background_color": _mode([o.get("background_color") for o in items]),
            "shape": _mode([o.get("shape") for o in items]),
            "unsafe_zone_hits": sum(1 for o in items if o.get("in_unsafe_zone")),
            "purposes": [f["value"] for f in _freq([o.get("purpose") for o in items], top=3)],
        })
    return out


def _resample(curve: list[float], n: int) -> list[float]:
    xs = np.linspace(0, 1, len(curve)); xt = np.linspace(0, 1, n)
    return [float(v) for v in np.interp(xt, xs, np.array(curve, dtype=float))]


def _animation_vocab(docs) -> list[dict]:
    rows: dict[tuple, dict] = {}
    for _, d in docs:
        for o in d.get("graphic_overlays") or []:
            for phase in ("in", "out"):
                m = _get(o, "motion", phase) or {}
                kind = m.get("kind") or o.get(f"animation_{phase}")
                if not kind or kind == "none":
                    continue
                key = (phase, kind)
                r = rows.setdefault(key, {"phase": phase, "animation": kind, "n": 0, "used_on": [],
                                          "durations": [], "easings": [], "offsets": [], "staggers": [],
                                          "overshoots": []})
                r["n"] += 1
                r["used_on"].append(o.get("type"))
                if m.get("curve") and len(m["curve"]) >= 3:      # measured progress curve (vidkit deep)
                    r.setdefault("curves", []).append(_resample(m["curve"], 16))
                    if kind == "scale" and m.get("from_scale") is not None:
                        r.setdefault("scale_from", []).append(m["from_scale"])
                r["durations"].append(m.get("duration_seconds"))
                r["easings"].append(m.get("easing"))
                r.setdefault("staggers", []).append(m.get("stagger_seconds"))
                r.setdefault("overshoots", []).append(m.get("overshoot_pct"))
                dx, dy = (m.get("from_dx"), m.get("from_dy")) if phase == "in" else (m.get("to_dx"), m.get("to_dy"))
                if dx is not None or dy is not None:
                    r["offsets"].append((dx or 0.0, dy or 0.0))
    total = sum(r["n"] for r in rows.values()) or 1
    out = []
    for r in sorted(rows.values(), key=lambda r: -r["n"]):
        offs = r.pop("offsets")
        stag, over = _spread(r.pop("staggers")), _spread(r.pop("overshoots"))
        curves = r.pop("curves", None)
        scale_from = r.pop("scale_from", None)
        out.append({**{k: v for k, v in r.items() if k not in ("durations", "easings", "used_on")},
                    "stagger_seconds": stag, "overshoot_pct": over,
                    "curve_16": [round(float(v), 3) for v in np.median(np.array(curves), axis=0)] if curves else None,
                    "measured_n": len(curves) if curves else 0,
                    "scale_from": _spread(scale_from) if scale_from else None,
                    "share": round(r["n"] / total, 2),
                    "used_on": [f["value"] for f in _freq(r["used_on"], top=4)],
                    "duration_seconds": _spread(r["durations"]),
                    "easing": _mode(r["easings"]),
                    "travel": {"dx": _spread([o[0] for o in offs]), "dy": _spread([o[1] for o in offs])} if offs else None})
    return out


def _motion_graphics(docs) -> list[dict]:
    by_dev: dict[str, dict] = {}
    for _, d in docs:
        for mg in d.get("motion_graphics") or []:
            dev = mg.get("device") or mg.get("name") or "other"
            r = by_dev.setdefault(dev, {"device": dev, "names": [], "n": 0, "how_to_rebuild": [], "params": []})
            r["n"] += len(mg.get("occurrences") or [1])
            r["names"].append(mg.get("name"))
            if mg.get("how_to_rebuild"):
                r["how_to_rebuild"].append(mg["how_to_rebuild"])
            if mg.get("params"):
                r["params"].append(mg["params"])
    for r in by_dev.values():
        r["names"] = [f["value"] for f in _freq(r["names"], top=3)]
        r["how_to_rebuild"] = r["how_to_rebuild"][:3]
        r["params"] = r["params"][:3]
    return sorted(by_dev.values(), key=lambda r: -r["n"])


def _asset_patterns(docs) -> dict:
    assets = [a for _, d in docs for a in (d.get("asset_inventory") or [])]
    return {
        "count": len(assets),
        "by_kind": _freq([a.get("sourcing_kind") for a in assets]),
        "art_styles": [f["value"] for f in _freq([a.get("art_style") for a in assets], top=5)],
        "transparency_required_share": round(sum(1 for a in assets if a.get("has_transparency")) / len(assets), 2) if assets else None,
        "animated_share": round(sum(1 for a in assets if a.get("is_animated")) / len(assets), 2) if assets else None,
        "palette": _freq([c for a in assets for c in (a.get("palette") or [])], top=8),
        "example_queries": [a["sourcing_hint"] for a in assets if a.get("sourcing_hint")][:12],
    }


def _beat_rules(docs) -> list[dict]:
    by_kind: dict[str, dict] = {}
    for _, d in docs:
        for b in d.get("beat_map") or []:
            k = b.get("trigger_kind") or "other"
            r = by_kind.setdefault(k, {"trigger_kind": k, "n": 0, "responses": [], "offsets": [], "durations": [], "phrases": []})
            r["n"] += 1
            r["responses"].append(b.get("response"))
            r["offsets"].append(b.get("offset_from_trigger_seconds"))
            r["durations"].append(b.get("response_duration_seconds"))
            r["phrases"].append(b.get("trigger_phrase"))
    out = []
    for r in sorted(by_kind.values(), key=lambda r: -r["n"]):
        out.append({"trigger_kind": r["trigger_kind"], "n": r["n"],
                    "typical_responses": r["responses"][:4],
                    "example_phrases": r["phrases"][:5],
                    "offset_from_trigger_seconds": _spread(r["offsets"]),
                    "response_duration_seconds": _spread(r["durations"])})
    return out


def _shot_grammar(docs) -> dict:
    """How shots themselves are chosen and joined, not just what sits on top of them."""
    cuts = [c for _, d in docs for c in (d.get("scene_cuts") or []) if isinstance(c, dict)]
    if not cuts:
        return {}
    broll = [c for c in cuts if c.get("is_broll")]
    return {
        "count": len(cuts),
        "shot_type": _freq([c.get("shot_type") for c in cuts]),
        "transition_in": _freq([c.get("transition_in") for c in cuts]),
        "camera_motion": _freq([c.get("camera_motion") for c in cuts]),
        "framing": _freq([c.get("framing") for c in cuts]),
        "broll_share": round(len(broll) / len(cuts), 2),
        "broll_seconds": _spread([(c.get("end_time") or 0) - (c.get("start_time") or 0) for c in broll]),
        "talking_head_seconds": _spread([(c.get("end_time") or 0) - (c.get("start_time") or 0)
                                         for c in cuts if c.get("shot_type") == "talking_head"]),
        "speed_ramps": _freq([c.get("speed_ramp") for c in cuts if c.get("speed_ramp") not in (None, "1x", "null")]),
        "why_cut_here": [f["value"] for f in _freq([c.get("why_here") for c in cuts], top=6)],
    }


def _broll_grammar(docs) -> dict:
    """Exactly how b-roll looks, arrives, moves, and leaves — the numbers a keyframing editor needs.

    Grouped by presentation mode, because a full-frame screen recording and a floating card obey
    different rules. Everything is a spread (median/p25/p75), never a single averaged number.
    """
    shots = [b for _, d in docs for b in (d.get("broll_shots") or []) if isinstance(b, dict)]
    n_docs = len(docs)
    docs_with = sum(1 for _, d in docs if d.get("broll_shots"))
    if not shots:
        return {"count": 0, "in_videos": 0, "note": "no broll pass in these analyses; run: vidkit style refresh <name> --passes broll"}
    per_video = [len(d.get("broll_shots") or []) for _, d in docs]
    dur = [_get(d, "metadata", "total_duration_seconds") or 0 for _, d in docs]
    on_screen = [sum(float(b.get("duration_seconds") or 0) for b in (d.get("broll_shots") or [])) for _, d in docs]
    share = [o / t for o, t in zip(on_screen, dur) if t]

    by_mode: dict[str, list] = defaultdict(list)
    for b in shots:
        by_mode[b.get("presentation") or "unknown"].append(b)

    def motion(items, phase):
        ms = [b.get(phase) or {} for b in items]
        pre = "from_" if phase == "enter" else "to_"
        return {
            "kind": _freq([m.get("kind") for m in ms]),
            "duration_seconds": _spread([m.get("duration_seconds") for m in ms]),
            "easing": _mode([m.get("easing") for m in ms]),
            "travel": {"dx": _spread([m.get(pre + "dx") for m in ms]), "dy": _spread([m.get(pre + "dy") for m in ms])},
            "scale": _spread([m.get(pre + "scale") for m in ms]),
            "opacity": _spread([m.get(pre + "opacity") for m in ms]),
            "overshoot_pct": _spread([m.get("overshoot_pct") for m in ms]),
        }

    modes = []
    for mode, items in sorted(by_mode.items(), key=lambda kv: -len(kv[1])):
        ft = [b.get("frame_treatment") or {} for b in items]
        hold = [b.get("hold") or {} for b in items]
        content = [b.get("content") or {} for b in items]
        trig = [b.get("trigger") or {} for b in items]
        modes.append({
            "presentation": mode, "count": len(items), "share": round(len(items) / len(shots), 2),
            "box": {k: _spread([_get(b, "box", k) for b in items]) for k in ("x", "y", "w", "h")},
            "speaker_visible_share": round(sum(1 for b in items if b.get("speaker_visible")) / len(items), 2),
            "duration_seconds": _spread([b.get("duration_seconds") for b in items]),
            "frame": {
                "corner_radius_pct": _spread([f.get("corner_radius_pct") for f in ft]),
                "border_px": _spread([_get(f, "border", "px") for f in ft]),
                "border_color": _mode([_get(f, "border", "color") for f in ft]),
                "shadow_share": round(sum(1 for f in ft if _get(f, "shadow", "enabled")) / len(ft), 2) if ft else None,
                "backdrop": _freq([f.get("backdrop") for f in ft]),
            },
            "enter": motion(items, "enter"),
            "hold": {"kind": _freq([h.get("kind") for h in hold]),
                     "rate_pct_per_second": _spread([h.get("rate_pct_per_second") for h in hold]),
                     "start_scale": _spread([h.get("start_scale") for h in hold]),
                     "end_scale": _spread([h.get("end_scale") for h in hold])},
            "exit": motion(items, "exit"),
            "content_kinds": _freq([c.get("kind") for c in content]),
            "animated_share": round(sum(1 for c in content if c.get("is_animated")) / len(content), 2) if content else None,
            "sites_shown": [f["value"] for f in _freq([c.get("site_or_app") for c in content], top=8)],
            "cursor_or_scroll": [c["cursor_or_scroll"] for c in content if c.get("cursor_or_scroll")][:4],
            "offset_from_trigger_seconds": _spread([t.get("offset_from_trigger_seconds") for t in trig]),
            "sync": _freq([b.get("sync") for b in items]),
            "examples": [c.get("describes") for c in content if c.get("describes")][:4],
        })
    derived = sum(1 for b in shots if b.get("derived"))
    return {
        "count": len(shots), "in_videos": docs_with, "of_videos": n_docs,
        "measured_from_pixels": derived, "from_model_pass": len(shots) - derived,
        "per_video": _spread(per_video),
        "screen_time_share": _spread(share),
        "gap_between_shots_seconds": _spread([
            float(b2.get("start_time") or 0) - float(b1.get("end_time") or 0)
            for _, d in docs
            for b1, b2 in zip(sorted((d.get("broll_shots") or []), key=lambda b: b.get("start_time") or 0),
                              sorted((d.get("broll_shots") or []), key=lambda b: b.get("start_time") or 0)[1:])]),
        "modes": modes,
    }


def _structure_template(docs) -> list[dict]:
    """The narrative shape, as fractions of the runtime, so a NEW script can be mapped onto it."""
    rows: dict[str, dict] = {}
    for _, d in docs:
        dur = _get(d, "metadata", "total_duration_seconds") or 0
        for b in d.get("structural_beats") or []:
            if not isinstance(b, dict) or not dur:
                continue
            name = b.get("name") or "other"
            r = rows.setdefault(name, {"section": name, "n": 0, "starts_pct": [], "durations": [],
                                       "duration_pct": [], "treatments": [], "what": []})
            st, en = float(b.get("start_time") or 0), float(b.get("end_time") or 0)
            r["n"] += 1
            r["starts_pct"].append(st / dur * 100)
            r["durations"].append(en - st)
            r["duration_pct"].append((en - st) / dur * 100)
            if b.get("visual_treatment"):
                r["treatments"].append(b["visual_treatment"])
            if b.get("what_happens"):
                r["what"].append(b["what_happens"])
    out = []
    for r in rows.values():
        out.append({"section": r["section"], "in_videos": r["n"],
                    "starts_at_pct": _spread(r["starts_pct"]), "seconds": _spread(r["durations"]),
                    "share_of_runtime_pct": _spread(r["duration_pct"]),
                    "visual_treatment": r["treatments"][:3], "what_happens": r["what"][:3]})
    return sorted(out, key=lambda r: (r["starts_at_pct"] or {}).get("median", 0))


def _grammar(docs) -> list[dict]:
    seen: dict[str, dict] = {}
    for i, (_, d) in enumerate(docs):
        for rule in d.get("edit_grammar") or []:
            if not isinstance(rule, str) or not rule.strip():
                continue
            # ponytail: exact-normalized text match only. Two videos phrasing the same rule differently stay
            # separate entries; swap in embeddings if the duplicate noise ever matters.
            key = _norm_rule(rule)
            r = seen.setdefault(key, {"rule": rule.strip(), "videos": set()})
            r["videos"].add(i)
    return sorted(({"rule": r["rule"], "in_videos": len(r["videos"])} for r in seen.values()),
                  key=lambda r: (-r["in_videos"], r["rule"]))


def _hook(docs) -> dict:
    hs = [d.get("hook_analysis") or {} for _, d in docs]
    hs = [h for h in hs if h]
    return {
        "types": _freq([h.get("hook_type") for h in hs]),
        "window_seconds": _spread([(h.get("end_time") or 0) - (h.get("start_time") or 0) for h in hs]),
        "text_on_screen_share": round(sum(1 for h in hs if h.get("text_on_screen")) / len(hs), 2) if hs else None,
        "techniques": [f["value"] for f in _freq([h.get("visual_technique") for h in hs], top=4)],
        "mechanisms": [h["why_it_holds"] for h in hs if h.get("why_it_holds")][:4],
        "structures": [h["structure_description"] for h in hs if h.get("structure_description")][:4],
    }


# ---------------- per-video cards ----------------
# Averaging erases the one reel that opens on a POV shot. Each reference keeps a card, but a card only
# records what that video does DIFFERENTLY from the shared profile, so adding near-identical references
# costs almost nothing and adding a genuinely new treatment costs one short card.

def _traits(d: dict) -> dict:
    """The comparable fingerprint of one video."""
    ov = [o for o in (d.get("graphic_overlays") or []) if isinstance(o, dict)]
    cuts = [c for c in (d.get("scene_cuts") or []) if isinstance(c, dict)]
    anims = set()
    for o in ov:
        for phase in ("in", "out"):
            k = _get(o, "motion", phase, "kind") or o.get(f"animation_{phase}")
            if k and k != "none":
                anims.add(f"{phase}:{k}")
    return {
        "overlay_types": {o.get("type") for o in ov if o.get("type")},
        "animations": anims,
        "devices": {m.get("device") for m in (d.get("motion_graphics") or []) if isinstance(m, dict) and m.get("device")},
        "asset_kinds": {a.get("sourcing_kind") for a in (d.get("asset_inventory") or [])
                        if isinstance(a, dict) and a.get("sourcing_kind")},
        "shot_types": {c.get("shot_type") for c in cuts if c.get("shot_type")},
        "camera": {c.get("camera_motion") for c in cuts if c.get("camera_motion")},
        "sections": {b.get("name") for b in (d.get("structural_beats") or []) if isinstance(b, dict) and b.get("name")},
        "hook_type": _get(d, "hook_analysis", "hook_type"),
        "highlight": {_get(o, "typography", "highlight_style") for o in ov} - {None, "none"},
    }


def _opens_with(d: dict) -> str:
    """How the first ~3 seconds look, in one line. This is how a human finds 'the POV one'."""
    bits = []
    first_cut = next((c for c in sorted((d.get("scene_cuts") or []), key=lambda c: c.get("start_time") or 0)
                      if isinstance(c, dict)), None)
    if first_cut:
        bits.append(f"{first_cut.get('shot_type') or 'shot'}"
                    + (f", {first_cut.get('camera_motion')}" if first_cut.get("camera_motion") not in (None, "static") else "")
                    + (f": {first_cut.get('description')}" if first_cut.get("description") else ""))
    early = sorted([o for o in (d.get("graphic_overlays") or [])
                    if isinstance(o, dict) and (o.get("start_time") or 0) <= 3.0],
                   key=lambda o: o.get("start_time") or 0)
    for o in early[:2]:
        if o.get("text_content"):
            bits.append(f'{o.get("type")} "{str(o["text_content"])[:60]}"')
        elif o.get("asset_description"):
            bits.append(f'{o.get("type")}: {str(o["asset_description"])[:60]}')
    ht = _get(d, "hook_analysis", "hook_type")
    if ht:
        bits.append(f"hook={ht}")
    return " | ".join(bits) or "not captured"


def _variants(docs, prof: dict) -> list[dict]:
    shared_rules = {_norm_rule(g["rule"]) for g in prof.get("grammar", []) if g.get("in_videos", 0) > 1}
    traits = [_traits(d) for _, d in docs]
    n = len(traits)
    # what counts as "shared": present in more than half the references
    shared = {}
    for key in ("overlay_types", "animations", "devices", "asset_kinds", "shot_types", "camera", "sections", "highlight"):
        counts = Counter(v for t in traits for v in t[key])
        shared[key] = {v for v, c in counts.items() if c > n / 2}

    out = []
    for i, ((r, d), t) in enumerate(zip(docs, traits)):
        only = {k: sorted(t[k] - shared[k]) for k in shared}
        missing = {k: sorted(shared[k] - t[k]) for k in shared}
        dur = _get(d, "metadata", "total_duration_seconds")
        pm = d.get("pacing_metrics") or {}
        card = {
            "id": f"v{i + 1}",
            "source": r.get("source"),
            "analysis": r["analysis"],
            "duration": dur,
            "opens_with": _opens_with(d),
            "hook_type": t["hook_type"],
            "sections": [b.get("name") for b in (d.get("structural_beats") or []) if isinstance(b, dict)],
            "shot_length_seconds": _spread(pm.get("shot_lengths") or []),
            "cuts": len(d.get("scene_cuts") or []),
            "overlays": len(d.get("graphic_overlays") or []),
            "tags": sorted({v for k in ("overlay_types", "devices", "shot_types", "highlight") for v in t[k]}
                           | ({t["hook_type"]} if t["hook_type"] else set())),
            # the whole point of the card:
            "only_this_one": {k: v for k, v in only.items() if v},
            "lacks_vs_style": {k: v for k, v in missing.items() if v},
            # only rules this video contributed that the shared grammar does not already carry
            "signature_rules": [g for g in (d.get("edit_grammar") or [])
                                if isinstance(g, str) and _norm_rule(g) not in shared_rules][:4],
            "complete": (d.get("_completeness") or {}).get("complete"),
        }
        out.append(card)
    return out


def _clusters(variants: list[dict]) -> list[dict]:
    """Group references whose fingerprints match, so the skill says it once instead of N times."""
    def sig(v):
        return (v.get("hook_type"), tuple(v["tags"]))
    groups: dict = {}
    for v in variants:
        groups.setdefault(sig(v), []).append(v["id"])
    return [{"members": ids, "hook_type": k[0], "tags": list(k[1])}
            for k, ids in groups.items() if len(ids) > 1]


def _divergences(prof: dict, n_docs: int) -> list[str]:
    out = []
    for ov in prof["overlay_vocabulary"]:
        if ov["in_videos"] < n_docs and n_docs > 1:
            out.append(f"{ov['type']} appears in only {ov['in_videos']}/{n_docs} references — treat as optional")
    for g in prof["grammar"]:
        if n_docs > 1 and g["in_videos"] == 1:
            out.append(f"one-off rule: {g['rule']}")
    cs = prof["consensus"]["shot_length_seconds"] or {}
    if cs.get("p75") and cs.get("p25") and cs["p25"] and cs["p75"] / cs["p25"] > 4:
        out.append(f"shot length is inconsistent across references (p25 {cs['p25']}s vs p75 {cs['p75']}s)")
    if prof["missing_analyses"]:
        out.append(f"{len(prof['missing_analyses'])} referenced analysis files are missing from disk")
    for r in prof["refs"]:
        if r.get("complete") is False:
            out.append(f"INCOMPLETE scan, this reference under-informs the style: {r['source']} "
                       f"({'; '.join(r.get('gaps') or []) or 'unknown gaps'})")
    return out[:20]


def _signature(prof: dict) -> list[str]:
    """The short human-readable 'this is the style' list. What goes at the top of the skill."""
    c, out = prof["consensus"], []
    if c["aspect_ratio"]:
        out.append(f"{c['aspect_ratio']} at {c['fps'] or '?'}fps, typically {(c['duration_seconds'] or {}).get('median')}s long.")
    sl = c["shot_length_seconds"] or {}
    if sl.get("median"):
        out.append(f"Cut every {sl['median']}s on average (range {sl.get('p25')}-{sl.get('p75')}s), pacing curve {c['pacing_curve'] or 'even'}.")
    if c["cut_silence_gaps_over_seconds"]:
        out.append(f"Remove any silence longer than {c['cut_silence_gaps_over_seconds']}s, {c['handle_seconds']}s handles.")
    ht = prof["hook"]["types"]
    if ht:
        out.append(f"Hook is a {ht[0]['value']} in the first {(prof['hook']['window_seconds'] or {}).get('median')}s"
                   + (", with text on screen." if (prof["hook"]["text_on_screen_share"] or 0) >= 0.5 else "."))
    secs = lambda sp: (f"{sp['median']}s" if (sp or {}).get("median") is not None else "unmeasured")
    for ov in prof["overlay_vocabulary"][:4]:
        anim = ov["animation_in"][0]["value"] if ov["animation_in"] else "cut"
        out.append(f"{ov['type']}: box x{ov['box']['x']} y{ov['box']['y']} w{ov['box']['w']} h{ov['box']['h']}, "
                   f"{anim} in over {secs(ov['motion_in_seconds'])}, lives {secs(ov['on_screen_seconds'])}, "
                   f"{ov['typography']['weight'] or ''} {ov['typography']['case'] or ''} {ov['typography']['color'] or ''}".strip())
    for a in prof["animation_vocabulary"][:3]:
        detail = ", ".join(x for x in (secs(a["duration_seconds"]) if (a["duration_seconds"] or {}).get("median") else "",
                                       a["easing"] or "") if x)
        out.append(f"{a['share'] * 100:.0f}% of {a['phase']} moves are {a['animation']}"
                   f"{' (' + detail + ')' if detail else ''} on {', '.join(a['used_on'])}.")
    for m in prof["motion_graphics"][:3]:
        out.append(f"Recurring device: {m['device']} ({m['n']}x). {(m['how_to_rebuild'] or [''])[0]}")
    ap = prof["asset_patterns"]
    if ap["count"]:
        kinds = ", ".join(f"{k['value']}" for k in ap["by_kind"][:3])
        out.append(f"Assets are mostly {kinds}; art style: {(ap['art_styles'] or ['unspecified'])[0]}.")
    return out


# ---------------- the skill ----------------

def cons_get(prof: dict, key: str, default):
    v = (prof.get("consensus") or {}).get(key)
    return v if v is not None else default

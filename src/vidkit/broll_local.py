"""Derive b-roll motion numbers from PIXELS, with no API call.

The broll pass asks the model to describe how each b-roll moment sits, arrives, moves and leaves. When that pass
cannot run (daily quota, timeouts), the scene list — which IS complete — says exactly where every b-roll span is,
and the cached video says what happened there. This measures the same fields the pass would have reported:
presentation and box from a diff against the previous shot, enter/exit from the transition contest, hold motion from
block-flow similarity fits, content from the (complete) asset inventory, trigger from the (complete) transcript.

Entries are marked derived: true and confidence ~0.55. They are numbers from the frames, not a model's judgement,
so they are the more trustworthy half of the pair where both exist.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .vision import decide as D
from .vision import frames as F
from .vision import measure as M

BROLL_TYPES = {"broll", "screen", "product", "text_card", "meme", "reaction"}
ANALYSIS_WIDTH = 320          # px; enough for boxes and flow, fast enough for 300 spans
HOLD_STEP = 0.25              # seconds between hold-motion samples
MAX_HOLD_SAMPLES = 12


def broll_scenes(data: dict) -> list[dict]:
    cands = [c for c in (data.get("scene_cuts") or []) if isinstance(c, dict)
             and (c.get("is_broll") or c.get("shot_type") in BROLL_TYPES)
             and c.get("start_time") is not None and c.get("end_time") is not None
             and float(c["end_time"]) - float(c["start_time"]) >= 0.2]
    # windowed scans leave nested spans (0.5-2.5 and 1.0-2.5 for one shot); keep the longest of any overlap group
    cands.sort(key=lambda c: (float(c["start_time"]), -(float(c["end_time"]) - float(c["start_time"]))))
    kept: list[dict] = []
    for c in cands:
        a, b = float(c["start_time"]), float(c["end_time"])
        if kept and a < float(kept[-1]["end_time"]) - 0.15:
            k = kept[-1]
            if b - a > float(k["end_time"]) - float(k["start_time"]):
                kept[-1] = c
            continue
        kept.append(c)
    return kept


# ---------------- pure helpers (unit-testable without video) ----------------

def presentation_from_box(box_pct: dict | None) -> str:
    if not box_pct:
        return "full_frame"
    area = (box_pct["w"] / 100) * (box_pct["h"] / 100)
    if area >= 0.85 or (box_pct["w"] >= 95 and box_pct["h"] >= 60):
        return "full_frame"
    # a card spans most of the width even when it is short; a pip is small in both directions
    if box_pct["w"] >= 45 or area >= 0.20:
        return "card"
    return "pip"


def content_for(scene: dict, assets: list[dict]) -> dict:
    t0, t1 = float(scene["start_time"]), float(scene["end_time"])
    hit = None
    for a in assets:
        if not isinstance(a, dict):
            continue
        if any(t0 - 0.5 <= float(t) <= t1 + 0.5 for t in (a.get("appears_at") or []) if isinstance(t, (int, float))):
            hit = a
            break
    kind_map = {"screenshot": "screenshot", "stock_video": "stock_video", "stock_photo": "stock_photo",
                "illustration": "illustration", "brand_logo": "logo", "generated_chart": "chart",
                "emoji": "illustration", "lottie": "illustration", "svg_icon": "illustration"}
    if hit:
        return {"kind": kind_map.get(hit.get("sourcing_kind"), hit.get("sourcing_kind") or "other"),
                "describes": hit.get("depicts") or scene.get("description"),
                "site_or_app": None, "is_animated": bool(hit.get("is_animated")), "asset_ref": hit.get("id")}
    st = scene.get("shot_type")
    return {"kind": "screen_recording" if st == "screen" else ("stock_video" if st == "broll" else st or "other"),
            "describes": scene.get("description"), "site_or_app": None,
            "is_animated": scene.get("camera_motion") not in (None, "static"), "asset_ref": None}


def trigger_for(scene: dict, transcript: list[dict]) -> dict:
    t0 = float(scene["start_time"])
    best = None
    for seg in transcript:
        if not isinstance(seg, dict) or not (seg.get("text") or "").strip():
            continue
        a, b = float(seg.get("start_time") or 0), float(seg.get("end_time") or 0)
        if a - 1.0 <= t0 <= b + 0.25:
            best = seg
            if a <= t0 <= b:
                break
    if not best:
        return {"phrase": None, "offset_from_trigger_seconds": None}
    return {"phrase": best["text"], "offset_from_trigger_seconds": round(t0 - float(best.get("start_time") or 0), 3)}


def hold_from_fits(fits: list[tuple[float, float, float]], step: float, frame_w: int, frame_h: int) -> dict:
    """fits: (scale, dx_px, dy_px) between consecutive samples `step` seconds apart."""
    if not fits:
        return {"kind": "static", "rate_pct_per_second": 0.0, "start_scale": 1.0, "end_scale": 1.0}
    scales = [s for s, _, _ in fits]
    total_scale = float(np.prod(scales))
    per_sec_scale = (total_scale ** (1 / (len(fits) * step)) - 1) * 100
    dx = sum(x for _, x, _ in fits) / (len(fits) * step) / frame_w * 100
    dy = sum(y for _, _, y in fits) / (len(fits) * step) / frame_h * 100
    if abs(per_sec_scale) >= 1.0 and abs(per_sec_scale) >= max(abs(dx), abs(dy)):
        return {"kind": "ken_burns_in" if per_sec_scale > 0 else "ken_burns_out",
                "rate_pct_per_second": round(abs(per_sec_scale), 2), "start_scale": 1.0, "end_scale": round(total_scale, 3)}
    if max(abs(dx), abs(dy)) >= 2.0:
        if abs(dy) >= abs(dx):
            kind = "scroll_down" if dy < 0 else "scroll_up"       # content moving up on screen = scrolling down
        else:
            kind = "pan_left" if dx < 0 else "pan_right"
        return {"kind": kind, "rate_pct_per_second": round(max(abs(dx), abs(dy)), 2), "start_scale": 1.0, "end_scale": 1.0}
    return {"kind": "static", "rate_pct_per_second": round(max(abs(dx), abs(dy), abs(per_sec_scale)), 2),
            "start_scale": 1.0, "end_scale": 1.0}


# ---------------- pixel measurements ----------------

def _box_pct(video: str, t0: float, duration: float) -> dict | None:
    """Where the new content sits: diff the shot's first settled frame against the frame just before it."""
    pre_start = max(0.0, t0 - 0.2)
    pre, _ = F.read_frames(video, pre_start, 0.12, ANALYSIS_WIDTH)
    post, _ = F.read_frames(video, t0 + min(0.25, duration * 0.3), 0.12, ANALYSIS_WIDTH)
    if len(pre) == 0 or len(post) == 0:
        return None
    mask = M.diff_mask(post[0], pre[0], thresh=24, clean=2)
    H, W = mask.shape
    if mask.mean() < 0.01:
        return None
    b = M.bbox(mask)
    if not b:
        return None
    x0, y0, x1, y1 = b
    return {"x": round(x0 / W * 100, 1), "y": round(y0 / H * 100, 1),
            "w": round((x1 - x0) / W * 100, 1), "h": round((y1 - y0) / H * 100, 1)}


def _edge(video: str, t: float, fps: float, reverse: bool = False) -> dict:
    """Classify the moment content arrives (or, reversed, leaves) at time t."""
    start = max(0.0, t - 0.25)
    frames, _ = F.read_frames(video, start, 0.6, ANALYSIS_WIDTH)
    out = {"kind": "cut", "duration_seconds": 0.0, "easing": "linear", "from_scale": 1.0, "from_opacity": 1.0,
           "from_dx": 0.0, "from_dy": 0.0, "overshoot_pct": 0.0}
    if len(frames) < 4:
        return out
    try:
        tr = D.analyze_transition(frames, fps=fps)
    except Exception:                                   # noqa: BLE001 - measurement, never fatal
        return out
    kind = (tr.get("ranking") or ["cut"])[0]
    res = (tr.get("results") or {}).get(kind) or {}
    curve = res.get("curve") or []
    moving = [i for i, v in enumerate(curve) if 0.02 < v < 0.98]
    dur = round(len(moving) / fps, 3) if moving else 0.0
    if kind == "cut" or dur == 0.0:
        return out
    out.update({"kind": {"dissolve": "fade", "wipe": "wipe", "slide": "slide_left"}.get(kind, kind),
                "duration_seconds": dur, "easing": "ease_out", "from_opacity": 0.0 if kind == "dissolve" else 1.0})
    if kind in ("slide", "wipe"):
        try:
            ap = D.analyze_appear(frames[::-1].copy() if reverse else frames, fps=fps)
            d = (ap.get("evidence") or {}).get("direction") or ""
            out["kind"] = {"left": "slide_right", "right": "slide_left", "up": "slide_down", "down": "slide_up"}.get(d, out["kind"])
            out["from_dx"] = 120.0 if "left" in out["kind"] else (-120.0 if "right" in out["kind"] else 0.0)
            out["from_dy"] = 120.0 if "up" in out["kind"] else (-120.0 if "down" in out["kind"] else 0.0)
        except Exception:                               # noqa: BLE001
            pass
    return out


def _hold(video: str, t0: float, t1: float) -> dict:
    n = max(2, min(MAX_HOLD_SAMPLES, int((t1 - t0) / HOLD_STEP) + 1))
    step = (t1 - t0) / (n - 1) if n > 1 else HOLD_STEP
    samples = []
    for i in range(n):
        fr, _ = F.read_frames(video, t0 + i * step, 0.05, ANALYSIS_WIDTH)
        if len(fr):
            samples.append(fr[0])
    if len(samples) < 2:
        return hold_from_fits([], step, ANALYSIS_WIDTH, ANALYSIS_WIDTH)
    H, W = samples[0].shape[:2]
    fits = []
    for a, b in zip(samples, samples[1:]):
        flow, valid, centers = M.block_flow(a, b, block=8, search=8)
        sim = M.fit_similarity(flow, valid, centers)
        if sim is None:
            continue
        s, cx, cy, resid = sim
        mf = M.mean_flow(a, b)
        dx, dy = (float(mf[0]), float(mf[1])) if mf is not None else (0.0, 0.0)
        fits.append((float(s), dx, dy))
    return hold_from_fits(fits, step, W, H)


def derive(data: dict, video: str, log=None) -> list[dict]:
    """One broll_shots entry per b-roll scene, measured from the frames."""
    say = log or (lambda _m: None)
    fps = float((data.get("metadata") or {}).get("fps") or 30)
    assets = data.get("asset_inventory") or []
    transcript = data.get("transcript") or []
    out = []
    scenes = broll_scenes(data)
    for i, sc in enumerate(scenes, 1):
        t0, t1 = float(sc["start_time"]), float(sc["end_time"])
        try:
            box = _box_pct(video, t0, t1 - t0)
            pres = presentation_from_box(box)
            if pres == "full_frame" or box is None:
                box = {"x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0}
            enter = _edge(video, t0, fps)
            hold = _hold(video, t0 + 0.1, t1 - 0.1) if t1 - t0 > 0.5 else hold_from_fits([], HOLD_STEP, 1, 1)
            exit_ = _edge(video, t1, fps, reverse=True)
        except Exception as e:                          # noqa: BLE001 - one bad span must not lose the rest
            say(f"    span {t0:.1f}-{t1:.1f}s: {str(e)[:60]}")
            continue
        content = content_for(sc, assets)
        trig = trigger_for(sc, transcript)
        out.append({
            "start_time": round(t0, 3), "end_time": round(t1, 3), "duration_seconds": round(t1 - t0, 3),
            # a diff cannot tell a full-frame replacement from b-roll behind a cut-out speaker, so it does
            # not claim to: None here means "unmeasured", and the model pass, when it runs, decides
            "presentation": pres, "box": box, "speaker_visible": True if pres != "full_frame" else None,
            "frame_treatment": {"corner_radius_pct": None, "border": {"color": None, "px": None},
                                "shadow": {"enabled": None, "softness": None}, "backdrop": "none"},
            "enter": enter, "hold": hold,
            "exit": {"kind": exit_["kind"], "to_dx": -exit_["from_dx"] or 0.0, "to_dy": -exit_["from_dy"] or 0.0,
                     "to_scale": exit_["from_scale"], "to_opacity": exit_["from_opacity"],
                     "duration_seconds": exit_["duration_seconds"], "easing": exit_["easing"]},
            "content": content, "trigger": trig,
            "sync": "cut" if enter["kind"] == "cut" else ("word" if trig.get("phrase") else "none"),
            "camera_motion_reported": sc.get("camera_motion"), "transition_reported": sc.get("transition_in"),
            "confidence": 0.55, "estimated": False, "derived": True, "source": "pixels",
        })
        if i % 10 == 0:
            say(f"    {i}/{len(scenes)} spans measured")
    return out


def silence_cuts_local(video: str, min_gap: float = 0.25, noise_db: float = -35.0, max_seconds: float | None = None) -> list[dict]:
    """ffmpeg silencedetect: the audible gaps between speech, no API tokens. Same shape as the structure pass."""
    import re
    import subprocess
    r = subprocess.run(["ffmpeg", "-v", "info", *(["-t", f"{max_seconds}"] if max_seconds else []), "-i", video, "-af", f"silencedetect=noise={noise_db}dB:d={min_gap}",
                        "-f", "null", "-"], capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    return [{"start_time": round(a, 3), "end_time": round(b, 3), "gap_seconds": round(b - a, 3),
             "kept_or_cut": "kept", "confidence": 0.9, "estimated": False, "source": "ffmpeg"}
            for a, b in zip(starts, ends)]

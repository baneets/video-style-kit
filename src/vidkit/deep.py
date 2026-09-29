"""vidkit deep: measure an edit from every frame, locally, then let Gemini NAME what was measured.

Layers:
  measure  - ffmpeg + numpy/OpenCV + Apple Vision (personmatte, ocr): shots, transitions, subject, background,
             every overlay element with a per-frame track and fitted in/hold/out motion, text.
  words    - mlx-whisper, word timestamps for every word.
  name     - Gemini (video + measured JSON): what each element is, purpose, tier, grammar. Never sets numbers.

Output: <name>.deep.json (everything) and <name>.analysis.json (the schema style.build already reads).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import config
from .vision import decide as D
from .vision import frames as F
from .vision import measure as M
from .vision import rects as R

TOOLS = Path(__file__).resolve().parents[2] / "tools"
WIDTH = 540                 # analysis width; boxes are stored as % so this never leaks into results
FPS = 30.0                  # every input is conformed to 30 fps so frame counts mean the same thing everywhere
MIN_AREA_FRAC = 0.0015      # a component smaller than this fraction of the frame is noise
MIN_LIFE = 4                # frames a graphic-cue tracklet must live to be an element
MIN_LIFE_DIFF = 8           # frames a diff-only tracklet must live (body-motion residue is shorter)
SETTLE_PX = 1.0             # centroid speed (px/frame) below which an element counts as at rest
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash"]   # cheapest first; next one on 404/429
GEMINI_MODEL = GEMINI_MODELS[0]
Log = Callable[[str], None]


def _r(x, n=3):
    if isinstance(x, (list, tuple)):
        return [_r(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), n)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =====================================================================================
# 1. frames + person mask (cached per video hash)
# =====================================================================================

def _cache_dir(video: str) -> Path:
    h = hashlib.sha256(Path(video).read_bytes()[:4_000_000] + str(os.path.getsize(video)).encode()).hexdigest()[:16]
    d = config.dirs()["cache"] / "deep" / h
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepare(video: str, max_seconds: float, log: Log) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Conform to WIDTH x auto @ 30fps, run personmatte, return (rgb frames, person masks or None, info)."""
    cd = _cache_dir(video)
    info = F.probe(video)
    w, h = info["width"], info["height"]
    tw = min(WIDTH, w)
    th = max(2, int(round(h * tw / w / 2)) * 2)
    small = cd / "small.mp4"
    if not small.exists():
        log(f"conforming to {tw}x{th} @ {FPS:g} fps")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-t", f"{max_seconds}", "-i", video, "-vf", f"scale={tw}:{th},fps={FPS:g}",
                        "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(small)], check=True)
    frames, _ = F.read_frames(str(small))
    masks = None
    packed = cd / "person.npz"
    if packed.exists():
        z = np.load(packed)
        masks = np.unpackbits(z["m"], axis=-1, count=tw).reshape(-1, th, tw).astype(bool)[: len(frames)]
    elif (TOOLS / "personmatte").exists():
        log("person segmentation (Apple Vision, on-device)")
        matte = cd / "matte.mov"
        t1 = time.time()
        r = subprocess.run([str(TOOLS / "personmatte"), str(small), str(matte)], capture_output=True, text=True,
                           env={**os.environ, "PERSONMATTE_QUALITY": os.environ.get("PERSONMATTE_QUALITY", "balanced")})
        log(f"personmatte {time.time() - t1:.0f}s")
        if r.returncode == 0 and matte.exists():
            out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(matte), "-vf", "alphaextract", "-f", "rawvideo",
                                  "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
            n = len(out) // (tw * th)
            masks = (np.frombuffer(out[: n * tw * th], np.uint8).reshape(n, th, tw) > 128)[: len(frames)]
            np.savez_compressed(packed, m=np.packbits(masks, axis=-1))
            matte.unlink(missing_ok=True)
        else:
            log(f"personmatte failed ({r.stderr.strip()[:120]}); continuing without a person mask")
    if masks is not None and len(masks) < len(frames):
        pad = np.repeat(masks[-1:], len(frames) - len(masks), axis=0)
        masks = np.concatenate([masks, pad])
    info.update({"analysis_width": tw, "analysis_height": th, "frames": len(frames), "fps_analysis": FPS, "cache": str(cd)})
    return frames, masks, info


# =====================================================================================
# 2. shots + transitions
# =====================================================================================

def _small_gray(frames: np.ndarray, w: int = 135) -> np.ndarray:
    import cv2
    h = int(round(frames.shape[1] * w / frames.shape[2]))
    return np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (w, h), interpolation=cv2.INTER_AREA) for f in frames]).astype(np.float32)


def find_shots(frames: np.ndarray, log: Log) -> list[dict]:
    g = _small_gray(frames)
    T = len(g)
    # a cut changes MOST of the frame; an overlay popping in changes a region. Score = fraction of pixels that moved.
    d1 = np.array([0.0] + [(np.abs(g[t] - g[t - 1]) > 22).mean() for t in range(1, T)])
    lag = 8
    d8 = np.array([0.0] * lag + [(np.abs(g[t] - g[t - lag]) > 22).mean() for t in range(lag, T)])
    cand = set()
    for t in range(1, T):
        if d1[t] > 0.45 and d1[t] > 3 * max(d1[max(0, t - 6): t].max(), 0.02):
            cand.add(t)
        elif d8[t] > 0.6 and d1[max(0, t - lag): t + 1].max() < 0.45 and d8[t] >= d8[max(0, t - 4): t + 5].max():
            cand.add(t)                               # slow change: dissolve / fast slide
    cuts = []
    for t in sorted(cand):
        if cuts and t - cuts[-1] < 10:
            continue
        cuts.append(t)
    shots = []
    starts = [0] + cuts
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else T
        tr = {"kind": "start", "duration_s": 0.0}
        if s > 0:
            w0, w1 = max(0, s - 10), min(T, s + 10)
            try:
                res = D.analyze_transition(frames[w0:w1], fps=FPS)
                kind = (res.get("ranking") or ["cut"])[0]
                kind = (res.get("results", {}).get(kind, {}).get("label") or kind)
                curve = res.get("results", {}).get((res.get("ranking") or ["cut"])[0], {}).get("curve") or []
                moving = sum(1 for v in curve if 0.02 < v < 0.98)
                tr = {"kind": kind, "duration_s": _r(moving / FPS), "ambiguous": bool(res.get("ambiguous")),
                      "reconstruction": _r(max(res.get("reconstruction_scores", {}).values() or [0.0]))}
                if res.get("note"):
                    tr["note"] = res["note"]
            except Exception as ex:                    # noqa: BLE001 measurement never fatal
                tr = {"kind": "cut", "duration_s": 0.0, "error": str(ex)[:80]}
        shots.append({"id": f"s{i + 1}", "start_frame": s, "end_frame": e, "start_time": _r(s / FPS), "end_time": _r(e / FPS),
                      "transition_in": tr})
    log(f"{len(shots)} shots")
    return shots


# =====================================================================================
# 3. subject, background, camera per shot
# =====================================================================================

def _face_boxes(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Apple Vision face rectangles (tools/ocr faces). Pixel boxes (x, y, w, h)."""
    import cv2
    tool = TOOLS / "ocr"
    if not tool.exists():
        return []
    H, W = frame.shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        cv2.imwrite(fh.name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        r = subprocess.run([str(tool), "faces", fh.name], capture_output=True, text=True)
    os.unlink(fh.name)
    try:
        j = json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return []
    return [(int(b["box"][0] * W), int(b["box"][1] * H), int(b["box"][2] * W), int(b["box"][3] * H)) for b in j.get("faces", []) if b.get("conf", 0) > 0.5]


def _bg_model(frames: np.ndarray, masks: np.ndarray | None, s: int, e: int) -> np.ndarray:
    idx = np.linspace(s, e - 1, min(45, e - s)).astype(int)
    stack = frames[idx].astype(np.float32)
    if masks is not None:
        import cv2
        k = np.ones((9, 9), np.uint8)
        m = np.stack([cv2.dilate(masks[i].astype(np.uint8), k) > 0 for i in idx])
        stack[m] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(stack, axis=0)
        hole = np.isnan(med[..., 0])
        if hole.any():
            known = med[~hole] if (~hole).any() else None
            if known is not None and len(known) > 100 and float(np.mean(np.var(known, axis=0))) < 400:
                med[hole] = np.median(known, axis=0)          # flat background (white / solid / gradient): paint it through
            else:
                med[hole] = np.median(frames[idx].astype(np.float32), axis=0)[hole]
        _bg_model.last_hole = hole
        return med
    _bg_model.last_hole = np.zeros(frames.shape[1:3], bool)
    return np.median(stack, axis=0)


def _dominant_colors(img: np.ndarray, k: int = 3) -> list[str]:
    import cv2
    px = img.reshape(-1, 3).astype(np.float32)
    px = px[np.random.default_rng(0).choice(len(px), min(4000, len(px)), replace=False)]
    _, labels, centers = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    order = np.argsort(-np.bincount(labels.ravel(), minlength=k))
    return ["#%02X%02X%02X" % tuple(int(v) for v in centers[i]) for i in order]


def share_backgrounds(shots: list[dict], bgs: dict[str, np.ndarray], masks: np.ndarray | None, frames: np.ndarray,
                      holes: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Shots with the same camera setup share one background (median of their medians). An overlay that lives for a
    whole shot is absent from the sibling shots, so the shared background reveals it; the shot's own median would
    have absorbed it. Handheld / moving shots keep their own model."""
    ids = [sh["id"] for sh in shots]
    H, W = frames.shape[1:3]
    small, pm = {}, {}
    import cv2
    for sh in shots:
        small[sh["id"]] = cv2.resize(bgs[sh["id"]], (W // 4, H // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
        pm[sh["id"]] = cv2.resize(masks[sh["start_frame"]: sh["end_frame"]: 5].any(0).astype(np.uint8), (W // 4, H // 4)) > 0 \
            if masks is not None else np.zeros((H // 4, W // 4), bool)
    cluster = {i: i for i in ids}
    for a in ids:
        for b in ids:
            if a >= b:
                continue
            ok = ~(pm[a] | pm[b])
            d = np.abs(small[a] - small[b])[ok].mean() if ok.any() else 999.0
            if d < 14.0:
                ra, rb = cluster[a], cluster[b]
                for k, v in cluster.items():
                    if v == rb:
                        cluster[k] = ra
    groups: dict[str, list[str]] = {}
    for k, v in cluster.items():
        groups.setdefault(v, []).append(k)
    out = dict(bgs)
    by_id = {sh["id"]: sh for sh in shots}
    for members in groups.values():
        members = [m for m in members if by_id[m]["camera"]["motion"] == "static"]
        if len(members) < 2:
            continue
        stack = np.stack([bgs[m].astype(np.float32) for m in members])
        if holes:
            for i, m in enumerate(members):
                stack[i][holes[m]] = np.nan
        with np.errstate(all="ignore"):
            shared = np.nanmedian(stack, axis=0)
        still = np.isnan(shared[..., 0])
        if still.any():                               # the person covered it in every member: fall back to plain median
            shared[still] = np.median(np.stack([bgs[m].astype(np.float32) for m in members]), axis=0)[still]
        for m in members:
            out[m] = shared
            by_id[m]["background"]["shared_with"] = [x for x in members if x != m]
    return out


def analyze_shot_layers(frames: np.ndarray, masks: np.ndarray | None, shot: dict) -> dict:
    """Subject presentation (raw vs cutout), framing, background kind + motion, camera motion for one shot."""
    s, e = shot["start_frame"], shot["end_frame"]
    H, W = frames.shape[1:3]
    bg = _bg_model(frames, masks, s, e)
    bg_u8 = np.clip(bg, 0, 255).astype(np.uint8)
    idx = np.linspace(s, e - 1, min(12, e - s)).astype(int)
    out: dict[str, Any] = {}

    # --- subject
    person_frac = float(np.mean([masks[i].mean() for i in idx])) if masks is not None else 0.0
    faces = []
    for i in idx[:: max(1, len(idx) // 6)]:
        fb = _face_boxes(frames[i])
        if fb:
            x, y, w, h = max(fb, key=lambda b: b[2] * b[3])
            faces.append((x + w / 2, y + h / 2, h))
    subject: dict[str, Any] = {"present": person_frac > 0.03, "area_pct": _r(person_frac * 100, 1)}
    if faces:
        cx, cy, fh = np.median(np.array(faces), axis=0)
        subject.update({"head_center": {"x": _r(cx / W * 100, 1), "y": _r(cy / H * 100, 1)}, "head_height_pct": _r(fh / H * 100, 1),
                        "headroom_pct": _r((cy - fh / 2) / H * 100, 1),
                        "thirds_zone": ("left" if cx < W / 3 else "right" if cx > 2 * W / 3 else "center") + "-" +
                                       ("top" if cy < H / 3 else "bottom" if cy > 2 * H / 3 else "middle")})
        # framing over time: head height per sample -> punch-ins without a cut
        hs = [f[2] for f in faces]
        if len(hs) >= 3 and max(hs) / max(1e-6, min(hs)) > 1.15:
            subject["reframes"] = {"head_height_min_pct": _r(min(hs) / H * 100, 1), "head_height_max_pct": _r(max(hs) / H * 100, 1)}
    if masks is not None and person_frac > 0.03:
        mb = M.bbox(masks[idx[len(idx) // 2]])
        if mb:
            x0, y0, x1, y1 = mb
            subject["body_box"] = {"x": _r(x0 / W * 100, 1), "y": _r(y0 / H * 100, 1), "w": _r((x1 - x0) / W * 100, 1), "h": _r((y1 - y0) / H * 100, 1)}
            subject["body_visible"] = "head" if (y1 - y0) < H * 0.35 else "bust" if (y1 - y0) < H * 0.6 else "waist_or_more"
        # cutout evidence: (a) edge halo / hard alpha, (b) background moves independently of the subject
        import cv2
        halo, spill = [], []
        for i in idx[::3]:
            m = masks[i].astype(np.uint8)
            ring = (cv2.dilate(m, np.ones((5, 5), np.uint8)) - m) > 0
            inner = (m - cv2.erode(m, np.ones((5, 5), np.uint8))) > 0
            if ring.sum() > 50 and inner.sum() > 50:
                f = frames[i].astype(np.float32)
                halo.append(float(np.abs(f[ring].mean(0) - f[inner].mean(0)).mean()))
                spill.append(float((f[inner][:, 1] - f[inner][:, [0, 2]].mean(1)).mean()))
        subject["cutout_evidence"] = {"edge_contrast": _r(np.mean(halo) if halo else None, 1),
                                      "green_spill": _r(np.mean(spill) if spill else None, 1)}

    # --- background
    var = float(M.spatial_variance(bg_u8))
    colors = _dominant_colors(bg_u8)
    flows = []
    for a, b in zip(idx, idx[1:]):
        inv = (~masks[a]) if masks is not None else None
        try:
            flow, valid, centers = M.block_flow(frames[a], frames[b], block=8, search=8, mask=inv)
            sim = M.fit_similarity(flow, valid, centers)
            mf = M.mean_flow(frames[a], frames[b], mask=inv)
            flows.append((sim[0] if sim else 1.0, float(mf[0]) if mf is not None else 0.0, float(mf[1]) if mf is not None else 0.0))
        except Exception:                             # noqa: BLE001
            pass
    step = (idx[1] - idx[0]) / FPS if len(idx) > 1 else 1.0
    if flows:
        sc = np.array([f[0] for f in flows]); dx = np.array([f[1] for f in flows]); dy = np.array([f[2] for f in flows])
        zoom_rate = float((sc.mean() - 1.0) / step * 100)
        pan_x = float(dx.mean() / W * 100 / step); pan_y = float(dy.mean() / H * 100 / step)
        jitter = float(np.hypot(dx - dx.mean(), dy - dy.mean()).mean() / W * 100)
    else:
        zoom_rate = pan_x = pan_y = jitter = 0.0
    if var < 400:                                     # solid / gradient: block flow on flat pixels is noise
        zoom_rate = pan_x = pan_y = jitter = 0.0
    if var < 40:
        kind = "solid"
    elif var < 400:
        kind = "gradient"
    elif jitter > 0.25:
        kind = "real_room_handheld"
    elif abs(zoom_rate) > 0.5 or abs(pan_x) > 0.5 or abs(pan_y) > 0.5:
        kind = "moving_image_or_video"
    else:
        kind = "static_scene"
    # flat colour background? measure on real frames (not the model): share of non-person pixels within 30 of the mode colour
    flat_color, flat_frac = None, 0.0
    try:
        fi = idx[len(idx) // 2]
        f = frames[fi].astype(np.int16)
        notp = ~masks[fi] if masks is not None else np.ones((H, W), bool)
        c0 = np.array([int(colors[0][1:3], 16), int(colors[0][3:5], 16), int(colors[0][5:7], 16)], np.int16)
        near = (np.abs(f - c0).max(axis=2) < 30) & notp
        flat_frac = float(near.sum() / max(1, notp.sum()))
        if flat_frac > 0.55:
            flat_color = colors[0]
    except Exception:                                 # noqa: BLE001
        pass
    if flat_color:
        kind = "solid" if var < 40 or flat_frac > 0.85 else "solid_textured"
        zoom_rate = pan_x = pan_y = jitter = 0.0
    background = {"kind": kind, "spatial_variance": _r(var, 0), "dominant_colors": colors, "flat_color": flat_color, "flat_fraction": _r(flat_frac, 2),
                  "motion": {"zoom_pct_per_s": _r(zoom_rate, 2), "pan_x_pct_per_s": _r(pan_x, 2), "pan_y_pct_per_s": _r(pan_y, 2),
                             "jitter_pct": _r(jitter, 2)}}
    out["background"] = background
    # presentation decision
    pres = "none"
    if subject["present"]:
        ce = subject.get("cutout_evidence", {})
        cut_signals = int(kind in ("solid", "gradient", "moving_image_or_video")) + int((ce.get("edge_contrast") or 0) > 45) + int((ce.get("green_spill") or 0) > 6)
        pres = "cutout" if cut_signals >= 2 else ("likely_cutout" if cut_signals == 1 and kind != "real_room_handheld" else "raw_camera")
    subject["presentation"] = pres
    out["subject"] = subject
    out["camera"] = {"motion": "handheld" if jitter > 0.25 else ("zoom" if abs(zoom_rate) > 0.5 else "pan" if max(abs(pan_x), abs(pan_y)) > 0.5 else "static"),
                     "zoom_pct_per_s": _r(zoom_rate, 2), "pan_pct_per_s": [_r(pan_x, 2), _r(pan_y, 2)]}
    out["_bg"] = bg_u8
    out["_hole"] = _bg_model.last_hole
    return out


# =====================================================================================
# 4. element discovery + tracking
# =====================================================================================

def _components(fg: np.ndarray, min_area: int) -> list[dict]:
    import cv2
    n, lab, stats, cents = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area:
            out.append({"box": (int(x), int(y), int(w), int(h)), "area": int(a), "c": (float(cents[i][0]), float(cents[i][1])), "label": i})
    return out, lab


def _contained(a, b) -> float:
    """Fraction of box a that lies inside box b."""
    ax0, ay0, aw, ah = a; bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0)); iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    return ix * iy / float(aw * ah + 1e-6)


def _iou(a, b) -> float:
    ax0, ay0, aw, ah = a; bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0)); iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter + 1e-6)


def graphic_cue(frame: np.ndarray) -> np.ndarray:
    """Pixels that look like a GRAPHIC rather than camera footage: flat (low local variance) and either saturated or
    near-pure white/black. A yellow callout, a white caption box, a black title bar all pass; skin, walls and shirts
    mostly do not (texture + mid-saturation). Rectangularity is checked per component later."""
    import cv2
    f = frame.astype(np.float32)
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.blur(g, (5, 5)); var = cv2.blur(g * g, (5, 5)) - mu * mu
    flat = var < 20.0
    mx, mn = f.max(2), f.min(2)
    sat = (mx - mn) / (mx + 1e-3)
    strong = (sat > 0.5) & (mx > 120)
    bright = mn > 235
    # ponytail: near-black flat patches are excluded on purpose; they were 80% shirt/shadow. A black title bar
    # still registers through the diff cue when it sits off the person, and through OCR when it carries text.
    return flat & (strong | bright)


def _rectangularity(mask_crop: np.ndarray) -> float:
    return float(mask_crop.mean())


def candidate_mask(frames: np.ndarray, masks: np.ndarray | None, bgf: np.ndarray, t: int) -> np.ndarray:
    """Per-frame overlay candidate pixels. See track_elements."""
    import cv2
    H, W = frames.shape[1:3]
    k3, k5 = np.ones((3, 3), np.uint8), np.ones((5, 5), np.uint8)
    dil = np.ones((25, 25), np.uint8)
    diff = np.abs(frames[t].astype(np.int16) - bgf).max(axis=2) > 30
    person = cv2.dilate(masks[t].astype(np.uint8), dil) > 0 if masks is not None else np.zeros((H, W), bool)
    # graphic pixels may sit ON the person (caption over the chest, inset over the shoulder): the flat/saturated
    # cue lifts the person exclusion, but only where the pixel also differs from the background.
    gcue = graphic_cue(frames[t]) & diff
    gcue = cv2.morphologyEx(gcue.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    # an overlay that occludes the speaker shows up as a HOLE in the person silhouette (Vision segments the
    # speaker, not the inset drawn over them): non-person pixels enclosed by person pixels.
    hole = np.zeros((H, W), bool)
    if masks is not None and masks[t].any():
        pm = masks[t].astype(np.uint8)
        pm = cv2.morphologyEx(pm, cv2.MORPH_CLOSE, k5)
        ff = pm.copy()
        cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 2)
        enclosed = (ff == 0)                     # not person, not reachable from the frame corner
        for corner in ((W - 1, 0), (0, H - 1), (W - 1, H - 1)):
            if ff[corner[1], corner[0]] == 0:
                ff2 = pm.copy(); cv2.floodFill(ff2, np.zeros((H + 2, W + 2), np.uint8), corner, 2)
                enclosed &= (ff2 == 0)
        hole = cv2.morphologyEx(enclosed.astype(np.uint8), cv2.MORPH_OPEN, k5) > 0
    d = (diff & ~person) | (gcue & person)
    d = cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_OPEN, k3)
    d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, k5) > 0
    return d, person, hole


def track_elements(frames: np.ndarray, masks: np.ndarray | None, shots: list[dict], bgs: dict[str, np.ndarray], log: Log) -> list[dict]:
    """One pass over the whole video. Per frame, candidate pixels = (differs from the shot's background model, person
    excluded) OR (looks like a graphic and is rectangular). Components are linked frame-to-frame into tracklets, which may
    cross shot boundaries (a persistent overlay is one element). The graphic cue is what catches overlays that live for the
    whole shot and would otherwise be absorbed into the background median."""
    import cv2
    T, H, W = frames.shape[:3]
    min_area = int(MIN_AREA_FRAC * H * W)
    k3, k5 = np.ones((3, 3), np.uint8), np.ones((5, 5), np.uint8)
    dil = np.ones((25, 25), np.uint8)
    shot_of = np.empty(T, dtype=object)
    for sh in shots:
        shot_of[sh["start_frame"]: sh["end_frame"]] = sh["id"]
    bg16 = {k: v.astype(np.int16) for k, v in bgs.items()}
    active: list[dict] = []
    done: list[dict] = []
    t_last = time.time()
    static = {sh["id"] for sh in shots if sh["camera"]["motion"] == "static"}
    flat = {sh["id"]: sh["background"]["flat_color"] for sh in shots if sh["background"].get("flat_color")}
    for t in range(T):
        bgf = bg16[shot_of[t]]
        if shot_of[t] in flat:
            # flat background: the whole frame is either background colour, person, or an overlay. No model needed,
            # so overlays that live for the entire shot are seen too.
            hx = flat[shot_of[t]]
            c0 = np.array([int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16)], np.int16)
            bgf = np.broadcast_to(c0, frames[t].shape).astype(np.int16)
        elif shot_of[t] not in static:
            comps = []                                # a moving shot has no usable background model: cards + OCR only
            for tr in active:
                tr["miss"] += 1
            still = []
            for tr in active:
                (done if tr["miss"] > 2 else still).append(tr)
            active = still
            continue
        d, person, hole = candidate_mask(frames, masks, bgf, t)
        comps = []
        # silhouette holes first: a RECTANGULAR hole in the person is an inset drawn over them (arm gaps are not rectangular)
        hcs, hlab = _components(hole, int(0.004 * H * W))
        for c in hcs:
            x, y, w, h = c["box"]
            sel = hlab[y:y + h, x:x + w] == c["label"]
            fill = float(sel.mean())
            if fill < 0.7 or w < 0.08 * W or h < 0.05 * H:
                continue
            px = frames[t, y:y + h, x:x + w][sel].astype(np.float32)
            c.update({"color": px.mean(0), "t": t, "src": "hole", "fill": fill, "on_person": 1.0})
            comps.append(c)
        cs, lab = _components(d, min_area)
        for c in cs:
            x, y, w, h = c["box"]
            if any(o["src"] == "hole" and _iou(o["box"], c["box"]) > 0.3 for o in comps):
                continue
            sel = lab[y:y + h, x:x + w] == c["label"]
            fill = float(sel.mean())
            on_person = float(person[y:y + h, x:x + w][sel].mean())
            px = frames[t, y:y + h, x:x + w][sel].astype(np.float32)
            c.update({"color": px.mean(0), "t": t, "src": "graphic" if on_person > 0.5 else "diff", "fill": fill, "on_person": on_person})
            comps.append(c)
        used = set()
        for tr in active:
            last = tr["obs"][-1]
            best, bi = 0.0, -1
            for j, c in enumerate(comps):
                if j in used or (c["src"] == "hole") != (last["src"] == "hole"):
                    continue
                v = _iou(last["box"], c["box"])
                if v > best:
                    best, bi = v, j
            if bi >= 0 and (best >= 0.3 or (best > 0.05 and np.abs(comps[bi]["color"] - last["color"]).mean() < 25)):
                tr["obs"].append(comps[bi]); used.add(bi); tr["miss"] = 0
            else:
                tr["miss"] += 1
        for j, c in enumerate(comps):
            if j not in used:
                active.append({"obs": [c], "miss": 0})
        still = []
        for tr in active:
            (done if tr["miss"] > 2 else still).append(tr)
        active = still
        if time.time() - t_last > 30:
            log(f"tracking frame {t}/{T}, {len(active)} live tracklets"); t_last = time.time()
    done += active
    els = []
    for tr in done:
        obs = tr["obs"]
        life = obs[-1]["t"] - obs[0]["t"] + 1
        graphic = sum(1 for o in obs if o["src"] in ("graphic", "hole")) / len(obs)
        if life < MIN_LIFE_DIFF:
            continue
        boxes = np.array([o["box"] for o in obs], np.float32)
        jitter = float(np.abs(np.diff(boxes[:, :2], axis=0)).mean()) if len(boxes) > 1 else 0.0
        if jitter > 2.5 and life < FPS:
            continue                                  # a short, wobbling blob = body motion residue
        if graphic > 0.5 and jitter > 1.5:
            continue                                  # a flat patch riding on the person (shirt print) moves with the body
        col = np.mean([o["color"] for o in obs], axis=0)
        mx, mn = float(col.max()), float(col.min())
        colourful = (mx - mn) / (mx + 1e-3) > 0.25 or mx > 150
        if not colourful and (life < 1.5 * FPS or jitter > 1.0):
            continue                                  # dark, dull, short or wobbly: shadow / hair / shirt, not a graphic
        bx = np.median(boxes, axis=0)
        touches_edge = bx[0] <= 1 or bx[1] <= 1 or bx[0] + bx[2] >= W - 1 or bx[1] + bx[3] >= H - 1
        if touches_edge and (bx[2] < 0.04 * W or bx[3] < 0.04 * H):
            continue                                  # sliver on the frame edge: reframing / letterbox artefact
        if life < 0.4 * FPS:
            continue
        els.append({"shot": shot_of[obs[0]["t"]], "obs": obs, "graphic_frac": _r(graphic, 2)})
    els.sort(key=lambda el: el["obs"][0]["t"])
    merged: list[dict] = []
    for el in els:
        for m in merged:
            if el["obs"][0]["t"] <= m["obs"][-1]["t"] + 3 and _iou(m["obs"][-1]["box"], el["obs"][0]["box"]) > 0.6:
                m["obs"] += el["obs"]; break
        else:
            merged.append(el)
    log(f"{len(done)} tracklets -> {len(merged)} elements")
    return merged


# =====================================================================================
# 5. motion curves per element
# =====================================================================================

EASINGS = {
    "linear": lambda t: t,
    "ease_out": lambda t: 1 - (1 - t) ** 3,
    "ease_in": lambda t: t ** 3,
    "ease_in_out": lambda t: 3 * t ** 2 - 2 * t ** 3,
    "ease_out_expo": lambda t: 1 - 2 ** (-10 * t) if t < 1 else 1.0,
    "spring_overshoot": lambda t: 1 - math.exp(-6 * t) * math.cos(4.5 * t),
    "snap": lambda t: 0.0 if t < 0.5 else 1.0,
}


def name_easing(curve: list[float]) -> tuple[str, float, float]:
    """Best named easing for a 0..1 progress curve; returns (name, rms, overshoot_pct)."""
    if len(curve) < 2:
        return "snap", 0.0, 0.0
    ts = np.linspace(0, 1, len(curve))
    best = ("linear", 9.0)
    for n, f in EASINGS.items():
        rms = float(np.sqrt(np.mean((np.array([f(t) for t in ts]) - np.array(curve)) ** 2)))
        if rms < best[1]:
            best = (n, rms)
    over = float(max(0.0, (max(curve) - 1.0) * 100))
    return best[0], _r(best[1]), _r(over, 1)


def _series(obs: list[dict], rest_area: float):
    cx = np.array([o["c"][0] for o in obs]); cy = np.array([o["c"][1] for o in obs])
    area = np.array([o["area"] for o in obs], np.float32)
    return cx, cy, np.sqrt(area / max(rest_area, 1.0))


def _settle_index(cx, cy, from_start=True) -> int:
    """First index (or last, reversed) after which the centroid stays still for 3 frames."""
    n = len(cx)
    rng = range(n - 3) if from_start else range(n - 1, 2, -1)
    for i in rng:
        seg = slice(i, i + 3) if from_start else slice(i - 3, i)
        v = np.hypot(np.diff(cx[seg]), np.diff(cy[seg]))
        if len(v) and v.max() < SETTLE_PX:
            return i
    return n - 1 if from_start else 0


def describe_motion(el: dict, frames: np.ndarray, bg: np.ndarray, shot: dict) -> dict:
    obs = el["obs"]
    H, W = frames.shape[1:3]
    T = len(obs)
    t_on, t_off = obs[0]["t"], obs[-1]["t"] + 1
    cx0, cy0, _ = _series(obs, 1.0)
    i_in = _settle_index(cx0, cy0, True)
    i_out = _settle_index(cx0, cy0, False)
    if i_out <= i_in:
        i_out = T - 1
    rest_slice = slice(i_in, max(i_in + 1, i_out))
    rest_area = float(np.median([o["area"] for o in obs[rest_slice]]))
    cx, cy, scale = _series(obs, rest_area)
    rx = float(np.median([o["box"][0] for o in obs[rest_slice]])); ry = float(np.median([o["box"][1] for o in obs[rest_slice]]))
    rw = float(np.median([o["box"][2] for o in obs[rest_slice]])); rh = float(np.median([o["box"][3] for o in obs[rest_slice]]))
    rest_c = (float(np.median(cx[rest_slice])), float(np.median(cy[rest_slice])))
    # opacity against the background model inside the rest box
    x0, y0, x1, y1 = int(rx), int(ry), int(rx + rw), int(ry + rh)
    final = frames[obs[i_in]["t"]]
    emask = M.diff_mask(final[y0:y1, x0:x1], bg[y0:y1, x0:x1])
    ops = []
    for o in obs:
        f = frames[o["t"]][y0:y1, x0:x1]
        try:
            ops.append(float(np.clip(M.blend_alpha(f.astype(np.float32), bg[y0:y1, x0:x1].astype(np.float32), final[y0:y1, x0:x1].astype(np.float32), emask), 0, 1)))
        except Exception:                             # noqa: BLE001
            ops.append(1.0)
    ops = np.array(ops) if ops else np.ones(T)

    def _phase(sl: slice, reverse: bool) -> dict:
        idx = list(range(T))[sl]
        if reverse:
            idx = idx[::-1]
        if len(idx) < 2:
            return {"kind": "cut", "duration_s": 0.0, "curve": []}
        d0 = math.hypot(cx[idx[0]] - rest_c[0], cy[idx[0]] - rest_c[1])
        travel = [math.hypot(cx[i] - rest_c[0], cy[i] - rest_c[1]) for i in idx]
        pos_curve = [1 - v / d0 for v in travel] if d0 > 2 else [1.0] * len(idx)
        sc_curve = [float(scale[i]) for i in idx]
        op_curve = [float(ops[i]) for i in idx]
        dx, dy = (cx[idx[0]] - rest_c[0]) / W * 100, (cy[idx[0]] - rest_c[1]) / H * 100
        moved = d0 > max(3.0, 0.02 * W)
        scaled = abs(sc_curve[0] - 1.0) > 0.12
        faded = op_curve[0] < 0.7
        if moved:
            kind = "slide"; curve = pos_curve
        elif scaled:
            kind = "scale"; curve = [(v - sc_curve[0]) / (1 - sc_curve[0]) if abs(1 - sc_curve[0]) > 1e-3 else 1.0 for v in sc_curve]
        elif faded:
            kind = "fade"; curve = [(v - op_curve[0]) / (1 - op_curve[0]) if abs(1 - op_curve[0]) > 1e-3 else 1.0 for v in op_curve]
        else:
            kind = "pop" if len(idx) <= 2 else "grow"   # grew in place (wipe/typewriter/mask) -> region-level contest below
            curve = [float(o) for o in np.array([obs[i]["area"] for i in idx]) / max(rest_area, 1)]
        ename, rms, over = name_easing(curve)
        direction = None
        if moved:
            direction = "from_" + ("left" if dx < -abs(dy) else "right" if dx > abs(dy) else "top" if dy < 0 else "bottom")
        return {"kind": kind, "direction": direction, "dx_pct": _r(dx, 1), "dy_pct": _r(dy, 1), "scale0": _r(sc_curve[0]),
                "opacity0": _r(op_curve[0]), "duration_s": _r(len(idx) / FPS), "easing": ename, "easing_rms": rms,
                "overshoot_pct": over, "curve": _r(curve, 3), "scale_curve": _r(sc_curve, 3), "opacity_curve": _r(op_curve, 3)}

    m_in = _phase(slice(0, i_in + 1), False)
    m_out = _phase(slice(i_out, T), True)
    # contest-based confirmation of the entry shape (wipe / typewriter / slide-from-behind / custom mask)
    pad = 8
    wx0, wy0 = max(0, int(min(o["box"][0] for o in obs[: i_in + 1])) - pad), max(0, int(min(o["box"][1] for o in obs[: i_in + 1])) - pad)
    wx1 = min(W, int(max(o["box"][0] + o["box"][2] for o in obs[: i_in + 1])) + pad)
    wy1 = min(H, int(max(o["box"][1] + o["box"][3] for o in obs[: i_in + 1])) + pad)
    if i_in >= 2 and (wx1 - wx0) > 8 and (wy1 - wy0) > 8 and (wx1 - wx0) * (wy1 - wy0) < 0.5 * W * H:
        win = np.concatenate([bg[None, wy0:wy1, wx0:wx1].astype(np.uint8), frames[t_on: t_on + i_in + 1, wy0:wy1, wx0:wx1]])
        if win.shape[2] > 200:
            import cv2
            sc = 200 / win.shape[2]
            win = np.stack([cv2.resize(f, (200, max(8, int(f.shape[0] * sc))), interpolation=cv2.INTER_AREA) for f in win])
        try:
            ap = D.analyze_appear(win, fps=FPS)
            m_in["contest"] = {"chosen": ap.get("chosen"), "margin": _r(ap.get("margin")), "ambiguous": ap.get("ambiguous"),
                               "reconstruction": _r(max((ap.get("reconstruction_scores") or {}).values() or [0.0]))}
            if ap.get("chosen") in ("wipe", "wipe_soft", "typewriter", "custom_mask", "slide_from_behind") and m_in["kind"] in ("grow", "pop"):
                m_in["kind"] = ap["chosen"]
        except Exception as ex:                        # noqa: BLE001
            m_in["contest"] = {"error": str(ex)[:80]}
    # hold
    hs = rest_slice
    hold: dict[str, Any] = {"kind": "still"}
    if i_out - i_in > 6:
        hx, hy, hsc = cx[hs], cy[hs], scale[hs]
        tt = np.arange(len(hx)) / FPS
        vx = float(np.polyfit(tt, hx, 1)[0]) / W * 100 if len(tt) > 1 else 0.0
        vy = float(np.polyfit(tt, hy, 1)[0]) / H * 100 if len(tt) > 1 else 0.0
        vs = float(np.polyfit(tt, hsc, 1)[0]) * 100 if len(tt) > 1 else 0.0
        det = hsc - np.polyval(np.polyfit(tt, hsc, 1), tt)
        amp = float(det.std() * 100)
        if amp > 1.5 and len(det) > 12:
            spec = np.abs(np.fft.rfft(det))
            k = int(np.argmax(spec[1:]) + 1)
            hold = {"kind": "pulse", "amplitude_pct": _r(amp * 1.41, 1), "cycle_s": _r(len(det) / FPS / k)}
        elif abs(vx) > 0.3 or abs(vy) > 0.3:
            hold = {"kind": "drift", "rate_pct_per_s": [_r(vx, 2), _r(vy, 2)]}
        elif abs(vs) > 0.8:
            hold = {"kind": "grow" if vs > 0 else "shrink", "rate_pct_per_s": _r(vs, 2)}
    # stop style
    if m_in["kind"] == "slide" and i_in > 1:
        d_rest = [math.hypot(cx[i] - rest_c[0], cy[i] - rest_c[1]) for i in range(i_in + 1)]
        d_first = d_rest[0]
        proj = [((cx[i] - rest_c[0]) * (cx[0] - rest_c[0]) + (cy[i] - rest_c[1]) * (cy[0] - rest_c[1])) / max(d_first, 1e-6) for i in range(i_in + 1)]
        over = -min(proj) / max(d_first, 1e-6) * 100
        m_in["stop_style"] = "overshoot" if over > 2 else ("settle" if m_in["easing"] in ("ease_out", "ease_out_expo") else "hard")
        m_in["overshoot_pct"] = _r(max(0.0, over), 1)
    # leaves via
    last = obs[-1]["box"]
    touches = ("left" if last[0] <= 1 else "right" if last[0] + last[2] >= W - 1 else "top" if last[1] <= 1 else "bottom" if last[1] + last[3] >= H - 1 else None)
    if any(abs(t_off - b) <= 1 for b in getattr(describe_motion, "_bounds", [shot["end_frame"]])):
        leaves = "covered_by_cut"
    elif m_out["kind"] == "slide" and touches:
        leaves = f"edge_{touches}"
    elif m_out["kind"] == "slide":
        leaves = "slide_" + (m_out["direction"] or "").replace("from_", "to_")
    elif m_out["kind"] == "fade":
        leaves = "fade"
    elif m_out["kind"] == "scale":
        leaves = "shrink"
    else:
        leaves = "cut"
    m_out["leaves_via"] = leaves
    track = [{"t": _r(o["t"] / FPS), "x": _r(o["box"][0] / W * 100, 1), "y": _r(o["box"][1] / H * 100, 1), "w": _r(o["box"][2] / W * 100, 1),
              "h": _r(o["box"][3] / H * 100, 1), "scale": _r(float(scale[i])), "opacity": _r(float(ops[i]))} for i, o in enumerate(obs)]
    mean_col = np.mean([o["color"] for o in obs[rest_slice]], axis=0)
    return {"life": {"t_on": _r(t_on / FPS), "t_off": _r(t_off / FPS), "frames": T},
            "box_rest": {"x": _r(rx / W * 100, 1), "y": _r(ry / H * 100, 1), "w": _r(rw / W * 100, 1), "h": _r(rh / H * 100, 1)},
            "in_unsafe_zone": bool(ry / H < 0.12 or (ry + rh) / H > 0.80),
            "colour": "#%02X%02X%02X" % tuple(int(v) for v in mean_col),
            "motion": {"in": m_in, "hold": hold, "out": m_out}, "track": track}


def _seq_motion(seq: list[dict], W: int, H: int, reverse: bool) -> dict:
    """Entry (or, reversed, exit) motion of a card from its per-frame (scale, dx, dy, score) fit sequence."""
    if reverse:
        seq = seq[::-1]
    # settle = first frame from which scale==1 and offset==0 hold for 3 frames
    settle = len(seq) - 1
    for i in range(len(seq) - 2):
        if all(abs(f["scale"] - 1.0) < 0.03 and abs(f["dx"]) <= 4 and abs(f["dy"]) <= 4 for f in seq[i: i + 3]):
            settle = i
            break
    move = seq[: settle + 1]
    if len(move) <= 1:
        return {"kind": "cut", "duration_s": 0.0, "easing": "snap", "curve": [], "scale0": 1.0, "dx_pct": 0.0, "dy_pct": 0.0,
                "opacity0": 1.0, "overshoot_pct": 0.0, "direction": None}
    s0, dx0, dy0 = move[0]["scale"], move[0]["dx"], move[0]["dy"]
    sc_curve = [f["scale"] for f in move]
    d0 = math.hypot(dx0, dy0)
    if abs(1 - s0) > 0.08:
        kind = "scale"; curve = [(v - s0) / (1 - s0) for v in sc_curve]
    elif d0 > 0.04 * W:
        kind = "slide"; curve = [1 - math.hypot(f["dx"], f["dy"]) / d0 for f in move]
    else:
        kind = "fade" if move[0]["score"] < 0.6 * move[-1]["score"] else "pop"
        curve = [f["score"] / max(move[-1]["score"], 1e-6) for f in move]
    ename, rms, over = name_easing(curve)
    direction = None
    if kind == "slide":
        direction = "from_" + ("left" if dx0 < -abs(dy0) else "right" if dx0 > abs(dy0) else "top" if dy0 < 0 else "bottom")
    return {"kind": kind, "direction": direction, "duration_s": _r(len(move) / FPS), "easing": ename, "easing_rms": rms,
            "overshoot_pct": _r(max(over, (max(sc_curve) - 1) * 100 if kind == "scale" else 0.0), 1),
            "scale0": _r(s0), "dx_pct": _r(dx0 / W * 100, 1), "dy_pct": _r(dy0 / H * 100, 1), "opacity0": _r(move[0]["score"] / max(move[-1]["score"], 1e-6)),
            "curve": _r(curve), "scale_curve": _r(sc_curve), "offset_curve": [[f["dx"], f["dy"]] for f in move],
            "stop_style": "overshoot" if max(sc_curve) > 1.03 else ("settle" if ename.startswith("ease_out") else "hard")}


def card_elements(frames: np.ndarray, shots: list[dict], log: Log) -> list[dict]:
    """Rectangular overlays (insets, cards) from persistent edges, with entry/exit fitted per frame."""
    H, W = frames.shape[1:3]
    tracks, E = R.track_rects(frames, log=log)
    out = []
    t1 = time.time()
    for tr in tracks:
        m = R.measure_entry_exit(E, tr, FPS)
        seq_in, seq_out = m["in"], m["out"]
        t_on = seq_in[0]["t"] if seq_in else tr["t_on"]
        t_off = (seq_out[-1]["t"] + 1) if seq_out else tr["t_off"]
        mi = _seq_motion(seq_in, W, H, reverse=False)
        mo = _seq_motion(seq_out, W, H, reverse=True)
        x, y, w, h = tr["box"]
        last = seq_out[-1] if seq_out else {"scale": 1.0, "dx": 0, "dy": 0}
        bx, by = x + w / 2 - w * last["scale"] / 2 + last["dx"], y + h / 2 - h * last["scale"] / 2 + last["dy"]
        touches = ("left" if bx <= 1 else "right" if bx + w * last["scale"] >= W - 1 else "top" if by <= 1 else "bottom" if by + h * last["scale"] >= H - 1 else None)
        if any(abs(t_off - sh["end_frame"]) <= 1 for sh in shots):
            leaves = "covered_by_cut"
        elif mo["kind"] == "slide":
            leaves = f"edge_{touches}" if touches else "slide_" + (mo["direction"] or "").replace("from_", "to_")
        else:
            leaves = {"fade": "fade", "scale": "shrink", "pop": "cut", "cut": "cut"}.get(mo["kind"], "cut")
        mo["leaves_via"] = leaves
        sid = next((sh["id"] for sh in shots if sh["start_frame"] <= t_on < sh["end_frame"]), shots[0]["id"])
        out.append({"kind": "card", "shot": sid, "life": {"t_on": _r(t_on / FPS), "t_off": _r(t_off / FPS), "frames": t_off - t_on},
                    "box_rest": {"x": _r(x / W * 100, 1), "y": _r(y / H * 100, 1), "w": _r(w / W * 100, 1), "h": _r(h / H * 100, 1)},
                    "in_unsafe_zone": bool(y / H < 0.12 or (y + h) / H > 0.80), "graphic_frac": 1.0,
                    "motion": {"in": mi, "hold": {"kind": "still"}, "out": mo},
                    "track": [{"t": _r(f["t"] / FPS), "scale": _r(f["scale"]), "dx": f["dx"], "dy": f["dy"], "edge_score": _r(f["score"], 2)}
                              for f in seq_in + seq_out], "rest_edge_score": _r(m["rest_score"], 2)})
    log(f"cards: {len(out)} measured ({time.time() - t1:.0f}s)")
    return out


# =====================================================================================
# 6. text (Apple Vision OCR on sampled full frames)
# =====================================================================================

def ocr_frames(frames: np.ndarray, every_s: float, log: Log) -> list[dict]:
    tool = TOOLS / "ocr"
    if not tool.exists():
        log("ocr tool missing (swiftc tools/ocr.swift -o tools/ocr); skipping text")
        return []
    import cv2
    step = max(1, int(every_s * FPS))
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for t in range(0, len(frames), step):
            p = f"{d}/{t:06d}.png"
            cv2.imwrite(p, cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR))
            paths.append(p)
        out = []
        for i in range(0, len(paths), 64):
            r = subprocess.run([str(tool)] + paths[i:i + 64], capture_output=True, text=True)
            for line in r.stdout.splitlines():
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = int(Path(j["file"]).stem)
                for ln in j.get("lines", []):
                    if ln.get("conf", 0) < 0.3 or len(ln["text"].strip()) < 2:
                        continue
                    out.append({"frame": t, "t": _r(t / FPS), "text": ln["text"], "conf": _r(ln["conf"], 2),
                                "box": {"x": _r(ln["box"][0] * 100, 1), "y": _r(ln["box"][1] * 100, 1), "w": _r(ln["box"][2] * 100, 1), "h": _r(ln["box"][3] * 100, 1)},
                                "words": [{"text": w["text"], "box": _r([v * 100 for v in w["box"]], 1)} for w in ln.get("words", [])]})
    log(f"ocr: {len(out)} text lines over {len(paths)} sampled frames")
    return out


def text_elements(ocr: list[dict], every_s: float) -> list[dict]:
    """Group OCR lines into text elements: same box (IoU>0.5) across consecutive samples = one element; text changes are events."""
    els: list[dict] = []
    for ln in sorted(ocr, key=lambda l: l["frame"]):
        b = (ln["box"]["x"], ln["box"]["y"], ln["box"]["w"], ln["box"]["h"])
        for el in els:
            if ln["t"] - el["t_off"] <= every_s * 1.5 and _iou(el["box"], b) > 0.45:
                el["t_off"] = ln["t"] + every_s
                if ln["text"] != el["texts"][-1]["text"]:
                    el["texts"].append({"t": ln["t"], "text": ln["text"]})
                el["samples"] += 1
                break
        else:
            els.append({"box": b, "t_on": ln["t"], "t_off": ln["t"] + every_s, "texts": [{"t": ln["t"], "text": ln["text"]}], "samples": 1,
                        "size_pct_of_height": ln["box"]["h"], "words": ln.get("words", [])})
    out = []
    for el in els:
        if el["samples"] < 2 and el["t_off"] - el["t_on"] < every_s * 2:
            continue
        if el["size_pct_of_height"] < 1.6:
            continue                                  # screen-recording body text (a GitHub page, a terminal), not an overlay
        txt = " ".join(t["text"] for t in el["texts"])
        kind = "caption" if len(el["texts"]) > 2 and el["box"][3] < 12 else "title"
        if len(el["texts"]) > 1 and all(t["text"].strip().replace(",", "").replace("%", "").replace("$", "").isdigit() for t in el["texts"]):
            kind = "counter"
        case = "upper" if txt.isupper() else "lower" if txt.islower() else "title" if txt.istitle() else "mixed"
        out.append({"kind": kind, "text": el["texts"][0]["text"], "text_events": el["texts"], "life": {"t_on": el["t_on"], "t_off": _r(el["t_off"])},
                    "box_rest": {"x": el["box"][0], "y": el["box"][1], "w": el["box"][2], "h": el["box"][3]},
                    "typography": {"size_pct_of_height": el["size_pct_of_height"], "case": case, "line_count": 1}})
    return out


def by_id_shot(shots: list[dict], t: float) -> dict:
    return next((sh for sh in shots if sh["start_time"] <= t < sh["end_time"]), shots[-1] if shots else {})


def text_track(frames: np.ndarray, bg: np.ndarray, tx: dict, every_s: float) -> list[dict]:
    """Per-frame observations of a text element inside its (padded) OCR box, so text gets the same motion analysis."""
    import cv2
    H, W = frames.shape[1:3]
    b = tx["box_rest"]
    pad = 0.02
    x0, y0 = int(max(0, (b["x"] / 100 - pad) * W)), int(max(0, (b["y"] / 100 - pad * 1.5) * H))
    x1, y1 = int(min(W, (b["x"] + b["w"]) / 100 * W + pad * W)), int(min(H, (b["y"] + b["h"]) / 100 * H + pad * 1.5 * H))
    if x1 - x0 < 6 or y1 - y0 < 4:
        return []
    f0, f1 = int(max(0, (tx["life"]["t_on"] - every_s - 0.1) * FPS)), int(min(len(frames), (tx["life"]["t_off"] + 0.1) * FPS))
    bgc = bg[y0:y1, x0:x1].astype(np.int16)
    obs = []
    for t in range(f0, f1):
        m = np.abs(frames[t, y0:y1, x0:x1].astype(np.int16) - bgc).max(axis=2) > 40
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        a = int(m.sum())
        if a < 0.002 * (x1 - x0) * (y1 - y0):
            if obs:
                break
            continue
        bb = M.bbox(m)
        if not bb:
            continue
        ys, xs = np.nonzero(m)
        obs.append({"box": (x0 + int(bb[0]), y0 + int(bb[1]), int(bb[2] - bb[0]), int(bb[3] - bb[1])), "area": a,
                    "c": (x0 + float(xs.mean()), y0 + float(ys.mean())), "t": t,
                    "color": frames[t, y0:y1, x0:x1][m].astype(np.float32).mean(0), "src": "text"})
    return obs if len(obs) >= MIN_LIFE else []


# =====================================================================================
# 7. words (mlx-whisper)
# =====================================================================================

def transcribe(video: str, log: Log, max_seconds: float | None = None) -> list[dict]:
    try:
        import mlx_whisper
    except ImportError:
        log("mlx_whisper not installed; no transcript")
        return []
    log("transcribing (mlx-whisper large-v3-turbo, on-device)")
    src = video
    if max_seconds:                                   # only the measured window: a long reference is not heard in full
        tmp = Path(tempfile.mkdtemp()) / "window.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-t", f"{max_seconds}", "-i", video, "-vn", "-ac", "1", "-ar", "16000",
                        str(tmp)], check=True)
        src = str(tmp)
    r = mlx_whisper.transcribe(src, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", word_timestamps=True)
    segs = []
    for sgm in r.get("segments", []):
        segs.append({"start_time": _r(sgm["start"]), "end_time": _r(sgm["end"]), "text": sgm["text"].strip(),
                     "word_times": [{"word": w["word"].strip(), "start_time": _r(w["start"]), "end_time": _r(w["end"])} for w in sgm.get("words", [])]})
    return segs


# =====================================================================================
# 8. density + tier + beats
# =====================================================================================

def density(elements: list[dict], shots: list[dict], duration: float) -> dict:
    n = max(1, int(duration * FPS))
    cover = np.zeros(n, bool); stack = np.zeros(n, np.int16)
    for el in elements:
        a, b = int(el["life"]["t_on"] * FPS), min(n, int(el["life"]["t_off"] * FPS))
        cover[a:b] = True; stack[a:b] += 1
    lens = [s["end_time"] - s["start_time"] for s in shots]
    kinds = {}
    for el in elements:
        k = el.get("motion", {}).get("in", {}).get("kind", el.get("kind"))
        kinds[k] = kinds.get(k, 0) + 1
    return {"elements_per_min": _r(len(elements) / max(duration, 1) * 60, 1), "overlay_coverage_pct": _r(cover.mean() * 100, 1),
            "max_simultaneous": int(stack.max()) if n else 0, "median_simultaneous_when_any": _r(float(np.median(stack[cover])) if cover.any() else 0, 1),
            "cuts_per_min": _r((len(shots) - 1) / max(duration, 1) * 60, 1), "avg_shot_s": _r(np.mean(lens) if lens else duration),
            "transition_kinds": sorted({s["transition_in"]["kind"] for s in shots[1:]}), "entry_kinds": kinds}


def tier(dens: dict, layers: list[dict], elements: list[dict]) -> dict:
    pts, ev = 0, []
    ent = {k for k in dens["entry_kinds"] if k not in ("pop", "cut", "grow")}
    if dens["elements_per_min"] >= 12: pts += 20; ev.append(f"{dens['elements_per_min']} elements/min")
    elif dens["elements_per_min"] >= 5: pts += 10
    if dens["overlay_coverage_pct"] >= 60: pts += 15; ev.append(f"overlays on screen {dens['overlay_coverage_pct']}% of runtime")
    if len(ent) >= 2: pts += 15; ev.append("animated entries: " + ", ".join(sorted(ent)))
    if any(el.get("motion", {}).get("hold", {}).get("kind") not in (None, "still") for el in elements): pts += 10; ev.append("elements keep moving while held")
    if any(k not in ("cut", "start") for k in dens["transition_kinds"]): pts += 10; ev.append("non-cut transitions: " + ", ".join(dens["transition_kinds"]))
    if any(l["subject"].get("presentation") in ("cutout", "likely_cutout") for l in layers): pts += 10; ev.append("speaker cut out over a replaced background")
    if any(abs(l["background"]["motion"]["zoom_pct_per_s"]) > 0.5 or abs(l["background"]["motion"]["pan_x_pct_per_s"]) > 0.5 for l in layers): pts += 10; ev.append("background moves (ken burns / parallax)")
    if dens["cuts_per_min"] >= 20: pts += 10; ev.append(f"{dens['cuts_per_min']} cuts/min")
    missing = []
    if dens["elements_per_min"] < 12: missing.append("more overlays (target >= 12/min)")
    if len(ent) < 2: missing.append("a second entry animation kind (slide + scale, or fade)")
    if not any(k not in ("cut", "start") for k in dens["transition_kinds"]): missing.append("at least one non-cut transition")
    if not any(abs(l["background"]["motion"]["zoom_pct_per_s"]) > 0.5 for l in layers): missing.append("slow push-in on the background/footage")
    return {"score": min(100, pts), "label": "high" if pts >= 65 else "mid" if pts >= 35 else "low", "evidence": ev, "missing_vs_high_end": missing}


def beats(elements: list[dict], transcript: list[dict]) -> list[dict]:
    words = [(w["start_time"], w["end_time"], w["word"]) for sgm in transcript for w in sgm.get("word_times", [])]
    out = []
    for el in elements:
        t = el["life"]["t_on"]
        near = [(abs(t - ws), ws, we, w) for ws, we, w in words if -1.0 <= t - ws <= 2.0]
        if not near:
            continue
        _, ws, we, w = min(near)
        out.append({"element": el["id"], "word": w, "word_time": ws, "offset_from_word_s": _r(t - ws), "response": el.get("text") or el.get("kind")})
    return out


# =====================================================================================
# 9. Gemini naming pass
# =====================================================================================

NAMING_PROMPT = """You are naming what a pixel-level analyzer already MEASURED in this video. The JSON below is ground truth for
timing, positions and motion; do not change any number. Your job is the parts pixels cannot know.

Return ONE JSON object:
{
 "format": {"content_type": "talking_head|tutorial|ad|vlog|other", "platform_guess": "reels|shorts|tiktok|youtube", "hook_type": "question|bold_claim|pattern_interrupt|visual_shock|curiosity_gap|other", "hook_structure": "how the first 3s are built"},
 "shots": [{"id": "s1", "shot_type": "talking_head|broll_full|broll_card|screen_recording|text_card|product|other", "content": "what is shown, 1 line", "is_broll": false, "site_or_source": null}],
 "elements": [{"id": "e1", "is_artifact": false, "depicts": "what it is", "asset_kind": "caption|title|callout|lottie|svg_icon|brand_logo|stock_photo|stock_video|screenshot|illustration|emoji|shape|chart|counter|arrow|other", "purpose": "why the editor put it there", "sourcing_hint": "search phrase to find an equivalent", "trigger_phrase": "spoken words it responds to or null", "style_notes": "font/colour/treatment in words"}],
 "asset_inventory": [{"id": "a1", "depicts": "...", "sourcing_kind": "...", "sourcing_hint": "...", "art_style": "...", "palette": ["#hex"], "elements": ["e3"]}],
 "motion_graphics": [{"name": "...", "device": "counter|progress_bar|pie_fill|arrow_draw_on|highlight_sweep|kinetic_type|mask_reveal|other", "elements": ["e5"], "how_to_rebuild": "one sentence"}],
 "tier": {"label": "low|mid|high", "why": "2 sentences grounded in the measured numbers", "what_a_higher_end_version_adds": ["..."]},
 "edit_grammar": ["10-15 imperative rules a new editor could apply to different footage; cite element/shot ids in brackets"],
 "speech_analysis": {"tone": "...", "delivery": "to_camera|voiceover|interview", "words_per_minute": 0, "transcript_summary": "2 sentences"}
}
Rules: quote on-screen text exactly as measured; every element id in the input must appear in "elements"; null beats a guess.
The analyzer is pixel-based and sometimes reports things that are NOT designed overlays: a patch of skin or shirt, a studio
light, a shadow, part of the background, a sliver at the frame edge, OCR noise from a screen recording. Look at the video at
that time and box: if nothing was deliberately added there by the editor, set "is_artifact": true (still give "depicts" saying
what it really is). Never write edit_grammar rules or motion_graphics from artifacts.

MEASURED:
"""


def _naming_schema():
    """Typed response schema: flash-lite otherwise echoes the input keys and skips the fields we need."""
    from typing import Optional
    from pydantic import BaseModel

    class Fmt(BaseModel):
        content_type: str; platform_guess: str; hook_type: str; hook_structure: str

    class Shot(BaseModel):
        id: str; shot_type: str; content: str; is_broll: bool; site_or_source: Optional[str] = None

    class El(BaseModel):
        id: str; is_artifact: bool; depicts: str; asset_kind: str; purpose: str
        sourcing_hint: Optional[str] = None; trigger_phrase: Optional[str] = None; style_notes: Optional[str] = None

    class Asset(BaseModel):
        id: str; depicts: str; sourcing_kind: str; sourcing_hint: str; art_style: str; palette: list[str]; elements: list[str]

    class MG(BaseModel):
        name: str; device: str; elements: list[str]; how_to_rebuild: str

    class Tier(BaseModel):
        label: str; why: str; what_a_higher_end_version_adds: list[str]

    class Speech(BaseModel):
        tone: str; delivery: str; words_per_minute: int; transcript_summary: str

    class Naming(BaseModel):
        format: Fmt; shots: list[Shot]; elements: list[El]; asset_inventory: list[Asset]; motion_graphics: list[MG]
        tier: Tier; edit_grammar: list[str]; speech_analysis: Speech
    return Naming


def gemini_name(video: str, measured: dict, log: Log) -> dict | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        log("GEMINI_API_KEY not set; skipping naming pass")
        return None
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)
    log(f"gemini: uploading {Path(video).name}")
    f = client.files.upload(file=video)
    while f.state.name == "PROCESSING":
        time.sleep(3)
        f = client.files.get(name=f.name)
    if f.state.name != "ACTIVE":
        log(f"gemini upload state {f.state.name}; skipping")
        return None
    slim = {k: v for k, v in measured.items() if k in ("metadata", "shots", "elements", "transcript", "density", "tier_measured")}
    slim["elements"] = [{k: v for k, v in el.items() if k != "track"} for el in slim.get("elements", [])]
    resp, used = None, None
    for model in GEMINI_MODELS:
        log(f"gemini: {model} naming {len(slim['elements'])} elements")
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type), NAMING_PROMPT + json.dumps(slim, separators=(",", ":"))],
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=_naming_schema(), temperature=0.2))
            used = model
            break
        except Exception as ex:                        # noqa: BLE001
            msg = str(ex)
            log(f"gemini {model}: {msg[:140]}")
            if "credits are depleted" in msg or "API key" in msg:
                break                                  # billing / auth: the next model will not help
    if resp is None:
        return None
    try:
        client.files.delete(name=f.name)
    except Exception:                                 # noqa: BLE001
        pass
    usage = getattr(resp, "usage_metadata", None)
    out = json.loads(resp.text)
    out["_usage"] = {"prompt_tokens": getattr(usage, "prompt_token_count", None), "output_tokens": getattr(usage, "candidates_token_count", None), "model": used}
    return out


# =====================================================================================
# 10. driver + conversion to the analysis schema
# =====================================================================================

def analyze(video: str, name: str | None = None, max_seconds: float = 240.0, ocr_every_s: float = 0.25, use_gemini: bool = True,
            log: Log = print) -> tuple[dict, dict]:
    t0 = time.time()
    frames, masks, info = prepare(video, max_seconds, log)
    dur = len(frames) / FPS
    shots = find_shots(frames, log)
    layers, elements, bgs, holes = [], [], {}, {}
    t1 = time.time()
    for sh in shots:
        lay = analyze_shot_layers(frames, masks, sh)
        bgs[sh["id"]] = lay.pop("_bg")
        holes[sh["id"]] = lay.pop("_hole")
        sh.update({"subject": lay["subject"], "background": lay["background"], "camera": lay["camera"]})
        layers.append(lay)
    # ponytail: no cross-shot background sharing; sibling shots are usually reframed a little and the lights ghost.
    # Overlays over the person come from the silhouette-hole / graphic / OCR cues instead.
    log(f"layers {time.time() - t1:.0f}s")
    cards = card_elements(frames, shots, log)
    t1 = time.time()
    describe_motion._bounds = [sh["end_frame"] for sh in shots]
    tracks = track_elements(frames, masks, shots, bgs, log)
    log(f"tracking {time.time() - t1:.0f}s")

    def _under_card(tr):
        t0, t1_ = tr["obs"][0]["t"] / FPS, tr["obs"][-1]["t"] / FPS
        b = tr["obs"][len(tr["obs"]) // 2]["box"]
        bx = (b[0] / W * 100, b[1] / H * 100, b[2] / W * 100, b[3] / H * 100)
        for c in cards:
            cb = (c["box_rest"]["x"], c["box_rest"]["y"], c["box_rest"]["w"], c["box_rest"]["h"])
            if c["life"]["t_on"] - 0.2 <= t0 and t1_ <= c["life"]["t_off"] + 0.2 and _contained(bx, cb) > 0.6:
                return True
        return False
    H, W = frames.shape[1:3]
    tracks = [tr for tr in tracks if not _under_card(tr)]
    elements.extend(cards)
    t1 = time.time()
    by_id = {sh["id"]: sh for sh in shots}
    for tr in tracks:
        sh = by_id[tr["shot"]]
        try:
            el = describe_motion(tr, frames, bgs[sh["id"]], sh)
        except Exception as ex:                        # noqa: BLE001
            log(f"element in {sh['id']} skipped: {ex}")
            continue
        el.update({"id": f"e{len(elements) + 1}", "shot": sh["id"], "kind": "graphic", "graphic_frac": tr["graphic_frac"]})
        elements.append(el)
    log(f"{len(elements)} graphic elements measured ({time.time() - t1:.0f}s)")
    ocr = ocr_frames(frames, ocr_every_s, log)
    texts = text_elements(ocr, ocr_every_s)
    # a shot with a wall of small text is a screen recording: record the density on the shot, keep only big lines
    for sh in shots:
        lines = [l for l in ocr if sh["start_time"] <= l["t"] < sh["end_time"]]
        per_sample = len(lines) / max(1, (sh["end_time"] - sh["start_time"]) / ocr_every_s)
        sh["screen_text"] = {"lines_per_frame": _r(per_sample, 1), "is_screen_recording": per_sample >= 8}
    texts = [tx for tx in texts if not (by_id_shot(shots, tx["life"]["t_on"]).get("screen_text", {}).get("is_screen_recording") and tx["typography"]["size_pct_of_height"] < 3.0)]
    # attach text to the graphic element it sits in, else make a text element
    for tx in texts:
        b = (tx["box_rest"]["x"], tx["box_rest"]["y"], tx["box_rest"]["w"], tx["box_rest"]["h"])
        host = None
        for el in elements:
            eb = (el["box_rest"]["x"], el["box_rest"]["y"], el["box_rest"]["w"], el["box_rest"]["h"])
            if el["life"]["t_on"] - 0.3 <= tx["life"]["t_on"] <= el["life"]["t_off"] and (_iou(eb, b) > 0.3 or _contained(b, eb) > 0.8):
                host = el; break
        if host and host.get("kind") == "card":
            host.setdefault("texts_inside", []).append({"text": tx["text"], "t_on": tx["life"]["t_on"], "box": tx["box_rest"]})
        elif host and not host.get("text"):
            host.update({"kind": tx["kind"], "text": tx["text"], "text_events": tx["text_events"], "typography": tx["typography"]})
        else:
            sid = next((s["id"] for s in shots if s["start_time"] <= tx["life"]["t_on"] < s["end_time"]), "s1")
            sh = by_id[sid]
            tr = text_track(frames, bgs[sid], tx, ocr_every_s)
            measured = None
            if tr:
                try:
                    measured = describe_motion({"obs": tr}, frames, bgs[sid], sh)
                except Exception:                      # noqa: BLE001
                    measured = None
            if measured:
                tx["motion"], tx["track"] = measured["motion"], measured["track"]
                tx["life"] = measured["life"]
            else:
                tx["motion"] = {"in": {"kind": "unmeasured_text", "duration_s": None}, "hold": {"kind": "still"}, "out": {"kind": "unmeasured_text", "leaves_via": "cut"}}
            tx.update({"id": f"e{len(elements) + 1}", "shot": sid,
                       "in_unsafe_zone": bool(tx["box_rest"]["y"] < 12 or tx["box_rest"]["y"] + tx["box_rest"]["h"] > 80)})
            elements.append(tx)
    elements.sort(key=lambda e: e["life"]["t_on"])
    for i, el in enumerate(elements):
        el["id"] = f"e{i + 1}"
    transcript = transcribe(video, log, dur)
    dens = density(elements, shots, dur)
    tr = tier(dens, layers, elements)
    from .broll_local import silence_cuts_local
    deep = {"schema": "deep/1", "built_at": _now(), "source": video, "name": name or Path(video).stem,
            "metadata": {"total_duration_seconds": _r(dur), "analyzed_seconds": _r(dur), "fps": FPS, "width": info["width"], "height": info["height"],
                         "aspect_ratio": "9:16" if info["height"] > info["width"] else "16:9", "person_mask": masks is not None},
            "shots": shots, "elements": elements, "transcript": transcript, "silence_cuts": silence_cuts_local(video, max_seconds=dur), "density": dens,
            "tier_measured": tr, "beats": beats(elements, transcript), "ocr_samples": len(ocr), "seconds_to_measure": _r(time.time() - t0, 1)}
    if use_gemini:
        try:
            deep["naming"] = gemini_name(video, deep, log)
        except Exception as ex:                        # noqa: BLE001
            log(f"gemini naming failed: {str(ex)[:200]}")
            deep["naming"] = None
    return deep, to_analysis(deep)


_KIND_IN = {"slide": "slide", "scale": "scale", "fade": "fade", "pop": "pop", "grow": "wipe", "wipe": "wipe", "wipe_soft": "wipe",
            "typewriter": "typewriter", "custom_mask": "wipe", "slide_from_behind": "slide", "unmeasured_text": "none"}


def _dir_suffix(m: dict) -> str:
    d = (m.get("direction") or "").replace("from_", "")
    return {"left": "slide_right", "right": "slide_left", "top": "slide_down", "bottom": "slide_up"}.get(d, "slide_up")


def to_analysis(deep: dict) -> dict:
    """The schema style.build / plan / conform already read, filled from measurements (+ Gemini names when present)."""
    nm = deep.get("naming") or {}
    names = {e["id"]: e for e in nm.get("elements", []) if isinstance(e, dict) and "id" in e}
    artifacts = {i for i, e in names.items() if e.get("is_artifact")}
    shot_names = {s["id"]: s for s in nm.get("shots", []) if isinstance(s, dict) and "id" in s}
    overlays, broll = [], []
    for el in deep["elements"]:
        if el["id"] in artifacts:
            continue
        mi, mh, mo = el["motion"]["in"], el["motion"]["hold"], el["motion"]["out"]
        n = names.get(el["id"], {})
        kind_in = _KIND_IN.get(mi.get("kind"), "none")
        ain = _dir_suffix(mi) if kind_in == "slide" else kind_in
        aout = ("slide_" + mo["leaves_via"].split("_")[-1]) if mo.get("leaves_via", "").startswith(("edge_", "slide_")) else \
               {"fade": "fade", "shrink": "scale", "cut": "cut", "covered_by_cut": "cut"}.get(mo.get("leaves_via"), "cut")
        typ = el.get("kind") if el.get("kind") in ("caption", "title", "counter") else \
              {"caption": "caption", "title": "title", "callout": "callout", "brand_logo": "logo", "emoji": "emoji", "shape": "shape", "arrow": "callout",
               "counter": "callout", "chart": "callout"}.get(n.get("asset_kind"), "callout")
        overlays.append({
            "id": el["id"], "start_time": el["life"]["t_on"], "end_time": el["life"]["t_off"], "type": typ,
            "text_content": el.get("text"), "asset_description": n.get("depicts"), "asset_ref": None,
            "spatial_box": el["box_rest"], "animation_in": ain, "animation_out": aout,
            "motion": {"in": {"kind": ain, "from_dx": mi.get("dx_pct", 0.0), "from_dy": mi.get("dy_pct", 0.0), "from_scale": mi.get("scale0", 1.0),
                              "from_opacity": mi.get("opacity0", 1.0), "from_rotation_deg": 0.0, "duration_seconds": mi.get("duration_s") or 0.0,
                              "easing": mi.get("easing", "linear"), "overshoot_pct": mi.get("overshoot_pct", 0.0), "curve": mi.get("curve"),
                              "stop_style": mi.get("stop_style")},
                       "hold": {"kind": mh.get("kind", "still"), "amplitude_pct": mh.get("amplitude_pct"), "cycle_seconds": mh.get("cycle_s"),
                                "rate_pct_per_second": mh.get("rate_pct_per_s")},
                       "out": {"kind": aout, "to_dx": mo.get("dx_pct"), "to_dy": mo.get("dy_pct"), "to_scale": mo.get("scale0"), "to_opacity": mo.get("opacity0"),
                               "to_rotation_deg": 0.0, "duration_seconds": mo.get("duration_s"), "easing": mo.get("easing", "linear"), "curve": mo.get("curve"),
                               "leaves_via": mo.get("leaves_via")}},
            "typography": el.get("typography"), "background_color": el.get("colour"), "colors": [el.get("colour")] if el.get("colour") else [],
            "shape": "none", "purpose": n.get("purpose"), "sourcing_hint": n.get("sourcing_hint"), "in_unsafe_zone": el.get("in_unsafe_zone", False),
            "focus_match": False, "confidence": 0.9 if mi.get("kind") != "unmeasured_text" else 0.6, "estimated": False, "measured": True})
    cuts = []
    for s in deep["shots"]:
        sn = shot_names.get(s["id"], {})
        st = sn.get("shot_type") or ("talking_head" if s["subject"].get("present") else "broll")
        is_b = bool(sn.get("is_broll")) or (not s["subject"].get("present"))
        cuts.append({"id": s["id"], "start_time": s["start_time"], "end_time": s["end_time"],
                     "transition_in": {"cut": "hard", "dissolve": "dissolve", "start": "none"}.get(s["transition_in"]["kind"], s["transition_in"]["kind"]),
                     "transition_duration_seconds": s["transition_in"].get("duration_s", 0.0),
                     "shot_type": st, "is_broll": is_b, "camera_motion": s["camera"]["motion"], "speed_ramp": None,
                     "description": sn.get("content"), "subject": s["subject"], "background": s["background"], "confidence": 0.9, "estimated": False})
        if is_b:
            broll.append({"start_time": s["start_time"], "end_time": s["end_time"], "duration_seconds": _r(s["end_time"] - s["start_time"]),
                          "presentation": "full_frame", "box": {"x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0}, "speaker_visible": False,
                          "content": {"shows": sn.get("content"), "site": sn.get("site_or_source")},
                          "enter": {"kind": s["transition_in"]["kind"], "duration_seconds": s["transition_in"].get("duration_s", 0.0)},
                          "hold": {"kind": "ken_burns" if abs(s["background"]["motion"]["zoom_pct_per_s"]) > 0.5 else "static",
                                   "rate_pct_per_second": s["background"]["motion"]["zoom_pct_per_s"]}, "exit": {"kind": "cut"}, "measured": True})
    lens = [c["end_time"] - c["start_time"] for c in cuts]
    dur = deep["metadata"]["total_duration_seconds"]
    beat_map = []
    from .plan import classify
    segs = deep.get("transcript") or []
    for b in deep["beats"]:
        el = next((o for o in overlays if o["id"] == b["element"]), None)
        if el is None:
            continue                                  # artifact or dropped element: no beat
        line = next((sg["text"] for sg in segs if sg["start_time"] - 0.05 <= b["word_time"] <= sg["end_time"] + 0.05), b["word"])
        kind, phrase = classify(line)
        beat_map.append({"trigger_phrase": phrase or b["word"], "trigger_kind": kind if kind != "none" else "other", "trigger_time": b["word_time"], "response": b["response"],
                         "response_overlay_index": next((i for i, o in enumerate(overlays) if o["id"] == b["element"]), None),
                         "offset_from_trigger_seconds": b["offset_from_word_s"],
                         "response_duration_seconds": _r(el["end_time"] - el["start_time"]) if el else None, "confidence": 0.8})
    fmt = nm.get("format", {})
    return {
        "metadata": {**deep["metadata"], "platform_guess": fmt.get("platform_guess"), "content_type": fmt.get("content_type"), "language": "en"},
        "hook_analysis": {"start_time": 0.0, "end_time": min(3.0, dur), "hook_type": fmt.get("hook_type"), "structure_description": fmt.get("hook_structure"),
                          "text_on_screen": next((o["text_content"] for o in overlays if o["start_time"] < 1.0 and o["text_content"]), None),
                          "spoken_line": (deep["transcript"][0]["text"] if deep["transcript"] else None), "confidence": 0.7},
        "pacing_metrics": {"total_cuts": max(0, len(cuts) - 1), "average_shot_length_seconds": _r(np.mean(lens) if lens else dur),
                           "cuts_per_minute": deep["density"]["cuts_per_min"], "curve_shape": None, "shot_lengths": _r(lens), "confidence": 0.95},
        "silence_cuts": deep["silence_cuts"], "scene_cuts": cuts, "graphic_overlays": overlays, "transcript": deep["transcript"],
        "speech_analysis": {**(nm.get("speech_analysis") or {}), "has_speech": bool(deep["transcript"])},
        "asset_inventory": nm.get("asset_inventory", []), "motion_graphics": nm.get("motion_graphics", []), "beat_map": beat_map,
        "structural_beats": [], "edit_grammar": nm.get("edit_grammar", []), "broll_shots": broll,
        "density": deep["density"], "tier": {**deep["tier_measured"], "model": nm.get("tier")},
        "artifacts_dropped": sorted(artifacts),
        "_completeness": {"complete": True, "problems": [], "measured_locally": True},
        "_warnings": [], "_vidkit": {"source": deep["source"], "prompt": "deep", "mode": "local+gemini" if nm else "local", "model": GEMINI_MODEL if nm else None,
                                     "container": {"duration": dur, "width": deep["metadata"]["width"], "height": deep["metadata"]["height"], "fps": FPS}},
    }


def save(deep: dict, analysis: dict, name: str) -> tuple[Path, Path]:
    d = config.dirs()["analyses"]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = "".join(c if c.isalnum() or c in "-_" else "-" for c in name.lower())
    p_deep, p_an = d / f"{stem}-{ts}.deep.json", d / f"{stem}-{ts}.analysis.json"
    p_deep.write_text(json.dumps(deep, indent=1))
    p_an.write_text(json.dumps(analysis, indent=1))
    return p_deep, p_an

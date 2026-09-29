"""Colour: measure a reference's actual colour, solve the ASC CDL that lands a clip on it, bake it into a .cube.

Why CDL: it IS the primary corrector (slope/offset/power per channel + saturation). Solved from percentiles it lands
the black point, midtone and white point of the source on the reference's, per channel. What it cannot copy is a
different curve SHAPE (a split tone that only touches the mids); the style's notes say so when the match error is
large. The same .cube goes into Resolve (SetLUT) and into the headless render (ffmpeg lut3d), so both agree.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import media

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _hsv(rgb: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2HSV).reshape(-1, 3)


def sample_frames(file: Path, n: int = 6, spans: list[tuple[float, float]] | None = None) -> list[np.ndarray]:
    """n frames spread across the file (or across `spans`), as float RGB 0-1 at 320 px wide."""
    dur = media.probe(file)["duration"]
    spans = spans or [(0.0, dur)]
    total = sum(b - a for a, b in spans) or dur
    out = []
    for k in range(n):
        want, acc = total * (k + 0.5) / n, 0.0
        for a, b in spans:
            if acc + (b - a) >= want:
                t = a + want - acc
                break
            acc += b - a
        else:
            t = dur / 2
        fr = media.grab_frame(file, min(max(0.0, t), max(0.0, dur - 0.05)), 320)
        if fr is not None:
            out.append(fr.astype(np.float32) / 255.0)
    if not out:
        raise SystemExit(f"could not read any frame from {file}")
    return out


def measure(frames: list[np.ndarray]) -> dict:
    """Everything a grade decision needs, measured."""
    px = np.concatenate([f.reshape(-1, 3) for f in frames], 0)
    luma = px @ LUMA
    hsv = _hsv(px)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    lo, mid, hi = (float(np.percentile(luma, q)) for q in (2, 50, 98))
    skin = ((h < 50) | (h > 340)) & (s > 0.15) & (s < 0.68) & (v > 0.18) & (v < 0.95)
    return {
        "black_2pct": round(lo, 4), "mid_50pct": round(mid, 4), "white_98pct": round(hi, 4),
        "mean_luma": round(float(luma.mean()), 4), "contrast_98_2": round(hi - lo, 4),
        "saturation_mean": round(float(s.mean()), 4),
        "warmth_r_minus_b": round(float((px[:, 0] - px[:, 2]).mean()), 4),
        "channel": {c: {"black": round(float(np.percentile(px[:, i], 2)), 4),
                        "mid": round(float(np.percentile(px[:, i], 50)), 4),
                        "white": round(float(np.percentile(px[:, i], 98)), 4)} for i, c in enumerate("rgb")},
        "skin": None if skin.sum() < 500 else {"hue_deg": round(float(h[skin].mean()), 1),
                                                "sat": round(float(s[skin].mean()), 3)},
    }


def solve_cdl(src: dict, ref: dict) -> dict:
    """Per channel: out = clip(in*slope + offset) ^ power, landing src's black/mid/white on ref's. For a given power
    the two endpoints give slope and offset in closed form, so power is the only free variable: found by a coarse
    scan then refinement on the midtone residual (plain bisection walks to an endpoint when no power reaches it)."""
    slope, offset, power = [], [], []
    for c in "rgb":
        s_, r_ = src["channel"][c], ref["channel"][c]
        bs, ms, ws = (max(1e-4, min(0.999, s_[k])) for k in ("black", "mid", "white"))
        br, mr, wr = (max(1e-4, min(0.999, r_[k])) for k in ("black", "mid", "white"))
        if ws - bs < 1e-3:
            slope.append(1.0); offset.append(br - bs); power.append(1.0)
            continue

        def fit(pw):
            B, Wt = br ** (1 / pw), wr ** (1 / pw)
            sl = (Wt - B) / (ws - bs)
            of = B - sl * bs
            return sl, of, max(1e-6, min(1.0, ms * sl + of)) ** pw - mr

        grid = [0.35 + i * (2.8 - 0.35) / 80 for i in range(81)]
        pw = min(grid, key=lambda q: abs(fit(q)[2]))
        step = (2.8 - 0.35) / 80
        for _ in range(40):
            step /= 2
            pw = min((max(0.35, pw - step), pw, min(2.8, pw + step)), key=lambda q: abs(fit(q)[2]))
        sl, of, _ = fit(pw)
        slope.append(float(np.clip(sl, 0.02, 8.0)))
        offset.append(float(np.clip(of, -0.4, 0.4)))
        power.append(float(np.clip(pw, 0.35, 2.8)))
    sat = float(np.clip(ref["saturation_mean"] / max(1e-3, src["saturation_mean"]), 0.3, 2.0))
    return {"slope": slope, "offset": offset, "power": power, "saturation": sat}


def soften(cdl: dict, strength: float) -> dict:
    """Blend a CDL toward identity: strength 1 = the full match, 0 = untouched."""
    k = max(0.0, min(1.0, strength))
    lerp = lambda a, b: a + (b - a) * k        # noqa: E731
    return {"slope": [lerp(1.0, v) for v in cdl["slope"]], "offset": [lerp(0.0, v) for v in cdl["offset"]],
            "power": [lerp(1.0, v) for v in cdl["power"]], "saturation": lerp(1.0, cdl["saturation"])}


def apply_cdl(rgb: np.ndarray, cdl: dict) -> np.ndarray:
    """rgb float 0-1 (..., 3) through the CDL (ASC order: slope, offset, clamp, power, then saturation on Rec.709 luma)."""
    sl, of, pw = (np.array(cdl[k], np.float32) for k in ("slope", "offset", "power"))
    x = np.clip(rgb * sl + of, 0.0, 1.0) ** pw
    y = (x @ LUMA)[..., None]
    return np.clip(y + (x - y) * cdl["saturation"], 0.0, 1.0)


def predicted_error(src: dict, ref: dict, cdl: dict) -> float:
    """Worst per-channel error (0-1) of where the CDL lands src's anchors vs the reference's."""
    err = 0.0
    for i, c in enumerate("rgb"):
        for k in ("black", "mid", "white"):
            v = np.clip(np.clip(src["channel"][c][k] * cdl["slope"][i] + cdl["offset"][i], 0, 1) ** cdl["power"][i], 0, 1)
            err = max(err, abs(float(v) - ref["channel"][c][k]))
    return round(err, 4)


def write_cube(cdl: dict, path: Path, size: int = 33, title: str = "video-style-kit") -> Path:
    """Bake the CDL into a 3D LUT (.cube, red varies fastest), readable by Resolve and ffmpeg lut3d."""
    g = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")            # r fastest when flattened in C order
    rgb = np.stack([r, gg, b], -1).reshape(-1, 3)
    out = apply_cdl(rgb, cdl)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    lines += [f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in out]
    path.write_text("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    # ponytail: the one runnable check: a CDL solved from stats lands the anchors, identity bakes to identity.
    rng = np.random.default_rng(0)
    src_px = rng.uniform(0.1, 0.7, (200, 200, 3)).astype(np.float32)
    ref_px = np.clip(src_px * 1.2 + 0.05, 0, 1) ** 1.1
    s, r = measure([src_px]), measure([ref_px])
    cdl = solve_cdl(s, r)
    assert predicted_error(s, r, cdl) < 0.02, predicted_error(s, r, cdl)
    ident = {"slope": [1, 1, 1], "offset": [0, 0, 0], "power": [1, 1, 1], "saturation": 1.0}
    x = rng.uniform(0, 1, (50, 3)).astype(np.float32)
    assert np.allclose(apply_cdl(x, ident), x, atol=1e-6)
    assert soften(cdl, 0.0)["slope"] == [1.0, 1.0, 1.0]
    print("grade.py self-check ok")

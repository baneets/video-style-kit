"""Parameterized hypothesis renderers + the reconstruction contest.

Every Hyp renders a frame for a progress value p in [0,1]. Fitting = per-frame argmin of residual over p
(coarse grid then refine), so each hypothesis gets its own optimal parameters before being scored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import measure as M

DIRS = {"right": (1, 0), "left": (-1, 0), "down": (0, 1), "up": (0, -1)}


# ---------- pixel ops ----------

def shift(img: np.ndarray, dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    """Integer translate with zero fill. Returns (shifted, valid_mask)."""
    H, W = img.shape[:2]
    out = np.zeros_like(img)
    valid = np.zeros((H, W), bool)
    xs0, xs1 = max(0, dx), min(W, W + dx)
    ys0, ys1 = max(0, dy), min(H, H + dy)
    if xs1 > xs0 and ys1 > ys0:
        out[ys0:ys1, xs0:xs1] = img[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
        valid[ys0:ys1, xs0:xs1] = True
    return out, valid


def warp(img: np.ndarray, Minv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear warp. Minv maps output homogeneous coords -> input coords. Returns (out float32, valid)."""
    H, W = img.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    ones = np.ones_like(xs)
    src = Minv @ np.stack([xs.ravel(), ys.ravel(), ones.ravel()])
    sx = (src[0] / src[2]).reshape(H, W)
    sy = (src[1] / src[2]).reshape(H, W)
    valid = (sx >= 0) & (sx <= W - 1) & (sy >= 0) & (sy <= H - 1)
    sx = np.clip(sx, 0, W - 1.001)
    sy = np.clip(sy, 0, H - 1.001)
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    im = img.astype(np.float32)
    if im.ndim == 2:
        im = im[..., None]
    out = (im[y0, x0] * (1 - fx) * (1 - fy) + im[y0, x0 + 1] * fx * (1 - fy)
           + im[y0 + 1, x0] * (1 - fx) * fy + im[y0 + 1, x0 + 1] * fx * fy)
    return (out if img.ndim == 3 else out[..., 0]), valid


def scale_matrix(s: float, cx: float, cy: float) -> np.ndarray:
    """output -> input for a scale by s about (cx, cy)."""
    s = max(s, 1e-3)
    return np.array([[1 / s, 0, cx - cx / s], [0, 1 / s, cy - cy / s], [0, 0, 1]], np.float32)


def rotation_matrix(theta: float, cx: float, cy: float) -> np.ndarray:
    c, s = np.cos(-theta), np.sin(-theta)
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float32)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], np.float32)
    return T2 @ R @ T1


def keystone_matrix(k: float, cx: float, cy: float, H: float, axis: str = "y") -> np.ndarray:
    """Perspective-like tilt: x scale varies linearly with y (axis='y') or y scale with x (axis='x'). output->input."""
    # forward: x' = cx + (x-cx)*(1 + k*(y-cy)/H)  -> a homography with a perspective row
    if axis == "y":
        Mf = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64)
        # perspective term: implement as projective matrix so it is a true homography
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float64)
        P = np.array([[1, 0, 0], [0, 1, 0], [0, k / H, 1]], np.float64)
        T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], np.float64)
        Mf = T2 @ P @ T1
    else:
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float64)
        P = np.array([[1, 0, 0], [0, 1, 0], [k / H, 0, 1]], np.float64)
        T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], np.float64)
        Mf = T2 @ P @ T1
    return np.linalg.inv(Mf).astype(np.float32)


def motion_blur(img: np.ndarray, angle: float, length: float) -> np.ndarray:
    n = max(int(round(length)), 1)
    if n <= 1:
        return img.astype(np.float32)
    acc = np.zeros(img.shape, np.float32)
    for i in range(n):
        t = i - (n - 1) / 2
        dx, dy = int(round(t * np.cos(angle))), int(round(t * np.sin(angle)))
        sh, _ = shift(img, dx, dy)
        # edge fill: reuse nearest content instead of black
        acc += np.where(_[..., None] if img.ndim == 3 else _, sh, img).astype(np.float32)
    return acc / n


def composite(bg: np.ndarray, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha.astype(np.float32)[..., None]
    return bg.astype(np.float32) * (1 - a) + fg.astype(np.float32) * a


def to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def proj(shape, d) -> np.ndarray:
    """Per-pixel coordinate along direction d (unit-ish)."""
    H, W = shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    return xs * d[0] + ys * d[1]


# ---------- hypotheses ----------

@dataclass
class Hyp:
    name: str
    render: Callable[[float], np.ndarray]     # p -> float32 frame (H,W,3)
    params: dict = field(default_factory=dict)
    simplicity: int = 5                        # lower = simpler; used only for the equivalence tie-break
    grid: np.ndarray = field(default_factory=lambda: np.linspace(0, 1, 17))
    label: str = ""                            # short human label for explain
    curve: str = "free"                        # "free": p fitted per frame; "linear": p = clip(rate*(t - t0)) fitted globally

    def __post_init__(self):
        self.label = self.label or self.name


@dataclass
class Obs:
    frames: np.ndarray            # (T,H,W,3) uint8, frames[0] = before, frames[-1] = after
    ref: np.ndarray               # "before" reference frame
    final: np.ndarray             # "after" frame
    emask: np.ndarray             # element mask on final (bool), or all-True for whole-frame transitions
    region: tuple | None = None   # (x0,y0,x1,y1) crop used for scoring


def _crop(a, region):
    if region is None:
        return a
    x0, y0, x1, y1 = region
    return a[y0:y1, x0:x1]


def fit_curve(hyp: Hyp, obs: Obs) -> tuple[np.ndarray, list[np.ndarray]]:
    """Optimal p per frame (coarse grid + refine), or a global linear-in-time curve. Returns (p_t, rendered)."""
    ps, rendered = [], []
    cache: dict[float, np.ndarray] = {}

    def R(p):
        key = round(float(p), 5)
        if key not in cache:
            cache[key] = hyp.render(float(p))
        return cache[key]

    if hyp.curve == "linear":
        T = len(obs.frames)
        obs_g = [M.gray(_crop(obs.frames[t], obs.region)) for t in range(1, T - 1)]
        n = max(len(hyp.grid) - 1, 1)
        best = (np.inf, None)
        for steps_per_frame in (0.25, 1 / 3, 0.5, 1.0, 2.0, 3.0):
            for t0 in range(-3, 4):
                ps_try = [min(1.0, max(0.0, round(steps_per_frame * (t - t0)) / n)) for t in range(1, T - 1)]
                e = sum(float(((M.gray(_crop(R(p), obs.region)) - g) ** 2).mean()) for p, g in zip(ps_try, obs_g))
                if e < best[0]:
                    best = (e, ps_try)
        hyp.params = dict(hyp.params, curve="linear")
        return np.array(best[1]), [R(p) for p in best[1]]

    for t in range(1, len(obs.frames) - 1):
        obs_g = M.gray(_crop(obs.frames[t], obs.region))
        best_p, best_e = 0.0, np.inf
        for p in hyp.grid:
            e = float(((M.gray(_crop(R(p), obs.region)) - obs_g) ** 2).mean())
            if e < best_e:
                best_p, best_e = float(p), e
        step = float(hyp.grid[1] - hyp.grid[0]) if len(hyp.grid) > 1 else 0
        if step > 0:
            for p in np.linspace(best_p - step, best_p + step, 9):
                if p < hyp.grid[0] or p > hyp.grid[-1]:
                    continue
                e = float(((M.gray(_crop(R(p), obs.region)) - obs_g) ** 2).mean())
                if e < best_e:
                    best_p, best_e = float(p), e
        ps.append(best_p)
        rendered.append(R(best_p))
    return np.array(ps), rendered


def score_frames(rendered: list[np.ndarray], obs: Obs, weights=(0.5, 0.3, 0.2)) -> dict:
    ss, ious, res = [], [], []
    ref_c = _crop(obs.ref, obs.region)
    for t, r in zip(range(1, len(obs.frames) - 1), rendered):
        o = _crop(obs.frames[t], obs.region)
        rc = to_u8(_crop(r, obs.region))
        ss.append(M.ssim(o, rc))
        ious.append(M.iou(M.diff_mask(o, ref_c), M.diff_mask(rc, ref_c)))
        res.append(min(1.0, M.rmse(o, rc) / 40.0))
    s, i, r = float(np.mean(ss)), float(np.mean(ious)), float(np.mean(res))
    return {"ssim": s, "iou": i, "residual": r,
            "combined": weights[0] * s + weights[1] * i + weights[2] * (1 - r)}


def contest(hyps: list[Hyp], obs: Obs, margin_threshold: float = 0.08, weights=(0.5, 0.3, 0.2)) -> dict:
    """Fit + render + score every hypothesis; rank; compute the relative margin between the top two."""
    results = {}
    for h in hyps:
        p, rendered = fit_curve(h, obs)
        sc = score_frames(rendered, obs, weights)
        results[h.name] = {"scores": sc, "curve": [round(float(x), 4) for x in p], "params": h.params,
                           "simplicity": h.simplicity, "label": h.label}
    order = sorted(results, key=lambda n: -results[n]["scores"]["combined"])
    # variants sharing a label (directions, softness) are one hypothesis: keep the best-fitting variant only
    best_of: dict[str, str] = {}
    for n in order:
        best_of.setdefault(results[n]["label"], n)
    keep = set(best_of.values())
    order = [n for n in order if n in keep]
    results = {n: v for n, v in results.items() if n in keep}
    best = results[order[0]]["scores"]["combined"]
    second = results[order[1]]["scores"]["combined"] if len(order) > 1 else 0.0
    margin = (best - second) / best if best > 0 else 0.0
    return {"ranking": order, "results": results, "margin": round(float(margin), 4),
            "ambiguous": bool(margin < margin_threshold), "margin_threshold": margin_threshold,
            "reconstruction_scores": {n: round(results[n]["scores"]["combined"], 4) for n in order}}

"""numpy measurement primitives: SSIM, IoU, block-matching optical flow, edge/blur/orientation stats."""
from __future__ import annotations

from collections import deque

import numpy as np

EPS = 1e-6


def gray(f: np.ndarray) -> np.ndarray:
    return f.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def box_filter(x: np.ndarray, r: int) -> np.ndarray:
    """Uniform (2r+1)^2 mean filter with reflect padding, via cumulative sums."""
    if r <= 0:
        return x.astype(np.float32)
    p = np.pad(x.astype(np.float32), r, mode="reflect")
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable gaussian on (H,W) or (H,W,C) float arrays."""
    if sigma < 0.15:
        return img.astype(np.float32)
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    x = img.astype(np.float32)
    pad = [(r, r), (r, r)] + [(0, 0)] * (x.ndim - 2)
    p = np.pad(x, pad, mode="reflect")
    out = np.zeros_like(p)
    for i, wgt in enumerate(k):
        out += wgt * np.roll(p, i - r, axis=0)
    out2 = np.zeros_like(out)
    for i, wgt in enumerate(k):
        out2 += wgt * np.roll(out, i - r, axis=1)
    return out2[r:-r, r:-r]


def ssim(a: np.ndarray, b: np.ndarray, r: int = 3) -> float:
    ga, gb = gray(a), gray(b)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = box_filter(ga, r), box_filter(gb, r)
    saa = box_filter(ga * ga, r) - mu_a**2
    sbb = box_filter(gb * gb, r) - mu_b**2
    sab = box_filter(ga * gb, r) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (saa + sbb + c2))
    return float(np.clip(s.mean(), -1, 1))


def iou(m1: np.ndarray, m2: np.ndarray) -> float:
    u = np.logical_or(m1, m2).sum()
    return 1.0 if u == 0 else float(np.logical_and(m1, m2).sum() / u)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((gray(a) - gray(b)) ** 2)))


def diff_mask(frame: np.ndarray, ref: np.ndarray, thresh: float = 28.0, clean: int = 1) -> np.ndarray:
    d = np.abs(frame.astype(np.int16) - ref.astype(np.int16)).max(axis=2) > thresh
    if clean:
        d = box_filter(d.astype(np.float32), clean) > 0.5
    return d


def bbox(mask: np.ndarray):
    """(x0, y0, x1, y1) exclusive, or None."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def centroid(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else None


def block_flow(a: np.ndarray, b: np.ndarray, block: int = 8, search: int = 6, mask: np.ndarray | None = None):
    """Block-matching flow a->b. Returns (flow (nby,nbx,2) in px [dx,dy], valid (nby,nbx), centers (nby,nbx,2))."""
    ga, gb = gray(a), gray(b)
    H, W = ga.shape
    nby, nbx = H // block, W // block
    ga = ga[: nby * block, : nbx * block]
    gbp = np.pad(gb, search, mode="edge")
    best = np.full((nby, nbx), np.inf, np.float32)
    flow = np.zeros((nby, nbx, 2), np.float32)
    blocks_a = ga.reshape(nby, block, nbx, block)
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            sh = gbp[search + dy: search + dy + nby * block, search + dx: search + dx + nbx * block]
            sad = np.abs(blocks_a - sh.reshape(nby, block, nbx, block)).mean(axis=(1, 3))
            better = sad < best - 1e-3
            best = np.where(better, sad, best)
            flow[better] = (dx, dy)
    texture = blocks_a.std(axis=(1, 3)) > 4.0
    valid = texture
    if mask is not None:
        m = mask[: nby * block, : nbx * block].reshape(nby, block, nbx, block).mean(axis=(1, 3)) > 0.95
        valid = valid & m
    ys, xs = np.mgrid[0:nby, 0:nbx]
    centers = np.stack([xs * block + block / 2, ys * block + block / 2], axis=-1).astype(np.float32)
    return flow, valid, centers


def mean_flow(a, b, mask=None, **kw):
    flow, valid, _ = block_flow(a, b, mask=mask, **kw)
    if valid.sum() == 0:
        return None
    return flow[valid].mean(axis=0)


def fit_similarity(flow, valid, centers):
    """flow ~= (s-1)(x - c). Returns (s, cx, cy, residual_px) or None."""
    if valid.sum() < 4:
        return None
    f = flow[valid].reshape(-1, 2)
    x = centers[valid].reshape(-1, 2)
    xm, fm = x.mean(axis=0), f.mean(axis=0)
    var = ((x - xm) ** 2).sum()
    if var < EPS:
        return None
    a = ((x - xm) * (f - fm)).sum() / var
    s = 1 + a
    if abs(a) < 1e-4:
        return 1.0, float(xm[0]), float(xm[1]), float(np.linalg.norm(f - fm, axis=1).mean())
    c = xm - fm / a
    res = np.linalg.norm(f - a * (x - c), axis=1).mean()
    return float(s), float(c[0]), float(c[1]), float(res)


def edge_width(profile: np.ndarray) -> float | None:
    """10%-90% transition width (samples) of a 1-D profile, or None if there is no edge."""
    p = np.asarray(profile, np.float32)
    total = p[-1] - p[0]
    if abs(total) < 12:
        return None
    frac = np.clip((p - p[0]) / total, 0, 1)
    i10 = int(np.argmax(frac >= 0.1))
    i90 = int(np.argmax(frac >= 0.9))
    return float(max(i90 - i10, 0) + 1)


def boundary_softness(g: np.ndarray, mask: np.ndarray, axis: int, side: str, half: int = 6) -> float | None:
    """Mean 10-90 width of the intensity profile across the mask's boundary. axis 1 = x, 0 = y. side 'hi' = max coord."""
    widths = []
    lines = range(mask.shape[0]) if axis == 1 else range(mask.shape[1])
    for i in lines:
        row = mask[i] if axis == 1 else mask[:, i]
        idx = np.nonzero(row)[0]
        if len(idx) == 0:
            continue
        pos = idx.max() if side == "hi" else idx.min()
        lo, hi = pos - half, pos + half + 1
        if lo < 0 or hi > len(row):
            continue
        prof = g[i, lo:hi] if axis == 1 else g[lo:hi, i]
        w = edge_width(prof)
        if w is not None:
            widths.append(w)
    return float(np.median(widths)) if widths else None


def spatial_variance(frame: np.ndarray) -> float:
    return float(gray(frame).var())


def blend_alpha(F, A, B, mask=None):
    """Least-squares alpha for F ~= a*A + (1-a)*B. Returns (alpha, rmse)."""
    f, a, b = gray(F), gray(A), gray(B)
    if mask is not None:
        f, a, b = f[mask], a[mask], b[mask]
    d = a - b
    den = float((d * d).sum())
    if den < EPS:
        return None, float(np.sqrt(((f - b) ** 2).mean()))
    alpha = float(((f - b) * d).sum() / den)
    res = float(np.sqrt(((f - (alpha * a + (1 - alpha) * b)) ** 2).mean()))
    return alpha, res


def hf_energy(g: np.ndarray, mask=None) -> float:
    e = np.abs(g - gaussian_blur(g, 1.0))
    return float(e[mask].mean() if mask is not None and mask.any() else e.mean())


def orientation_energy(g: np.ndarray, n: int = 8, mask=None):
    """Directional gradient energy. Returns (energies, angles_rad, anisotropy, min_angle)."""
    gy, gx = np.gradient(g)
    angles = np.arange(n) * np.pi / n
    e = []
    for th in angles:
        v = np.abs(gx * np.cos(th) + gy * np.sin(th))
        e.append(float(v[mask].mean() if mask is not None and mask.any() else v.mean()))
    e = np.array(e)
    aniso = float((e.max() - e.min()) / (e.max() + e.min() + EPS))
    return e, angles, aniso, float(angles[int(e.argmin())])


def column_runs(occ: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a 1-D bool array as (start, end_exclusive)."""
    runs, start = [], None
    for i, v in enumerate(occ):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(occ)))
    return runs


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected labelling. Small frames only (BFS in python)."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    n = 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0]:
            continue
        n += 1
        q = deque([(y0, x0)])
        lab[y0, x0] = n
        while q:
            y, x = q.popleft()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = n
                    q.append((ny, nx))
    return lab, n


def fit_line_residual(pts: np.ndarray) -> tuple[float, float]:
    """PCA line fit. Returns (mean perpendicular distance, angle_rad)."""
    if len(pts) < 3:
        return 0.0, 0.0
    c = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    return float(np.abs((pts - c) @ n).mean()), float(np.arctan2(d[1], d[0]))


def fit_circle_residual(pts: np.ndarray) -> tuple[float, tuple[float, float], float]:
    """Kasa algebraic circle fit. Returns (mean radial residual, center, radius)."""
    if len(pts) < 4:
        return 0.0, (0.0, 0.0), 0.0
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = np.sqrt(max(sol[2] + cx * cx + cy * cy, EPS))
    res = np.abs(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r).mean()
    return float(res), (float(cx), float(cy)), float(r)


def mtf_axis_ratio(sharp_g: np.ndarray, blurred_g: np.ndarray, band=(0.08, 0.30)) -> tuple[float, float]:
    """Kernel MTF estimate |B(f)|/|S(f)| averaged over a mid band along the fx axis and along the fy axis.
    Content anisotropy cancels in the ratio; compression loss is isotropic and cancels between the two."""
    H, W = sharp_g.shape
    win = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    S = np.abs(np.fft.fft2((sharp_g - sharp_g.mean()) * win))
    B = np.abs(np.fft.fft2((blurred_g - blurred_g.mean()) * win))
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    ratio = B / (S + 1e-3)
    sel_x = (np.abs(fx) >= band[0]) & (np.abs(fx) <= band[1]) & (np.abs(fy) < 0.03)
    sel_y = (np.abs(fy) >= band[0]) & (np.abs(fy) <= band[1]) & (np.abs(fx) < 0.03)
    rx = float(np.median(ratio[np.broadcast_to(sel_x, ratio.shape)]))
    ry = float(np.median(ratio[np.broadcast_to(sel_y, ratio.shape)]))
    return rx, ry

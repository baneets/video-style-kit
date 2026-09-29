"""Rectangular overlays (insets, cards, boxes) found from PERSISTENT AXIS-ALIGNED EDGES, no colour or person assumption.

A card drawn over footage has two long horizontal and two long vertical edges that stay put for many frames while
everything else in the frame moves. Background structure (a door frame, a table edge) is also a persistent rectangle,
but it lasts the whole shot on every shot; an overlay's lifetime is a strict subset. The entry/exit animation is
recovered by re-fitting the settled rectangle at scale s and offset (dx, dy) in the frames before/after its persistent
phase and reading the best-fit parameters per frame.
"""
from __future__ import annotations

import numpy as np

WIN = 9                     # frames a border must persist to count as a static edge
MIN_SIDE = 0.06             # min rectangle side as a fraction of the frame's width/height
MIN_PERIMETER_COVER = 0.55  # fraction of the rect perimeter that must be persistent edge


def edges(frame: np.ndarray) -> np.ndarray:
    import cv2
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(g, 60, 140) > 0


def persistent(edge_stack: np.ndarray, frac: float = 0.8) -> np.ndarray:
    return edge_stack.mean(axis=0) >= frac


def _runs_mask(pe: np.ndarray, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Long horizontal and vertical runs of persistent edge pixels (small gaps bridged)."""
    import cv2
    hk = np.ones((1, max(3, int(0.03 * W))), np.uint8)
    vk = np.ones((max(3, int(0.03 * H)), 1), np.uint8)
    u8 = pe.astype(np.uint8)
    h = cv2.morphologyEx(cv2.morphologyEx(u8, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8)), cv2.MORPH_OPEN, hk)
    v = cv2.morphologyEx(cv2.morphologyEx(u8, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8)), cv2.MORPH_OPEN, vk)
    return h > 0, v > 0


def perimeter_cover(pe: np.ndarray, box: tuple[int, int, int, int], tol: int = 2) -> float:
    """Fraction of a rectangle's perimeter that lies on persistent edge pixels (within tol px)."""
    x, y, w, h = box
    H, W = pe.shape
    band = lambda y0, y1, x0, x1: pe[max(0, y0): min(H, y1), max(0, x0): min(W, x1)]
    top = band(y - tol, y + tol + 1, x, x + w).any(axis=0).mean() if w > 0 else 0
    bot = band(y + h - tol, y + h + tol + 1, x, x + w).any(axis=0).mean() if w > 0 else 0
    lef = band(y, y + h, x - tol, x + tol + 1).any(axis=1).mean() if h > 0 else 0
    rig = band(y, y + h, x + w - tol, x + w + tol + 1).any(axis=1).mean() if h > 0 else 0
    return float((top * w + bot * w + lef * h + rig * h) / (2 * (w + h)))


def find_rects(pe: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Axis-aligned rectangles whose perimeter is mostly persistent edge."""
    import cv2
    H, W = pe.shape
    hm, vm = _runs_mask(pe, W, H)
    frame_mask = (hm | vm).astype(np.uint8)
    # bridge corners so the four sides connect, then take each connected structure's bounding rect
    frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(frame_mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w < MIN_SIDE * W or h < MIN_SIDE * H:
            continue
        if perimeter_cover(pe, (x, y, w, h)) >= MIN_PERIMETER_COVER:
            out.append((int(x), int(y), int(w), int(h)))
    # dedupe nested duplicates
    out.sort(key=lambda b: -b[2] * b[3])
    keep = []
    for b in out:
        if not any(_iou(b, k) > 0.7 for k in keep):
            keep.append(b)
    return keep


def _iou(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter + 1e-6)


def track_rects(frames: np.ndarray, step: int = 3, log=None) -> list[dict]:
    """Rect tracks over the whole video: {box, t_on, t_off (frames), samples}. Full-video rectangles are dropped."""
    T = len(frames)
    E = np.stack([edges(f) for f in frames])
    tracks: list[dict] = []
    active: list[dict] = []
    for t in range(WIN, T, step):
        pe = persistent(E[t - WIN: t + 1])
        rects = find_rects(pe)
        used = set()
        for tr in active:
            best, bi = 0.0, -1
            for j, r in enumerate(rects):
                if j in used:
                    continue
                v = _iou(tr["box"], r)
                if v > best:
                    best, bi = v, j
            if bi >= 0 and best > 0.6:
                tr["t_off"] = t; tr["samples"] += 1; tr["miss"] = 0; used.add(bi)
            else:
                tr["miss"] += 1
        for j, r in enumerate(rects):
            if j not in used:
                active.append({"box": r, "t_on": t - WIN, "t_off": t, "samples": 1, "miss": 0})
        still = []
        for tr in active:
            (tracks if tr["miss"] > 3 else still).append(tr)
        active = still
    tracks += active
    tracks.sort(key=lambda tr: tr["t_on"])
    merged: list[dict] = []
    for tr in tracks:                                 # the content inside a card cuts and briefly hides a side: re-join
        for m in merged:
            if tr["t_on"] <= m["t_off"] + int(1.0 * 30) and _iou(m["box"], tr["box"]) > 0.6:
                m["t_off"] = max(m["t_off"], tr["t_off"]); m["samples"] += tr["samples"]; m["boxes"].append(tr["box"]); break
        else:
            merged.append({**tr, "boxes": [tr["box"]]})
    out = []
    for tr in merged:
        life = tr["t_off"] - tr["t_on"]
        if life < 12 or tr["samples"] < 3:
            continue
        if life > 0.95 * T:
            continue                                  # background structure, present for the whole video
        tr["box"] = tuple(int(v) for v in np.median(np.array(tr["boxes"]), axis=0))
        out.append(tr)
    if log:
        log(f"rects: {len(out)} rectangular overlays")
    return out, E


def fit_transform(E: np.ndarray, box: tuple[int, int, int, int], t: int, scales=np.linspace(0.3, 1.25, 39),
                  dxs=range(-24, 25, 4), dys=range(-24, 25, 4)) -> dict:
    """Best (scale about the box centre, dx, dy) for the box's perimeter on frame t's edges; score = perimeter cover."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    pe = E[t]
    H, W = pe.shape
    best = {"scale": 1.0, "dx": 0, "dy": 0, "score": 0.0}
    for s in scales:
        sw, sh = int(round(w * s)), int(round(h * s))
        if sw < 6 or sh < 6:
            continue
        for dx in dxs:
            for dy in dys:
                bx, by = int(round(cx - sw / 2 + dx)), int(round(cy - sh / 2 + dy))
                if bx + sw < 0 or by + sh < 0 or bx > W or by > H:
                    continue
                sc = perimeter_cover(pe, (bx, by, sw, sh), tol=1)
                if sc > best["score"]:
                    best = {"scale": float(s), "dx": int(dx), "dy": int(dy), "score": sc}
    return best


def measure_entry_exit(E: np.ndarray, tr: dict, fps: float, max_back_s: float = 1.5) -> dict:
    """Walk backwards from the settled phase fitting the rect per frame until it is not found; same forwards for exit."""
    T = len(E)
    box = tuple(tr["box"])
    rest = fit_transform(E, box, min(T - 1, tr["t_on"] + WIN + 2), scales=[1.0], dxs=[0], dys=[0])["score"]
    thr = max(0.45, 0.7 * rest)

    def walk(start: int, direction: int) -> list[dict]:
        seq = []
        t = start
        misses = 0
        prev = {"scale": 1.0, "dx": 0, "dy": 0}
        while 0 <= t < T and len(seq) < int(max_back_s * fps):
            # local search around the previous frame's fit: a card moves a little per frame, never teleports
            sc = np.clip(np.arange(prev["scale"] - 0.15, prev["scale"] + 0.151, 0.025), 0.2, 1.4)
            f = fit_transform(E, box, t, scales=sc, dxs=[prev["dx"] + o for o in range(-16, 17, 4)], dys=[prev["dy"] + o for o in range(-16, 17, 4)])
            prev = {"scale": f["scale"], "dx": f["dx"], "dy": f["dy"]} if f["score"] >= thr else prev
            if f["score"] < thr:
                misses += 1
                if misses >= 2:
                    break
            else:
                misses = 0
            seq.append({"t": t, **f})
            t += direction
        while seq and seq[-1]["score"] < thr:
            seq.pop()
        return seq

    back = walk(tr["t_on"] + WIN, -1)[::-1]       # chronological, ends at settle
    fwd = walk(tr["t_off"], +1)
    return {"in": back, "out": fwd, "rest_score": rest}

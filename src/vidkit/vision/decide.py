"""Pair-specific decisive tests + orchestration. Every analyze_* returns a fragment dict:

  {"question", "chosen", "rejected", "margin", "ambiguous", "evidence", "reconstruction_scores", "note", "curves", ...}

The reconstruction contest (hyps.contest) is the backstop; the decisive tests are the evidence.
"""
from __future__ import annotations

import numpy as np

from . import measure as M
from .measure import gaussian_blur
from .hyps import (DIRS, Hyp, Obs, composite, contest, keystone_matrix, motion_blur, proj,
                   rotation_matrix, scale_matrix, shift, warp)

DEFAULT_MARGIN = 0.08
# transforms are simpler than masks (a translate reproduces with one keyframe pair; a mask needs a matte)
SIMPLICITY = {"fade": 1, "slide": 2, "slide_from_behind": 3, "scale": 4, "wipe": 5, "iris": 6, "blur_in": 7,
              "custom_mask": 9, "cut": 1, "dissolve": 2, "dip": 3, "slide_over": 4, "push": 5}


def _r(x, n=3):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: _r(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v, n) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return round(float(x), n)


def _finish(question: str, res: dict, evidence: dict, note: str | None = None, alias: dict | None = None,
            equivalent: list[str] | None = None) -> dict:
    ranking = res["ranking"]
    alias = alias or {}
    name = lambda n: alias.get(n, n)  # noqa: E731
    top = ranking[0]
    chosen = name(top)
    ambiguous = res["ambiguous"]
    if ambiguous:
        if equivalent:
            chosen = name(equivalent[0])
            note = (note or "") + (f" Hypotheses {[name(e) for e in equivalent]} reproduce this element identically "
                                   f"(no interior detail to track); emitting the simpler one, {chosen}.")
            ambiguous = False
        else:
            chosen = None
            note = (note or "") + (f" Ambiguous: {name(ranking[0])} vs {name(ranking[1])} within margin "
                                   f"{res['margin']:.3f} < {res['margin_threshold']}; both reported, no winner declared.")
    r0 = res["results"][top]
    best_score = r0["scores"]["combined"]
    if best_score < 0.65:
        note = (note or "") + (f" Poor absolute fit (best reconstruction {best_score:.2f}): no hypothesis reproduces these frames well. "
                               "Either the element's own content is changing (video inset, live camera) or the window/crop is wrong.")
    return {
        "question": question,
        "fit_quality": round(float(best_score), 4),
        "chosen": chosen,
        "candidates": [name(n) for n in ranking[:2]],
        "rejected": [name(n) for n in ranking if name(n) != chosen],
        "margin": res["margin"],
        "margin_threshold": res["margin_threshold"],
        "ambiguous": ambiguous,
        "evidence": evidence,
        "reconstruction_scores": {name(k): v for k, v in res["reconstruction_scores"].items()},
        "component_scores": {name(k): {kk: _r(vv) for kk, vv in v["scores"].items()} for k, v in res["results"].items()},
        "curve": r0["curve"],
        "params": {k: _r(v) if isinstance(v, (float, int, list, tuple, np.floating)) else v for k, v in r0["params"].items()},
        "note": note.strip() if note else None,
    }


def _region(masks, pad=4):
    union = np.logical_or.reduce(masks)
    b = M.bbox(union)
    if b is None:
        return None
    H, W = union.shape
    return (max(0, b[0] - pad), max(0, b[1] - pad), min(W, b[2] + pad), min(H, b[3] + pad))


def _direction(c0, c1):
    if c0 is None or c1 is None:
        return "right", 0.0
    v = np.array([c1[0] - c0[0], c1[1] - c0[1]])
    n = float(np.linalg.norm(v))
    if n < 1.5:
        return "right", n
    best = max(DIRS, key=lambda k: float(np.dot(v / n, DIRS[k])))
    return best, n


# =====================================================================================
# Section 2 / 6: appear family (slide, wipe, iris, custom mask, fade, blur-in, slide-from-behind, scale)
# =====================================================================================

def appear_hyps(frames, ref, final, emask, dname, region, candidates=None) -> list[Hyp]:
    H, W = emask.shape
    d = DIRS[dname]
    P = proj((H, W), d)
    core = _core(emask)
    pe = P[core]
    lead, trail = float(pe.max()), float(pe.min())
    entry = float(P.min())
    L_slide = lead - entry + 1
    L_el = lead - trail + 1
    finalf = final.astype(np.float32)
    emf = emask.astype(np.float32)
    cx, cy = M.centroid(emask)
    ys, xs = np.mgrid[0:H, 0:W]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    R = float(dist[emask].max())

    def r_slide(p, clip_behind=False, L=L_slide):
        off = -(1 - p) * L
        dx, dy = int(round(off * d[0])), int(round(off * d[1]))
        img, valid = shift(finalf, dx, dy)
        m, _ = shift(emf, dx, dy)
        a = (m > 0.5) & valid
        if clip_behind:
            a &= P >= trail
        return composite(ref, img, a)

    def r_wipe(p, soft=0.0):
        b = trail + p * L_el
        if soft < 0.5:
            a = emask & (P <= b)
        else:
            a = emf * np.clip((b - P) / soft + 0.5, 0, 1)
        return composite(ref, finalf, a)

    def r_iris(p):
        return composite(ref, finalf, emask & (dist <= p * R))

    def r_fade(p):
        return composite(ref, finalf, emf * p)

    smax = 0.08 * min(H, W)

    premult = finalf * emf[..., None]   # element on transparent black: blur the layer WITH its alpha

    def r_blur(p):
        s = (1 - p) * smax
        return composite(ref, gaussian_blur(premult, s), np.clip(gaussian_blur(emf, s), 0, 1))

    def r_scale(p):
        s = max(p, 0.02)
        img, valid = warp(finalf, scale_matrix(s, cx, cy))
        m, _ = warp(emf, scale_matrix(s, cx, cy))
        return composite(ref, img, (m > 0.5) & valid)

    all_h = {
        "fade": Hyp("fade", r_fade, {"channel": "opacity"}, SIMPLICITY["fade"]),
        "slide": Hyp("slide", r_slide, {"direction": dname, "travel_px": L_slide}, SIMPLICITY["slide"]),
        "wipe": Hyp("wipe", r_wipe, {"direction": dname, "softness_px": 0.0}, SIMPLICITY["wipe"]),
        "wipe_soft": Hyp("wipe_soft", lambda p: r_wipe(p, 4.0), {"direction": dname, "softness_px": 4.0}, SIMPLICITY["wipe"], label="wipe"),
        "iris": Hyp("iris", r_iris, {"center": (cx, cy), "radius_px": R}, SIMPLICITY["iris"]),
        "slide_from_behind": Hyp("slide_from_behind", lambda p: r_slide(p, True, L_el),
                                 {"direction": dname, "occluder_edge": "element trailing edge", "travel_px": L_el},
                                 SIMPLICITY["slide_from_behind"]),
        "blur_in": Hyp("blur_in", r_blur, {"max_sigma_px": smax}, SIMPLICITY["blur_in"]),
        "scale": Hyp("scale", r_scale, {"center": (cx, cy)}, SIMPLICITY["scale"]),
    }
    if candidates:
        return [all_h[c] for c in candidates if c in all_h]
    return list(all_h.values())


def _core(mask: np.ndarray) -> np.ndarray:
    """Mask restricted to rows/cols carrying >= 25% of the peak mass: drops compression slivers at frame edges."""
    cm, rm = mask.sum(axis=0), mask.sum(axis=1)
    keep = np.outer(rm >= 0.25 * rm.max(), cm >= 0.25 * cm.max()) if mask.any() else mask
    out = mask & keep
    return out if out.sum() >= 0.5 * mask.sum() else mask


def _boundary_points(visible, emask):
    unrevealed = emask & ~visible
    if unrevealed.sum() == 0 or visible.sum() == 0:
        return None
    near = M.box_filter(unrevealed.astype(np.float32), 1) > 0
    b = visible & emask & near
    ys, xs = np.nonzero(b)
    return np.stack([xs, ys], axis=1).astype(np.float32) if len(xs) >= 4 else None


def appear_evidence(frames, masks, emask, dname, fps=30.0) -> dict:
    d = DIRS[dname]
    H, W = emask.shape
    P = proj((H, W), d)
    area = emask.sum()
    active = [t for t in range(len(frames)) if 0.05 * area < masks[t].sum() < 0.98 * area]
    entry = float(P.min())
    leads, trails, cents = [], [], []
    for t in active:
        pm = P[masks[t]]
        leads.append(float(pm.max())); cents.append(M.centroid(masks[t]))
        if pm.min() > entry + 0.5:          # a mask touching the frame edge says nothing about the trailing edge
            trails.append(float(pm.min()))
    # interior flow vs boundary velocity
    flows, bvels, feat = [], [], []
    for t in range(len(frames) - 1):
        inter = masks[t] & masks[t + 1]
        if inter.sum() < 64:
            continue
        core = M.box_filter(inter.astype(np.float32), 3) > 0.99
        flow, valid, _ = M.block_flow(frames[t], frames[t + 1], block=6, search=14, mask=core)
        if valid.sum() == 0:
            continue
        f = flow[valid]
        flows.append(float((f @ np.array(d)).mean()))
        # strongest-texture block = "tracked feature"
        g = M.gray(frames[t])
        nb = valid.shape
        stds = g[: nb[0] * 6, : nb[1] * 6].reshape(nb[0], 6, nb[1], 6).std(axis=(1, 3))
        stds[~valid] = -1
        iy, ix = np.unravel_index(int(stds.argmax()), stds.shape)
        feat.append(float(np.linalg.norm(flow[iy, ix])))
        if masks[t].sum() > 0.05 * area:
            bvels.append(float(P[masks[t + 1]].max() - P[masks[t]].max()))
    line_res, circ_res = [], []
    for t in active[len(active) // 3: max(len(active) // 3 + 1, 2 * len(active) // 3)]:
        pts = _boundary_points(masks[t], emask)
        if pts is None:
            continue
        line_res.append(M.fit_line_residual(pts)[0])
        circ_res.append(M.fit_circle_residual(pts)[0])
    lr = float(np.median(line_res)) if line_res else None
    cr = float(np.median(circ_res)) if circ_res else None
    shape = None
    if lr is not None:
        shape = "line" if lr <= 1.6 else ("circle" if cr is not None and cr <= 1.6 else "custom")
    # softness of the moving boundary vs a static element edge
    gmid = M.gray(frames[active[len(active) // 2]]) if active else None
    axis = 1 if d[0] else 0
    side = "hi" if (d[0] + d[1]) > 0 else "lo"
    moving_soft = M.boundary_softness(gmid, masks[active[len(active) // 2]], axis, side) if gmid is not None else None
    static_soft = M.boundary_softness(M.gray(frames[-1]), emask, axis, "lo" if side == "hi" else "hi")
    # off-canvas origin: extrapolate lead back to the first active frame - 1
    origin = None
    if len(leads) >= 2:
        a, b = np.polyfit(active, leads, 1)
        origin = float(a * (active[0] - 1) + b)
    pe = P[emask]
    # section 6 channels
    alphas, widths, hfs = [], [], []
    for t in range(len(frames)):
        al, _ = M.blend_alpha(frames[t], frames[-1], frames[0], emask)
        alphas.append(al)
        g = M.gray(frames[t])
        widths.append(M.boundary_softness(g, emask, 1, "lo"))
        hfs.append(M.hf_energy(g, emask))
    alphas_v = [a for a in alphas if a is not None]
    widths_v = [w for w in widths if w is not None]
    core = M.box_filter(emask.astype(np.float32), 2) > 0.99
    texture_std = float(M.gray(frames[-1])[core].std()) if core.any() else 0.0
    areas = [float(m.sum() / max(area, 1)) for m in masks]
    gfin = M.gray(frames[-1])
    pop = any(a < 0.08 and b > 0.9 and np.abs(M.gray(frames[i + 1]) - gfin)[emask].mean() < 12
              for i, (a, b) in enumerate(zip(areas, areas[1:])))
    return {
        "visible_area_fraction_curve": _r(areas),
        "pops_in_one_frame": bool(pop),
        "direction": dname,
        "element_bbox_px": M.bbox(emask),
        "element_texture_std": _r(texture_std),
        "featureless": bool(texture_std < 6.0),
        "interior_flow_magnitude_px": _r(np.mean(np.abs(flows))) if flows else None,
        "boundary_velocity_px": _r(np.mean(np.abs(bvels))) if bvels else None,
        "trailing_edge_pinned": bool(np.std(trails) < 1.5) if len(trails) >= 2 else None,
        "trailing_edge_std_px": _r(np.std(trails)) if len(trails) >= 2 else None,
        "leading_edge_positions_px": _r(leads),
        "extrapolated_lead_at_start_px": _r(origin),
        "final_trailing_edge_px": _r(pe.min()),
        "origin_outside_final_bounds": (origin < float(pe.min()) - 0.1 * (float(pe.max()) - float(pe.min()))) if origin is not None else None,
        "tracked_feature_displacement_px": _r(np.median(feat)) if feat else None,
        "boundary_edge_softness_px": _r(moving_soft),
        "static_edge_softness_px": _r(static_soft),
        "boundary_line_residual_px": _r(lr),
        "boundary_circle_residual_px": _r(cr),
        "boundary_shape": shape,
        "opacity_curve": _r(alphas),
        "opacity": "animated" if alphas_v and (max(alphas_v) - min(alphas_v)) > 0.3 else "static",
        "edge_width_curve_px": _r(widths),
        "blur": "animated" if len(widths_v) >= 2 and (max(widths_v) - min(widths_v)) > 1.5 else "static",
        "hf_energy_curve": _r(hfs),
        "position": "animated" if len(cents) >= 2 and np.linalg.norm(np.array(cents[0]) - np.array(cents[-1])) > 2 else "static",
        "frames_active": len(active),
        "fps": fps,
    }


def analyze_appear(frames: np.ndarray, margin: float = DEFAULT_MARGIN, candidates=None, fps: float = 30.0,
                   question: str = "appear", alias: dict | None = None) -> dict:
    ref, final = frames[0], frames[-1]
    emask = M.diff_mask(final, ref)
    if emask.sum() < 20:
        return {"question": question, "chosen": None, "ambiguous": True, "note": "no element detected (final frame == reference)",
                "evidence": {}, "reconstruction_scores": {}, "margin": 0.0, "rejected": [], "candidates": []}
    masks = [M.diff_mask(f, ref) for f in frames]
    area = emask.sum()
    first = next((t for t in range(len(frames)) if masks[t].sum() > 0.02 * area), 0)
    dname, _ = _direction(M.centroid(masks[first]) if masks[first].sum() else None, M.centroid(emask))
    ev = appear_evidence(frames, masks, emask, dname, fps)
    if ev["pops_in_one_frame"]:
        return {"question": question, "chosen": "pop", "candidates": ["pop"], "rejected": [], "margin": 1.0, "margin_threshold": margin,
                "ambiguous": False, "evidence": ev, "reconstruction_scores": {"pop": 1.0}, "curve": [], "params": {},
                "note": "Element appears fully formed between two consecutive frames (no partially-visible frame): a pop/cut-in, nothing to disambiguate."}
    region = _region(masks)
    hyps = appear_hyps(frames, ref, final, emask, dname, region, candidates)
    mid = masks[len(frames) // 2]
    revealed_ok = bool(mid.sum() > 30 and np.abs(M.gray(frames[len(frames) // 2]) - M.gray(final))[mid].mean() < 20)
    ev["revealed_content_matches_final"] = revealed_ok
    if ev["boundary_shape"] == "custom" and revealed_ok and (candidates is None or "custom_mask" in candidates):
        # data-driven matte: content pinned, observed mask per frame. Gated on the boundary-shape test so it
        # cannot overfit wipes/irises. Emit the boundary polygon per keyframe.
        obs_masks = masks

        def r_custom(p):
            t = int(round(p * (len(frames) - 1)))
            return composite(ref, final.astype(np.float32), obs_masks[t] & emask)
        polys = []
        for t in range(len(frames)):
            pts = _boundary_points(masks[t], emask)
            polys.append(_r(pts[:: max(1, len(pts) // 24)].tolist(), 1) if pts is not None else None)
        hyps.append(Hyp("custom_mask", r_custom, {"boundary_polygons_per_frame": polys}, SIMPLICITY["custom_mask"],
                        grid=np.linspace(0, 1, len(frames))))
    obs = Obs(frames, ref, final, emask, region)
    res = contest(hyps, obs, margin)
    equivalent = None
    note = None
    if res["ambiguous"] and ev["featureless"]:
        near = [n for n in res["ranking"] if res["results"][n]["scores"]["combined"] >= res["results"][res["ranking"][0]]["scores"]["combined"] * (1 - margin)]
        motion_like = [n for n in near if n in ("slide", "wipe", "wipe_soft", "slide_from_behind")]
        if len(motion_like) >= 2:
            equivalent = sorted(motion_like, key=lambda n: SIMPLICITY.get(n.replace("_soft", ""), 9))
    return _finish(question, res, ev, note, alias, equivalent)


# =====================================================================================
# Section 4 / 5 / 7: transitions (cut, dissolve, dip-to-color, flash, push, slide-over)
# =====================================================================================

def transition_evidence(frames) -> dict:
    A, B = frames[0], frames[-1]
    var = [M.spatial_variance(f) for f in frames]
    alphas, resid = [], []
    for f in frames:
        a, r = M.blend_alpha(f, A, B)
        alphas.append(a); resid.append(r)
    vmin_t = int(np.argmin(var))
    vmax = max(var)
    hold = int(sum(1 for v in var if v < 0.15 * vmax))
    dip_color = [float(c) for c in frames[vmin_t].reshape(-1, 3).mean(axis=0)]
    # per-pixel trajectory monotonicity on a sample
    rng = np.random.default_rng(0)
    H, W = A.shape[:2]
    ys, xs = rng.integers(0, H, 300), rng.integers(0, W, 300)
    traj = np.stack([M.gray(f)[ys, xs] for f in frames], axis=1)  # (300, T)
    dif = np.diff(traj, axis=1)
    sign = np.sign(dif) * (np.abs(dif) > 3)
    mono = []
    for row in sign:
        nz = row[row != 0]
        mono.append(bool(len(nz) == 0 or (nz > 0).all() or (nz < 0).all()))
    interior = list(range(1, len(frames) - 1))
    blend_frames = [t for t in interior if alphas[t] is not None and 0.05 < alphas[t] < 0.95 and resid[t] < 12]
    return {
        "spatial_variance_curve": _r(var, 1),
        "variance_min_over_max": _r(min(var) / vmax) if vmax > 0 else None,
        "blend_alpha_curve": _r(alphas),
        "blend_fit_residual_curve": _r(resid, 2),
        "blend_fit_frames": blend_frames,
        "dissolve_duration_frames": len(blend_frames),
        "monotonic_pixel_fraction": _r(np.mean(mono)),
        "dip_color_rgb": _r(dip_color, 1),
        "dip_hold_frames": hold,
        "is_flash": bool(hold <= 1 and min(var) < 0.15 * vmax),
    }


def transition_hyps(frames, dip_color) -> list[Hyp]:
    A, B = frames[0].astype(np.float32), frames[-1].astype(np.float32)
    H, W = A.shape[:2]
    c = np.array(dip_color, np.float32)[None, None, :]

    def r_cut(p):
        return B if p >= 0.5 else A

    def r_dissolve(p):
        return A * (1 - p) + B * p

    def r_dip(p):
        if p < 0.5:
            b = p * 2
            return A * (1 - b) + c * b
        b = (p - 0.5) * 2
        return c * (1 - b) + B * b

    hyps = [Hyp("cut", r_cut, {}, SIMPLICITY["cut"], grid=np.array([0.0, 1.0])),
            Hyp("dissolve", r_dissolve, {}, SIMPLICITY["dissolve"]),
            Hyp("dip", r_dip, {"dip_color_rgb": _r(dip_color, 1)}, SIMPLICITY["dip"])]
    for dname, d in DIRS.items():
        L = W if d[0] else H

        def r_push(p, d=d, L=L):
            off = int(round((1 - p) * L))
            b_img, b_valid = shift(B, off * d[0], off * d[1])
            a_img, a_valid = shift(A, (off - L) * d[0], (off - L) * d[1])
            return np.where(b_valid[..., None], b_img, a_img)

        def r_over(p, d=d, L=L):
            off = int(round((1 - p) * L))
            b_img, b_valid = shift(B, off * d[0], off * d[1])
            return np.where(b_valid[..., None], b_img, A)
        hyps.append(Hyp(f"push_{dname}", r_push, {"direction": dname}, SIMPLICITY["push"], label="push"))
        hyps.append(Hyp(f"slide_over_{dname}", r_over, {"direction": dname}, SIMPLICITY["slide_over"], label="slide_over"))
    return hyps


def analyze_transition(frames: np.ndarray, margin: float = DEFAULT_MARGIN, fps: float = 30.0) -> dict:
    ev = transition_evidence(frames)
    hyps = transition_hyps(frames, ev["dip_color_rgb"])
    pruned = []
    A, B = frames[0], frames[-1]
    interior = list(range(1, len(frames) - 1))
    all_a_or_b = all(min(M.rmse(frames[t], A), M.rmse(frames[t], B)) < 3.0 for t in interior)
    if all_a_or_b:
        # every intermediate frame IS A or B: every transition hypothesis degenerates to a cut. Decided by the
        # section-5 test, not by a contest between identical renders.
        k = next((t for t in interior if M.rmse(frames[t], B) < M.rmse(frames[t], A)), len(frames) - 1)
        res = {"ranking": ["cut"], "results": {"cut": {"scores": {"ssim": 1.0, "iou": 1.0, "residual": 0.0, "combined": 1.0},
               "curve": [0.0 if t < k else 1.0 for t in interior], "params": {"cut_frame": k}, "simplicity": 1, "label": "cut"}},
               "margin": 1.0, "ambiguous": False, "margin_threshold": margin, "reconstruction_scores": {"cut": 1.0}}
        return _finish("transition", res, ev, f"Hard cut at frame {k}: no intermediate frame differs from A or B, so dissolve/dip/push/slide-over all degenerate to a cut.")
    if ev["dissolve_duration_frames"] == 0:
        hyps = [h for h in hyps if h.name != "dissolve"]
        pruned.append("dissolve pruned: no frame blends A and B with alpha strictly inside (0.05, 0.95).")
    if ev["variance_min_over_max"] is not None and ev["variance_min_over_max"] > 0.35:
        hyps = [h for h in hyps if h.name != "dip"]
        pruned.append("dip pruned: spatial variance never collapses toward a uniform frame.")
    pruned = " ".join(pruned) or None
    obs = Obs(frames, frames[0], frames[-1], np.ones(frames.shape[1:3], bool), None)
    res = contest(hyps, obs, margin)
    alias = {n: res["results"][n]["label"] for n in res["results"]}
    note = None
    top = alias[res["ranking"][0]]
    if top == "dip":
        kind = "flash" if ev["is_flash"] else "dip"
        c = ev["dip_color_rgb"]
        lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        to = "black" if lum < 40 else ("white" if lum > 215 else "color")
        alias[res["ranking"][0]] = f"{kind}_to_{to}"
        note = f"Recovered dip color rgb{tuple(int(v) for v in c)} -> record as a $palette slot, not a literal. Hold {ev['dip_hold_frames']} frame(s)."
    if top == "dissolve":
        note = f"Dissolve spans {ev['dissolve_duration_frames']} blended frame(s) = {ev['dissolve_duration_frames'] / fps:.3f}s at {fps:g} fps."
    if top == "cut" and ev["dissolve_duration_frames"] > 0:
        note = f"Contest says cut but {ev['dissolve_duration_frames']} frame(s) fit a blend; check for a very short dissolve."
    if pruned:
        note = (note + " " if note else "") + pruned
    return _finish("transition", res, ev, note, alias)


# =====================================================================================
# Section 3 / 10: background zoom vs element scale vs compound (also dolly vs scale)
# =====================================================================================

def analyze_zoom(frames: np.ndarray, element_box: tuple | None = None, margin: float = DEFAULT_MARGIN, tol: float = 0.04) -> dict:
    f0, fT = frames[0], frames[-1]
    H, W = f0.shape[:2]
    if element_box is None:
        b = M.bbox(M.diff_mask(fT, f0))
        element_box = b or (W // 3, H // 3, 2 * W // 3, 2 * H // 3)
    x0, y0, x1, y1 = element_box
    emask0 = np.zeros((H, W), bool); emask0[y0:y1, x0:x1] = True
    bg_mask = ~(M.box_filter(emask0.astype(np.float32), 6) > 0)
    # background similarity per frame
    s_bg, centers, resids, nvalid = [1.0], [], [], []
    for t in range(1, len(frames)):
        flow, valid, cen = M.block_flow(frames[t - 1], frames[t], block=8, search=8, mask=bg_mask)
        nvalid.append(int(valid.sum()))
        fit = M.fit_similarity(flow, valid, cen)
        if fit is None:
            s_bg.append(s_bg[-1]); continue
        s, cx, cy, r = fit
        s_bg.append(s_bg[-1] * s); resids.append(r)
        if abs(s - 1) > 0.004:
            centers.append((cx, cy))
    bg_trackable = bool(nvalid and np.median(nvalid) >= 8 and resids and np.median(resids) < 1.5)
    s_bg_T = float(s_bg[-1])
    cbg = tuple(float(np.median([c[i] for c in centers])) for i in range(2)) if centers else (W / 2, H / 2)
    # stabilize and measure the element's residual scale (area of its diff mask vs frame 0 after unwarping)
    ecx, ecy = (x0 + x1) / 2, (y0 + y1) / 2
    s_el, hf = [1.0], []
    g0 = M.gray(f0)
    hf0 = M.hf_energy(g0, emask0)
    prev = f0
    el_region = M.box_filter(emask0.astype(np.float32), 4) > 0
    for t in range(1, len(frames)):
        stab, valid = warp(frames[t], scale_matrix(1 / s_bg[t], cbg[0], cbg[1]))
        stab = np.clip(stab, 0, 255).astype(np.uint8)
        # residual element motion between consecutive camera-stabilized frames, as a similarity fit on element blocks
        grow = M.box_filter(emask0.astype(np.float32), int(8 * max(s_el[-1], 1.0))) > 0
        flow, valid_b, cen = M.block_flow(prev, stab, block=6, search=8, mask=grow)
        fit = M.fit_similarity(flow, valid_b, cen)
        s_el.append(s_el[-1] * (fit[0] if fit else 1.0))
        prev = stab
        hf.append(M.hf_energy(M.gray(frames[t]), M.diff_mask(frames[t], f0) | emask0))
    s_el_T = float(s_el[-1])
    bg_moving = abs(s_bg_T - 1) > tol
    el_moving = abs(s_el_T - 1) > tol
    f0f = f0.astype(np.float32)

    def r_bg(p):
        s = 1 + p * (s_bg_T - 1)
        img, valid = warp(f0f, scale_matrix(s, cbg[0], cbg[1]))
        return np.where(valid[..., None], img, f0f)

    def r_el(p, base=None):
        s = 1 + p * (s_el_T - 1)
        src = f0f if base is None else base
        img, valid = warp(src, scale_matrix(s, ecx, ecy))
        m, _ = warp(emask0.astype(np.float32), scale_matrix(s, ecx, ecy))
        return composite(src, img, (m > 0.5) & valid)

    def r_compound(p):
        return r_el(p, r_bg(p))

    hyps = []
    if bg_moving or not el_moving:
        hyps.append(Hyp("bg_zoom", r_bg, {"scale_end": _r(s_bg_T), "center": _r(cbg, 1)}, 1, label="bg_zoom"))
    if el_moving or not bg_moving:
        hyps.append(Hyp("element_scale", r_el, {"scale_end": _r(s_el_T), "center": (_r(ecx, 1), _r(ecy, 1))}, 2))
    if bg_moving and el_moving:
        hyps.append(Hyp("compound", r_compound, {"bg_scale_end": _r(s_bg_T), "element_scale_end": _r(s_el_T)}, 3))
    if len(hyps) == 1:
        hyps.append(Hyp("static", lambda p: f0f, {}, 0))
    obs = Obs(frames, f0, fT, np.ones((H, W), bool), None)
    res = contest(hyps, obs, margin)
    hf_trend = _r(np.polyfit(range(len(hf)), hf, 1)[0] / max(hf0, 1e-3)) if len(hf) >= 3 else None
    ev = {
        "bg_scale_curve": _r(s_bg), "bg_scale_end": _r(s_bg_T), "bg_zoom_center_px": _r(cbg, 1),
        "bg_fit_residual_px": _r(np.mean(resids)) if resids else None,
        "element_scale_curve_after_stabilizing": _r(s_el), "element_scale_end": _r(s_el_T),
        "element_center_px": (_r(ecx, 1), _r(ecy, 1)),
        "center_distance_px": _r(np.hypot(cbg[0] - ecx, cbg[1] - ecy), 1),
        "layer_scale_ratio": _r(s_el_T / s_bg_T) if s_bg_T else None,
        "hf_energy_trend_while_scaling": hf_trend,
        "sharpness_signature": ("re-rendered graphic (sharpness kept)" if hf_trend is not None and hf_trend > -0.02 else
                                "digitally enlarged (sharpness lost)" if hf_trend is not None else None),
        "background_moving": bg_moving, "element_moving": el_moving, "tolerance": tol,
    }
    ev["bg_valid_blocks_median"] = _r(np.median(nvalid), 1) if nvalid else 0
    ev["bg_trackable"] = bg_trackable
    note = None
    if not bg_trackable:
        res["ambiguous"] = True
        note = ("Background has too little trackable texture (or the flow fit is too noisy) to separate camera motion from "
                "element motion; the measured scale cannot be attributed. Not decided.")
        return _finish("zoom", res, ev, note)
    if bg_moving and el_moving:
        note = ("Camera moves AND the element moves: the fragment separates them (bg_scale = camera, element_scale = element). "
                "With only two layers this is also what a dolly-in with parallax looks like; check layer_scale_ratio.")
    elif bg_moving:
        note = "Whole frame scales about one center: the camera is punching in, the element is not animating."
    return _finish("zoom", res, ev, note)


# =====================================================================================
# Section 8: typewriter vs text wipe vs word-by-word stagger
# =====================================================================================

def analyze_text(frames: np.ndarray, margin: float = DEFAULT_MARGIN, fps: float = 30.0) -> dict:
    ref, final = frames[0], frames[-1]
    emask = M.diff_mask(final, ref, clean=0)
    b = M.bbox(emask)
    if b is None:
        return {"question": "text_reveal", "chosen": None, "ambiguous": True, "note": "no caption detected", "evidence": {},
                "reconstruction_scores": {}, "margin": 0.0, "rejected": [], "candidates": []}
    x0, y0, x1, y1 = b
    colmass = emask[y0:y1, x0:x1].sum(axis=0)
    occ = colmass > 0.35 * colmass.max()      # bleed columns from compression carry little mass
    runs = [(s + x0, e + x0) for s, e in M.column_runs(occ)]
    gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
    med_gap = float(np.median(gaps)) if gaps else 0.0
    word_gap = max(3.0, 2.0 * med_gap)
    words, cur = [], [runs[0]] if runs else []
    for i, g in enumerate(gaps):
        if g >= word_gap:
            words.append(cur); cur = []
        cur.append(runs[i + 1])
    if cur:
        words.append(cur)
    glyph_ends = [e for _, e in runs]
    word_ends = [w[-1][1] for w in words]
    H, W = emask.shape
    xs = np.mgrid[0:H, 0:W][1]
    finalf = final.astype(np.float32)

    def reveal(bound):
        return composite(ref, finalf, emask & (xs < bound))

    hyps = [
        Hyp("text_wipe", lambda p: reveal(x0 + p * (x1 - x0)), {"direction": "right"}, 1, grid=np.linspace(0, 1, max(17, (x1 - x0) // 2))),
        Hyp("typewriter", lambda p: reveal(x0 if p == 0 else glyph_ends[min(len(glyph_ends) - 1, int(round(p * len(glyph_ends))) - 1)]),
            {"glyph_count": len(runs)}, 2, grid=np.linspace(0, 1, len(glyph_ends) + 1), curve="linear"),
        Hyp("word_stagger", lambda p: reveal(x0 if p == 0 else word_ends[min(len(word_ends) - 1, int(round(p * len(word_ends))) - 1)]),
            {"word_count": len(words)}, 3, grid=np.linspace(0, 1, len(word_ends) + 1), curve="linear"),
    ]
    # evidence: reveal boundary per frame
    bounds, partial = [], 0
    for f in frames:
        vis = M.diff_mask(f, ref, clean=0) & emask
        bb = M.bbox(vis)
        bx = bb[2] if bb else x0
        bounds.append(bx)
        for s_, e_ in runs:
            if s_ < bx < e_:
                cols = vis[:, s_:e_].any(axis=0)
                k = int(cols.sum())
                prefix = bool(cols[:k].all()) if k else False     # a half-drawn glyph is its left part, contiguous
                if prefix and 0.25 < k / len(cols) < 0.75:
                    partial += 1
                break
    inc = np.diff(bounds)
    glyph_w = float(np.median([e - s for s, e in runs])) if runs else 0
    word_w = float(np.median([w[-1][1] - w[0][0] for w in words])) if words else 0
    steps = [float(i) for i in inc if i > 0]
    plateaus = int(sum(1 for i in inc if i == 0))
    step_frames = [i + 1 for i, v in enumerate(inc) if v > 0]
    stagger = float(np.mean(np.diff(step_frames))) / fps if len(step_frames) >= 2 else None
    ev = {
        "reveal_boundary_px": [int(v) for v in bounds], "boundary_increments_px": [int(v) for v in inc],
        "plateau_frames": plateaus, "mean_step_px": _r(np.mean(steps)) if steps else None,
        "glyph_width_px": _r(glyph_w), "word_width_px": _r(word_w), "glyph_count": len(runs), "word_count": len(words),
        "partial_glyph_frames": partial,
        "continuous_advance": bool(plateaus == 0 and steps and np.std(steps) < 0.35 * np.mean(steps) and np.mean(steps) < glyph_w + med_gap + 1),
        "stagger_seconds": _r(stagger, 4), "fps": fps,
    }
    obs = Obs(frames, ref, final, emask, (max(0, x0 - 4), max(0, y0 - 4), min(W, x1 + 4), min(H, y1 + 4)))
    note = f"stagger_seconds={ev['stagger_seconds']} (per-step rhythm) from {len(step_frames)} steps."
    backtrack = any(i < -2 for i in inc)
    word_pitch = float(np.median([b[-1][1] - a[-1][1] for a, b in zip(words, words[1:])])) if len(words) > 1 else word_w
    oversized = bool(steps and max(steps) > 1.3 * max(word_pitch, word_w, glyph_w + med_gap + 1))
    ev["boundary_tracking_reliable"] = not (backtrack or oversized)
    if backtrack or oversized:
        res = contest(hyps, obs, margin)
        res["ambiguous"] = True
        why = "boundary moved backwards" if backtrack else "a reveal step exceeds the largest segmented unit (word)"
        return _finish("text_reveal", res, ev, note + f" Reveal-boundary tracking unreliable ({why}); glyph/word segmentation cannot be trusted under this compression. Not decided.")
    if partial == 0 and plateaus + len(steps) > 0:
        # a wipe sampled exactly on glyph boundaries is pixel-identical to a typewriter; with no partial glyph in any
        # frame the discrete hypotheses are the honest ones, so the (nested) continuous wipe is pruned.
        hyps = [h for h in hyps if h.name != "text_wipe"]
        note += " text_wipe pruned: no frame shows a partially rendered glyph."
    res = contest(hyps, obs, margin)
    best = res["results"][res["ranking"][0]]["scores"]["combined"]
    if best < 0.92:
        # clean/moderate winners reconstruct at >= 0.95; below this the glyph segmentation itself is compromised
        res["ambiguous"] = True
        note += f" Best reconstruction only {best:.2f}: glyph/word segmentation unreliable under this compression. Not decided."
    return _finish("text_reveal", res, ev, note)


# =====================================================================================
# Section 9: motion blur vs defocus blur
# =====================================================================================

def analyze_blur(sharp: np.ndarray, blurred: np.ndarray, margin: float = DEFAULT_MARGIN, mask=None) -> dict:
    g = M.gray(blurred)
    e, angles, aniso, amin = M.orientation_energy(g, 8, mask)
    sharp_f = sharp.astype(np.float32)
    frames = np.stack([sharp, blurred, blurred])   # interior frame = the blurred one
    Lmax, smax = 16.0, 6.0
    hyps = [Hyp("defocus", lambda p: gaussian_blur(sharp_f, p * smax), {"max_sigma_px": smax}, 1)]
    for k, th in enumerate(angles):
        hyps.append(Hyp(f"motion_{k}", lambda p, th=th: motion_blur(sharp_f, th, 1 + p * Lmax),
                        {"angle_deg": _r(np.degrees(th), 1)}, 2, label="motion"))
    obs = Obs(frames, sharp, blurred, np.ones(g.shape, bool), None)
    res = contest(hyps, obs, margin)
    alias = {n: res["results"][n]["label"] for n in res["results"]}
    rx, ry = M.mtf_axis_ratio(M.gray(sharp), g)
    loss_ratio = rx / max(ry, 1e-3)     # kernel MTF along fx vs fy: 1 = isotropic (defocus); far from 1 = motion
    reliable = min(rx, ry) < 0.8                 # some real loss happened; if both stay ~1 the evidence was compressed away
    both_dead = max(rx, ry) < 0.12                # both axes wiped out: isotropic by construction
    iso = bool(reliable and (both_dead or 0.7 < loss_ratio < 1.43))
    ev = {"orientation_energy": _r(e, 2), "orientation_angles_deg": _r(np.degrees(angles), 1), "anisotropy": _r(aniso),
          "min_energy_angle_deg": _r(np.degrees(amin), 1), "blur_direction_deg": _r(np.degrees(amin), 1) if aniso > 0.15 else None,
          "mtf_ratio_x_over_y": _r(loss_ratio), "mtf_x": _r(rx), "mtf_y": _r(ry), "mtf_reliable": bool(reliable), "hf_loss_isotropic": iso}
    note = None
    top = alias[res["ranking"][0]]
    if not reliable:
        res["ambiguous"] = True
        note = (f"MTF band unreliable (fx {rx:.2f}, fy {ry:.2f}): the reference and the blurred frame carry the same detail, "
                "so no kernel can be discriminated (compression has removed the evidence). Not decided.")
        return _finish("blur_type", res, ev, note, alias)
    if res["ambiguous"] and ((top == "defocus" and iso) or (top == "motion" and not iso)):
        res["ambiguous"] = False
        note = (f"Contest margin {res['margin']:.3f} is under {margin}, but the decisive test agrees with the leader: directional "
                f"MTF ratio fx/fy {loss_ratio:.2f} ({'isotropic' if iso else 'anisotropic'}). Treat as confirmed-by-evidence, not by reconstruction.")
    return _finish("blur_type", res, ev, note, alias)


# =====================================================================================
# Section 10: rotation vs perspective skew
# =====================================================================================

def _edge_slopes(mask):
    rows = [(y, np.nonzero(mask[y])[0]) for y in range(mask.shape[0]) if mask[y].any()]
    if len(rows) < 4:
        return None
    ys = np.array([y for y, _ in rows], float)
    left = np.array([r.min() for _, r in rows], float)
    right = np.array([r.max() for _, r in rows], float)
    k = max(2, len(ys) // 5)
    sel = slice(k, -k) if len(ys) > 3 * k else slice(None)  # avoid the corner rows
    sl = np.polyfit(ys[sel], left[sel], 1)[0]
    sr = np.polyfit(ys[sel], right[sel], 1)[0]
    cols = [(x, np.nonzero(mask[:, x])[0]) for x in range(mask.shape[1]) if mask[:, x].any()]
    xs = np.array([x for x, _ in cols], float)
    top = np.array([c.min() for _, c in cols], float)
    bot = np.array([c.max() for _, c in cols], float)
    k2 = max(2, len(xs) // 5)
    sel2 = slice(k2, -k2) if len(xs) > 3 * k2 else slice(None)
    st = np.polyfit(xs[sel2], top[sel2], 1)[0]
    sb = np.polyfit(xs[sel2], bot[sel2], 1)[0]
    return {"left": sl, "right": sr, "top": st, "bottom": sb}


def analyze_shape(frames: np.ndarray, bg: np.ndarray, margin: float = DEFAULT_MARGIN) -> dict:
    f0 = frames[0].astype(np.float32)
    H, W = f0.shape[:2]
    emask0 = M.diff_mask(frames[0], bg)
    c = M.centroid(emask0) or (W / 2, H / 2)
    emf = emask0.astype(np.float32)

    def r_warp(Minv):
        img, valid = warp(f0, Minv)
        m, _ = warp(emf, Minv)
        return composite(bg, img, (m > 0.5) & valid)

    th_max, k_max = np.radians(50), 1.2
    hyps = [
        Hyp("rotation_ccw", lambda p: r_warp(rotation_matrix(p * th_max, *c)), {"max_deg": 50}, 1, label="rotation"),
        Hyp("rotation_cw", lambda p: r_warp(rotation_matrix(-p * th_max, *c)), {"max_deg": -50}, 1, label="rotation"),
        Hyp("keystone_y+", lambda p: r_warp(keystone_matrix(p * k_max, c[0], c[1], H, "y")), {"axis": "y"}, 2, label="perspective"),
        Hyp("keystone_y-", lambda p: r_warp(keystone_matrix(-p * k_max, c[0], c[1], H, "y")), {"axis": "y"}, 2, label="perspective"),
        Hyp("keystone_x+", lambda p: r_warp(keystone_matrix(p * k_max, c[0], c[1], W, "x")), {"axis": "x"}, 2, label="perspective"),
        Hyp("keystone_x-", lambda p: r_warp(keystone_matrix(-p * k_max, c[0], c[1], W, "x")), {"axis": "x"}, 2, label="perspective"),
    ]
    obs = Obs(frames, frames[0], frames[-1], np.ones((H, W), bool), _region([M.diff_mask(f, bg) for f in frames], 6))
    res = contest(hyps, obs, margin)
    alias = {n: res["results"][n]["label"] for n in res["results"]}
    sl = _edge_slopes(M.diff_mask(frames[-1], bg))
    ev = {"edge_slopes_final": _r(sl) if sl else None}
    if sl:
        ev["left_right_parallel_delta"] = _r(abs(sl["left"] - sl["right"]))
        ev["top_bottom_parallel_delta"] = _r(abs(sl["top"] - sl["bottom"]))
        ev["opposite_edges_parallel"] = bool(ev["left_right_parallel_delta"] < 0.08 and ev["top_bottom_parallel_delta"] < 0.08)
    return _finish("rotation_vs_perspective", res, ev, None, alias)


# =====================================================================================
# Section 10: exit vs occluded (reverse of appear, restricted candidates)
# =====================================================================================

def analyze_exit(frames: np.ndarray, margin: float = DEFAULT_MARGIN, fps: float = 30.0) -> dict:
    alias = {"slide": "exit_slide", "slide_from_behind": "occluded_by_static_edge", "wipe": "wipe_out",
             "wipe_soft": "wipe_out_soft", "fade": "fade_out", "scale": "scale_out"}
    frag = analyze_appear(frames[::-1].copy(), margin, ["slide", "slide_from_behind", "wipe", "fade", "scale"], fps,
                          "exit_vs_occluded", alias)
    ev = frag.get("evidence", {})
    if ev:
        ev["occluding_boundary_stationary"] = ev.get("trailing_edge_pinned")
        ev["note_on_direction"] = "measured on the time-reversed frames: 'trailing edge' = the edge where content disappears"
    return frag


# =====================================================================================
# Section 10: real whip-pan vs synthetic whip transition (evidence-only)
# =====================================================================================

def analyze_whip(frames: np.ndarray, margin: float = DEFAULT_MARGIN) -> dict:
    an, hf, rm = [], [], []
    for f in frames:
        g = M.gray(f)
        an.append(M.orientation_energy(g)[2]); hf.append(M.hf_energy(g))
    for t in range(len(frames) - 1):
        rm.append(M.rmse(frames[t], frames[t + 1]))
    peak = int(np.argmin(hf))
    med = float(np.median(rm)) if rm else 0
    spike_t = int(np.argmax(rm)) if rm else 0
    cut_at_peak = bool(med > 0 and rm[spike_t] > 2.2 * med)   # a synthetic whip hides one hard cut somewhere in the blur
    # parallax: flow spread across blocks on the pre-blur frames
    flow, valid, _ = M.block_flow(frames[0], frames[1], block=8, search=8)
    flow2, valid2, _ = M.block_flow(frames[1], frames[2], block=8, search=8)
    consistent = valid & valid2 & (np.linalg.norm(flow - flow2, axis=2) <= 2.0)   # noise-driven matches are not repeatable
    if consistent.sum() >= 6 and consistent.sum() >= 0.3 * max(valid.sum(), 1) and np.linalg.norm(flow[consistent], axis=1).mean() > 1.0:
        mags = np.linalg.norm(flow[consistent], axis=1)
        parallax = float(mags.std() / (mags.mean() + 1e-6))
    else:
        parallax = None
    has_parallax = bool(parallax is not None and parallax > 0.25)
    # continuity: does the scene after the peak match the scene before, shifted? (real pan: yes)
    a, b = frames[max(0, peak - 3)], frames[min(len(frames) - 1, peak + 3)]
    mf = M.mean_flow(a, b, block=8, search=12)
    cont = None
    if mf is not None:
        sh, _ = shift(a, int(round(mf[0])), int(round(mf[1])))
        cont = M.ssim(sh, b)
    px = 0.5 if parallax is None else min(parallax / 0.5, 1.0)   # unknown parallax = no vote either way
    s_synth = 0.5 * cut_at_peak + 0.5 * (1 - px)
    s_real = 0.5 * (not cut_at_peak) + 0.5 * px
    scores = {"synthetic_whip": _r(s_synth), "real_whip_pan": _r(s_real)}
    order = sorted(scores, key=lambda k: -scores[k])
    mg = (scores[order[0]] - scores[order[1]]) / scores[order[0]] if scores[order[0]] > 0 else 0
    ev = {"anisotropy_curve": _r(an), "hf_energy_curve": _r(hf, 2), "peak_blur_frame": peak, "frame_diff_curve": _r(rm, 1),
          "cut_frame": spike_t if cut_at_peak else None, "cut_at_peak": cut_at_peak, "parallax_flow_spread": _r(parallax), "has_parallax": has_parallax,
          "scene_continuity_ssim_across_peak": _r(cont)}
    amb = mg < margin
    return {"question": "whip_real_vs_synthetic", "chosen": None if amb else order[0], "candidates": order, "rejected": order[1:] if not amb else [],
            "margin": _r(mg, 4), "margin_threshold": margin, "ambiguous": amb, "evidence": ev, "reconstruction_scores": None,
            "evidence_scores": scores,
            "note": "Evidence-only: no layered reconstruction for camera pans. Synthetic whips usually hide a hard cut at peak blur; real pans show parallax."}


# =====================================================================================
# Section 10: drop shadow vs second element
# =====================================================================================

def analyze_shadow(frames: np.ndarray, bg: np.ndarray, margin: float = DEFAULT_MARGIN) -> dict:
    """Shadow = pure darkening (every channel scaled by the same factor), blurred, shaped like its owner, at a fixed
    offset. A second element carries its own colour/texture. Owner = largest self-lit component."""
    offsets, shape_corrs, pure_fracs, contrasts, second_comp = [], [], [], [], []
    owner_bb = cand_bb = None
    bgf = bg.astype(np.float32) + 1.0
    bgg = M.gray(bg)
    for f in frames:
        D = M.gray(f) - bgg
        ratio = (f.astype(np.float32) + 1.0) / bgf
        pure_dark = (D < -8) & (ratio.std(axis=2) < 0.12) & (ratio.mean(axis=2) < 0.95)
        loose = M.diff_mask(f, bg, thresh=14)
        self_lit = loose & ~pure_dark
        self_lit = M.box_filter(self_lit.astype(np.float32), 1) > 0.5
        lab, n = M.label_components(self_lit)
        if n == 0:
            continue
        sizes = sorted(((lab == i).sum(), i) for i in range(1, n + 1))[::-1]
        owner = lab == sizes[0][1]
        second_comp.append(bool(len(sizes) > 1 and sizes[1][0] > 0.25 * sizes[0][0]))
        cand = loose & pure_dark & ~(M.box_filter(owner.astype(np.float32), 2) > 0)
        ob = M.bbox(owner)
        owner_bb = ob
        if cand.sum() < 30:
            shape_corrs.append(0.0); pure_fracs.append(0.0); contrasts.append(1.0)
            continue
        cand_bb = M.bbox(cand)
        dark = np.clip(-D, 0, None) * cand
        om = gaussian_blur(owner.astype(np.float32), 2.0)
        best = (-1.0, (0, 0))
        for dy in range(-24, 25, 2):
            for dx in range(-24, 25, 2):
                sh, _ = shift(om, dx, dy)
                sel = cand | (sh > 0.05)
                a, b = sh[sel], dark[sel]
                if sel.sum() < 30 or a.std() < 1e-3 or b.std() < 1e-3:
                    continue
                c = float(np.corrcoef(a, b)[0, 1])
                if c > best[0]:
                    best = (c, (dx, dy))
        shape_corrs.append(best[0]); offsets.append(best[1])
        pure_fracs.append(float(pure_dark[cand].mean()))
        contrasts.append(float(D[cand].std() / max(float(D[owner].std()), 1e-3)))
    if not shape_corrs:
        return {"question": "shadow_vs_second_element", "chosen": None, "ambiguous": True, "note": "no components found",
                "evidence": {}, "reconstruction_scores": None, "margin": 0.0, "rejected": [], "candidates": []}
    sc = float(np.median(shape_corrs))
    off = np.array(offsets, float) if offsets else None
    fixed = bool(off is not None and len(off) >= 2 and off.std(axis=0).max() < 2.5)
    pure = float(np.median(pure_fracs))
    low_contrast = bool(np.median(contrasts) < 0.5)
    has_second = bool(np.mean(second_comp) > 0.5)
    s_shadow = 0.4 * max(sc, 0) + 0.2 * fixed + 0.2 * pure + 0.2 * low_contrast
    s_second = 0.4 * (1 - max(sc, 0)) + 0.2 * has_second + 0.2 * (1 - pure) + 0.2 * (not low_contrast)
    scores = {"drop_shadow": _r(s_shadow), "second_element": _r(s_second)}
    order = sorted(scores, key=lambda k: -scores[k])
    mg = (scores[order[0]] - scores[order[1]]) / scores[order[0]] if scores[order[0]] > 0 else 0
    amb = mg < margin
    ev = {"shape_correlation_at_best_offset": _r(sc), "offset_px": _r(off.mean(axis=0), 1) if off is not None else None,
          "offset_fixed": fixed, "pure_darkening_fraction": _r(pure), "candidate_contrast_ratio": _r(np.median(contrasts)),
          "candidate_low_contrast": low_contrast, "second_self_lit_component": has_second,
          "owner_bbox": owner_bb, "candidate_bbox": cand_bb}
    return {"question": "shadow_vs_second_element", "chosen": None if amb else order[0], "candidates": order,
            "rejected": order[1:] if not amb else [], "margin": _r(mg, 4), "margin_threshold": margin, "ambiguous": amb,
            "evidence": ev, "reconstruction_scores": None, "evidence_scores": scores,
            "note": "Shadow = pure darkening (equal channel scaling), blurred, owner-shaped, fixed offset. Second element = self-lit component with its own colour."}


# =====================================================================================
# Section 10: speed ramp vs native slow motion
# =====================================================================================

def analyze_speed(frames: np.ndarray, margin: float = DEFAULT_MARGIN) -> dict:
    T = len(frames)
    diffs = [M.rmse(frames[t], frames[t + 1]) for t in range(T - 1)]
    dup_thr = max(0.75, 0.2 * float(np.median(diffs))) if diffs else 0.75
    dups = [d < dup_thr for d in diffs]
    ghost = []
    for t in range(1, T - 1):
        a, r = M.blend_alpha(frames[t], frames[t - 1], frames[t + 1])
        motion = M.rmse(frames[t - 1], frames[t + 1])
        ghost.append(bool(a is not None and 0.2 < a < 0.8 and r < 0.35 * motion and motion > 6))
    dup_ratio = float(np.mean(dups)) if dups else 0.0
    ghost_ratio = float(np.mean(ghost)) if ghost else 0.0
    # at most every other frame can be a duplicate/ghost of its neighbours, so a ratio of 0.5 is "all of them"
    dup_s, ghost_s = min(1.0, 2 * dup_ratio), min(1.0, 2 * ghost_ratio)
    scores = {"speed_ramp_duplicated_frames": _r(dup_s), "speed_ramp_flow_interpolated": _r(ghost_s),
              "native_slow_motion": _r(max(0.0, 1 - max(dup_s, ghost_s)))}
    order = sorted(scores, key=lambda k: -scores[k])
    mg = (scores[order[0]] - scores[order[1]]) / scores[order[0]] if scores[order[0]] > 0 else 0
    amb = mg < margin
    ev = {"duplicate_frame_ratio": _r(dup_ratio), "exact_duplicate_pairs": int(sum(dups)),
          "blend_ghost_frame_ratio": _r(ghost_ratio), "frame_diff_curve": _r([M.rmse(frames[t], frames[t + 1]) for t in range(T - 1)], 1)}
    return {"question": "speed_ramp_vs_native", "chosen": None if amb else order[0], "candidates": order,
            "rejected": order[1:] if not amb else [], "margin": _r(mg, 4), "margin_threshold": margin, "ambiguous": amb,
            "evidence": ev, "reconstruction_scores": None, "evidence_scores": scores,
            "note": "Duplicated frames => frame-hold ramp; frames that are blends of their neighbours => optical-flow interpolation; neither => native slow motion (or a ramp this test cannot see)."}


# =====================================================================================
# Plain-language explanation of a fragment
# =====================================================================================

_EV_TEXT = {
    "interior_flow_magnitude_px": "mean interior optical flow along the motion axis: {v} px/frame",
    "boundary_velocity_px": "visible-boundary velocity: {v} px/frame",
    "tracked_feature_displacement_px": "strongest interior feature moved {v} px/frame (moving = slide, fixed = wipe)",
    "trailing_edge_pinned": "trailing edge pinned: {v} (pinned = wipe/mask, moving = slide)",
    "origin_outside_final_bounds": "extrapolated origin outside the final bounds: {v} (true = slide-in)",
    "boundary_edge_softness_px": "moving-boundary 10-90 edge width: {v} px",
    "static_edge_softness_px": "element's own static edge width: {v} px",
    "boundary_shape": "reveal boundary shape: {v} (line = wipe, circle = iris, custom = matte)",
    "opacity": "opacity channel: {v}", "blur": "blur channel: {v}", "position": "position channel: {v}",
    "featureless": "element is featureless (no interior detail): {v}",
    "pops_in_one_frame": "element goes from absent to fully visible between two consecutive frames: {v}",
    "variance_min_over_max": "spatial variance collapses to {v} of its max (near 0 = dip, high = dissolve)",
    "dissolve_duration_frames": "frames that fit a*A+(1-a)*B strictly between A and B: {v}",
    "monotonic_pixel_fraction": "fraction of sampled pixels moving monotonically A->B: {v}",
    "dip_color_rgb": "recovered dip colour rgb: {v}", "dip_hold_frames": "frames held at the dip: {v}", "is_flash": "flash (spike, no plateau): {v}",
    "bg_scale_end": "background similarity scale at the end: {v}", "element_scale_end": "element scale after stabilising the camera: {v}",
    "center_distance_px": "distance between the camera zoom centre and the element centre: {v} px",
    "sharpness_signature": "sharpness while scaling: {v}", "layer_scale_ratio": "element/background scale ratio (1 = same, else parallax): {v}",
    "partial_glyph_frames": "frames showing a partially rendered glyph: {v} (any = text wipe)",
    "plateau_frames": "frames where the reveal boundary did not move: {v}", "mean_step_px": "mean boundary step: {v} px",
    "glyph_width_px": "glyph width: {v} px", "word_width_px": "word width: {v} px", "stagger_seconds": "per-step rhythm: {v} s",
    "anisotropy": "gradient-energy anisotropy: {v} (high = directional/motion, low = isotropic/defocus)",
    "blur_direction_deg": "blur direction: {v} deg",
    "opposite_edges_parallel": "opposite edges stay parallel: {v} (parallel = rotation, converging = perspective)",
    "occluding_boundary_stationary": "boundary where content disappears is stationary: {v} (stationary = occlusion)",
    "cut_at_peak": "hard cut hidden at peak blur: {v}", "has_parallax": "parallax between depth layers: {v}",
    "shape_correlation_at_best_offset": "candidate darkening correlates with the owner's blurred shape: {v}",
    "offset_fixed": "owner->candidate offset constant over time: {v}", "pure_darkening_fraction": "candidate pixels that are pure darkening: {v}",
    "second_self_lit_component": "a second self-lit component exists: {v}",
    "duplicate_frame_ratio": "exact duplicate consecutive frames: {v}", "blend_ghost_frame_ratio": "frames that are blends of their neighbours: {v}",
}


def explain(frag: dict) -> str:
    lines = [f"Question: {frag.get('question')}"]
    cands = frag.get("candidates") or []
    scores = frag.get("reconstruction_scores") or frag.get("evidence_scores") or {}
    if scores:
        lines.append("Considered: " + ", ".join(f"{k} ({v:.3f})" for k, v in scores.items()))
    elif cands:
        lines.append("Considered: " + ", ".join(cands))
    if frag.get("chosen"):
        lines.append(f"Chosen: {frag['chosen']} by margin {frag.get('margin')} (threshold {frag.get('margin_threshold')})"
                     + ("" if frag.get("reconstruction_scores") else " [evidence-only, no reconstruction contest]"))
    else:
        lines.append(f"No winner: ambiguous between {cands[:2]} (margin {frag.get('margin')} < {frag.get('margin_threshold')})")
    if frag.get("rejected"):
        lines.append("Rejected: " + ", ".join(frag["rejected"]))
    ev = frag.get("evidence") or {}
    shown = [(_EV_TEXT[k].format(v=ev[k])) for k in _EV_TEXT if k in ev and ev[k] is not None]
    if shown:
        lines.append("Evidence:")
        lines += ["  - " + t for t in shown]
    if frag.get("params"):
        lines.append(f"Fitted parameters: {frag['params']}")
    if frag.get("curve"):
        lines.append(f"Fitted progress curve (per frame): {frag['curve']}")
    if frag.get("note"):
        lines.append(f"Note: {frag['note']}")
    return "\n".join(lines)


def find_fragments(obj, path="") -> list[tuple[str, dict]]:
    out = []
    if isinstance(obj, dict):
        if "question" in obj and ("chosen" in obj or "candidates" in obj):
            out.append((path or "$", obj))
        for k, v in obj.items():
            out += find_fragments(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += find_fragments(v, f"{path}[{i}]")
    return out

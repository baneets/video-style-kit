"""The cut: which source seconds survive, in edit order. Pure Python over words + loudness; no editor involved."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import media, words as W

DANGLING = {"and", "so", "but", "or", "because", "cause", "um", "uh", "the", "a", "an", "my"}
CONNECTORS = {"and", "so", "but", "because", "cause", "like", "um", "uh", "ugh", "yeah", "okay", "then", "anyway"}


@dataclass
class Range:
    """One kept stretch of one source file, in SOURCE seconds."""
    file: int
    t0: float
    t1: float
    label: str = ""

    @property
    def dur(self) -> float:
        return self.t1 - self.t0


@dataclass
class Plan:
    files: list[Path]                                            # levelled copies the ranges index into
    fps: float = 30.0
    words: dict[int, list[dict]] = field(default_factory=dict)   # per file, source seconds
    ranges: list[Range] = field(default_factory=list)            # edit order
    removed: list[str] = field(default_factory=list)             # what was cut, for the report
    edit_words: list[dict] = field(default_factory=list)         # words in EDIT seconds

    @property
    def length(self) -> float:
        return sum(r.dur for r in self.ranges)

    def offsets(self) -> list[float]:
        out, t = [], 0.0
        for r in self.ranges:
            out.append(t)
            t += r.dur
        return out

    def frames(self) -> list[tuple[int, int]]:
        """Each range's (record start, length) in timeline frames, rounded ONCE on the running total so the
        timeline, the overlay and the headless render agree to the frame."""
        out, acc = [], 0.0
        for r in self.ranges:
            a = int(round(acc * self.fps))
            acc += r.dur
            out.append((a, int(round(acc * self.fps)) - a))
        return out

    @property
    def n_frames(self) -> int:
        return int(round(self.length * self.fps))

    def to_json(self) -> dict:
        return {"files": [str(f) for f in self.files], "fps": self.fps, "length_s": round(self.length, 3),
                "ranges": [{"file": r.file, "start": round(r.t0, 3), "end": round(r.t1, 3), "label": r.label} for r in self.ranges],
                "removed": self.removed}


def load_keep(path: Path, n_files: int) -> list[Range]:
    """keep.json: [{"file": 0, "start": 12.3, "end": 18.9, "label": "hook"}, ...] in the order they should play."""
    rows = json.loads(Path(path).read_text())
    out = []
    for i, r in enumerate(rows):
        f = int(r.get("file", 0))
        if not 0 <= f < n_files:
            raise SystemExit(f"keep.json row {i}: file {f} but only {n_files} file(s) were given")
        a, b = float(r["start"]), float(r["end"])
        if b - a < 0.2:
            raise SystemExit(f"keep.json row {i}: end {b} is not after start {a}")
        out.append(Range(f, a, b, str(r.get("label", ""))))
    return out


def sentence_safe(plan: Plan, keep: list[Range], tail: float = 0.30) -> list[Range]:
    """A hand-picked range never ends inside a word or on a dangling 'and'/'so': the end moves forward to the end of
    the word it cuts through, then back off any dangling connector. The start moves back to the start of a word it
    cuts through. Ranges never jump into another range of the same file."""
    out = []
    for n, r in enumerate(keep):
        ws = plan.words.get(r.file, [])
        t0, t1 = r.t0, r.t1
        inside = next((w for w in ws if w["start"] < t0 < w["end"]), None)
        if inside:
            t0 = inside["start"] - 0.04
        if n == 0:                          # the video's first word is never "and"/"so"/"but"/"um": start on the next one
            first = [w for w in ws if w["end"] > t0 + 0.05 and w["start"] < t1]
            while len(first) > 3 and W.bare(first[0]["word"]) in CONNECTORS:
                t0 = max(first[0]["end"], first[1]["start"] - 0.04)
                first = first[1:]
        last = [w for w in ws if w["start"] < t1 and w["end"] > t0]
        if last and last[-1]["end"] > t1:
            t1 = last[-1]["end"]
        while len(last) > 2 and W.bare(last[-1]["word"]) in DANGLING:
            last = last[:-1]
            t1 = last[-1]["end"]
        nxt = next((w["start"] for w in ws if w["start"] >= t1 - 1e-3), None)
        t1 = t1 + min(tail, max(0.0, (nxt - t1 - 0.02) if nxt is not None else tail))
        others = [q for q in keep if q is not r and q.file == r.file]
        t1 = min([t1] + [q.t0 for q in others if q.t0 >= r.t1 - 0.01])
        t0 = max([t0] + [q.t1 for q in others if q.t1 <= r.t0 + 0.01])
        out.append(Range(r.file, max(0.0, t0), t1, r.label))
    return out


def filler_blob(curve: np.ndarray, t: float, reach: float = 0.45, max_len: float = 0.8):
    """The filler's own sound: the voiced blob nearest `t` (whisper's time can sit 0.3 s off), padded 20 ms."""
    win = media.WIN
    i0, i1 = max(0, int((t - 1.5) / win)), min(len(curve), int((t + 1.5) / win))
    seg = curve[i0:i1]
    if len(seg) < 5:
        return None
    on = seg > max(np.percentile(seg, 15) + 10, np.percentile(seg, 90) - 14)
    blobs, k = [], 0
    while k < len(on):
        if on[k]:
            j = k
            while j < len(on) and on[j]:
                j += 1
            blobs.append(((i0 + k) * win, (i0 + j) * win))
            k = j
        else:
            k += 1
    near = [b for b in blobs if b[1] - b[0] <= max_len and (b[0] <= t <= b[1] or min(abs(b[0] - t), abs(b[1] - t)) <= reach)]
    if not near:
        return None
    inside = [b for b in near if b[0] <= t <= b[1]]
    a, b = inside[0] if inside else min(near, key=lambda b: abs((b[0] + b[1]) / 2 - t))
    return (a - 0.02, b + 0.02)


def tight_cut(plan: Plan, keep: list[Range] | None, pause_cap: float = 0.45, breath: float = 0.25,
              head: float = 0.10, tail: float = 0.18, fillers: bool = True, min_piece: float = 0.35) -> Plan:
    """Inside every kept range (the whole files when none are given): drop fillers and stumbles ("I, I"), shrink any
    pause over `pause_cap` to `breath`, snap every new edge to a loudness dip so no word is clipped."""
    base = keep or [Range(i, 0.0, media.probe(f)["duration"], "all") for i, f in enumerate(plan.files)]
    out = []
    for rg in base:
        ws = [w for w in plan.words.get(rg.file, []) if w["end"] > rg.t0 and w["start"] < rg.t1]
        if not ws:
            plan.removed.append(f"{rg.label or round(rg.t0, 1)}: no speech found, dropped")
            continue
        curve = media.rms_curve(plan.files[rg.file])
        cuts = [(s + breath / 2, e - breath / 2) for s, e in media.silences(curve, pause_cap)
                if e > rg.t0 and s < rg.t1 and e - s > pause_cap + breath]
        if cuts:
            plan.removed.append(f"{len(cuts)} pause(s) shortened to {breath:.2f}s in {rg.label or 'the take'}")
        if fillers:
            for i, w in enumerate(ws):
                b = W.bare(w["word"])
                nx = ws[i + 1] if i + 1 < len(ws) else None
                if nx and b in W.STUMBLE and b == W.bare(nx["word"]) and nx["start"] - w["end"] < 0.6:
                    cuts.append((max(ws[i - 1]["end"] if i else rg.t0, w["start"] - 0.04), nx["start"] - 0.02))
                    plan.removed.append(f"stumble {w['word']!r} at {w['start']:.1f}s")
                elif W.FILLER_RE.match(re.sub(r"[^a-z]", "", w["word"].lower()) or "x"):
                    lo = ws[i - 1]["end"] if i else rg.t0
                    hi = ws[i + 1]["start"] if i + 1 < len(ws) else rg.t1
                    blob = filler_blob(curve, (w["start"] + w["end"]) / 2)
                    cuts.append(blob if blob else (max(lo, w["start"] - 0.04), min(hi, w["end"] + 0.04)))
                    plan.removed.append(f"filler {w['word']!r} at {w['start']:.1f}s")
        t0 = max(rg.t0, ws[0]["start"] - head) if keep is None else rg.t0
        t1 = min(rg.t1, ws[-1]["end"] + tail) if keep is None else rg.t1
        pieces, t = [], t0
        for a, b in sorted(cuts):
            a, b = max(a, t0), min(b, t1)
            if b <= t or a >= t1:
                continue
            if a - t >= 0.10:
                pieces.append((t, a))
            t = max(t, b)
        if t1 - t >= 0.10:
            pieces.append((t, t1))
        merged = []                          # a piece too short to show (one quick word before a pause) keeps its pause
        for a, b in pieces:                  # instead of vanishing: it joins the next piece
            if merged and merged[-1][1] - merged[-1][0] < min_piece:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        if len(merged) > 1 and merged[-1][1] - merged[-1][0] < min_piece:
            last = merged.pop()
            merged[-1] = (merged[-1][0], last[1])
        pieces = merged
        for j, (a, b) in enumerate(pieces):
            if j:
                a = media.snap_to_dip(a, curve, ws)
            if j < len(pieces) - 1:
                b = media.snap_to_dip(b, curve, ws)
            if b - a >= 0.10:
                out.append(Range(rg.file, a, b, rg.label if not j else f"{rg.label}+{j}"))
    plan.ranges = out
    return plan


def head_zero(plan: Plan) -> Plan:
    """The video starts on the first word: one frame before the measured speech onset."""
    if not plan.ranges:
        return plan
    r = plan.ranges[0]
    t = media.onset(media.rms_curve(plan.files[r.file]), max(0.0, r.t0 - 0.5))
    if any(r.t0 - 0.5 < w["end"] <= r.t0 + 0.1 and w["start"] < r.t0 - 0.05 for w in plan.words.get(r.file, [])):
        t = r.t0                                  # continuous speech: never reach back onto the previous word
    r.t0 = max(0.0, min(r.t0, t) - 1 / plan.fps) if t < r.t0 else max(r.t0, t - 0.08)
    return plan


def tail_hold(plan: Plan, hold: float = 0.30) -> Plan:
    """The last range ends `hold` seconds after its last word, never reaching the next word."""
    if not plan.ranges:
        return plan
    r = plan.ranges[-1]
    ws = [w for w in plan.words.get(r.file, []) if r.t0 <= w["start"] < r.t1 and w["end"] <= r.t1 + 0.05]
    if ws:
        nxt = min((w["start"] for w in plan.words.get(r.file, []) if w["start"] > ws[-1]["start"] + 1e-3), default=float("inf"))
        dur = media.probe(plan.files[r.file])["duration"]
        r.t1 = min(dur, ws[-1]["end"] + hold, nxt - 0.02)
    return plan


def sliver_clean(plan: Plan, min_s: float = 0.2) -> Plan:
    """Nothing shorter than `min_s` reaches the timeline (a 3-frame flash reads as a glitch)."""
    plan.ranges = [r for r in plan.ranges if r.dur >= min_s]
    return plan


def edit_words(plan: Plan) -> list[dict]:
    """Every surviving word, re-timed into EDIT seconds (a word must lie at least half inside a kept range)."""
    out, t = [], 0.0
    for r in plan.ranges:
        for w in plan.words.get(r.file, []):
            a, b = max(w["start"], r.t0), min(w["end"], r.t1)
            if b - a >= 0.5 * (w["end"] - w["start"]) and b > a:
                out.append({"word": w["word"], "start": round(t + a - r.t0, 3), "end": round(t + b - r.t0, 3)})
        t += r.dur
    return out


if __name__ == "__main__":
    # ponytail: the one runnable check for the pure bookkeeping (no ffmpeg).
    ws = [{"word": "hi", "start": 1.0, "end": 1.3}, {"word": "there", "start": 5.0, "end": 5.4}]
    p = Plan([Path("a")], 30, {0: ws}, [Range(0, 0.9, 1.4), Range(0, 4.9, 5.6)])
    ew = edit_words(p)
    assert [w["word"] for w in ew] == ["hi", "there"] and abs(ew[1]["start"] - 0.6) < 1e-6, ew
    assert p.frames() == [(0, 15), (15, 21)] and p.n_frames == 36, p.frames()
    print("cut.py self-check ok")

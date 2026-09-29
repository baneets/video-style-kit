"""stylekit edit: your footage + a style -> a finished video (Resolve timeline + render, or headless).

Order of work (every step prints what it measured):
  1. voice-level each file (-16 LUFS copy, video untouched) and transcribe it (on-device, cached)
  2. the cut: your keep.json ranges (sentence-safe) or the whole files; fillers, stumbles and long pauses out;
     the first frame is the first word; the end holds on room tone
  3. framing: scale to the style's aspect, face-centred; punch-ins at the style's rhythm
  4. grade: each file's colour matched to the reference's (CDL baked into a .cube), at the style's strength
  5. text: captions / hook / titles rendered by Remotion as one transparent clip, positioned by the style
  6. music (your file only): looped/trimmed to the cut, levelled under the voice, fades baked in
  7. build + render in DaVinci Resolve Studio (V1 talk, V2 b-roll, V3 text, A1 voice with J-cuts, A2 music), or the
     same edit with ffmpeg when Resolve Studio is not reachable
  8. finish: loudness to the style's target, a review sheet of the result, report.json
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from . import cut as C, faces, grade, headless, media, overlays, paths, recipe as R, resolve_build as RB, words as W


def _frames_to(t: float, fps: float) -> int:
    return int(round(t * fps))


def face_track(file: Path, every: float = 2.0) -> list[tuple[float, tuple[float, float, float] | None]]:
    """Face (centre x, centre y, height) sampled every `every` seconds over the whole file, cached."""
    cache = media.work_dir(file) / f"faces3_{every:g}.json"
    if cache.exists():
        return [(t, tuple(c) if c else None) for t, c in json.loads(cache.read_text())]
    dur = media.probe(file)["duration"]
    ts = list(np.arange(0.3, max(0.4, dur - 0.1), every))[:400]
    cs = faces.face_centres(file, ts)
    cache.write_text(json.dumps([[t, c] for t, c in zip(ts, cs)]))
    return list(zip(ts, cs))


def centre_in(track, t0: float, t1: float) -> tuple[tuple[float, float], float | None]:
    """(face centre, face height) over a span: the median of the samples inside it, else the nearest sample."""
    got = [c for t, c in track if t0 - 0.5 <= t <= t1 + 0.5 and c]
    if not got:
        near = sorted(((abs(t - (t0 + t1) / 2), c) for t, c in track if c), key=lambda x: x[0])
        got = [near[0][1]] if near else []
    if not got:
        return (0.5, 0.45), None
    return (float(np.median([c[0] for c in got])), float(np.median([c[1] for c in got]))), float(np.median([c[2] for c in got]))


def prepare_bed(music: Path, length_s: float, target_lufs: float, fade_in: float, fade_out: float, out: Path) -> Path:
    """The music as its own file: looped/trimmed to the cut, levelled to `target_lufs`, fades baked in."""
    media.run(["ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-i", music, "-t", f"{length_s:.3f}",
               "-af", f"loudnorm=I={target_lufs}:TP=-2:LRA=11,afade=t=in:d={fade_in},"
                      f"afade=t=out:st={max(0.0, length_s - fade_out):.3f}:d={fade_out}",
               "-ar", "48000", "-ac", "2", out], check=True)
    return out


def finish_audio(src: Path, out: Path, target: float) -> Path:
    """Final loudness to `target` LUFS, true peak -1 dB (two-pass loudnorm), picture copied untouched."""
    import re
    r = media.run(["ffmpeg", "-v", "info", "-i", src, "-af", f"loudnorm=I={target}:TP=-1:LRA=11:print_format=json",
                   "-f", "null", "-"], capture_output=True, text=True).stderr
    m = json.loads(re.findall(r"\{[^{}]*\}", r)[-1])
    af = (f"loudnorm=I={target}:TP=-1:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    media.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy", "-af", af,
               "-ar", "48000", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", out], check=True)
    return out


def review_sheet(video: Path, out: Path, n: int = 12) -> Path:
    from . import sheets
    dur = media.probe(video)["duration"]
    tiles = [sheets._tile(media.grab_frame(video, t, 480), 240, f"{t:.1f}s") for t in np.linspace(0.05, max(0.1, dur - 0.25), n)]
    return sheets.grid(tiles, 6, out, f"{video.name}: {dur:.1f}s")


def assemble_audio(plan: C.Plan, out: Path) -> Path:
    """The cut's voice alone (no J-cut shift), for re-transcribing the captions against what is actually heard."""
    parts = []
    for i, r in enumerate(plan.ranges):
        p = out.parent / f"asm{i:04d}.wav"
        media.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{r.t0:.4f}", "-t", f"{r.dur:.4f}", "-i", plan.files[r.file],
                   "-vn", "-ac", "1", "-ar", "16000", p], check=True)
        parts.append(p)
    lst = out.parent / "asm.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
    media.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
    for p in parts:
        p.unlink()
    return out


def load_json_rows(path: str | None, what: str) -> list[dict]:
    if not path:
        return []
    try:
        rows = json.loads(Path(path).expanduser().read_text())
    except (OSError, json.JSONDecodeError) as ex:
        raise SystemExit(f"{what} {path}: {ex}")
    if not isinstance(rows, list):
        raise SystemExit(f"{what} {path} must be a JSON list")
    return rows


def edit(files: list[str], style: str, name: str | None = None, keep: str | None = None, titles: str | None = None,
         broll: str | None = None, hook: str | None = None, music: str | None = None, engine: str = "auto",
         captions: bool = True, plan_only: bool = False, aspect: str | None = None) -> dict:
    t_start = time.time()
    rec = R.load(style)
    slug = paths.slug(style)
    srcs = [Path(f).expanduser().resolve() for f in files]
    for f in srcs:
        if not f.exists():
            raise SystemExit(f"no such file: {f}")
    name = paths.slug(name or srcs[0].stem)
    work = paths.WORK / "edits" / f"{slug}__{name}"
    work.mkdir(parents=True, exist_ok=True)
    aspect = aspect or rec["canvas"]["aspect"]
    if aspect not in R.ASPECTS:
        raise SystemExit(f"aspect must be one of {list(R.ASPECTS)}")
    out_w, out_h = R.ASPECTS[aspect]

    info = [media.probe(f) for f in srcs]
    for f, i in zip(srcs, info):
        if not i["width"]:
            raise SystemExit(f"{f} has no video stream")
        if not i["has_audio"]:
            raise SystemExit(f"{f} has no sound: the talking footage needs its audio (b-roll goes in --broll)")
    rate_label, fps, rational = media.standard_rate(float(rec["canvas"].get("fps") or info[0]["fps"]))
    print(f"edit {name!r} in style {slug!r}: {len(srcs)} file(s), {aspect} {out_w}x{out_h} @ {rate_label} fps")

    # 1. voice + words
    voices = []
    for f, i in zip(srcs, info):
        v = media.voice_copy(f, float(rec["audio"]["voice_lufs"]))
        if abs(media.standard_rate(i["fps"])[1] - fps) > 0.01:
            print(f"  {f.name}: {i['fps']:.3f} fps differs from the timeline; conforming a copy")
            v = media.conform_fps(v, rational)
        voices.append(v)
    words = {}
    for k, v in enumerate(voices):
        print(f"  transcribing {srcs[k].name} (on-device; cached after the first time)")
        words[k] = W.transcribe(v)
    plan = C.Plan(voices, fps, words)

    # 2. the cut
    ck = rec["cut"]
    keep_r = C.load_keep(Path(keep).expanduser(), len(srcs)) if keep else None
    if keep_r:
        keep_r = C.sentence_safe(plan, keep_r)
    C.tight_cut(plan, keep_r, float(ck["pause_cap_s"]), float(ck["breath_s"]), fillers=bool(ck["remove_fillers"]))
    C.head_zero(plan)
    C.tail_hold(plan, float(ck["tail_hold_s"]))
    C.sliver_clean(plan, float(ck["min_shot_s"]))
    if not plan.ranges:
        raise SystemExit("nothing survived the cut: no speech was found in the kept ranges")
    mapped = C.edit_words(plan)
    (work / "plan.json").write_text(json.dumps(plan.to_json(), indent=1))
    (work / "cut_transcript.txt").write_text(
        "\n".join(f"{w['start']:7.2f}  {w['word']}" for w in mapped))
    print(f"  cut: {len(plan.ranges)} pieces, {plan.length:.1f}s (from {sum(i['duration'] for i in info):.1f}s); "
          + "; ".join(plan.removed[:6]) + (" ..." if len(plan.removed) > 6 else ""))
    if plan_only:
        print(f"  plan only: {work / 'plan.json'} and {work / 'cut_transcript.txt'} (edit-time words for titles/b-roll)")
        return {"plan": str(work / "plan.json"), "transcript": str(work / "cut_transcript.txt")}

    # caption words: re-transcribe what the cut actually sounds like (source-file word times drift on long takes)
    ew = mapped
    try:
        (work / "cut.words.json").unlink(missing_ok=True)
        heard = W.transcribe(assemble_audio(plan, work / "cut.wav"), work / "cut.words.json")
        # HEAR the first word: source word times run early, so a start placed after "but" can still open on it
        first = plan.ranges[0]
        if len(heard) > 3 and W.bare(heard[0]["word"]) in C.CONNECTORS and heard[1]["start"] - 0.04 < first.dur - 0.3:
            d = max(heard[0]["end"], heard[1]["start"] - 0.04)
            first.t0 += d
            print(f"  opening word {heard[0]['word']!r} heard at the start; the video now starts on {heard[1]['word']!r}")
            heard = [{**w, "start": round(w["start"] - d, 3), "end": round(w["end"] - d, 3)} for w in heard[1:]]
            mapped = C.edit_words(plan)
            (work / "plan.json").write_text(json.dumps(plan.to_json(), indent=1))
        if len(heard) >= 0.6 * len(mapped):
            ew = heard
    except Exception as ex:                                  # noqa: BLE001
        print(f"  (caption re-timing skipped: {str(ex)[:120]}; using the source word times)")

    # 3. framing + 4. grade
    frames = plan.frames()
    avoid = bool(rec["captions"].get("avoid_face", True))
    need_faces = avoid or any(abs(i["width"] / i["height"] - out_w / out_h) > 0.02 for i in info) or rec["punch_in"]["every"]
    tracks = {k: face_track(v) for k, v in enumerate(voices)} if need_faces else {}
    bands = []                                            # (edit t0, edit t1, face top, face bottom) per piece
    cubes, cdls, gerr = {}, {}, {}
    if rec["grade"]["mode"] == "match":
        ref = rec["grade"]["reference"]
        for k, v in enumerate(voices):
            spans = [(r.t0, r.t1) for r in plan.ranges if r.file == k]
            if not spans:
                continue
            src_stats = grade.measure(grade.sample_frames(v, 6, spans))
            full = grade.solve_cdl(src_stats, ref)
            cdl = grade.soften(full, float(rec["grade"]["strength"]))
            cdls[k] = cdl
            cubes[k] = str(grade.write_cube(cdl, work / f"grade_{slug}_{name}_{k}.cube", title=f"{slug} {name} {k}"))
            gerr[k] = grade.predicted_error(src_stats, ref, full)
            print(f"  grade {srcs[k].name}: matched to the reference at {rec['grade']['strength']:.0%} "
                  f"(full-match worst channel error {gerr[k]:.3f}{', a CDL cannot reach this look fully' if gerr[k] > 0.05 else ''})")
    every, pz = int(rec["punch_in"]["every"] or 0), float(rec["punch_in"]["zoom"])
    # J-cut: the next piece's voice starts `j` frames before its picture; its audio stays in sync (it is simply
    # extended earlier) and the previous piece's audio ends that much sooner.
    j = _frames_to(float(ck["j_cut_s"]), fps)
    leads = [j if n_ and _frames_to(r.t0, fps) >= j and frames[n_ - 1][1] > 2 * j else 0 for n_, r in enumerate(plan.ranges)]
    ranges = []
    for n_, (r, (rec_f, len_f)) in enumerate(zip(plan.ranges, frames)):
        i = info[r.file]
        zoom = pz if every and n_ % every == every - 1 else 1.0
        centre, fh = centre_in(tracks[r.file], r.t0, r.t1) if r.file in tracks else ((0.5, 0.45), None)
        win = faces.window(i["width"], i["height"], out_w, out_h, centre, zoom)
        band = faces.face_band(centre, fh, win, out_h)
        if band:
            bands.append((rec_f / fps, (rec_f + len_f) / fps, *band))
        src_in = _frames_to(r.t0, fps)
        nxt = leads[n_ + 1] if n_ + 1 < len(leads) else 0
        ranges.append({"file": r.file, "t0": src_in / fps, "src_in_f": src_in, "src_len_f": len_f,
                       "rec_f": rec_f, "len_f": len_f, "has_audio": info[r.file]["has_audio"],
                       "a_in_f": src_in - leads[n_], "a_len_f": len_f + leads[n_] - nxt, "a_rec_f": rec_f - leads[n_],
                       "zoom": round(win["scale"] / min(out_w / i["width"], out_h / i["height"]), 5),
                       "pan": round(win["pan"], 2), "tilt": round(win["tilt"], 2),
                       "sw": win["w"], "sh": win["h"], "left": win["left"], "top": win["top"], "lut": cubes.get(r.file),
                       "cdl": cdls.get(r.file)})

    # b-roll (your clips), placed by phrase or time, graded to the same reference
    broll_rows = []
    for j, b in enumerate(load_json_rows(broll, "broll")):
        bf = Path(b["file"]).expanduser().resolve()
        if not bf.exists():
            raise SystemExit(f"b-roll file not found: {bf}")
        at = overlays.find_phrase(ew, b["at_phrase"]) if "at_phrase" in b else b.get("at")
        if at is None:
            print(f"  b-roll {bf.name}: phrase {b.get('at_phrase')!r} is not in the cut; skipped")
            continue
        bi = media.probe(bf)
        b_fps = bi["fps"] or fps
        dur = min(float(b.get("dur", 2.0)), plan.length - float(at), max(0.1, bi["duration"] - float(b.get("start", 0.0))))
        if dur < 0.3:
            continue
        win = faces.window(bi["width"], bi["height"], out_w, out_h, (0.5, 0.5), 1.0)
        lut, bcdl = None, None
        if rec["grade"]["mode"] == "match":
            st = grade.measure(grade.sample_frames(bf, 4, [(float(b.get("start", 0.0)), float(b.get("start", 0.0)) + dur)]))
            bcdl = grade.soften(grade.solve_cdl(st, rec["grade"]["reference"]), float(rec["grade"]["strength"]))
            lut = str(grade.write_cube(bcdl, work / f"grade_{slug}_{name}_broll{j}.cube"))
        broll_rows.append({"file": str(bf), "t0": float(b.get("start", 0.0)), "src_in_f": _frames_to(float(b.get("start", 0.0)), b_fps),
                           "src_len_f": _frames_to(dur, b_fps), "rec_f": _frames_to(float(at), fps), "len_f": _frames_to(dur, fps),
                           "zoom": round(win["scale"] / min(out_w / bi["width"], out_h / bi["height"]), 5),
                           "sw": win["w"], "sh": win["h"], "left": win["left"], "top": win["top"], "lut": lut, "cdl": bcdl})

    # 5. text
    title_rows, warn = overlays.resolve_titles(load_json_rows(titles, "titles"), ew, plan.length, float(rec["titles"]["hold_s"]))
    for w_ in warn:
        print("  " + w_)
    if hook and not rec["hook"].get("enabled", False):
        print("  note: this style's references had no hook title; showing yours anyway (styled from recipe.hook)")
    props = overlays.build_props(rec, out_w, out_h, fps, plan.n_frames, ew, hook, title_rows, captions)
    moved = overlays.avoid_faces(props, bands) if avoid else 0
    if moved:
        print(f"  text: {moved} caption/title line(s) moved off the speaker's face")
    overlay = overlays.render(props, work / "overlay.mov") if overlays.has_text(props) else None

    # 6. music
    bed = None
    if music:
        mf = Path(music).expanduser().resolve()
        if not mf.exists():
            raise SystemExit(f"music file not found: {mf}")
        target = float(rec["audio"]["voice_lufs"]) - float(rec["audio"]["music_under_voice_db"])
        bed = prepare_bed(mf, plan.length, target, float(rec["audio"]["music_fade_in_s"]), float(rec["audio"]["music_fade_out_s"]),
                          work / "music_bed.wav")
        print(f"  music: {mf.name} levelled to {target:.0f} LUFS ({rec['audio']['music_under_voice_db']:.0f} dB under the voice)")
    elif rec["audio"].get("music"):
        print("  note: the references had music under the voice; pass --music <your track> to match that")

    job = {"project": f"{slug} - {name}", "timeline": f"{name} ({slug})", "w": out_w, "h": out_h, "fps": fps,
           "fps_label": rate_label, "fps_rational": rational, "n_frames": plan.n_frames, "files": [str(v) for v in voices],
           "ranges": ranges, "broll": broll_rows, "overlay": str(overlay) if overlay else None,
           "music": str(bed) if bed else None}
    (work / "job.json").write_text(json.dumps(job, indent=1))

    # 7. build + render
    out_dir = paths.EXPORTS / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    used, raw, resolve_note = "headless", None, ""
    if engine in ("auto", "resolve"):
        r = RB.launch_and_connect() if engine == "resolve" else RB.connect()
        if r is None and engine == "auto":
            r = RB.launch_and_connect(60)
        if RB.is_studio(r):
            print("  building the timeline in DaVinci Resolve Studio")
            job_r = dict(job)
            job_r["fps"] = rate_label
            built = RB.build(r, job_r)
            print(f"  Resolve: project {job['project']!r}, {built['pieces']} pieces, {built['graded']} graded; rendering")
            raw = RB.render(built["project"], built["timeline"], work, name, out_w, out_h)
            used = "resolve"
        else:
            resolve_note = ("DaVinci Resolve Studio is not reachable (not installed, not running, the free edition, or "
                            "external scripting is off in Preferences > System > General)")
            if engine == "resolve":
                raise SystemExit(resolve_note)
            print(f"  {resolve_note}; rendering the same edit headless with ffmpeg")
    if raw is None:
        raw = headless.render({**job, "fps": fps}, work / "headless", work / f"{name}.headless.mp4")
    final = finish_audio(raw, out_dir / f"{name}.mp4", float(rec["audio"]["final_lufs"]))

    # 8. check the result
    got = media.probe(final)
    loud = media.lufs(final)
    sheet = review_sheet(final, work / "review.jpg")
    report = {"output": str(final), "engine": used, "resolve_note": resolve_note or None, "style": slug,
              "length_s": round(got["duration"], 2), "expected_s": round(plan.n_frames / fps, 2),
              "size": f"{got['width']}x{got['height']}", "fps": rate_label, "lufs": loud, "pieces": len(ranges),
              "removed": plan.removed, "broll": len(broll_rows), "titles": len(title_rows), "hook": hook,
              "captions": bool(props["captions"] and props["captions"]["rows"]),
              "grade_error_full_match": gerr, "review_sheet": str(sheet), "work": str(work),
              "seconds": round(time.time() - t_start, 1)}
    (work / "report.json").write_text(json.dumps(report, indent=1))
    problems = []
    if abs(got["duration"] - plan.n_frames / fps) > 0.25:
        problems.append(f"length {got['duration']:.2f}s but the cut is {plan.n_frames / fps:.2f}s")
    if (got["width"], got["height"]) != (out_w, out_h):
        problems.append(f"size {got['width']}x{got['height']} but the style is {out_w}x{out_h}")
    if loud is None or abs(loud - float(rec["audio"]["final_lufs"])) > 1.5:
        problems.append(f"loudness {loud} LUFS, target {rec['audio']['final_lufs']}")
    report["problems"] = problems
    (work / "report.json").write_text(json.dumps(report, indent=1))
    print(f"\nDONE in {report['seconds']:.0f}s ({used}): {final}")
    print(f"  {report['length_s']}s, {report['size']}, {rate_label} fps, {loud} LUFS; review sheet: {sheet}")
    if problems:
        print("  PROBLEMS: " + "; ".join(problems))
    return report

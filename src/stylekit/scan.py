"""stylekit scan: reference video(s) -> a measured style folder + a Claude Code skill.

  styles/<style>/
    recipe.json            the numbers the editor uses (measured where possible; hand-tunable, never overwritten)
    recipe.measured.json   the latest measurement, always rewritten (compare it with recipe.json after a rescan)
    STYLE.md               the style in words: written by Claude from the sheets (template until then)
    SKILL.md               the Claude Code skill (linked as ~/.claude/skills/style-<style>)
    refs.json              where each reference came from
    scan/ref<N>-<id>/      deep.json (every measurement), analysis.json, audio.json, sheets/*.jpg
    scan/profile.json      all references aggregated (vidkit style build)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import grade, media, paths, recipe, sheets


def fetch(ref: str, cookies: str | None = None) -> Path:
    """A local file stays where it is; a link is downloaded once (yt-dlp) into .work/downloads."""
    if not ref.startswith(("http://", "https://")):
        p = Path(ref).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"no such file: {p}")
        return p
    dl = paths.WORK / "downloads"
    dl.mkdir(parents=True, exist_ok=True)
    import hashlib
    stem = hashlib.sha1(ref.encode()).hexdigest()[:12]
    have = [p for p in dl.glob(f"{stem}.*") if p.suffix in (".mp4", ".mkv", ".webm", ".mov")]
    if have:
        return have[0]
    cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-playlist", "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
           "--merge-output-format", "mp4", "-o", str(dl / f"{stem}.%(ext)s"), ref]
    if cookies:
        cmd[3:3] = ["--cookies-from-browser", cookies]
    print(f"  downloading {ref}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    have = [p for p in dl.glob(f"{stem}.*") if p.suffix in (".mp4", ".mkv", ".webm", ".mov")]
    if r.returncode or not have:
        hint = "" if cookies else " (Instagram/TikTok often need a login: add --cookies chrome, or download the video and pass the file)"
        raise SystemExit(f"could not download {ref}{hint}:\n{(r.stderr or r.stdout).strip()[-600:]}")
    return have[0]


def measure_audio(video: Path, deep: dict) -> dict | None:
    """Is there music under the voice, and how far under? Level in the pauses between words vs under the words."""
    if not media.probe(video)["has_audio"]:
        return None
    curve = media.rms_curve(video)
    t = np.arange(len(curve)) * media.WIN
    words = [(w["start_time"], w["end_time"]) for s in deep.get("transcript") or [] for w in s.get("word_times", [])]
    if not words:                                             # no speech at all: whatever is there is the music
        lvl = float(np.median(curve)) if len(curve) else -90.0
        return {"music": lvl > -45, "speech_db": lvl, "gap_db": lvl, "under_db": 0.0, "note": "no speech found"}
    speech = np.zeros(len(curve), bool)
    for a, b in words:
        speech[(t >= a) & (t <= b)] = True
    gap = np.zeros(len(curve), bool)
    for (a0, a1), (b0, _) in zip(words, words[1:]):
        if b0 - a1 >= 0.25:
            gap[(t >= a1 + 0.08) & (t <= b0 - 0.08)] = True
    sdb = float(np.median(curve[speech])) if speech.any() else -90.0
    gdb = float(np.median(curve[gap])) if gap.sum() >= 5 else float(np.percentile(curve, 5))
    music = gdb > -48 and (sdb - gdb) < 30
    return {"music": bool(music), "speech_db": round(sdb, 1), "gap_db": round(gdb, 1), "under_db": round(sdb - gdb, 1)}


def aggregate_profile(sdir: Path, analyses: list[Path]) -> Path | None:
    """vidkit's cross-reference aggregate (structure, beat rules, animation vocabulary): zero cost, disk only."""
    from vidkit import style as vstyle
    name = sdir.name
    try:
        vstyle.create(name)
        st = vstyle.load_style(name)
        st["refs"] = []
        vstyle.save_style(name, st)
        for a in analyses:
            vstyle.add_ref(name, str(a), a, {"prompt": "deep", "complete": True, "problems": []})
        prof = vstyle.build(name)
    except Exception as ex:                                   # noqa: BLE001  (the aggregate is a bonus, never a blocker)
        print(f"  (profile aggregate skipped: {str(ex)[:160]})")
        return None
    out = sdir / "scan" / "profile.json"
    out.write_text(json.dumps(prof, indent=1))
    return out


def link_skill(sdir: Path) -> Path:
    link = paths.SKILLS / f"style-{sdir.name}"
    paths.SKILLS.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == sdir.resolve():
            return link
        if link.is_symlink():
            link.unlink()
        else:
            raise SystemExit(f"{link} exists and is not a link made by this kit; move it away and re-run")
    link.symlink_to(sdir, target_is_directory=True)
    return link


def write_docs(sdir: Path, display: str, refs: list[dict], notes: str) -> None:
    """SKILL.md (always refreshed: it only points at files) and STYLE.md (written once; Claude fills it in)."""
    slug = sdir.name
    tmpl = (paths.TEMPLATES / "STYLE_SKILL.md").read_text()
    sub = {"{{SLUG}}": slug, "{{NAME}}": display, "{{KIT}}": str(paths.ROOT), "{{N}}": str(len(refs)),
           "{{STYLE_DIR}}": str(sdir)}
    for k, v in sub.items():
        tmpl = tmpl.replace(k, v)
    (sdir / "SKILL.md").write_text(tmpl)
    sp = sdir / "STYLE.md"
    if not sp.exists():
        s = (paths.TEMPLATES / "STYLE_GUIDE.md").read_text()
        for k, v in sub.items():
            s = s.replace(k, v)
        s = s.replace("{{NOTES}}", notes.strip() or "(none given)")
        s = s.replace("{{REFS}}", "\n".join(f"- `{r['id']}`: {r['source']}" for r in refs))
        sp.write_text(s)


def scan(refs: list[str], name: str, gemini: bool = False, max_seconds: float = 240.0, cookies: str | None = None,
         notes: str = "") -> Path:
    t_start = time.time()
    slug = paths.slug(name)
    sdir = paths.style_dir(name)
    (sdir / "scan").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("VIDKIT_HOME", str(paths.WORK / "vidkit"))
    from vidkit import deep as vdeep

    old = json.loads((sdir / "refs.json").read_text()) if (sdir / "refs.json").exists() else []
    rows = {r["source"]: r for r in old}
    videos = []
    for r in refs:
        v = fetch(r, cookies)
        videos.append(v)
        src = r if r.startswith("http") else str(v)
        if src not in rows:
            rows[src] = {"id": f"ref{len(rows) + 1}-{media.file_key(v)[:6]}", "source": src, "local": str(v)}
        rows[src]["local"] = str(v)
    (sdir / "refs.json").write_text(json.dumps(list(rows.values()), indent=1))

    deeps, analyses, audio, caps_all, frames = [], [], [], [], []
    for row in rows.values():
        v = Path(row["local"])
        if not v.exists():
            print(f"  {row['id']}: the local copy {v} is gone; skipping it (scan the source again to bring it back)")
            continue
        rdir = sdir / "scan" / row["id"]
        rdir.mkdir(parents=True, exist_ok=True)
        dp = rdir / "deep.json"
        if dp.exists():
            d = json.loads(dp.read_text())
            print(f"  {row['id']}: already measured")
        else:
            info = media.probe(v)
            print(f"  {row['id']}: measuring {v.name} ({info['duration']:.0f}s, {info['width']}x{info['height']}) "
                  f"— every frame, on this Mac; about {max(1, int(min(info['duration'], max_seconds) * 12 / 60))} min")
            d, a = vdeep.analyze(str(v), row["id"], max_seconds=max_seconds, use_gemini=gemini,
                                 log=lambda m, i=row["id"]: print(f"    {i}: {m}", flush=True))
            dp.write_text(json.dumps(d, indent=1))
            (rdir / "analysis.json").write_text(json.dumps(a, indent=1))
        if not (rdir / "analysis.json").exists():
            (rdir / "analysis.json").write_text(json.dumps(vdeep.to_analysis(d), indent=1))
        deeps.append(d)
        analyses.append(rdir / "analysis.json")
        caps, _ = recipe.split_text_elements(d)
        caps_all.append(caps)
        au = measure_audio(v, d)
        (rdir / "audio.json").write_text(json.dumps(au, indent=1))
        audio.append(au)
        if not (rdir / "sheets" / "overview.jpg").exists():
            print(f"  {row['id']}: contact sheets")
            sheets.make_all(v, d, caps, rdir / "sheets")
        frames += grade.sample_frames(v, 6)
    if not deeps:
        raise SystemExit("no reference could be measured")

    colour = grade.measure(frames)
    (sdir / "scan" / "colour.json").write_text(json.dumps(colour, indent=1))
    aggregate_profile(sdir, analyses)
    values, prov = recipe.derive(deeps, colour, audio)
    rp, fresh = recipe.write_measured(sdir, values, prov)
    display = name if name != slug else slug.replace("-", " ").title()
    write_docs(sdir, display, list(rows.values()), notes)
    link = link_skill(sdir)
    print(f"\nscanned {len(deeps)} reference(s) in {time.time() - t_start:.0f}s")
    print(f"  style folder : {sdir}")
    print(f"  recipe       : {rp.name} ({'new' if fresh else 'kept your edited copy; the new numbers are in recipe.measured.json'})")
    print(f"  skill        : {link} -> /style-{slug}")
    print("  sheets       : " + ", ".join(str(p.relative_to(sdir)) for p in sorted(sdir.glob('scan/*/sheets/*.jpg'))[:8]))
    print(f"\nNEXT (Claude): finish the style: follow {paths.ROOT / 'WRITING_A_STYLE.md'}")
    return sdir


def list_styles() -> list[dict]:
    out = []
    for d in sorted(paths.STYLES.glob("*")):
        if (d / "recipe.json").exists():
            refs = json.loads((d / "refs.json").read_text()) if (d / "refs.json").exists() else []
            written = (d / "STYLE.md").exists() and "WRITE_ME" not in (d / "STYLE.md").read_text()
            out.append({"style": d.name, "refs": len(refs), "style_md_written": written,
                        "skill": (paths.SKILLS / f"style-{d.name}").is_symlink()})
    return out


def remove(name: str) -> None:
    sdir = paths.style_dir(name)
    link = paths.SKILLS / f"style-{sdir.name}"
    if link.is_symlink():
        link.unlink()
    if sdir.exists():
        shutil.rmtree(sdir)

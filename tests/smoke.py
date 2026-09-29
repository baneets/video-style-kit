#!/usr/bin/env python3
"""End-to-end smoke test: make a synthetic reference (captions burned in at 70 % height, synced to real speech from
macOS `say`), scan it into a throwaway style, edit a synthetic 16:9 "talking" clip into that 9:16 style, and check the
result. Uses no user media and leaves nothing behind (the style, its skill link and the export are removed).

  .venv/bin/python tests/smoke.py [--engine auto|resolve|headless] [--keep]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT / "src"))
STYLE = "zz-smoke-test"

LINES = ["this is how", "you edit videos", "with your own style", "every single day"]
TALK = ["Here is the thing.", "Um, most people never start.", "They wait for the perfect moment.",
        "But you just have to begin today."]


def sh(*cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def say(text: str, out: Path) -> float:
    aiff = out.with_suffix(".aiff")
    sh("say", "-o", aiff, text)
    sh("ffmpeg", "-y", "-v", "error", "-i", aiff, "-ar", "48000", "-ac", "1", out)
    aiff.unlink()
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                                capture_output=True, text=True).stdout)


def caption_png(text: str, out: Path, W: int = 720, H: int = 1280) -> Path:
    """One caption line drawn with Pillow (Homebrew's ffmpeg has no drawtext): white bold, black outline, 70 % down."""
    from PIL import Image, ImageDraw, ImageFont
    font = None
    for f in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        if Path(f).exists():
            font = ImageFont.truetype(f, 64)
            break
    font = font or ImageFont.load_default(size=64)
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((W / 2, H * 0.70), text, font=font, fill=(255, 255, 255, 255), anchor="mm", stroke_width=6,
           stroke_fill=(0, 0, 0, 255))
    im.save(out)
    return out


def make_reference(d: Path) -> Path:
    """720x1280, one caption line per spoken phrase, white bold text with a black border at 70 % height."""
    parts, t, spans = [], 0.0, []
    for i, line in enumerate(LINES):
        wav = d / f"r{i}.wav"
        dur = say(line, wav)
        parts.append(wav)
        spans.append((caption_png(line.upper(), d / f"c{i}.png"), t, t + dur + 0.25))
        t += dur + 0.25
        sil = d / f"s{i}.wav"
        sh("ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "0.25", sil)
        parts.append(sil)
    lst = d / "ref.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    sh("ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, d / "ref.wav")
    ins = ["-f", "lavfi", "-i", f"color=c=0x35506e:s=720x1280:r=30:d={t:.3f}",
           "-f", "lavfi", "-i", f"color=c=0xd9a066:s=260x360:r=30:d={t:.3f}", "-i", d / "ref.wav"]
    graph, v = ["[0:v][1:v]overlay=(W-w)/2:H*0.22[b0]"], "[b0]"
    for k, (png, a, b) in enumerate(spans):
        ins += ["-loop", "1", "-i", png]
        graph.append(f"{v}[{3 + k}:v]overlay=0:0:shortest=1:enable='between(t,{a:.3f},{b:.3f})'[b{k + 1}]")
        v = f"[b{k + 1}]"
    out = d / "reference.mp4"
    sh("ffmpeg", "-y", "-v", "error", *ins, "-filter_complex", ";".join(graph), "-map", v, "-map", "2:a",
       "-t", f"{t:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out)
    return out


def make_talk(d: Path) -> Path:
    """1280x720 at 30 fps, speech with long pauses between sentences (the editor should tighten them)."""
    parts = []
    for i, line in enumerate(TALK):
        wav = d / f"t{i}.wav"
        say(line, wav)
        parts.append(wav)
        sil = d / f"ts{i}.wav"
        sh("ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "1.4", sil)
        parts.append(sil)
    lst = d / "talk.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    sh("ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, d / "talk.wav")
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(d / "talk.wav")],
                               capture_output=True, text=True).stdout)
    out = d / "talk.mp4"
    sh("ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s=1280x720:r=30:d={dur:.3f}", "-i", d / "talk.wav",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="auto", choices=["auto", "resolve", "headless"])
    ap.add_argument("--keep", action="store_true", help="keep the throwaway style and export for inspection")
    a = ap.parse_args()
    if shutil.which("say") is None:
        sys.exit("this test needs macOS `say`")
    from stylekit import edit, paths, recipe, scan
    fails = []
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        print("1/4 making a synthetic reference and a synthetic talking clip")
        ref, talk = make_reference(d), make_talk(d)
        print("2/4 scanning the reference (about 2 minutes)")
        scan.remove(STYLE)
        sdir = scan.scan([str(ref)], STYLE, notes="smoke test")
        rec = recipe.load(STYLE)
        cap = rec["captions"]
        print(f"    measured captions: enabled={cap['enabled']} y={cap.get('y')} size={cap.get('size')} case={cap.get('case')}")
        if not cap["enabled"]:
            fails.append("the scan did not find the burned-in captions")
        elif not 0.62 <= float(cap["y"]) <= 0.78:
            fails.append(f"captions measured at y={cap['y']}, drawn at 0.70")
        if rec["canvas"]["aspect"] != "9:16":
            fails.append(f"aspect measured {rec['canvas']['aspect']}, reference is 9:16")
        if not (paths.SKILLS / f"style-{STYLE}").is_symlink():
            fails.append("the style's skill link was not created")
        for s in ("overview", "hook", "text"):
            if not list(sdir.glob(f"scan/*/sheets/{s}.jpg")):
                fails.append(f"no {s} contact sheet")
        print("3/4 editing the talking clip in that style")
        rep = edit.edit([str(talk)], STYLE, name="smoke", hook="THE SMOKE TEST", engine=a.engine)
        print("4/4 checking the result")
        src_len = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                        str(talk)], capture_output=True, text=True).stdout)
        if rep["problems"]:
            fails += rep["problems"]
        if rep["size"] != "1080x1920":
            fails.append(f"output {rep['size']}, style is 1080x1920")
        if not rep["captions"]:
            fails.append("no captions were drawn")
        if not rep["length_s"] < src_len - 2.0:
            fails.append(f"the long pauses were not tightened ({rep['length_s']}s from {src_len:.1f}s)")
        if not any("um" in r.lower() for r in rep["removed"]):
            fails.append("the 'um' was not removed")
        print(json.dumps({k: rep[k] for k in ("output", "engine", "length_s", "size", "fps", "lufs", "pieces")}, indent=1))
        if not a.keep:
            scan.remove(STYLE)
            shutil.rmtree(paths.EXPORTS / STYLE, ignore_errors=True)
            shutil.rmtree(Path(rep["work"]), ignore_errors=True)
            if rep["engine"] == "resolve":
                from stylekit import resolve_build
                resolve_build.delete_project(resolve_build.connect(), f"{STYLE} - smoke")
    if fails:
        print("SMOKE TEST FAILED:\n  - " + "\n  - ".join(fails))
        sys.exit(1)
    print(f"SMOKE TEST PASSED ({rep['engine']})")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()

"""stylekit: scan a reference video into an editing style, then edit your footage in it.

  stylekit doctor                                   check every dependency, say exactly what to fix
  stylekit scan <video or link>... --name "my style" measure the reference(s), make styles/<style>/ + its skill
  stylekit styles                                   list your styles
  stylekit transcript <footage>...                  sentences with times (to choose what to keep)
  stylekit calibrate <style>                        the style's text drawn over a reference frame, to compare
  stylekit edit <footage>... --style <style>        cut, frame, grade, caption, music, build in Resolve, render
  stylekit link                                     re-link every style's skill into ~/.claude/skills
  stylekit remove <style>                           delete a style and its skill link
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_scan(a):
    from . import scan
    scan.scan(a.refs, a.name, gemini=a.gemini, max_seconds=a.max_seconds, cookies=a.cookies, notes=a.notes or "")


def cmd_styles(a):
    from . import scan
    rows = scan.list_styles()
    if not rows:
        print("no styles yet: stylekit scan <video> --name \"my style\"")
    for r in rows:
        print(f"{r['style']:28} refs {r['refs']:2}  STYLE.md {'written' if r['style_md_written'] else 'NOT WRITTEN YET'}  "
              f"skill {'/style-' + r['style'] if r['skill'] else 'not linked (stylekit link)'}")


def cmd_transcript(a):
    from . import media, words as W
    for k, f in enumerate(a.files):
        p = Path(f).expanduser().resolve()
        if not p.exists():
            sys.exit(f"no such file: {p}")
        ws = W.transcribe(media.voice_copy(p))
        print(f"# file {k}: {p.name} ({media.probe(p)['duration']:.1f}s)")
        sent, t0 = [], None
        for i, w in enumerate(ws):
            if t0 is None:
                t0 = w["start"]
            sent.append(w["word"])
            gap = ws[i + 1]["start"] - w["end"] if i + 1 < len(ws) else 9
            if w["word"][-1:] in ".?!" or gap > 0.8 or len(sent) >= 30:
                print(f"[{k}] {t0:7.2f} - {w['end']:7.2f}  {' '.join(sent)}")
                sent, t0 = [], None


def cmd_calibrate(a):
    from . import calibrate
    calibrate.run(a.style)


def cmd_edit(a):
    from . import edit
    rep = edit.edit(a.files, a.style, name=a.name, keep=a.keep, titles=a.titles, broll=a.broll, hook=a.hook,
                    music=a.music, engine=a.engine, captions=not a.no_captions, plan_only=a.plan_only, aspect=a.aspect)
    if a.json:
        print(json.dumps(rep, indent=1))


def cmd_link(a):
    from . import paths, scan
    n = 0
    for d in sorted(paths.STYLES.glob("*")):
        if (d / "SKILL.md").exists():
            print(f"  {scan.link_skill(d)} -> {d}")
            n += 1
    print(f"{n} style skill(s) linked")


def cmd_remove(a):
    from . import paths, scan
    d = paths.style_dir(a.style)
    if not d.exists():
        sys.exit(f"no style {a.style!r}")
    if not a.yes:
        sys.exit(f"this deletes {d} (its scans, recipe and STYLE.md) and its skill link. Re-run with --yes to confirm.")
    scan.remove(a.style)
    print(f"removed {d}")


def cmd_doctor(a):
    from . import doctor
    sys.exit(0 if doctor.run(fix_hint=True) else 1)


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)      # numpy empty-slice noise from the scanner      # progress shows up live even when output goes to a file
    ap = argparse.ArgumentParser(prog="stylekit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="reference video(s) -> a style")
    s.add_argument("refs", nargs="+", help="video files or links (YouTube, Instagram, TikTok...)")
    s.add_argument("--name", required=True, help="what you call this style")
    s.add_argument("--notes", help="your own words about the style (saved into STYLE.md)")
    s.add_argument("--gemini", action="store_true", help="also let Gemini name the elements (needs GEMINI_API_KEY; costs a few cents)")
    s.add_argument("--cookies", help="browser to borrow a login from for downloads, e.g. chrome (Instagram needs it)")
    s.add_argument("--max-seconds", type=float, default=240.0, help="measure at most this much of each reference")
    s.set_defaults(fn=cmd_scan)

    s = sub.add_parser("styles", help="list styles")
    s.set_defaults(fn=cmd_styles)

    s = sub.add_parser("transcript", help="what is said in your footage, with times")
    s.add_argument("files", nargs="+")
    s.set_defaults(fn=cmd_transcript)

    s = sub.add_parser("calibrate", help="draw the style's text over a reference frame to compare")
    s.add_argument("style")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("edit", help="edit footage in a style")
    s.add_argument("files", nargs="+", help="your talking footage (in the order it should play when no --keep)")
    s.add_argument("--style", required=True)
    s.add_argument("--name", help="output name (default: the first file's name)")
    s.add_argument("--keep", help='keep.json: [{"file":0,"start":12.3,"end":18.9,"label":"hook"}, ...] in play order')
    s.add_argument("--hook", help="hook title text shown from the first frame")
    s.add_argument("--titles", help='titles.json: [{"text":"3 RULES","at_phrase":"three rules","hold":2.0}, ...]')
    s.add_argument("--broll", help='broll.json: [{"file":"clip.mp4","at_phrase":"my desk","dur":2.5,"start":0}, ...]')
    s.add_argument("--music", help="your music file (looped/levelled under the voice)")
    s.add_argument("--aspect", choices=["9:16", "16:9", "1:1", "4:5"], help="override the style's aspect")
    s.add_argument("--engine", choices=["auto", "resolve", "headless"], default="auto",
                   help="auto = Resolve Studio when reachable, else ffmpeg")
    s.add_argument("--no-captions", action="store_true")
    s.add_argument("--plan-only", action="store_true", help="only cut; write plan.json + the cut's transcript")
    s.add_argument("--json", action="store_true", help="print the report as JSON")
    s.set_defaults(fn=cmd_edit)

    s = sub.add_parser("link", help="re-link style skills into ~/.claude/skills")
    s.set_defaults(fn=cmd_link)

    s = sub.add_parser("remove", help="delete a style")
    s.add_argument("style")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_remove)

    s = sub.add_parser("doctor", help="check the installation")
    s.set_defaults(fn=cmd_doctor)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

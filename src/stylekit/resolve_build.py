"""Build the edit as a real DaVinci Resolve timeline through Resolve's scripting API, then render it.

Tracks: V1 the talking footage (one item per kept range, face-framed, punch-ins, style grade as a LUT),
        V2 your b-roll, V3 the text overlay (captions/hook/titles, alpha), A1 voice (J-cut), A2 music.

Facts this relies on (measured on Resolve Studio 21.1):
  - AppendToTimeline with mediaType 1 (video) and 2 (audio) places the halves separately, so a J-cut is the audio
    entry starting a few frames earlier in the source; recordFrame is ABSOLUTE (the timeline starts at 01:00:00:00).
  - SetProperty works for ZoomX/ZoomY/Pan/Tilt and AudioVolume.
  - The grade goes on with SetCDL (node 1), the same slope/offset/power/saturation the headless render bakes into a
    .cube; SetLUT would need Resolve's system LUT folder (/Library/..., not ~/Library) and a RefreshLUTList.
  - SetRenderSettings with one unknown key rejects the whole dict; TargetDir must be absolute.
External scripting needs DaVinci Resolve STUDIO (the free app has no scripting since 21.1); without it the kit
renders the same edit headless and writes a timeline file the free app can import.
"""
from __future__ import annotations

import os

import sys
import time
from pathlib import Path

from . import paths

API = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
LIB = "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"


def connect(wait_s: float = 0.0):
    """The running Resolve, or None (not installed, not running, not Studio, or a modal dialog is open)."""
    os.environ.setdefault("RESOLVE_SCRIPT_API", API)
    os.environ.setdefault("RESOLVE_SCRIPT_LIB", LIB)
    mods = os.path.join(os.environ["RESOLVE_SCRIPT_API"], "Modules")
    if mods not in sys.path:
        sys.path.append(mods)
    if not Path(os.environ["RESOLVE_SCRIPT_LIB"]).exists():
        return None
    try:
        import DaVinciResolveScript as dvr
    except ImportError:
        return None
    t_end = time.time() + wait_s
    while True:
        r = dvr.scriptapp("Resolve")
        if r is not None or time.time() >= t_end:
            return r
        time.sleep(2)


def launch_and_connect(wait_s: float = 90.0):
    r = connect()
    if r is None and Path("/Applications/DaVinci Resolve/DaVinci Resolve.app").exists():
        os.system('open -a "DaVinci Resolve"')
        r = connect(wait_s)
    return r


def is_studio(r) -> bool:
    return r is not None and "Studio" in (r.GetProductName() or "")


def new_project(r, base: str, w: int, h: int, fps: int):
    pm = r.GetProjectManager()
    name, n = base, 1
    p = pm.CreateProject(name)
    while not p and n < 50:
        n += 1
        name = f"{base} ({n})"
        p = pm.CreateProject(name)
    if not p:
        raise SystemExit("Resolve could not create a project (is a modal dialog open in Resolve?)")
    for k, v in (("timelineResolutionWidth", str(w)), ("timelineResolutionHeight", str(h)),
                 ("timelineFrameRate", str(fps)), ("timelinePlaybackFrameRate", str(fps)),
                 ("timelineInputResMismatchBehavior", "scaleToFit")):
        p.SetSetting(k, v)
    if p.GetSetting("timelineResolutionWidth") != str(w):
        raise SystemExit(f"Resolve refused the {w}x{h} timeline size")
    return p


def import_media(r, p, files: list[Path]) -> dict[Path, object]:
    items = r.GetMediaStorage().AddItemListToMediaPool([str(f) for f in files]) or []
    by_path = {}
    for it in items:
        by_path[Path(it.GetClipProperty("File Path"))] = it
    missing = [f for f in files if f not in by_path]
    if missing:
        raise SystemExit(f"Resolve could not import: {', '.join(map(str, missing))}")
    return by_path


def cdl_dict(cdl: dict) -> dict:
    """grade.py's CDL -> the dict TimelineItem.SetCDL takes (node 1, the primary)."""
    j = lambda v: " ".join(f"{x:.5f}" for x in v)          # noqa: E731
    return {"NodeIndex": "1", "Slope": j(cdl["slope"]), "Offset": j(cdl["offset"]), "Power": j(cdl["power"]),
            "Saturation": f"{cdl['saturation']:.5f}"}


def fit_zoom(src_w: int, src_h: int, out_w: int, out_h: int) -> float:
    """With 'scale entire image to fit', a clip is scaled by min(out/src); filling the frame needs this extra zoom."""
    fit = min(out_w / src_w, out_h / src_h)
    fill = max(out_w / src_w, out_h / src_h)
    return fill / fit


def build(r, job: dict) -> dict:
    """job (written by edit.py): project, timeline, w, h, fps, files, ranges [{file, src_in_f, src_len_f, rec_f,
    a_in_f, a_len_f, a_rec_f (the J-cut audio span), zoom, pan, tilt, cdl}], broll [{file, src_in_f, src_len_f, rec_f, zoom, cdl}], overlay (path|None),
    music (path|None), out_dir. Returns {project, timeline, items} for rendering."""
    p = new_project(r, job["project"], job["w"], job["h"], job["fps"])
    mp = p.GetMediaPool()
    files = [Path(f) for f in job["files"]]
    extra = [Path(b["file"]) for b in job["broll"]] + [Path(x) for x in (job.get("overlay"), job.get("music")) if x]
    pool = import_media(r, p, list(dict.fromkeys(files + extra)))
    tl = mp.CreateEmptyTimeline(job["timeline"]) or sys.exit("Resolve could not create the timeline")
    p.SetCurrentTimeline(tl)
    while tl.GetTrackCount("video") < 3:
        tl.AddTrack("video")
    while tl.GetTrackCount("audio") < 2:
        tl.AddTrack("audio", "stereo")
    s = tl.GetStartFrame()
    v_entries, a_entries = [], []
    for rg in job["ranges"]:
        clip = pool[files[rg["file"]]]
        v_entries.append({"mediaPoolItem": clip, "startFrame": rg["src_in_f"], "endFrame": rg["src_in_f"] + rg["src_len_f"],
                          "recordFrame": s + rg["rec_f"], "trackIndex": 1, "mediaType": 1})
        if rg.get("has_audio", True):
            a_entries.append({"mediaPoolItem": clip, "startFrame": rg["a_in_f"], "endFrame": rg["a_in_f"] + rg["a_len_f"],
                              "recordFrame": s + rg["a_rec_f"], "trackIndex": 1, "mediaType": 2})
    vids = mp.AppendToTimeline(v_entries) or []
    auds = mp.AppendToTimeline(a_entries) or [] if a_entries else []
    if len(vids) != len(v_entries) or len(auds) != len(a_entries):
        raise SystemExit(f"Resolve placed {len(vids)}/{len(v_entries)} video and {len(auds)}/{len(a_entries)} audio pieces")

    graded = 0
    for it, rg in zip(vids, job["ranges"]):
        it.SetProperty("ZoomX", rg["zoom"]); it.SetProperty("ZoomY", rg["zoom"])
        it.SetProperty("Pan", rg["pan"]); it.SetProperty("Tilt", rg["tilt"])
        if rg.get("cdl"):
            graded += bool(it.SetCDL(cdl_dict(rg["cdl"])))

    for b in job["broll"]:
        got = mp.AppendToTimeline([{"mediaPoolItem": pool[Path(b["file"])], "startFrame": b["src_in_f"],
                                    "endFrame": b["src_in_f"] + b["src_len_f"], "recordFrame": s + b["rec_f"],
                                    "trackIndex": 2, "mediaType": 1}]) or []
        if got:
            got[0].SetProperty("ZoomX", b["zoom"]); got[0].SetProperty("ZoomY", b["zoom"])
            if b.get("cdl"):
                graded += bool(got[0].SetCDL(cdl_dict(b["cdl"])))
    if job.get("overlay"):
        ov = pool[Path(job["overlay"])]
        n = int(ov.GetClipProperty("Frames") or job["n_frames"])
        mp.AppendToTimeline([{"mediaPoolItem": ov, "startFrame": 0, "endFrame": n, "recordFrame": s,
                              "trackIndex": 3, "mediaType": 1}])
    if job.get("music"):
        mu = pool[Path(job["music"])]
        n = int(mu.GetClipProperty("Frames") or job["n_frames"])
        mp.AppendToTimeline([{"mediaPoolItem": mu, "startFrame": 0, "endFrame": min(n, job["n_frames"]),
                              "recordFrame": s, "trackIndex": 2, "mediaType": 2}])
    p.SetCurrentTimeline(tl)
    r.OpenPage("edit")
    pm = r.GetProjectManager()
    pm.SaveProject()
    return {"project": p, "timeline": tl, "graded": graded, "pieces": len(vids)}


def render(p, tl, out_dir: Path, name: str, w: int, h: int) -> Path:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{name}.resolve.*"):
        old.unlink()
    p.SetCurrentTimeline(tl)
    codecs = p.GetRenderCodecs("mp4") or {}
    codec = next((c for d, c in codecs.items() if "264" in d or "264" in c), None) or "H264"
    if not p.SetCurrentRenderFormatAndCodec("mp4", codec):
        raise SystemExit(f"Resolve refused mp4/{codec}; render codecs available: {codecs}")
    ok = p.SetRenderSettings({"SelectAllFrames": True, "TargetDir": str(out_dir), "CustomName": f"{name}.resolve",
                              "FormatWidth": w, "FormatHeight": h, "ExportVideo": True, "ExportAudio": True})
    if not ok:
        raise SystemExit("Resolve refused the render settings")
    job = p.AddRenderJob() or sys.exit("Resolve could not add the render job")
    p.StartRendering(job)
    while p.IsRenderingInProgress():
        time.sleep(2)
    st = p.GetRenderJobStatus(job) or {}
    if st.get("JobStatus") != "Complete":
        raise SystemExit(f"Resolve render ended with {st}")
    outs = sorted(out_dir.glob(f"{name}.resolve*.mp4"), key=lambda f: f.stat().st_mtime)
    if not outs:
        raise SystemExit(f"Resolve said the render completed but no file is in {out_dir}")
    return outs[-1]


def delete_project(r, name: str) -> bool:
    """Remove a project the kit made (closing it first when it is open)."""
    if r is None:
        return False
    pm = r.GetProjectManager()
    cur = pm.GetCurrentProject()
    if cur and cur.GetName() == name:
        pm.CloseProject(cur)
    return bool(pm.DeleteProject(name))

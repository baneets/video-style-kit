"""The same edit rendered with ffmpeg alone: for machines without Resolve Studio scripting, and for fast previews.

Same inputs as the Resolve build (the job dict), same frame counts, same LUTs, same overlay clip, same J-cuts, so the
two engines agree to the frame. Each kept range is rendered to its own intermediate (exact frame count), then
concatenated; audio is cut from the same ranges shifted by the J-cut lead.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .media import run


def _vf(item: dict, w: int, h: int, fps: str) -> str:
    """scale-to-fill x zoom, crop the face window, grade, conform."""
    f = [f"scale={item['sw']:.0f}:{item['sh']:.0f}:flags=lanczos", f"crop={w}:{h}:{item['left']:.0f}:{item['top']:.0f}"]
    if item.get("lut"):
        f.append(f"lut3d=file='{item['lut']}':interp=tetrahedral")
    f += [f"fps={fps}", "setsar=1", "format=yuv420p"]
    return ",".join(f)


def _seg(src: Path, t0: float, n_frames: int, fps: str, vf: str, out: Path) -> None:
    pre = max(0.0, t0 - 2.0)
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{pre:.3f}", "-i", src, "-ss", f"{t0 - pre:.3f}", "-an", "-vf", vf,
         "-frames:v", str(n_frames), "-c:v", "libx264", "-preset", "fast", "-crf", "14", "-r", str(fps), out], check=True)
    got = int(run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                   "stream=nb_read_packets", "-of", "csv=p=0", out], capture_output=True, text=True).stdout.split()[0] or 0)
    if got < n_frames:            # the source ran out (last frame of a file): hold the last frame to the exact length
        tmp = out.with_suffix(".pad.mp4")
        run(["ffmpeg", "-y", "-v", "error", "-i", out, "-vf", f"tpad=stop_mode=clone:stop={n_frames - got}",
             "-frames:v", str(n_frames), "-c:v", "libx264", "-preset", "fast", "-crf", "14", tmp], check=True)
        tmp.replace(out)


def render(job: dict, work: Path, out: Path) -> Path:
    w, h, fps = job["w"], job["h"], job["fps"]
    rate = job.get("fps_rational") or str(fps)          # exact rate for ffmpeg (60000/1001, not 59.94)
    work.mkdir(parents=True, exist_ok=True)
    files = [Path(f) for f in job["files"]]
    segs, auds = [], []
    for i, rg in enumerate(job["ranges"]):
        src = files[rg["file"]]
        seg = work / f"v{i:04d}.mp4"
        _seg(src, rg["t0"], rg["len_f"], rate, _vf(rg, w, h, rate), seg)
        segs.append(seg)
        a = work / f"a{i:04d}.wav"
        dur = rg["a_len_f"] / fps                       # J-cut span: starts early, ends where the next one starts early
        if rg.get("has_audio", True):
            run(["ffmpeg", "-y", "-v", "error", "-ss", f"{rg['a_in_f'] / fps:.4f}", "-i", src, "-t", f"{dur:.4f}",
                 "-vn", "-ac", "2", "-ar", "48000", "-af", f"apad=whole_dur={dur:.4f},afade=t=in:d=0.004,"
                 f"afade=t=out:st={max(0.0, dur - 0.004):.4f}:d=0.004", a], check=True)
        else:
            run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", f"{dur:.4f}", a], check=True)
        auds.append(a)
    (work / "v.txt").write_text("".join(f"file '{p.name}'\n" for p in segs))
    (work / "a.txt").write_text("".join(f"file '{p.name}'\n" for p in auds))
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", work / "v.txt", "-c", "copy", work / "base.mp4"], check=True)
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", work / "a.txt", "-c", "pcm_s16le", work / "voice.wav"], check=True)

    ins = ["-i", work / "base.mp4", "-i", work / "voice.wav"]
    graph, v = [], "[0:v]"
    k = 2
    for bi, b in enumerate(job["broll"]):
        ins += ["-ss", f"{b['t0']:.3f}", "-t", f"{b['len_f'] / fps:.4f}", "-i", b["file"]]
        graph.append(f"[{k}:v]{_vf(b, w, h, rate)},setpts=PTS-STARTPTS+{b['rec_f'] / fps:.4f}/TB[b{bi}]")
        graph.append(f"{v}[b{bi}]overlay=eof_action=pass:enable='between(t,{b['rec_f'] / fps:.4f},{(b['rec_f'] + b['len_f']) / fps:.4f})'[vb{bi}]")
        v = f"[vb{bi}]"
        k += 1
    if job.get("overlay"):
        ins += ["-i", job["overlay"]]
        graph.append(f"{v}[{k}:v]overlay=format=auto:eof_action=pass,format=yuv420p[vo]")
        v = "[vo]"
        k += 1
    a = "[1:a]"
    if job.get("music"):
        ins += ["-i", job["music"]]
        graph.append(f"[1:a][{k}:a]amix=inputs=2:duration=first:normalize=0[am]")
        a = "[am]"
        k += 1
    graph.append(f"{v}null[vout]")
    graph.append(f"{a}anull[aout]")
    run(["ffmpeg", "-y", "-v", "error", *ins, "-filter_complex", ";".join(graph), "-map", "[vout]", "-map", "[aout]",
         "-frames:v", str(job["n_frames"]), "-r", rate, "-c:v", "libx264", "-preset", "medium", "-crf", "17",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", out], check=True)
    return out

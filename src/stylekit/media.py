"""ffmpeg / ffprobe measurements. All local, all free."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from . import paths

WIN = 0.04          # seconds per loudness window


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], **kw)


def probe(file: Path) -> dict:
    """{duration, width, height, fps, has_audio} of a media file."""
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate",
               "-of", "json", file], capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"cannot read {file}: {out.stderr.strip()[:200]}")
    j = json.loads(out.stdout)
    v = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), {})
    n, _, d = (v.get("r_frame_rate") or "0/1").partition("/")
    fps = float(n) / float(d or 1) if float(n or 0) else 0.0
    return {"duration": float(j.get("format", {}).get("duration") or 0), "width": int(v.get("width") or 0),
            "height": int(v.get("height") or 0), "fps": fps,
            "has_audio": any(s.get("codec_type") == "audio" for s in j.get("streams", []))}


RATES = [("23.976", 24000 / 1001, "24000/1001"), ("24", 24.0, "24"), ("25", 25.0, "25"),
         ("29.97", 30000 / 1001, "30000/1001"), ("30", 30.0, "30"), ("48", 48.0, "48"), ("50", 50.0, "50"),
         ("59.94", 60000 / 1001, "60000/1001"), ("60", 60.0, "60")]


def standard_rate(fps: float) -> tuple[str, float, str]:
    """The standard rate nearest a source rate: (Resolve label, exact value, ffmpeg rational). The timeline runs at
    the footage's own rate, so every source frame is exactly one timeline frame (no gaps, no drift)."""
    return min(RATES, key=lambda r: abs(r[1] - (fps or 30.0)))


def conform_fps(file: Path, rational: str) -> Path:
    """A copy of `file` at another frame rate (only for a second talking file shot at a different rate)."""
    out = work_dir(file) / f"conform_{rational.replace('/', '_')}.mov"
    if not out.exists():
        tmp = out.with_suffix(".tmp.mov")
        run(["ffmpeg", "-y", "-v", "error", "-i", file, "-map", "0:v:0", "-map", "0:a:0?", "-vf", f"fps={rational}",
             "-c:v", "libx264", "-crf", "12", "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", tmp], check=True)
        tmp.rename(out)
    return out


def file_key(file: Path) -> str:
    """Short stable id for a file (size + first 2 MB), so caches survive renames."""
    with open(file, "rb") as fh:
        head = fh.read(2_000_000)
    return hashlib.sha1(head + str(file.stat().st_size).encode()).hexdigest()[:12]


def work_dir(file: Path) -> Path:
    d = paths.WORK / "files" / f"{file.stem[:40]}-{file_key(file)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def rms_curve(file: Path) -> np.ndarray:
    """RMS level (dBFS) per 40 ms window over the whole file. One ffmpeg pass, cached."""
    cache = work_dir(file) / "rms40.npy"
    if cache.exists():
        return np.load(cache)
    n = int(48000 * WIN)
    out = run(["ffmpeg", "-v", "error", "-i", file, "-vn", "-af",
               f"aresample=48000,asetnsamples={n},astats=metadata=1:reset=1,"
               "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-", "-f", "null", "-"],
              capture_output=True, text=True).stdout
    v = np.array([float(x) if x != "-inf" else -90.0 for x in re.findall(r"RMS_level=(-?[\d.]+|-inf)", out)])
    np.save(cache, v)
    return v


def silences(curve: np.ndarray, min_d: float) -> list[tuple[float, float]]:
    """Pauses longer than `min_d`. The threshold comes from THIS file's own floor and speech level: a fixed dB
    threshold misses every pause on one room and finds dozens on another."""
    if not len(curve):
        return []
    lo, hi = np.percentile(curve, 10), np.percentile(curve, 90)
    quiet = curve < hi - 0.45 * (hi - lo)
    out, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * WIN >= min_d:
                out.append((start * WIN, i * WIN))
            start = None
    if start is not None and (len(curve) - start) * WIN >= min_d:
        out.append((start * WIN, len(curve) * WIN))
    return out


def snap_to_dip(t: float, curve: np.ndarray, words: list[dict], radius: float = 0.12) -> float:
    """Move a cut point to the quietest 40 ms window within +-radius that is not inside a word."""
    lo, hi = max(0, int((t - radius) / WIN)), min(len(curve) - 1, int((t + radius) / WIN))
    best, best_v = t, None
    for i in range(lo, hi + 1):
        tt = i * WIN + WIN / 2
        if any(w["start"] < tt < w["end"] for w in words):
            continue
        if best_v is None or curve[i] < best_v:
            best, best_v = tt, curve[i]
    return best


def onset(curve: np.ndarray, t_from: float, floor_db: float = -30.0, hold_s: float = 0.10) -> float:
    """First moment after `t_from` where the level stays above `floor_db` for `hold_s`."""
    need = max(1, int(hold_s / WIN))
    run_ = 0
    for i in range(int(t_from / WIN), len(curve)):
        run_ = run_ + 1 if curve[i] > floor_db else 0
        if run_ >= need:
            return (i - need + 1) * WIN
    return t_from


def lufs(file: Path, t0: float | None = None, dur: float | None = None) -> float | None:
    """Integrated loudness of a file (or a span of it); None when there is no audio."""
    cut = (["-ss", f"{t0:.3f}"] if t0 is not None else []) + (["-t", f"{dur:.3f}"] if dur is not None else [])
    r = run(["ffmpeg", "-v", "info", *cut, "-i", file, "-vn", "-af", "ebur128=peak=none", "-f", "null", "-"],
            capture_output=True, text=True)
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
    return float(m[-1]) if m else None


def voice_copy(file: Path, target: float = -16.0) -> Path:
    """The file with its video stream untouched and the voice levelled to `target` LUFS, peaks limited (two-pass
    loudnorm). Camera audio often sits near -26 LUFS with -3 dB peaks, so a flat gain would clip loud words."""
    out = work_dir(file) / f"voice{int(abs(target))}.mov"
    if out.exists():
        return out
    if not probe(file)["has_audio"]:
        return file
    r = run(["ffmpeg", "-v", "info", "-i", file, "-af", f"loudnorm=I={target}:TP=-1.5:LRA=11:print_format=json",
             "-f", "null", "-"], capture_output=True, text=True).stderr
    m = json.loads(re.findall(r"\{[^{}]*\}", r)[-1])
    af = (f"loudnorm=I={target}:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=false")
    tmp = out.with_suffix(".tmp.mov")
    base = ["ffmpeg", "-y", "-v", "error", "-i", file, "-map", "0:v:0", "-map", "0:a:0"]
    tail = ["-af", af, "-ar", "48000", "-c:a", "pcm_s16le", tmp]
    if run(base + ["-c:v", "copy"] + tail).returncode:          # a codec .mov cannot carry (VP9, AV1): re-encode it
        run(base + ["-c:v", "libx264", "-crf", "14", "-preset", "fast", "-pix_fmt", "yuv420p"] + tail, check=True)
    tmp.rename(out)
    return out


def grab_frame(file: Path, t: float, width: int | None = None) -> np.ndarray | None:
    """One decoded frame as RGB uint8 (fast seek to t-1, then an accurate 1 s seek)."""
    from PIL import Image
    import io
    pre = max(0.0, t - 1.0)
    vf = ["-vf", f"scale={width}:-2"] if width else []
    raw = run(["ffmpeg", "-v", "error", "-ss", f"{pre:.3f}", "-i", file, "-ss", f"{t - pre:.3f}", "-frames:v", "1",
               *vf, "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    if not raw:
        return None
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))

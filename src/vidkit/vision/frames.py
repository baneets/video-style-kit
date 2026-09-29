"""Frame IO through ffmpeg (already required for yt-dlp). numpy arrays are (T, H, W, 3) uint8."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

LEVELS = ("clean", "moderate", "heavy")


def probe(path: str) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate:format=duration", "-of", "json", path],
                       capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": s["width"], "height": s["height"], "fps": float(num) / float(den or 1),
            "duration": float(j["format"].get("duration", 0))}


def read_frames(path: str, start: float = 0.0, duration: float | None = None, width: int | None = None) -> tuple[np.ndarray, float]:
    """Decode a window of a video to RGB frames. `width` downsamples (keeps aspect, even height)."""
    info = probe(path)
    w, h = info["width"], info["height"]
    if width and width < w:
        h = max(2, int(round(h * width / w / 2)) * 2)
        w = width
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", path, "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(out) // (w * h * 3)
    return np.frombuffer(out[: n * w * h * 3], np.uint8).reshape(n, h, w, 3).copy(), info["fps"]


def write_video(frames: np.ndarray, path: str, fps: float = 30, crf: int | None = None) -> None:
    t, h, w, _ = frames.shape
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    cmd += ["-crf", str(crf)] if crf is not None else ["-qp", "0"]
    cmd += [path]
    subprocess.run(cmd, input=frames.tobytes(), capture_output=True, check=True)


def roundtrip(frames: np.ndarray, crf: int, fps: float = 30) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        p = str(Path(d) / "rt.mp4")
        write_video(frames, p, fps, crf)
        out, _ = read_frames(p)
    n = min(len(out), len(frames))
    return out[:n]


def degrade(frames: np.ndarray, level: str, seed: int = 0) -> np.ndarray:
    """clean = untouched; moderate = h264 crf 28; heavy = gaussian noise sigma 12 + h264 crf 40."""
    if level == "clean":
        return frames
    if level == "moderate":
        return roundtrip(frames, 28)
    if level == "heavy":
        rng = np.random.default_rng(seed)
        noisy = np.clip(frames.astype(np.float32) + rng.normal(0, 12, frames.shape), 0, 255).astype(np.uint8)
        return roundtrip(noisy, 40)
    raise ValueError(f"unknown level {level!r}; use one of {LEVELS}")

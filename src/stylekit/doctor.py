"""stylekit doctor: every dependency checked, with the exact fix for anything missing."""
from __future__ import annotations

import importlib
import os
import platform
import plistlib
import shutil
import subprocess
from pathlib import Path

from . import paths, resolve_build as RB

OK, WARN, FAIL = "ok  ", "warn", "FAIL"


def _line(state: str, what: str, detail: str = "") -> None:
    print(f"[{state}] {what}" + (f": {detail}" if detail else ""))


def resolve_version() -> str | None:
    pl = Path("/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Info.plist")
    if not pl.exists():
        return None
    try:
        return plistlib.loads(pl.read_bytes()).get("CFBundleShortVersionString")
    except Exception:                                 # noqa: BLE001
        return "unknown"


def run(fix_hint: bool = True) -> bool:
    good = True
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        _line(FAIL, "Apple Silicon Mac", f"this is {platform.system()} {platform.machine()}; the scanner uses Apple Vision + MLX")
        good = False
    else:
        _line(OK, "Apple Silicon Mac", platform.mac_ver()[0])
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool):
            _line(OK, tool)
        else:
            _line(FAIL, tool, "brew install ffmpeg")
            good = False
    for mod, pip in (("numpy", "numpy"), ("cv2", "opencv-python-headless"), ("mlx_whisper", "mlx-whisper"),
                     ("PIL", "pillow"), ("yt_dlp", "yt-dlp"), ("vidkit.deep", "-e ."), ("stylekit.edit", "-e .")):
        try:
            importlib.import_module(mod)
            _line(OK, f"python: {mod}")
        except Exception as ex:                       # noqa: BLE001
            _line(FAIL, f"python: {mod}", f"{str(ex)[:80]} -> ./install.sh (or .venv/bin/pip install {pip})")
            good = False
    for t in ("ocr", "personmatte"):
        if (paths.TOOLS / t).exists():
            _line(OK, f"Apple Vision tool {t}")
        else:
            _line(FAIL, f"Apple Vision tool {t}", f"swiftc -O tools/{t}.swift -o tools/{t}  (needs: xcode-select --install)")
            good = False
    node = shutil.which("node")
    if not node:
        _line(FAIL, "node", "brew install node")
        good = False
    else:
        v = subprocess.run([node, "-v"], capture_output=True, text=True).stdout.strip()
        major = int(v.lstrip("v").split(".")[0] or 0)
        _line(OK if major >= 18 else FAIL, "node", v + ("" if major >= 18 else " (need 18+: brew upgrade node)"))
        good &= major >= 18
    if (paths.REMOTION / "node_modules" / "remotion").exists():
        _line(OK, "Remotion (text overlays)")
    else:
        _line(FAIL, "Remotion (text overlays)", "cd remotion && npm ci")
        good = False
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub" / "models--mlx-community--whisper-large-v3-turbo"
    _line(OK if hf.exists() else WARN, "whisper model", "cached" if hf.exists() else "downloads (~1.6 GB) on first use")
    ver = resolve_version()
    if not ver:
        _line(WARN, "DaVinci Resolve", "not found in /Applications; edits will render headless (install Resolve from blackmagicdesign.com)")
    else:
        r = RB.connect()
        if r is None:
            _line(WARN, f"DaVinci Resolve {ver}", "installed but not reachable right now: open it, and set Preferences > System > "
                  "General > External scripting using = Local, then run doctor again (edits still work headless)")
        elif RB.is_studio(r):
            RB.STATUS.unlink(missing_ok=True)
            _line(OK, f"DaVinci Resolve {ver}", f"{r.GetProductName()} scripting connected: edits build real timelines")
        else:
            _line(WARN, f"DaVinci Resolve {ver}", f"{r.GetProductName()}: the free edition has no scripting; edits render headless "
                  "and you can import the result into Resolve")
    link = paths.SKILLS / "scan-video-style"
    if link.exists():
        _line(OK, "Claude Code skill /scan-video-style", str(link))
    else:
        _line(FAIL, "Claude Code skill /scan-video-style", "./install.sh (links skills/scan-video-style into ~/.claude/skills)")
        good = False
    _line(OK if os.environ.get("GEMINI_API_KEY") else WARN, "GEMINI_API_KEY (optional)",
          "set" if os.environ.get("GEMINI_API_KEY") else "not set; only needed for `scan --gemini`")
    print("\nall required parts are working" if good else "\nfix the FAIL lines above, then run: stylekit doctor")
    return good

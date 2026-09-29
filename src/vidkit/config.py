"""Where the scanner keeps its working files. Everything lives inside the kit folder (git-ignored)."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def home() -> Path:
    return Path(os.environ.get("VIDKIT_HOME", REPO_ROOT / ".work" / "vidkit")).expanduser()


def dirs() -> dict[str, Path]:
    h = home()
    d = {k: h / k for k in ("analyses", "cache")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d

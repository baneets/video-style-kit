"""Every path the kit uses, in one place. Everything lives inside the kit folder except the skill links."""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STYLES = ROOT / "styles"                 # one folder per style the user makes (git-ignored)
WORK = ROOT / ".work"                    # caches, transcripts, intermediate renders (git-ignored)
EXPORTS = ROOT / "exports"               # finished videos (git-ignored)
REMOTION = ROOT / "remotion"
TOOLS = ROOT / "tools"                   # Apple Vision helpers compiled by install.sh
TEMPLATES = ROOT / "templates"
SKILLS = Path(os.environ.get("CLAUDE_SKILLS_DIR", Path.home() / ".claude" / "skills")).expanduser()


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not s:
        raise SystemExit(f"style name {name!r} has no letters or digits")
    return s


def style_dir(name: str) -> Path:
    return STYLES / slug(name)

#!/usr/bin/env bash
# video-style-kit installer. Safe to run again at any time (every step skips what is already done).
#
#   ./install.sh            everything, including the DaVinci Resolve MCP server for Claude Code
#   ./install.sh --no-mcp   skip the Resolve MCP server
#
# Needs: an Apple Silicon Mac, Homebrew, Xcode Command Line Tools. Installs: ffmpeg, node, python 3.12 (brew, only
# if missing), a Python venv (.venv), the Apple Vision helpers (tools/), Remotion (remotion/node_modules), the
# on-device whisper model, and the Claude Code skill /scan-video-style.
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$KIT"
MCP=1
for a in "$@"; do
  case "$a" in
    --no-mcp) MCP=0 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

say()  { printf '\n==> %s\n' "$*"; }
die()  { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. the machine
say "Checking the Mac"
[ "$(uname -s)" = "Darwin" ] || die "this kit runs on macOS (it uses Apple Vision and MLX)."
[ "$(uname -m)" = "arm64" ] || die "this kit needs an Apple Silicon Mac (M1 or newer)."
echo "macOS $(sw_vers -productVersion), $(uname -m)"

if ! xcode-select -p >/dev/null 2>&1; then
  die "Xcode Command Line Tools are missing. Run:  xcode-select --install   (a window opens; click Install), then run ./install.sh again."
fi
if ! swiftc --version >/dev/null 2>&1; then
  die "swiftc does not run (usually the Xcode licence was never accepted). Run:  sudo xcodebuild -license accept   then ./install.sh again."
fi

if ! command -v brew >/dev/null 2>&1; then
  for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do [ -x "$b" ] && eval "$("$b" shellenv)" && break; done
fi
command -v brew >/dev/null 2>&1 || die "Homebrew is missing. Install it (it asks for your Mac password):
  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"
then run ./install.sh again."

# ---------------------------------------------------------------- 2. command-line tools
say "ffmpeg, node, python 3.12"
need=()
command -v ffmpeg >/dev/null 2>&1 || need+=(ffmpeg)
if command -v node >/dev/null 2>&1; then
  [ "$(node -v | sed 's/v\([0-9]*\).*/\1/')" -ge 18 ] || need+=(node)
else
  need+=(node)
fi
PY=""
for c in python3.12 python3.11 python3.13 /opt/homebrew/bin/python3.12; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then
    PY="$(command -v "$c")"; break
  fi
done
[ -n "$PY" ] || need+=(python@3.12)
if [ ${#need[@]} -gt 0 ]; then
  echo "brew install ${need[*]}"
  brew install "${need[@]}"
fi
[ -n "$PY" ] || PY="$(brew --prefix)/bin/python3.12"
"$PY" -c 'import venv' || die "python at $PY cannot make virtual environments"
echo "ffmpeg: $(ffmpeg -version | head -1 | cut -d' ' -f3)  node: $(node -v)  python: $("$PY" -V)"

# ---------------------------------------------------------------- 3. python environment
say "Python environment (.venv)"
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -e ".[gemini]"
.venv/bin/python -c "import cv2, mlx_whisper, numpy, PIL, yt_dlp, vidkit.deep, stylekit.edit" || die "python packages did not import (see the error above)"
echo "ok"

# ---------------------------------------------------------------- 4. Apple Vision helpers
say "Apple Vision helpers (text + face + person detection, on-device)"
for t in ocr personmatte; do
  if [ ! -x "tools/$t" ] || [ "tools/$t.swift" -nt "tools/$t" ]; then
    if ! swiftc -O "tools/$t.swift" -o "tools/$t" 2>.work-swift.log; then cat .work-swift.log; rm -f .work-swift.log; die "could not compile tools/$t.swift"; fi
    rm -f .work-swift.log
  fi
  [ -x "tools/$t" ] || die "could not compile tools/$t.swift"
done
echo "ok"

# ---------------------------------------------------------------- 5. Remotion (text overlays)
say "Remotion (draws captions and titles)"
( cd remotion && npm ci --no-audit --no-fund --loglevel=error )
( cd remotion && npx --yes remotion browser ensure >/dev/null )
echo "ok"

# ---------------------------------------------------------------- 6. whisper model
say "Speech-to-text model (mlx whisper large-v3-turbo, ~1.6 GB, once)"
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("mlx-community/whisper-large-v3-turbo")
print("ok")
PY

# ---------------------------------------------------------------- 7. Claude Code skills
say "Claude Code skill /scan-video-style"
SK="${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}"
mkdir -p "$SK"
sed "s#{{KIT}}#$KIT#g" skills/scan-video-style/SKILL.md.tmpl > skills/scan-video-style/SKILL.md
if [ -L "$SK/scan-video-style" ] || [ ! -e "$SK/scan-video-style" ]; then
  ln -sfn "$KIT/skills/scan-video-style" "$SK/scan-video-style"
else
  echo "note: $SK/scan-video-style already exists and is not a link; left it alone"
fi
mkdir -p styles exports .work
.venv/bin/stylekit link
echo "ok"

# ---------------------------------------------------------------- 8. Resolve MCP server (optional)
if [ "$MCP" = 1 ]; then
  say "DaVinci Resolve MCP server for Claude Code (lets Claude drive Resolve directly for hand tweaks)"
  if ! command -v claude >/dev/null 2>&1; then
    echo "skipped: the 'claude' command is not on PATH (install Claude Code first, then run ./install.sh again)"
  elif claude mcp get davinci-resolve >/dev/null 2>&1; then
    echo "a 'davinci-resolve' MCP server is already registered; left it as it is"
  else
    MCPD="$KIT/vendor/davinci-resolve-mcp"
    if [ ! -d "$MCPD/.git" ]; then
      git -c advice.detachedHead=false clone -q --depth 1 --branch v4.7.10 https://github.com/samuelgursky/davinci-resolve-mcp.git "$MCPD"
    fi
    [ -x "$MCPD/venv/bin/python" ] || "$PY" -m venv "$MCPD/venv"
    "$MCPD/venv/bin/python" -m pip install -q --upgrade pip
    "$MCPD/venv/bin/python" -m pip install -q "mcp[cli]>=1.30,<2"
    [ -f "$MCPD/requirements.txt" ] && "$MCPD/venv/bin/python" -m pip install -q -r "$MCPD/requirements.txt"
    API="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
    LIB="/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
    claude mcp add davinci-resolve -s user \
      -e "RESOLVE_SCRIPT_API=$API" -e "RESOLVE_SCRIPT_LIB=$LIB" -e "PYTHONPATH=$API/Modules" \
      -- "$MCPD/venv/bin/python" "$MCPD/src/server.py"
    echo "registered MCP server 'davinci-resolve' (restart Claude Code to load it)"
  fi
fi

# ---------------------------------------------------------------- 9. check
say "Checking everything"
.venv/bin/stylekit doctor || true
cat <<EOF

Installed. In DaVinci Resolve (Studio) set once: Preferences > System > General > External scripting using = Local.
Then, in Claude Code:  "scan this video <file or link> and call the style <name>"
Command line:          $KIT/.venv/bin/stylekit --help
EOF

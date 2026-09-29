# SETUP.md: instructions for Claude Code

You are Claude Code, and the user wants this kit installed on their Mac. Follow these steps in order. Run every
command yourself; stop and ask the user only where a step says so (a password or a click only they can give).
Say what you are doing in one line per step.

## What gets installed

- A Python environment inside this folder (`.venv`) with the scanner (on-device, frame by frame) and the editor
- Two small Apple Vision helpers compiled from `tools/*.swift` (text, faces, person mask; on-device)
- Remotion inside `remotion/` (draws captions and titles as a transparent video layer)
- The on-device speech-to-text model (mlx whisper large-v3-turbo, about 1.6 GB, downloaded once)
- The Claude Code skill `/scan-video-style` (a link in `~/.claude/skills`)
- Optional: the open-source DaVinci Resolve MCP server (samuelgursky/davinci-resolve-mcp, MIT) registered with
  Claude Code, so you can also drive Resolve directly for hand tweaks
- Homebrew packages only if missing: ffmpeg, node, python@3.12

Nothing is uploaded anywhere. The user's videos, scans and styles stay in this folder (`styles/`, `exports/`, git-ignored).

## Requirements (check, don't assume)

1. **Apple Silicon Mac** (M1 or newer), macOS 13+: `uname -m` must print `arm64`.
2. **DaVinci Resolve** in `/Applications`. **Resolve Studio** lets the kit build the edit as a real Resolve
   timeline. The free edition works too, but its scripting is locked (Resolve 21.1+), so edits are rendered
   with ffmpeg instead and the user can import the video into Resolve. Tell the user which one they have.
3. **Xcode Command Line Tools**: `xcode-select -p` prints a path. If not, ask the user to run
   `xcode-select --install` and click Install in the window that opens; wait for them.
4. **Homebrew**: `command -v brew`. If missing, the user must install it (it asks for their password). Ask them
   to type this in the Claude Code prompt:
   `! /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
5. About 5 GB free disk.

## Install

From this folder:

```bash
./install.sh
```

It is safe to run again; it skips what is done. It takes 3-10 minutes (mostly the speech model download and npm).
Use `./install.sh --no-mcp` if the user does not want the Resolve MCP server. If it stops with `STOP:`, the line
says exactly what to do; do it (or ask the user if it needs a password/click), then run `./install.sh` again.

## Resolve settings (once, by the user, only for Resolve Studio)

Ask the user to open DaVinci Resolve, then **DaVinci Resolve > Preferences > System > General > External scripting
using: Local**, click Save, and restart Resolve. Without this the kit cannot reach Resolve (it will still render
headless).

## Check

```bash
.venv/bin/stylekit doctor
```
Every line should be `[ok  ]`. `[warn]` on Resolve means edits render headless (fine, but tell the user why).
`[FAIL]` lines each print their fix.

## Prove it works (2 minutes, recommended)

Make a small test clip and edit it (no style needed from the user yet; this uses a throwaway style):

```bash
.venv/bin/python tests/smoke.py
```
It builds a synthetic reference and a synthetic "talking" clip, scans the reference, edits the clip in that style,
checks the output (length, size, loudness, text drawn), and deletes the throwaway style. It prints `SMOKE TEST PASSED`.

## Then tell the user how to use it

- "Scan this video <file or link> and call the style <name>": measures it and makes `/style-<name>`
- "Edit <my footage> in <name> style" (optionally with a hook line, their music file, their b-roll clips)
- Their styles are in `styles/`, finished videos in `exports/<style>/`, and each edit is also a Resolve project
  named `<style> - <video name>` when Resolve Studio is used.

Restart Claude Code once after installing so it loads the new skill (and the MCP server).

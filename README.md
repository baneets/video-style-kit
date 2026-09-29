# video-style-kit

**Drop in a video whose editing you like. Get that editing style as a Claude Code skill. Edit your own footage in it,
built in DaVinci Resolve.**

```
you:    scan this reel https://www.youtube.com/shorts/... and call the style "clean talking head"
claude: measures it frame by frame on your Mac, writes the style, makes /style-clean-talking-head
you:    edit ~/Movies/today.mp4 in clean talking head style, hook "I quit my job for this", music ~/Music/bed.mp3
claude: cuts, frames, grades, captions, builds the timeline in Resolve, renders exports/clean-talking-head/today.mp4
```

Nothing ships with the kit except the method: no videos, no styles. Every style is made by scanning a video you
choose, and your references, styles and exports stay on your Mac.

## Install

Open Claude Code in this folder and say: **"read SETUP.md and set this up"**. Or yourself:

```bash
git clone https://github.com/baneets/video-style-kit.git
cd video-style-kit
./install.sh
.venv/bin/stylekit doctor
```

You need an **Apple Silicon Mac**, **DaVinci Resolve** (Studio builds real timelines; the free edition gets the same
edit rendered by ffmpeg), [Claude Code](https://claude.com/claude-code), Homebrew and the Xcode command line tools.
The installer adds ffmpeg, node and Python 3.12 if missing. Details: [SETUP.md](SETUP.md).

## What a scan measures (all on-device, free)

From every frame of the reference: every cut and transition, shot length, speaker framing and push-ins, every piece
of on-screen text (where, how big, what case, how it enters/holds/leaves, how long), which text is captions (it
matches the spoken words) and how many words show at once, the hook title, words per minute and the pauses the
editor left, the colour (black/mid/white per channel, saturation), and whether music runs under the voice and how
far under. Claude then looks at contact sheets of the reference to name the font and colours, checks its numbers by
drawing its captions over the reference frame, and writes the style out in words (`STYLE.md`).

## What an edit does

1. levels the voice and transcribes it (mlx-whisper, on-device)
2. cuts: the pieces Claude picked (hook first), fillers ("um", "uh"), stumbles and long pauses out at the
   reference's pace, first frame = first word, J-cuts between pieces
3. frames it to the style's shape (9:16 / 16:9 / 1:1 / 4:5), face-centred, with the reference's push-in rhythm
4. matches your footage's colour to the reference (an ASC CDL baked into a LUT, at the style's strength)
5. draws captions, hook and titles with [Remotion](https://www.remotion.dev) as one transparent layer, in the
   reference's position, size, font and motion
6. lays your music under the voice at the reference's level, and your b-roll where Claude placed it
7. builds it in **DaVinci Resolve Studio** (a project per edit: V1 footage, V2 b-roll, V3 text, A1 voice, A2 music),
   renders, normalises loudness to -14 LUFS, and makes a review sheet Claude checks before handing it over

Feedback on a cut ("captions bigger", "less zoom") changes the style itself, so the next edit is right too.

## Commands

```
stylekit doctor                                 check the install; every problem comes with its fix
stylekit scan <file|link>... --name "style"     reference(s) -> styles/<style>/ + the /style-<style> skill
stylekit calibrate <style>                      our text drawn over the reference's, to compare
stylekit styles                                 list styles
stylekit transcript <footage>...                what was said, with times
stylekit edit <footage>... --style <style> [--keep keep.json] [--hook "..."] [--titles titles.json]
              [--broll broll.json] [--music song.mp3] [--aspect 9:16] [--engine auto|resolve|headless] [--plan-only]
```
Files for `--keep`, `--titles`, `--broll`: [EDITING.md](EDITING.md). Every number a style uses is in its
`recipe.json` and can be edited by hand (table in EDITING.md).

## Layout

```
SETUP.md               install instructions written for Claude Code
EDITING.md             the edit process every style skill follows
WRITING_A_STYLE.md     how Claude finishes a style after a scan
install.sh             one-command install (re-runnable)
skills/scan-video-style/   the Claude Code skill: "scan this video"
src/vidkit/            the frame-by-frame scanner (shots, text, motion, faces, words)
src/stylekit/          scan -> recipe, cut, framing, grade, overlays, Resolve build, headless render
remotion/              the text renderer (captions / hook / titles, transparent ProRes 4444)
tools/*.swift          Apple Vision helpers (text, faces, person mask), compiled on install
templates/             the per-style SKILL.md and STYLE.md templates
styles/  exports/      yours, git-ignored
tests/smoke.py         end-to-end test on synthetic media
```

## Limits, plainly

- macOS on Apple Silicon only (the scanner uses Apple Vision and MLX).
- Resolve's scripting is Studio-only in Resolve 21.1+; with the free edition the edit is rendered by ffmpeg with
  the same cut, framing, grade, text and audio, and you can import it into Resolve.
- The editor reproduces cuts, pacing, framing, push-ins, captions, titles, colour and music level. Animated graphics,
  stickers, cut-out speakers and 3D moves in a reference are described in the style's `STYLE.md` so you (or Claude
  through the Resolve MCP server) can add them by hand.
- Colour matching moves black/mid/white and saturation per channel; a look built on curves that bend only the
  mid-tones is approximated, and the scan says how far off it is.
- Fonts are matched to the closest of ~30 Google fonts (or any font installed on your Mac), fetched at render time.
- Downloading references from Instagram/TikTok usually needs `--cookies chrome` (your browser login), or download
  the video yourself and pass the file. Only scan videos you have the right to use as a reference.

## Credits

Remotion (remotion.dev; free for individuals and small teams, see its license), OpenCV, MLX Whisper, ffmpeg,
yt-dlp, and the optional [DaVinci Resolve MCP server](https://github.com/samuelgursky/davinci-resolve-mcp).
License: MIT (see LICENSE).

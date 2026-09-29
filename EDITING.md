# Editing footage in a style (the process Claude follows)

Every style skill (`/style-<name>`) points here. `$KIT` is the folder this file is in; the command is
`$KIT/.venv/bin/stylekit`. Run commands from anywhere; every path may be absolute.

## 0. Before the first edit of a session

`$KIT/.venv/bin/stylekit doctor` — every line should be `ok` (a `warn` on Resolve means the edit renders headless).
Read the style's `STYLE.md` and `recipe.json` (the skill that sent you here names the folder).

## 1. Know what you are making

From the user's message (ask only for what is missing and needed):
- the footage file(s): the talking-head recording(s). B-roll clips are separate (step 3).
- the style (its folder name), and the length they want (default: like the references, `recipe.json` →
  `pacing.reference_seconds`)
- their music file, if the style has music (`recipe.json` → `audio.music: true`). Never download music.

## 2. Read what was said

```bash
$KIT/.venv/bin/stylekit transcript /path/to/footage.mp4 [more files...]
```
Prints every sentence as `[file] start - end  text`, source seconds. First run transcribes on the Mac (a few seconds
per minute of footage; cached after that).

## 3. Plan the edit (this is most of the work)

Write these small JSON files next to the footage or in `/tmp` (all optional):

**keep.json**: the pieces to use, in the order they should play. Without it the whole file is used, tightened.
```json
[{"file": 0, "start": 41.2, "end": 47.9, "label": "hook"},
 {"file": 0, "start": 3.1, "end": 40.6, "label": "body"}]
```
Rules: start the video on its strongest line (the hook first, even if it was said later); cut tangents, repeats and
false starts; end on a complete sentence. Starts/ends are snapped to whole words automatically; fillers ("um", "uh"),
stumbles ("I, I") and long pauses inside each piece are removed automatically at the style's pace.

**Hook text** (`--hook "..."`): when `recipe.json` → `hook.enabled` is true, the style opens on a text title; write
one in the style's voice (short, same case as the references, the promise of the video).

**titles.json**: extra on-screen text on key words, as often as the references do (`pacing.text_elements_per_min`
and STYLE.md). Anchor each on the words that trigger it:
```json
[{"text": "3 RULES", "at_phrase": "three rules", "hold": 2.0},
 {"text": "$10K/MONTH", "at_phrase": "ten thousand a month", "y": 0.3}]
```
`at_phrase` must be words that survive the cut (check with `--plan-only`, below); `at` (seconds of the edit) also works.

**broll.json**: the user's own b-roll clips laid over the talking head:
```json
[{"file": "/path/desk.mp4", "at_phrase": "my desk", "dur": 2.5, "start": 0.0}]
```

Check the cut before rendering (fast, nothing rendered):
```bash
$KIT/.venv/bin/stylekit edit footage.mp4 --style <style> --keep keep.json --plan-only
```
It prints the cut length and writes `cut_transcript.txt` (the words with EDIT times) to use for titles and b-roll.

## 4. Edit

```bash
$KIT/.venv/bin/stylekit edit footage.mp4 --style <style> --name my-video \
    --keep keep.json --hook "..." --titles titles.json --broll broll.json --music song.mp3
```
With DaVinci Resolve **Studio** running (or startable), this builds a real timeline in a new Resolve project
(`<style> - <name>`): V1 the footage (face-framed, punch-ins, graded with a per-file LUT), V2 b-roll, V3 the text
(one transparent clip), A1 voice with J-cuts, A2 music; then renders it. Without Studio scripting it renders the
same edit with ffmpeg. Either way the finished file is `$KIT/exports/<style>/<name>.mp4`, loudness-normalised.

Takes roughly: first transcription ~10-20 s per minute of footage, text overlay ~1 s per second of video, render
about real time.

## 5. Look before handing it over

- Open the review sheet it prints (`review.jpg`): frames across the video. Check the captions sit where the style
  puts them, never over the face (lines that would cover it are moved automatically), the hook is readable, the
  framing keeps the face in.
- The editor LISTENS to the finished cut: if the first word heard is "but"/"so"/"um" it starts on the next word.
- Read the `PROBLEMS:` line if one is printed (length, size or loudness off) and fix before delivering.
- Check the first frame is the first word and the last line is complete (`cut_transcript.txt`).

## 6. Deliver

Tell the user: where the file is, its length, the Resolve project name (they can open it and tweak anything), what
was cut (fillers/pauses count, any lines you dropped), and which style numbers came from a single reference.

## 7. Feedback edits the style

"Captions bigger", "less zoom", "warmer" → change the style's `recipe.json` (the numbers) or `STYLE.md` (the
rules) in the same turn, then re-run the same command. A rule the user states for the style goes into STYLE.md
under "The user's own rules". Never keep a correction only in the conversation.

## What each recipe number does

| key | effect |
|---|---|
| `canvas.aspect` | 9:16, 16:9, 1:1 or 4:5 output; footage of another shape is face-centred and cropped |
| `cut.pause_cap_s` / `cut.breath_s` | pauses longer than the cap are shortened to the breath |
| `cut.remove_fillers` | um/uh/stumbles out |
| `cut.j_cut_s` | the next line's voice starts this much before its picture |
| `cut.tail_hold_s` | room tone held after the last word |
| `punch_in.every` / `.zoom` | every Nth piece pushed in by `zoom` (0 = never) |
| `captions.*` | font, weight, size (fraction of frame height), y (0 top, 1 bottom), colour, stroke, box, case, words_per_row (words on one line), lines (lines shown together), mode (`row` or `reveal` word by word), highlight of the spoken word, enter/exit motion, avoid_face (move a line off the face) |
| `hook.*` / `titles.*` | same text keys, plus `hold_s` (a hook that stays the whole video: set it longer than the video, e.g. 600) |
| `grade.mode` / `.strength` | `match` = footage colour matched to the references (0-1 strength), `none` = untouched |
| `audio.*` | voice level, final loudness, music level under the voice, fades |

Fonts: any of `Montserrat, Inter, Poppins, Anton, Bebas Neue, Oswald, Roboto, Archivo Black, League Spartan,
DM Sans, Playfair Display, Lora, Permanent Marker, Barlow Condensed, Figtree, Manrope, Space Grotesk,
Instrument Serif, Kanit, Rubik, Plus Jakarta Sans, Outfit, Bangers, Lato, Open Sans, Raleway, Work Sans, Nunito,
Merriweather` (downloaded at render time), or the exact name of a font installed on the Mac.
Motion kinds: `cut, fade, pop, scale, slide_up, slide_down, slide_left, slide_right, wipe, typewriter`; easings
`linear, ease_in, ease_out, ease_in_out, ease_out_expo, spring, smooth`.

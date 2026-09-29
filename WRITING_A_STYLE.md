# Finishing a style after a scan (Claude's writer step)

`stylekit scan` measures everything pixels and sound can tell (shot rhythm, where text sits and how big, how it
moves, how fast the words come, the colour, whether music runs under the voice) and writes `recipe.json`. It cannot
name a font or be sure of a colour, and it does not write the style in words. You do that now, by LOOKING. About
10 minutes. `$S` is the style folder the scan printed (`$KIT/styles/<style>`).

## 1. Look at the references

For each `$S/scan/ref*/sheets/`: open (Read) `overview.jpg` (one frame per shot), `hook.jpg` (first 3 s),
`text.jpg` (every text element cropped; `CAPTION` = it matched the spoken words) and `motion.jpg` (entries frame by
frame). Also read `$S/recipe.json` → `provenance` and `pacing`, and `$S/scan/profile.json` → `signature`.

## 2. Fix what pixels cannot tell, in `$S/recipe.json`

For `captions`, `hook` and `titles` (only the parts the references use):
- `font`: the closest from the list in `$KIT/EDITING.md` (heavy geometric sans → Montserrat 800/900; condensed caps →
  Anton / Bebas Neue / Oswald; clean UI sans → Inter; serif → Playfair Display / Instrument Serif). `weight` 400-900.
- `color`, `stroke_color` + `stroke_width` (outline as a fraction of the font size, ~0.06-0.15), `shadow` 0-1, `box`
  (`{"color": "#000000", "opacity": 0.8, "padding": 0.22, "radius": 0.2}` or `null`), `case`.
- `highlight`: when the spoken word changes colour or gets a box: `{"color": "#FFE14D"}` or `{"box_color": "#7B2FF7"}`;
  otherwise `null`.
- Keep the measured `y`, `size`, `words_per_row`, `mode`, `enter`, `exit` unless the sheets clearly contradict them
  (OCR can mistake a sticker or a screenshot for text: judge from `text.jpg`).
- `hook.enabled`: true only if the references open on a title that is not a caption.

Then check your numbers against the references:
```bash
$KIT/.venv/bin/stylekit calibrate <style>
```
and Read `$S/scan/calibrate.jpg`: the right half draws OUR text on top of THEIR frame. Adjust `size`, `y`, `weight`,
`letter_spacing`, `stroke_width` until ours sits on theirs; 2-4 rounds is normal.

## 3. Write `$S/STYLE.md`

Replace every section of the template (and delete the `WRITE_ME` comment). Concrete and short: numbers with where they
came from ("captions at 70 % of the height (measured, 3 references)"), what you saw in the sheets, and a "What the kit
cannot copy automatically" section for anything the editor does not do itself (animated graphics, stickers, cut-out
speaker over a background, 3D moves): describe each well enough to add by hand in Resolve. Put the user's own words
under "The user's own rules". Say plainly when a number rests on one reference.

## 4. Hand it over

Tell the user the style is ready, what defines it (3-5 bullets), and how to use it: "edit <file> in <name> style"
or `/style-<name>` (a new Claude Code session lists the new skill; in this session, just ask). Offer to edit a clip
now.

## Adding references later

`stylekit scan <new video> --name <same name>` measures only the new ones, re-aggregates, writes the new numbers to
`recipe.measured.json` and keeps your edited `recipe.json`. Compare the two and move over the numbers that changed
by more than a little; keep every value the user asked for.

---
name: style-{{SLUG}}
description: Edit video in the "{{NAME}}" style, a style made by scanning {{N}} reference video(s) with video-style-kit. Use when the user asks to edit footage "in {{NAME}} style", "like the {{NAME}} reference", "make this a {{NAME}} video", names this style, or runs /style-{{SLUG}}. Cuts the talking footage, frames it, matches the colour, draws the captions/titles, adds their music, and builds the edit in DaVinci Resolve (Remotion draws the text), then renders one file.
---

# {{NAME}} style

- Kit: `{{KIT}}` (command: `{{KIT}}/.venv/bin/stylekit`)
- This style: `{{STYLE_DIR}}`

1. Read `{{STYLE_DIR}}/STYLE.md` first: the style in words. The user's own rules at its top outrank everything else.
2. `{{STYLE_DIR}}/recipe.json` holds the numbers the editor uses (caption size and place, pacing, grade, music level).
   `provenance` in it says which numbers were measured from the references.
3. Then follow `{{KIT}}/EDITING.md` with `--style {{SLUG}}`: transcript, plan (keep / hook / titles / b-roll), edit,
   look at the result, deliver.
4. Feedback on a cut changes THIS style in the same turn (recipe.json for numbers, STYLE.md for rules), then re-render.

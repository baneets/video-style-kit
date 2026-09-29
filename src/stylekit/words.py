"""Word-level transcripts (on-device mlx-whisper) and caption grouping."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .media import run, work_dir

WHISPER = "mlx-community/whisper-large-v3-turbo"
FILLER_RE = re.compile(r"^(u+h+m*|u+m+|a+h+|e+r+m*|h+m+|m+h*m+)$")      # um, uh, ah, erm, hmm
STUMBLE = {"i", "i'm", "we", "we're", "that's", "that", "the", "it", "it's", "and", "so", "like", "you", "he", "she",
           "they", "a", "to", "but", "my", "is", "was", "just", "this"}


def bare(word: str) -> str:
    return re.sub(r"[^a-z']", "", word.lower())


def transcribe(file: Path, cache: Path | None = None, language: str | None = None) -> list[dict]:
    """[{word, start, end}] in the file's own seconds, cached next to the file's work folder."""
    cache = cache or work_dir(file) / "words.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import mlx_whisper
    wav = cache.with_suffix(".wav")
    run(["ffmpeg", "-y", "-v", "error", "-i", file, "-vn", "-ac", "1", "-ar", "16000", wav], check=True)
    r = mlx_whisper.transcribe(str(wav), path_or_hf_repo=WHISPER, word_timestamps=True,
                               condition_on_previous_text=False, language=language,
                               # a prompt full of fillers makes whisper WRITE the fillers, so they can be cut
                               initial_prompt="Umm, so, uh, I was like, hmm... ah, you know, I mean, um, yeah.")
    words = [{"word": w["word"].strip(), "start": round(w["start"], 3), "end": round(w["end"], 3)}
             for s in r.get("segments", []) for w in s.get("words", []) if w["word"].strip()]
    words = drop_ghost_words(words, wav)
    wav.unlink(missing_ok=True)
    cache.write_text(json.dumps(words))
    return words


def drop_ghost_words(words: list[dict], wav: Path, floor_db: float = -42.0) -> list[dict]:
    """Whisper writes words over silence and room tone. A word whose span is quieter than `floor_db` is dropped."""
    raw = run(["ffmpeg", "-v", "error", "-i", wav, "-ac", "1", "-ar", "8000", "-f", "s16le", "-"], capture_output=True).stdout
    a = np.frombuffer(raw, dtype=np.int16).astype(float) / 32768
    keep = []
    for w in words:
        i0, i1 = int(max(0, w["start"] - 0.05) * 8000), int((w["end"] + 0.05) * 8000)
        seg = a[i0:max(i1, i0 + 400)]
        if seg.size and 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9) >= floor_db:
            keep.append(w)
    return keep


def caption_rows(words: list[dict], per_row: int = 3, last_hold: float = 0.20, max_gap: float = 1.0) -> list[dict]:
    """Group words into on-screen rows: `per_row` words a row, a sentence end or a pause starts a new row, each row
    holds until the next begins (no flicker), the last one holds `last_hold` past its word. Fillers never show."""
    words = [w for w in words if not FILLER_RE.match(bare(w["word"]) or "x")]
    groups, i = [], 0
    while i < len(words):
        grp = words[i:i + max(1, per_row)]
        for k in range(1, len(grp)):
            if grp[k - 1]["word"][-1:] in ".?!" or grp[k]["start"] - grp[k - 1]["end"] > max_gap:
                grp = grp[:k]
                break
        groups.append(grp)
        i += len(grp)
    rows = []
    for gi, grp in enumerate(groups):
        on = grp[0]["start"]
        off = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else grp[-1]["end"] + last_hold
        if off - grp[-1]["end"] > max_gap:
            off = grp[-1]["end"] + last_hold
        rows.append({"start": round(on, 3), "end": round(max(on + 0.15, off), 3),
                     "words": [{"text": w["word"], "start": w["start"], "end": w["end"]} for w in grp]})
    return rows


if __name__ == "__main__":
    # ponytail: the one runnable check for the pure part.
    ws = [{"word": w, "start": i * 0.4, "end": i * 0.4 + 0.3} for i, w in enumerate("I sold two companies. um Here is how".split())]
    rows = caption_rows(ws, 3)
    assert [" ".join(x["text"] for x in r["words"]) for r in rows] == ["I sold two", "companies.", "Here is how"], rows
    assert all(rows[i]["end"] == rows[i + 1]["start"] for i in range(len(rows) - 1)), "rows must tile"
    print("words.py self-check ok")

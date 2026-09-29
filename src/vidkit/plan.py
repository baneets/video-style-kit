"""Classify a spoken line by what kind of visual it asks for (enumeration, number, cta, question, ...)."""
from __future__ import annotations

import re

# how a spoken line is classified. First match wins, so order matters: the specific patterns come first.
TRIGGERS: list[tuple[str, re.Pattern]] = [
    ("enumeration", re.compile(r"\b(\d+|one|two|three|four|five|six|seven)\s+(things|ways|steps|reasons|tips|"
                               r"mistakes|rules|secrets|lessons)\b|\b(first|second|third|finally|lastly)\b", re.I)),
    ("number", re.compile(r"(?<![\w.])(\$\s?\d[\d,.]*|\d[\d,.]*\s?(%|x|k\b|million|billion|hours?|minutes?|days?|"
                          r"weeks?|months?|years?)|\d[\d,.]{2,})", re.I)),
    ("cta", re.compile(r"\b(follow|comment|dm me|link in bio|subscribe|save this|share this|click|sign up|"
                       r"try it|check it out)\b", re.I)),
    ("question", re.compile(r"\?\s*$|^\s*(what|why|how|who|when|where|do you|have you|ever wonder)\b", re.I)),
    ("objection", re.compile(r"\b(but|however|the problem is|the catch|except|unless|actually no|"
                             r"here'?s the thing|wrong)\b", re.I)),
    ("reveal", re.compile(r"\b(this is|introducing|meet|it'?s called|the tool is|turns out|watch this|"
                          r"look at this|here'?s)\b", re.I)),
    ("claim", re.compile(r"\b(you can|you'?ll|nobody|everyone|most people|the truth|never|always|"
                         r"guaranteed|fastest|best|only way)\b", re.I)),
    ("transition_word", re.compile(r"^\s*(so|now|then|next|after that|meanwhile|and then)\b", re.I)),
]


def classify(text: str) -> tuple[str, str | None]:
    """Returns (trigger_kind, the matched phrase). 'none' when the line asks for no visual."""
    if not text or not text.strip():
        return "none", None
    for kind, rx in TRIGGERS:
        m = rx.search(text)
        if m:
            return kind, m.group(0).strip()
    return "none", None

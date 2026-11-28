"""Checks for the conversational generator. Numeric agreement is necessary, not entailment."""

from decimal import Decimal
import re

from .generation import resolve_citations
from .text import prefix_within


def recent_conversation(history, tokenizer):
    turns = []
    for turn in history[-2:]:
        role = turn["role"]
        if role not in {"assistant", "user"}:
            raise ValueError("expected a past user or assistant turn")
        content = " ".join(turn["content"].split()).replace("<|", "< |")
        if content:
            turns.append(f"{role}: {prefix_within(content, tokenizer, 96, special=False)}")
    return "\n".join(turns)


# Capture telephone numbers as a unit, so matching the area code alone cannot pass.
_LITERAL = re.compile(
    r"(?<!\w)(?P<phone>(?:1[- .])?(?:\(\d{3}\)\s*|\d{3}[- .])\d{3}[- .]\d{4})(?!\w)"
    r"|(?<!\w)(?P<currency>[$£€]\s*)?(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"(?:\s*(?P<unit>%|percent\b|dollars?\b|days?\b|weeks?\b|months?\b|years?\b|hours?\b))?(?!\w)",
    re.I,
)


def quantities(text):
    values = set()
    for match in _LITERAL.finditer(text):
        if match["phone"]:
            digits = re.sub(r"\D", "", match["phone"])
            values.add(("phone", digits[-10:]))
        else:
            unit = (match["unit"] or "").lower().rstrip("s")
            unit = {"percent": "%", "dollar": "$"}.get(unit, unit)
            currency = (match["currency"] or "").strip()
            values.add((currency or unit, Decimal(match["number"].replace(",", ""))))
    return values


def check_cited_values(text, prompt, evidence):
    """Each group of numeric claims must occur in its cited visible source(s)."""
    visible = {e["evidence_id"]: quantities(e["text"][:prompt.visible_characters[e["evidence_id"]]])
               for e in evidence if e["evidence_id"] in prompt.visible_characters}
    start = 0
    for group in re.finditer(r"\[[^\[\]\r\n]+\]", text):
        ids, _ = resolve_citations(group.group(), prompt)
        if not ids:
            continue
        if not quantities(text[start:group.start()]) <= set().union(*(visible[k] for k in ids)):
            return False
        start = group.end()
    return not text[start:].strip(" \t\r\n.,;:!?")


def copied_example(text, prompt, evidence):
    """Catch the two concrete example facts in v2 when their cited passage lacks necessary details."""
    bodies = {e["evidence_id"]: e["text"][:prompt.visible_characters[e["evidence_id"]]]
              for e in evidence if e["evidence_id"] in prompt.visible_characters}
    examples = [(r"\b(?:you can )?renew online or at an office\b",
                 lambda s: all(word in s.lower() for word in ("renew", "online", "office"))),
                (r"\bthe fee is\s*\$\s*25(?:\.00)?(?!\d|[.,]\d)",
                 lambda s: ("$", Decimal(25)) in quantities(s))]
    for pattern, supported in examples:
        for occurrence in re.finditer(pattern, text, re.I):
            tail = text[occurrence.end():]
            # Allow the citation before or immediately after sentence punctuation.
            group = re.match(r"^[\s.!?]*((?:\[[^\[\]\n]+\][ \t]*)+)", tail)
            ids = resolve_citations(group[1], prompt)[0] if group else []
            if not any(supported(bodies[i]) for i in ids):
                return True
    return False

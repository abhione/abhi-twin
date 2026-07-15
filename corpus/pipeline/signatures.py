"""Email signature + quoted-text stripping. Host: mac | spark.

Aggressive by design: the spec's failure mode #4 is the persona LoRA overfitting
to boilerplate ("Best, Abhi"). eval/persona.py measures the residual boilerplate
rate; this module is the first line of defense.
"""

from __future__ import annotations

import re

# a line that *starts* a signature block when seen in the tail of the message
_SIG_STARTERS = re.compile(
    r"^\s*(--\s*$"
    r"|best[,!]?\s*$|best regards[,!]?\s*$|regards[,!]?\s*$|kind regards[,!]?\s*$"
    r"|thanks[,!]?\s*$|thank you[,!]?\s*$|thanks so much[,!]?\s*$|cheers[,!]?\s*$"
    r"|sent from my .*$|get outlook for .*$)",
    re.IGNORECASE,
)

_QUOTE_INTRO = re.compile(r"^\s*on .{0,120} wrote:\s*$", re.IGNORECASE)
_ORIG_MSG = re.compile(r"^\s*-{2,}\s*(original|forwarded) message\s*-{2,}", re.IGNORECASE)
# how far from the end of the message a sign-off may start and still be stripped
_SIG_TAIL_LINES = 6


def strip_quoted(text: str) -> str:
    """Drop quoted reply text: `> ` lines, 'On ... wrote:' intros, forwarded blocks."""
    out: list[str] = []
    for line in text.splitlines():
        if _ORIG_MSG.match(line) or _QUOTE_INTRO.match(line):
            break  # everything below is the quoted thread
        if line.lstrip().startswith(">"):
            continue
        out.append(line)
    return "\n".join(out)


def strip_signature(text: str) -> str:
    """Cut the message at a signature marker near the tail ('-- ' cuts anywhere)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not _SIG_STARTERS.match(line):
            continue
        near_tail = len(lines) - i <= _SIG_TAIL_LINES
        if line.strip().startswith("--") or near_tail:
            lines = lines[:i]
            break
    return "\n".join(lines)


def clean_email_body(text: str) -> str:
    """strip quoted -> strip signature -> collapse runs of blank lines."""
    text = strip_signature(strip_quoted(text))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

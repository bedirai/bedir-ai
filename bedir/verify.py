"""The quote-hash gate, as an independent client-side check.

A claim in a Bedir debate counts only if its quote appears in the source
document word for word. The server locates each quote, records where it
starts and ends, and returns the SHA-256 hash of the exact span. This module
repeats that work on your machine, so you never have to trust the server's
word for it.

Matching rules (identical to the server):
- Runs of whitespace count as one space.
- Curly and straight quote marks are the same; en and em dashes equal "-".
- Case is ignored. Nothing else is normalised: no stemming, no paraphrase.
- A quote must be at least 15 characters long after normalisation.

Offsets are UTF-16 code units, the same unit JavaScript uses, so they match
the server's ``start`` and ``end`` fields even for text with emoji.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

MIN_QUOTE = 15
MAX_QUOTE = 400

_CHAR_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00ad": "",
}
_EDGE_QUOTES = "\"'“”‘’„«» \t\r\n"


@dataclass(frozen=True)
class Match:
    """Where a quote sits in the document."""

    verified: bool
    quote: str
    start: int | None = None  # UTF-16 offset, inclusive
    end: int | None = None  # UTF-16 offset, exclusive
    sha256: str | None = None


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalise(text: str) -> tuple[str, list[int]]:
    """Return the normalised text and, for each of its characters, the index in ``text``."""
    out: list[str] = []
    origin: list[int] = []
    space = False
    for i, ch in enumerate(text):
        ch = _CHAR_MAP.get(ch, ch)
        if not ch:
            continue
        if ch.isspace():
            if space or not out:
                continue
            ch = " "
            space = True
        else:
            space = False
        out.append(ch.lower()[0])
        origin.append(i)
    if out and out[-1] == " ":
        out.pop()
        origin.pop()
    return "".join(out), origin


def clean_quote(quote: str) -> str:
    """Strip wrapping quote marks and a leading or trailing ellipsis."""
    q = (quote or "").strip().strip(_EDGE_QUOTES)
    q = re.sub(r"^(\.\.\.|…)\s*|\s*(\.\.\.|…)$", "", q)
    return q.strip()[:MAX_QUOTE]


def _utf16_offsets(text: str) -> list[int]:
    """UTF-16 offset of every code point in ``text``, plus the total length."""
    offsets = [0]
    for ch in text:
        offsets.append(offsets[-1] + (2 if ord(ch) > 0xFFFF else 1))
    return offsets


def locate(document: str, quote: str) -> Match:
    """Find ``quote`` in ``document`` using the gate's matching rules."""
    q = clean_quote(quote)
    needle, _ = _normalise(q)
    if len(needle) < MIN_QUOTE:
        return Match(False, q)
    hay, origin = _normalise(document)
    at = hay.find(needle)
    if at < 0:
        return Match(False, q)
    first = origin[at]
    last = origin[at + len(needle) - 1] + 1
    exact = document[first:last]
    u16 = _utf16_offsets(document)
    return Match(True, exact, u16[first], u16[last], sha256(exact))


def slice_utf16(document: str, start: int, end: int) -> str:
    """Slice ``document`` by UTF-16 offsets, as JavaScript's ``String.slice`` does."""
    raw = document.encode("utf-16-le")
    return raw[2 * start : 2 * end].decode("utf-16-le")


def check(document: str, claim: dict) -> tuple[bool, str]:
    """Re-verify one claim returned by the server.

    Returns ``(ok, reason)``. ``ok`` is True when the server's verdict on the
    quote agrees with an independent check on this machine.
    """
    quote = claim.get("quote") or ""
    if not claim.get("verified"):
        local = locate(document, quote)
        if local.verified:
            return False, "server eclipsed a quote that is in the document"
        return True, "eclipsed: not in the document"
    start, end, digest = claim.get("start"), claim.get("end"), claim.get("sha256")
    if not isinstance(start, int) or not isinstance(end, int) or not digest:
        return False, "verified claim without offsets or hash"
    span = slice_utf16(document, start, end)
    if span != quote:
        return False, "text at the reported offsets differs from the quote"
    if sha256(span) != digest:
        return False, "hash does not match the quoted span"
    return True, "verbatim, hash matches"

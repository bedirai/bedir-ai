"""Minimal client for the public Bedir debate API. Standard library only."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterator

API = "https://bedirsavasi.com/api/debate"
LANGS = ("en", "tr", "de")


class DebateError(RuntimeError):
    """Raised when the API refuses or fails a debate."""


def debate(question: str, document: str, lang: str = "en", api: str = API, timeout: float = 240) -> Iterator[dict]:
    """Run one debate and yield its events as they stream in.

    Events, in order: ``start``, ``sides``, ``rebuttals``, ``verdict``, ``done``.
    An ``error`` event, or an HTTP error, raises :class:`DebateError`.
    """
    if lang not in LANGS:
        raise ValueError(f"lang must be one of {LANGS}")
    body = json.dumps({"lang": lang, "question": question, "document": document}).encode("utf-8")
    request = urllib.request.Request(
        api,
        data=body,
        method="POST",
        headers={"content-type": "application/json", "user-agent": "bedir-demo/0.1"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:200]
        if error.code == 429:
            raise DebateError("daily demo limit reached for your network; try again tomorrow") from None
        if error.code == 422:
            raise DebateError("question must be 8+ characters and the document 200 to 12,000 characters") from None
        raise DebateError(f"HTTP {error.code}: {detail}") from None
    with response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "error":
                raise DebateError(f"debate failed: {event.get('code')}")
            yield event

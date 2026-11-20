"""Terminal demo: run a live Bedir debate and verify every quote locally.

    python -m bedir --sample en-commercial-lease
    python -m bedir --document memo.txt --question "Should we ship it?" --lang en
    python -m bedir --replay examples/debates/en-commercial-lease.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

from .client import API, DebateError, debate
from .verify import check

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "examples" / "documents"
COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
WIDTH = min(shutil.get_terminal_size((100, 20)).columns, 100)


def paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


SUN = lambda t: paint(t, "38;5;214")  # noqa: E731
TIDE = lambda t: paint(t, "38;5;111")  # noqa: E731
DIM = lambda t: paint(t, "2")  # noqa: E731
OK = lambda t: paint(t, "38;5;114")  # noqa: E731
BAD = lambda t: paint(t, "38;5;209")  # noqa: E731
BOLD = lambda t: paint(t, "1")  # noqa: E731

PHASES = {
    "start": "Hilal         both sides build their strongest case",
    "sides": "İlk Dördün    every quote checked against the text and hashed",
    "rebuttals": "Tutulma       cross-examination; unverified claims go dark",
    "verdict": "Bedir         the judge weighs what survived",
}


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=indent)


def show_claim(document: str, claim: dict, colour, label: str = "") -> tuple[int, int]:
    ok, reason = check(document, claim)
    mark = OK("✓") if claim.get("verified") and ok else (BAD("✕") if not ok else DIM("◌"))
    print(wrap(f"{mark} {label}{claim.get('text', '')}", "    "))
    if claim.get("quote"):
        print(colour(wrap(f"“{claim['quote']}”", "      ")))
    detail = f"sha256 {claim['sha256'][:12]}… @{claim['start']}" if claim.get("verified") else ""
    print(DIM(f"      {reason}{' · ' + detail if detail else ''}"))
    return int(ok), int(bool(claim.get("verified")))


def render(events: list[dict], document: str, question: str) -> int:
    """Print a debate. Returns the number of local verification failures."""
    failures = 0
    for event in events:
        kind = event.get("type")
        if kind in PHASES:
            print(DIM(f"\n── {PHASES[kind]}"))
        if kind == "sides":
            for key, name, colour in (("for", "FOR", SUN), ("against", "AGAINST", TIDE)):
                side = event[key]
                print(f"\n  {colour(BOLD(name))}  {side.get('thesis', '')}")
                for claim in side.get("claims", []):
                    ok, _ = show_claim(document, claim, colour)
                    failures += 1 - ok
        elif kind == "rebuttals":
            for key, name, colour in (("for", "FOR", SUN), ("against", "AGAINST", TIDE)):
                reb = event.get(key)
                if reb:
                    ok, _ = show_claim(document, reb, colour, label=f"{name} rebuts: ")
                    failures += 1 - ok
        elif kind == "verdict":
            p = int(event.get("p_for", 50))
            bar_width = max(20, WIDTH - 24)
            filled = round(bar_width * p / 100)
            print(f"\n  {SUN('█' * filled)}{TIDE('█' * (bar_width - filled))}")
            print(f"  {SUN(f'FOR {p}%')}{' ' * max(1, bar_width - 14)}{TIDE(f'{100 - p}% AGAINST')}")
            print("\n" + wrap(BOLD(event.get("verdict", ""))))
            if event.get("cruxes"):
                print(DIM("\n  Where the sides split"))
                for crux in event["cruxes"]:
                    print(wrap(f"• {crux}", "    "))
            if event.get("flip"):
                print(DIM("\n  What would change the verdict"))
                print(wrap(event["flip"], "    "))
        elif kind == "done":
            seconds = event.get("ms", 0) / 1000
            print(DIM(f"\n── {event.get('verified')} of {event.get('total')} quotes verified by the server · {seconds:.1f} s"))
    return failures


def summary(failures: int) -> None:
    status = OK("all quotes re-verified on this machine") if failures == 0 else BAD(f"{failures} local check(s) failed")
    print(f"── {status}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bedir", description="Run a live Bedir debate and verify every quote locally.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sample", help="name of a file in examples/documents, without .txt")
    src.add_argument("--document", type=Path, help="path to a text file (200 to 12,000 characters)")
    src.add_argument("--replay", type=Path, help="print a recorded debate from examples/debates")
    parser.add_argument("--question", help="a yes-or-no question about the document")
    parser.add_argument("--lang", choices=("en", "tr", "de"), help="language of the debate (default: from the sample, else en)")
    parser.add_argument("--save", type=Path, help="write the full transcript to this JSON file")
    parser.add_argument("--api", default=API, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.replay:
        record = json.loads(args.replay.read_text("utf-8"))
        print(BOLD(f"\n{record['question']}") + DIM(f"   [{record['lang']} · recorded {record['recorded_at']}]"))
        failures = render(record["events"], record["document"], record["question"])
        summary(failures)
        return 1 if failures else 0

    lang, question = args.lang, args.question
    if args.sample:
        meta = {m["file"][:-4]: m for m in json.loads((DOCS / "index.json").read_text("utf-8"))}
        if args.sample not in meta:
            parser.error(f"unknown sample; choose from: {', '.join(sorted(meta))}")
        document = (DOCS / f"{args.sample}.txt").read_text("utf-8").strip()
        lang = lang or meta[args.sample]["lang"]
        question = question or meta[args.sample]["question"]
    else:
        document = args.document.read_text("utf-8").strip()
    lang = lang or "en"
    if not question:
        parser.error("--question is required with --document")

    print(BOLD(f"\n{question}") + DIM(f"   [{lang} · live at {args.api}]"))
    events: list[dict] = []
    failures = 0
    started = time.time()
    try:
        for event in debate(question, document, lang, api=args.api):
            events.append(event)
            failures += render([event], document, question)
    except DebateError as error:
        print(BAD(f"\n{error}"), file=sys.stderr)
        return 2
    summary(failures)
    if args.save:
        record = {
            "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "api": args.api,
            "lang": lang,
            "question": question,
            "document": document,
            "wall_seconds": round(time.time() - started, 1),
            "events": events,
        }
        args.save.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", "utf-8")
        print(DIM(f"saved {args.save}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

<p align="center">
  <img src="docs/img/hero.png" alt="BedirAI: every answer meets its strongest opponent" width="100%">
</p>

# BedirAI

**Adversarial reasoning models. Every answer meets its strongest opponent.**

Bedir makes two AI debaters argue opposite answers to a hard question about a document, at full strength. Every claim must quote the document word for word. Each quote is checked on the server and hashed with SHA-256, and any claim whose quote is not in the text is *eclipsed* before the judge sees it. A separate judge model, Mizan, weighs only the evidence that survives and returns a probability, the points where the sides split and the evidence that would change its verdict.

*Bedir* means full moon. The moon is full only in opposition, when it stands 180° from the sun. An argument is complete only when it faces the strongest case against it.

[**Try the live Arena →**](https://bedirsavasi.com/arena/) &nbsp;·&nbsp; [Protocol and API](docs/protocol.md) &nbsp;·&nbsp; [Models](https://bedirsavasi.com/models/) &nbsp;·&nbsp; [Research](https://bedirsavasi.com/research/)

This repository holds the public demo: a command-line client for the live debate API, an independent quote-hash verifier, recorded debates in English, Turkish and German, and the sample documents behind them.

## The Arena

<p align="center">
  <img src="docs/img/arena.png" alt="A live debate in the Bedir Arena: two sides with verified quotes and Mizan's verdict" width="85%">
</p>

## Run a debate from your terminal

Python 3.9 or later, no dependencies.

```bash
git clone https://github.com/bedirai/bedir-ai.git
cd bedir-ai
python3 -m bedir --sample en-commercial-lease
```

Use your own text (200 to 12,000 characters) and a yes-or-no question:

```bash
python3 -m bedir --document memo.txt --question "Should we approve the vendor?" --lang en
```

Replay a recorded debate without calling the API:

```bash
python3 -m bedir --replay examples/debates/tr-dort-gunluk-hafta.json
```

The client re-checks every quote on your machine: the text at the reported offsets must equal the quote, and its SHA-256 must equal the server's hash. A short excerpt:

```text
Can the tenant end the lease early without paying the penalty?   [en]

── İlk Dördün    every quote checked against the text and hashed

  FOR  The tenant is entitled to terminate the lease without penalty because the landlord
       failed to repair a structural defect within thirty days of written notice.
    ✓ Clause 7.2 explicitly permits termination without penalty if the landlord fails to
      complete repairs within thirty days.
      “If the Landlord fails to complete such repairs within thirty (30) days of the notice,
      the Tenant may terminate this Lease without penalty.”
      verbatim, hash matches · sha256 824cdf8855c7… @447

  AGAINST  The tenant cannot terminate the lease without penalty because the statutory
           notice requirement was not met.
    ✓ The lease explicitly requires written notice delivered by registered mail or hand,
      which the tenant failed to do.
      “Notice of termination shall be given in writing and delivered by registered mail or by hand.”
      verbatim, hash matches · sha256 d07f0f4e0f45… @770

── Bedir         the judge weighs what survived

  FOR 75%                                                     25% AGAINST

── 8 of 8 quotes verified by the server · 14.5 s
── all quotes re-verified on this machine
```

A JavaScript version for Node.js 18+ is in [`examples/javascript/debate.mjs`](examples/javascript/debate.mjs).

## Verify quotes in your own code

```python
from bedir import debate, check

document = open("examples/documents/en-four-day-week.txt").read()
for event in debate("Should the city move all of its office staff to a four-day week?", document):
    if event["type"] == "sides":
        for claim in event["for"]["claims"] + event["against"]["claims"]:
            ok, reason = check(document, claim)
            print("✓" if ok else "✕", reason, "|", claim["quote"])
    elif event["type"] == "verdict":
        print("Mizan:", event["p_for"], "% for |", event["verdict"])
```

`bedir.locate(document, quote)` finds a quote with the same rules the server uses and returns its UTF-16 offsets and hash. The rules are deliberately strict: whitespace, quote marks and dashes are normalised and case is ignored; nothing else is. A paraphrase never passes. See [`bedir/verify.py`](bedir/verify.py) and its tests:

```bash
python3 -m unittest -v
```

## Recorded debates

Real runs against the live API on 20 November 2026, in [`examples/debates/`](examples/debates):

| File | Language | Question | Quotes verified | Mizan |
|---|---|---|---|---|
| [`en-commercial-lease.json`](examples/debates/en-commercial-lease.json) | English | Can the tenant end the lease early without paying the penalty? | 8 of 8 | 75% for |
| [`tr-dort-gunluk-hafta.json`](examples/debates/tr-dort-gunluk-hafta.json) | Turkish | Should the city move all office staff to a four-day week? | 8 of 8 | 35% for |
| [`de-ki-kreditmodell.json`](examples/debates/de-ki-kreditmodell.json) | German | Should the bank deploy the vendor's AI credit-scoring model? | 8 of 8 | 35% for |

Each file holds the question, the full document and every streamed event, so anyone can re-verify every hash offline.

## Sample documents

Nine synthetic case files, three topics in three languages, in [`examples/documents/`](examples/documents): a commercial lease dispute, a four-day-week pilot and a model-risk review of an AI credit-scoring model. All names and figures are invented.

## The models

| Model | Role |
|---|---|
| **Bedir** | Debater. Argues either side at full strength; trained by self-play RL against itself, rewarded by the judge, penalised for any claim without a verified quote. |
| **Mizan** | Judge. Weighs two opposing cases and returns a calibrated probability; trained on real outcomes and scored with the Brier rule. |
| **Tutulma** | Red team. Attacks the judge with rhetoric, length and anchoring tricks; every attack that works becomes training data for Mizan. |
| **Hilal** | Compact 8B model that runs the whole debate on a single server, with open weights. |

The live Arena runs Bedir's protocol on open-weight models through Cloudflare Workers AI. Bedir's own debater and judge take over as they finish training. Read more on the [models page](https://bedirsavasi.com/models/) and the [training program](https://bedirsavasi.com/training/).

## Use and limits

- The public API has a daily limit per network. Do not send personal or confidential data; submitted text is processed once and not stored.
- A verified quote proves that the words appear in the document. It does not prove that a claim is true.
- Bedir shows arguments and evidence. It is not legal, financial or professional advice and does not predict the outcome of any case.

## License

Apache License 2.0. See [LICENSE](LICENSE).

**Contact:** merhaba@bedirsavasi.com · [bedirsavasi.com](https://bedirsavasi.com)

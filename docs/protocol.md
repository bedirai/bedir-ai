# The Bedir debate protocol

Bedir answers a yes-or-no question about a document by making two debaters argue opposite sides, and letting a separate judge weigh only the evidence that survives verification. The protocol runs in four phases, named after the phases of the moon.

| Phase | Name | What happens |
|---|---|---|
| 1 | **Hilal** (crescent) | Each side writes a one-sentence thesis and up to three claims. Every claim carries one quote copied from the document. |
| 2 | **İlk Dördün** (first quarter) | Every quote is matched against the document. A match records the start and end offsets and the SHA-256 hash of the exact span. |
| 3 | **Tutulma** (eclipse) | Each side rebuts the other's weakest verified claim, again with a quote. Claims whose quotes are not in the document are *eclipsed*: they are shown, struck through, and never reach the judge. |
| 4 | **Bedir** (full moon) | The judge, from a different model family, sees only the verified claims, in random order, and never the document. It returns a probability for YES, the cruxes and the evidence that would change its verdict. |

## Design rules

- **Both sides, always.** There is no mode that argues only one side.
- **The judge never sees the document**, only quoted evidence. Debate helps a judge most when it relies on the debaters for information ([Kenton et al., 2024](https://arxiv.org/abs/2407.04622)).
- **Short turns.** Claims are capped at 35 words and rebuttals at 60. Uncapped critics learn to game a weaker judge ([Kenton et al., 2026](https://arxiv.org/abs/2608.17776)).
- **Deterministic evidence gate.** No amount of rhetoric can pass a quote that is not in the text. See [`bedir/verify.py`](../bedir/verify.py) for the exact matching rules.
- **Randomised order.** The judge sees the two sides in random order to reduce position bias.

## API

`POST https://bedirsavasi.com/api/debate`

```json
{ "lang": "en", "question": "Can the tenant end the lease early without paying the penalty?", "document": "..." }
```

| Field | Rule |
|---|---|
| `lang` | `en`, `tr` or `de`: the language of theses, claims and verdict. Quotes stay in the document's wording. |
| `question` | 8 to 300 characters; a yes-or-no question. |
| `document` | 200 to 12,000 characters of plain text. Do not send personal or confidential data. |

The public demo allows a small number of debates per network per day; extra requests receive HTTP 429. Submitted text is processed once and not stored.

### Response

`application/x-ndjson`: one JSON event per line, streamed as each phase finishes.

```text
{"type":"start","remaining":11}
{"type":"sides","for":{"thesis":"...","claims":[Claim, ...]},"against":{...}}
{"type":"rebuttals","for":Claim|null,"against":Claim|null}
{"type":"verdict","p_for":75,"verdict":"...","cruxes":["..."],"flip":"..."}
{"type":"done","verified":8,"total":8,"ms":14512,"remaining":11,"engine":{...}}
```

A `Claim` is:

```json
{
  "text": "Clause 7.2 explicitly permits termination without penalty if the landlord fails to complete repairs within thirty days.",
  "quote": "If the Landlord fails to complete such repairs within thirty (30) days of the notice, the Tenant may terminate this Lease without penalty.",
  "verified": true,
  "start": 447,
  "end": 585,
  "sha256": "824cdf8855c70f0643e4dcf13cff56453285a3a5c1536b8466c082d7f2b8891a"
}
```

`start` and `end` are UTF-16 offsets (JavaScript string indices). For an eclipsed claim, `verified` is `false` and the offsets and hash are absent. Rebuttals add `target`, the number of the opponent's claim they answer.

An `{"type":"error"}` event means the debate could not be completed.

## Engine

The public Arena runs this protocol on open-weight models through Cloudflare Workers AI: one model family debates, another judges. Bedir's own debater (Bedir) and judge (Mizan) replace them as their training completes.

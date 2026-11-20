# Contributing to BedirAI

We welcome contributions to the quote-hash verifier, RL reward functions, evaluation metrics, data-lineage checks, multilingual sample cases and hardware-tuned training recipes.

## Local verification before opening a PR

This repository has zero external Python dependencies (`Python >= 3.9`). Run the full suite:

```bash
python3 -m unittest -v
python3 -m bedir.eval --debates examples/debates
python3 -m bedir.eval --validate-bench benchmarks/mizan_bench_sample.jsonl
python3 -m training.compute_model --phase all --sensitivity
python3 -m training.scaling.benchmark_harness training/scaling/schema_example.json
```

## Contribution guidelines

- **Keep `bedir/` dependency-free** (Python standard library only) so the verifier, reward hooks and evaluation harness run reproducibly in any training or CI container.
- **Preserve strict quote matching:** never add stemming, fuzzy matching or paraphrase tolerance to `bedir/verify.py`.
- **Clean data lineage:** never commit personal data, unmasked court filings or outputs from closed frontier model APIs into training samples or benchmarks.
- **Honest reporting:** clearly label synthetic schema examples vs. measured runs.

All contributions are licensed under Apache License 2.0.

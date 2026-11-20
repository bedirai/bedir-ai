# Security & Responsible Disclosure

## Reporting a vulnerability

If you discover a security issue in this repository, the quote-hash verifier or the public Arena API (`https://bedirsavasi.com/api/debate`), please email **merhaba@bedirsavasi.com** with `[SECURITY]` in the subject line.

We acknowledge reports within two working days and coordinate fixes before public disclosure.

## Safety & dual-use policy

- **Quote-hash gate integrity:** Any input that causes `bedir.verify.locate()` or `bedir.verify.check()` to falsely verify a quote that does not appear verbatim in the source document is treated as a high-severity bug.
- **No one-sided persuasion mode:** BedirAI protocols always argue both sides.
- **Red-team model handling (`Tutulma`):** Persuader/red-team weights trained to exploit judge blind spots are kept private and are never published; only aggregate hack-rate metrics and hardened judge checkpoints are released.
- **Data privacy:** Do not submit personal, confidential or privileged text to the public Arena demo.

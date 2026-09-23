# Ground truth — Aerarium Sistemi

`aerarium_ground_truth.csv` holds one expected verdict per requirement (70) for
`Aerarium_Sistemi_Regulation_EN.pdf` (Internal Regulation, v1.0, March 2026),
with confidence, the sections relied on and a rationale.

## Criteria

- **CONFORME** — every obligation of the requirement is met by a practice the
  document states is in place.
- **PARZIALMENTE_CONFORME** — at least one obligation of the requirement is met
  by a practice in place, but not all.
- **NON_CONFORME** — no obligation is met: the document is silent, mentions the
  topic only tangentially, describes a plan, or states the practice is absent.
- **NON_APPLICABILE** — the requirement does not apply to the organization.

A practice that exists in the document but addresses a different object than
the requirement (e.g. human validation of alerts cited for an impact
assessment) does not count.

## Limits

Built by a single annotator (an LLM, from the document text only), not with the
two-reviewer protocol of the thesis. Use it as a cross-check, not as a
replacement for φ\*. The `confidenza` column flags the borderline calls.

## Usage

```bash
python scripts/compare_ground_truth.py \
  --report report.json \
  --ground-truth docs/validation/aerarium_ground_truth.csv
```

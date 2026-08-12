# MoA-CAS — Preprocessing Progress Report

**Branch:** `aswathi-dataset` · **Status:** Preprocessing complete

## Summary

Built and ran a data preprocessing pipeline that turns raw MUCS 2021
recordings into clean, language-tagged manifests ready for training.
All four MUCS code-switched splits and both monolingual baseline checks
have been downloaded, processed, and verified — not just written.

## Datasets processed

| Dataset | Split | Sentences | Hours | Code-mixed | Status |
|---|---|---|---|---|---|
| MUCS Hindi-English | test | 3,132 | 5.14h | 80.0% | ✅ done |
| MUCS Hindi-English | train | 52,769 | 88.98h | 83.9% | ✅ done |
| MUCS Bengali-English | test | 4,275 | 7.00h | 87.5% | ✅ done |
| MUCS Bengali-English | train | 26,563 | 45.50h | 85.6% | ✅ done |
| IndicVoices Hindi (monolingual check) | 200 samples | — | 0.31h | 1.5% | ✅ done |
| IndicVoices Bengali (monolingual check) | 200 samples | — | 0.41h | 9.0% | ✅ done |

**Totals: 86,739 code-switched sentences, ~146.6 hours, real 16kHz audio cached to disk.**

## Validation

- Word/language-tagging logic checked against the source paper's own
  published stats (Biswas et al. 2025, Table 1) on the same MUCS
  Hindi-English test split — within ~2–6% on every metric.
- Train vs. test code-mixed ratios land within a few points of each other
  per language pair — internally consistent.
- Monolingual baselines confirmed genuinely clean (1.5–9% code-mixed, vs.
  80–88% for the real code-switched sets).

## What was built (`aswathi-dataset` branch)

- `src/moa_cas/preprocessing/` — audio extraction/resampling + word-level
  Hindi/Bengali/English tagging + code-switch detection
- `src/moa_cas/pipeline/` — manifest schema, build orchestration, splits
- `configs/datasets.yaml` — registry driving `scripts/build_manifest.py`
- 11 passing unit tests, no real audio required
- Full docs: [PIPELINE.md](PIPELINE.md) (how it works), [WHAT_WAS_DONE.md](WHAT_WAS_DONE.md) (detailed run log)

## Known limitations (by design, not oversights)

- No Tamil/Telugu — MUCS has no code-switched audio for those languages
- No held-out dev split yet (helper exists, unused — MUCS's own train/test used as-is)
- No augmentation / synthetic data generation (deferred to a later phase)

## Next steps

1. Decide how to handle the Dravidian-CS gap (different corpus, or the
   synthetic TTS augmentation from the project doc)
2. Merge `aswathi-dataset` into `main` when ready
3. Begin the MoA-CAS architecture itself — adapters, router, two-stage
   training — using these manifests as input (needs a compute decision first)

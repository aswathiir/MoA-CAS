# 01 — Baseline Benchmark and Problem Analysis

The motivating experiment for MoA-CAS: measure how badly a strong
monolingual Indic ASR model degrades on code-switched speech, and
establish precisely *what* causes that degradation.

---

## 1. Work done

Benchmarked **IndicConformer-600M** (`ai4bharat/indic-conformer-600m-multilingual`,
CTC greedy decoding) against two contrasting test sets:

| Run | Dataset | Samples | Purpose |
|---|---|---|---|
| A | IndicVoices Hindi (`valid` split, streamed) | 100 | Clean monolingual reference point |
| B | MUCS 2021 Hindi-English test | 100 | Code-switched condition |

Both were decoded with `lang="hi"` — the model has no English mode at all
(English is not one of the 22 scheduled Indian languages it supports), so
Hindi decoding is the only option available for the code-mixed set.

Beyond running the benchmark, this document adds a **decomposition
analysis** that was not part of the original experiment: an attempt to
determine how much of the measured gap is genuinely attributable to
code-switching versus other confounds. That analysis produced the most
consequential finding in this document (Section 4).

---

## 2. What was achieved

### Headline numbers (recomputed directly from `results/*.csv`)

| Condition | WER | CER | Perfect transcripts | Hallucinations (WER > 100%) |
|---|---|---|---|---|
| IndicVoices Hindi (monolingual) | **7.93%** | 4.16% | 36 / 100 | 0 / 100 |
| MUCS Hindi-English (code-mixed) | **57.14%** | 56.13% | 0 / 100 | 11 / 100 |

A **7.2× degradation** in WER between the two conditions.

### Monolingual performance by speaking style

The clean condition degrades predictably with spontaneity, which is a
useful sanity check that the model and metric behave sensibly:

| Scenario | WER | n |
|---|---|---|
| Read (scripted) | 4.42% | 17 |
| Extempore (spontaneous) | 6.63% | 53 |
| Conversation | 12.20% | 30 |

### Qualitative failure modes on code-mixed speech

Three distinct behaviours appear when English enters the audio:

1. **Phonetic transliteration** — English words are rendered into
   Devanagari by sound rather than recognised as English:
   `document` → `डॉक्यूमेंट`, `formatting` → `फॉर्मेटिंग`. The model
   has no English output vocabulary, so this is the only thing it *can*
   do. Under exact-match WER these all count as errors.
2. **Hallucination** — 11% of utterances produced *more* wrong words
   than the reference contains (WER above 100%), meaning the model
   invented content rather than failing quietly.
3. **Loss of alignment** — the output drifts to transcribing a different
   part of the audio than the reference covers (see Section 4).

---

## 3. Method followed

- **Decoding**: CTC greedy decoding, no external language model, no
  beam search. This measures the acoustic model's own behaviour without
  a language model masking or compensating for its weaknesses.
- **Metrics**: WER and CER via `jiwer`. Both reference and hypothesis
  pass through the same normalisation first — NFC Unicode
  normalisation, lowercasing, punctuation stripped, whitespace
  collapsed. Normalisation is deliberately applied identically to both
  sides so it cannot bias the comparison.
- **Sampling**: the first 100 utterances of each test set, in manifest
  order (not randomly sampled).
- **Determinism**: CTC greedy decoding is deterministic; re-running the
  benchmark reproduces the identical CSV byte-for-byte. This was
  verified in practice — the 57.14% figure has been independently
  reproduced.

---

## 4. Critical analysis — what the gap actually measures

**This section revises the headline interpretation and should be read
before citing the 7.9% → 57.1% figure as a "code-switching gap".**

The natural reading of the headline result is that code-switching causes
the 49-point WER increase. The data does not fully support that.

### 4a. English density does predict higher error — but only partly

Joining the benchmark outputs back against the per-utterance language
tags from the preprocessing pipeline:

| English share of utterance | Mean WER | n |
|---|---|---|
| 0% (pure Hindi) | 53.90% | 12 |
| 1–20% English | 45.77% | 25 |
| 20–40% English | 51.54% | 39 |
| > 40% English | **79.71%** | 24 |

Pearson correlation between English word share and WER: **r = 0.333**
(moderate, positive). Heavily English-laden utterances clearly fail
harder. So code-switching *is* a genuine, measurable effect.

### 4b. But pure-Hindi utterances in MUCS also fail — at 53.9% WER

This is the crucial observation. Utterances in the MUCS test set that
contain **zero English words** still score **53.90% WER**, against
**7.93%** for Hindi in IndicVoices. Roughly 46 of the 49 gap points are
therefore present *before any code-switching is involved at all*.

The difference between those two monolingual conditions is not language
— it is domain and recording conditions. MUCS is spoken-tutorial and
technical-lecture audio (software walkthroughs, with dense technical
vocabulary, much of it transliterated), recorded differently from
IndicVoices' elicited speech.

### 4c. A second confound: the corpus's own segmentation granularity

MUCS's `segments` file tiles each recording **contiguously at whole-second
boundaries** — every segment ends exactly where the next begins, with
zero gaps, and not a single non-integer boundary across the split. Since
real speech boundaries do not land on integer seconds, each segment
necessarily contains fragments of neighbouring utterances whose words
are not in its reference transcript.

This is measurable in the outputs. Across the benchmarked sample:

- **39%** of hypotheses begin with words belonging to the *previous*
  utterance's reference.
- **10%** of hypotheses are materially shorter than their reference
  (the utterance's own ending was cut off).
- Utterances showing either artifact average **62.43%** WER, versus
  **53.37%** for those showing neither.

This was checked against our own extraction code and **is not a bug on
our side** — the pipeline cuts exactly the `[start, end]` windows the
corpus specifies. It is inherent to the MUCS annotation, and it places a
hard floor under achievable WER on this test set.

### 4d. Revised decomposition

| Contribution | Approximate magnitude |
|---|---|
| Domain shift (tutorial/technical speech vs. elicited speech) | The bulk — pure-Hindi MUCS sits at ~54% WER |
| Coarse segmentation artifacts | ~9 WER points on the ~44% of utterances affected |
| Code-switching proper | Real but secondary — ~54% → ~80% as English share rises past 40% |

**Implication for the project**: MoA-CAS targets the code-switching
component. Even if it worked perfectly, it would not close a 49-point
gap, because most of that gap is not code-switching. This does not
invalidate the project — the code-switching effect is real and
substantial — but the motivating claim needs restating honestly, and the
evaluation needs a fair control (Section 6).

---

## 5. Possible outcome

Projections, clearly labelled as such — none of this is measured yet.

- **If MoA-CAS addresses only code-switching**, the plausible target is
  recovering most of the code-switching increment: roughly **57% → 45–50%
  WER** on this test set. Reaching the ~8% range would additionally
  require in-domain adaptation, which is a separate problem from the one
  the architecture is designed to solve.
- **With a fair in-domain control** (Section 6), the measurable
  improvement attributable to MoA-CAS should look considerably more
  favourable than it does against the current IndicVoices reference,
  because the domain confound would be removed from the comparison
  rather than charged against the architecture.
- **The equity metric** (WER disparity between language pairs) may be
  distorted by an asymmetry in the data itself: Bengali-English is
  **31.6%** English by word count against Hindi-English's **25.3%**. If
  Bengali-English scores worse, part of that will be higher English
  density rather than any dialectal or demographic inequity. Density
  needs controlling for before disparity can be attributed.

---

## 6. Final works left

1. **Establish a fair in-domain monolingual control.** Use the
   pure-Hindi subset of MUCS (7,083 utterances available) as the
   monolingual baseline instead of IndicVoices. This isolates
   code-switching from domain shift and is the single most important
   fix to the experimental design.
2. **Benchmark the full test sets.** Only 100 of 3,132 Hindi-English
   utterances have been evaluated. Report confidence intervals — with
   n=100 and per-utterance WER ranging 0.08–2.11, the headline mean
   carries meaningful uncertainty.
3. **Benchmark Bengali-English at all.** It has never been run; there is
   currently no baseline for the second language pair.
4. **Quantify the segmentation floor.** Re-segment a sample with forced
   alignment or VAD and re-benchmark, to measure how much WER is
   irreducible given the corpus annotation. Without this, every future
   improvement number is contaminated by the same artifact.
5. **Separate transliteration from true error.** A transliteration-aware
   scoring pass (or a romanisation-normalised WER) would show how much
   of the code-mixed error is the model correctly hearing the word but
   writing it in the wrong script — which is a fundamentally different
   failure from not recognising it at all, and matters for deciding what
   the adapters actually need to learn.
6. **RNNT comparison.** Only CTC decoding was benchmarked; the model also
   supports RNNT, which may behave differently on code-mixed input.

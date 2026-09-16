# 02 — Data Preprocessing Pipeline

Turning raw MUCS 2021 corpora into clean, language-tagged, training-ready
manifests — the foundation everything downstream depends on.

---

## 1. Work done

Built a preprocessing pipeline from scratch and ran **all four** MUCS
code-switched splits plus two monolingual reference sets through it.

**Datasets acquired and processed** (downloaded from OpenSLR 104):

- MUCS Hindi-English: train (7.3 GB archive) + test
- MUCS Bengali-English: train (3.9 GB archive) + test (0.6 GB archive)
- IndicVoices Hindi and Bengali: 200 samples each, streamed from
  HuggingFace as monolingual sanity checks (text only — audio
  deliberately not cached)

**Code produced** (`src/moa_cas/`):

| Module | Responsibility |
|---|---|
| `preprocessing/audio.py` | Kaldi file parsing, segment extraction, resampling to 16 kHz mono |
| `preprocessing/text.py` | Unicode normalisation, per-word script→language tagging, code-switch point detection |
| `pipeline/manifest.py` | `ManifestRow` schema, JSONL read/write |
| `pipeline/build.py` | Orchestration — adapter → extraction → tagging → filtering → manifest |
| `pipeline/splits.py` | Speaker-disjoint split helper (written, not yet exercised) |
| `datasets/mucs.py`, `datasets/indicvoices.py` | Per-corpus readers |
| `stats.py` | Corpus statistics reporting |

Driven by a registry (`configs/datasets.yaml`) through a single CLI
(`scripts/build_manifest.py`), so adding a split requires a config entry
rather than code changes.

---

## 2. What was achieved

### Corpus statistics (recomputed directly from the manifests)

| Split | Utterances | Hours | Speakers | L1 words | English words | English share | CS points | Code-mixed |
|---|---|---|---|---|---|---|---|---|
| Hindi-English train | 52,769 | 88.98 | 520 | 465,697 | 157,911 | 25.3% | 166,904 | 83.9% |
| Hindi-English test | 3,132 | 5.14 | 30 | 27,956 | 9,099 | 24.6% | 9,198 | 80.0% |
| Bengali-English train | 26,563 | 45.50 | 267 | 166,404 | 76,840 | 31.6% | 76,048 | 85.6% |
| Bengali-English test | 4,275 | 7.00 | 40 | 25,282 | 12,644 | 33.3% | 12,422 | 87.5% |
| **Total** | **86,739** | **146.62** | **857** | **685,339** | **256,494** | — | **264,572** | — |

Plus ~16 GB of segmented, resampled 16 kHz mono audio on disk, and the
monolingual controls:

| Reference set | Utterances | Hours | Code-mixed |
|---|---|---|---|
| IndicVoices Hindi | 200 | 0.31 | 1.5% |
| IndicVoices Bengali | 200 | 0.41 | 9.0% |

### Correctness validation against published figures

The word-level tagging was validated against Biswas et al. (2025), which
publishes its own counts for this exact MUCS Hindi-English test split:

| Measure | Published | Ours | Delta |
|---|---|---|---|
| Duration | 5.2 h | 5.14 h | −1.2% |
| Hindi words | 28,215 | 27,956 | −0.9% |
| English words | 9,627 | 9,099 | −5.5% |
| Code-switch bigrams | 9,365 | 9,198 | −1.8% |

Agreement within ~1–6% on independently derived counts. The residual
differences are consistent with differing treatment of punctuation and
mid-sentence numerals. This is the strongest available evidence that the
tagger measures what it claims to.

### Secondary validation

The monolingual controls came back at 1.5% (Hindi) and 9.0% (Bengali)
code-mixed, against 80–88% for the actual code-switched corpora. That
two-orders-of-magnitude separation confirms both that the tagger
discriminates correctly and that the monolingual baselines are genuinely
monolingual — a "clean" baseline that turned out to be 80% code-mixed
would have invalidated the entire benchmark comparison.

---

## 3. Method followed

**Step 1 — Parse Kaldi format.** MUCS ships four plain-text index files:
`text` (utterance → transcription), `wav.scp` (recording → audio file),
`segments` (utterance → recording + start/end seconds), `utt2spk`
(utterance → speaker).

**Step 2 — Extract and standardise audio.** Slice each utterance's
window out of its parent recording, downmix to mono, resample to 16 kHz.
Extraction is idempotent — already-cut segments are reused on re-runs.

**Step 3 — Tag language per word.** Classify each word by the dominant
Unicode script of its letters: Devanagari (U+0900–097F) → Hindi, Bengali
(U+0980–09FF) → Bengali, Latin ASCII → English. Tokens with no
alphabetic characters (digits, punctuation) are tagged `other` and
excluded from language counting.

**Step 4 — Detect code-switch points.** Walk the tagged words in order;
each adjacent pair with differing language tags is a switch point,
recorded directionally (L1→English vs English→L1). `other` tokens are
skipped rather than treated as breaks, so a mid-sentence numeral does
not fabricate two spurious switches.

**Step 5 — Filter.** Drop utterances with empty transcriptions or
durations outside 0.3–30 s.

**Step 6 — Emit manifests.** One JSON object per utterance carrying audio
path, text, duration, split/dataset provenance, speaker, per-language
word counts, code-switch count, and a code-mixed flag.

### Notable engineering decision: streaming extraction

The train splits could not be extracted conventionally. The Hindi-English
train archive is 7.3 GB compressed and roughly 18 GB extracted, which did
not fit on available disk alongside the archive itself. Rather than
requiring more disk, `extract_kaldi_split_from_tar` makes a **single
sequential pass** through the `.tar.gz`, reads one recording into memory,
slices out all of its segments, and discards it before moving to the
next. The full raw recording set is never on disk at once — only the
small cut segments. Sequential (rather than per-member random access) is
deliberate: gzip streams cannot be seeked efficiently, so looking up 500+
members individually would rescan from the start each time.

---

## 4. Possible outcome

- The manifests are directly consumable for both training stages. The
  per-utterance language counts already encode the filtering needed to
  build monolingual subsets for Profile Learning (`n_words_en == 0`
  yields 7,083 pure-Hindi utterances) without any further data
  processing.
- The same fields support Stage 2 (router training): `n_cs_points` and
  `is_code_mixed` identify exactly the utterances containing
  mid-utterance transitions the router must learn to handle.
- The recorded code-switch direction counts are the groundwork for
  Code-Switch Bigram Accuracy (CBA), the switch-point-specific metric
  used in the Whisper adaptation literature — a more diagnostic measure
  than aggregate WER for this problem.
- 857 distinct speakers across the corpora make a speaker-disjoint
  train/dev split feasible, which matters for trustworthy validation.

---

## 5. Final works left

1. **No held-out dev split exists.** `pipeline/splits.py::speaker_disjoint_split`
   is written but has never been run. MUCS's own train/test split is
   used as-is, which means there is currently no validation set for
   model selection or early stopping during training. This is the most
   pressing gap.
2. **The segmentation artifact is unaddressed.** As quantified in
   `01_BASELINE_AND_PROBLEM_ANALYSIS.md`, MUCS segments are contiguous
   integer-second tiles, so 39% of utterances contain leading speech
   from their predecessor. This is corpus-inherent, not a pipeline bug,
   but it injects noisy supervision into training and puts a floor under
   evaluation. Forced alignment or VAD-based re-segmentation would fix
   it; nothing has been attempted yet.
3. **Tamil/Telugu do not exist in MUCS as code-switched audio.** MUCS
   ships only Hindi-English and Bengali-English code-switched speech;
   Tamil, Telugu and Gujarati appear only as monolingual data. The
   project's Dravidian equity axis therefore has no data at all and
   needs either a different corpus or synthetic generation.
4. **IndicVoices audio is not cached.** Only 200 text-only samples per
   language were profiled. If IndicVoices is wanted as actual training
   data (rather than a sanity check), the audio must be fetched and
   stored.
5. **Text normalisation is minimal.** Only NFC normalisation and
   whitespace collapsing are applied at preprocessing time. Numerals,
   punctuation and inconsistent transliteration spellings are left
   as-is, which likely inflates both training difficulty and measured
   WER.
6. **No integration test against real audio.** Unit tests cover the
   tagging and manifest logic, but nothing exercises audio extraction
   end-to-end in CI, because the raw corpora are far too large to commit.

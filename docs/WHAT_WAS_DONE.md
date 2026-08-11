# Preprocessing — what was actually done

Plain record of what was run, on what data, and what came out. For how the
code works, see [PIPELINE.md](PIPELINE.md).

## In simple terms

All four MUCS code-switched splits (Hindi-English and Bengali-English,
train and test) have been downloaded, cut into per-sentence audio, tagged
word-by-word for language, and written into manifests. Both monolingual
IndicVoices baselines (Hindi and Bengali) were pulled and checked too.
Everything below was actually run and its output verified — not just
written and assumed to work.

## Steps followed, in order

1. **Downloaded the raw MUCS packages** from OpenSLR 104 — Hindi-English
   train (7.3GB) and test (already local), Bengali-English train (3.9GB)
   and test (0.6GB).
2. **Parsed the Kaldi files** — `text`, `wav.scp`, `segments`, `utt2spk` —
   for each split.
3. **Cut and resampled the audio** — sliced each sentence out of its parent
   recording, converted to 16kHz mono, saved as its own WAV file. For the
   two train splits, the extracted audio wouldn't fit on disk next to the
   downloaded archive, so recordings were streamed and sliced straight out
   of the `.tar.gz` one at a time instead of extracting everything first
   (see PIPELINE.md's "known gotcha" section for why).
4. **Tagged every word's language** by Unicode script (Devanagari → Hindi,
   Bengali script → Bengali, Latin → English) and found every code-switch
   point between them.
5. **Filtered out unusable clips** — empty transcriptions or audio outside
   the 0.3–30 second range.
6. **Wrote one manifest per split** — text, audio path, duration, and all
   the word/language-switch counts from step 4.
7. **Streamed 200 samples each of clean Hindi and Bengali from
   IndicVoices** (no MUCS involvement) through the same word-tagging step,
   as a check that the "monolingual baseline" datasets are actually clean.

## What came out

### MUCS code-switched (all four splits, real recordings)

| | Hindi-English test | Hindi-English train | Bengali-English test | Bengali-English train |
|---|---|---|---|---|
| Sentences kept | 3,132 | 52,769 | 4,275 | 26,563 |
| Audio duration | 5.14h | 88.98h | 7.00h | 45.50h |
| L1 words | 27,956 | 465,697 | 25,282 | 166,404 |
| English words | 9,099 | 157,911 | 12,644 | 76,840 |
| Code-switch points | 9,198 | 166,904 | 12,422 | 76,048 |
| % code-mixed | 80.0% | 83.9% | 87.5% | 85.6% |

Total: **86,739 sentences, ~146.6 hours**, all four splits landing in the
same 80–88% code-mixed range — consistent with each other, which is a
reasonable sanity check on its own (train and test of the same language
pair should look similar; they do).

### IndicVoices monolingual baseline (sanity check, 200 samples each)

| | Hindi | Bengali |
|---|---|---|
| Duration | 0.31h | 0.41h |
| % code-mixed | 1.5% | 9.0% |

Both come back overwhelmingly monolingual, as a "clean baseline" should.
Bengali is somewhat less clean than Hindi (9% vs 1.5% code-mixed) — still
a legitimate baseline, just a real difference worth knowing about rather
than a bug: some natural code-mixing shows up even in "monolingual"
spontaneous speech, more so in Bengali than Hindi in this sample.

## How we know the tagging is correct

The Whisper adaptation paper (Biswas et al. 2025) published its own word
and code-switch counts for this exact same MUCS Hindi-English test split.
Ours:

| | Their Table 1 | Ours |
|---|---|---|
| Duration | 5.2h | 5.14h |
| Hindi words | 28,215 | 27,956 |
| English words | 9,627 | 9,099 |
| Code-switch bigrams | 9,365 | 9,198 |

All four numbers land within ~2–6% of theirs on the same data — close
enough to trust the word-tagging logic is measuring what it claims to
measure, not just producing plausible-looking output. (Small gaps are
expected: minor differences in how the two of us treat punctuation and
mid-sentence numerals.) No published train-split numbers exist to compare
against, but the train/test consistency above (83.9% vs 80.0% code-mixed
for Hindi-English) is the same kind of check.

## What was NOT run

- Any augmentation or synthetic data generation — out of scope for this
  phase; that's Track 2 of the Whisper paper, for later.
- A held-out **dev** split carved out of the train data — the
  `speaker_disjoint_split` helper exists (see PIPELINE.md) but hasn't been
  exercised yet; MUCS's own train/test split was used as-is.
- Nothing for Tamil/Telugu — MUCS has no code-switched audio for those
  languages at all (monolingual only), so there was nothing to run.

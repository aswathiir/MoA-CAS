# Preprocessing — what was actually done

Plain record of what was run, on what data, and what came out. For how the
code works, see [PIPELINE.md](PIPELINE.md).

## In simple terms

We took the raw MUCS Hindi-English recordings (the same ones the baseline
benchmark used) and ran them through the new pipeline instead of the old
one-off script. Separately, we pulled a small sample of clean monolingual
Hindi speech to confirm the "clean baseline" really is clean. Both are
proven to work end-to-end, not just written.

## Steps followed, in order

1. **Parsed the raw MUCS Kaldi files** — `text`, `wav.scp`, `segments`,
   `utt2spk` — for the Hindi-English test split (3,136 sentences).
2. **Cut and resampled the audio** — sliced each sentence out of its parent
   recording, converted to 16kHz mono, saved as its own WAV file.
3. **Tagged every word's language** by Unicode script (Devanagari →
   Hindi, Latin → English) and found every code-switch point between them.
4. **Filtered out unusable clips** — 4 sentences dropped (empty
   transcription or audio outside the 0.3–30 second range).
5. **Wrote the manifest** — one line per sentence, with text, audio path,
   duration, and all the word/language-switch counts from step 3.
6. **Streamed 200 clean Hindi samples from IndicVoices** (no MUCS
   involvement) through the same word-tagging step, as a check that the
   "monolingual baseline" dataset is actually monolingual.

## What came out

| | MUCS Hindi-English (test) | IndicVoices Hindi (sample) |
|---|---|---|
| Source | Real recordings, OpenSLR 104 | HuggingFace, streamed |
| Sentences kept | 3,132 (of 3,136) | 200 |
| Audio duration | 5.14 hours | 0.31 hours |
| Hindi words | 27,956 | 3,009 |
| English words | 9,099 | 5 |
| Code-switch points | 9,198 | 5 |
| % of sentences code-mixed | 80.0% | 1.5% |

The right-hand column is the point of running IndicVoices at all: a
"monolingual baseline" dataset that turned out to be 80% code-mixed would
mean the whole benchmark comparison was invalid. 1.5% confirms it's clean.

## How we know the tagging is correct

The Whisper adaptation paper (Biswas et al. 2025) published its own word
and code-switch counts for this exact same MUCS test split. Ours:

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
mid-sentence numerals.)

## What was NOT run

- MUCS Bengali-English — no audio downloaded locally yet.
- IndicVoices Bengali — same tagging code, just not exercised.
- The full 90-hour MUCS Hindi-English **train** split — only the 5-hour
  test split is on disk.
- Any augmentation or synthetic data generation — out of scope for this
  phase.

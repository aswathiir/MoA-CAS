# Data pipeline 

No research framing here — just what the code in `src/moa_cas/` and
`scripts/build_manifest.py` actually does, in order, and where to find each
piece if you need to change it.

## What it does, in one paragraph

You point it at a folder of raw recordings + transcripts. It cuts each
individual sentence's audio out of the big recording file, converts it to a
standard format (16kHz, mono), reads the matching line of text, marks every
*word* as Hindi/Bengali/English based on what alphabet it's written in,
counts how often the speaker switches language mid-sentence, throws out
clips that are broken/empty/too short/too long, and writes one clean line
per sentence into a single output file (the "manifest") that the benchmark
script reads.

## Step by step

**1. Raw data layout (Kaldi format)**
MUCS ships its data in a format called Kaldi. Four plain-text files describe
everything:
- `text` — sentence ID → transcription
- `wav.scp` — recording ID → path to the big WAV file it lives in
- `segments` — sentence ID → which recording + start/end time in seconds
- `utt2spk` — sentence ID → speaker ID (used later for train/dev splitting)

**2. Cut and standardize the audio** — [`preprocessing/audio.py`](../src/moa_cas/preprocessing/audio.py)
Reads those four files, and for each sentence: slices the [start, end]
window out of its parent recording, downmixes to mono, resamples to 16kHz
(the format the model expects), and saves it as its own small WAV file.
This is offline, one-time work — it caches the cut files so re-running
doesn't re-slice audio that's already there.

For a large split (e.g. the 90-hour Hindi-English train set), extracting
every raw recording to disk first can need more space than you have free
alongside the downloaded archive. `extract_kaldi_split_from_tar` handles
this case: it makes one sequential pass through the `.tar.gz`, reads each
recording into memory, slices out all of that recording's sentences, and
discards it before moving to the next — the full raw recording set is
never sitting on disk at once, only the small cut-up sentences are.

**3. Tag every word's language** — [`preprocessing/text.py`](../src/moa_cas/preprocessing/text.py)
For each transcription, splits it into words and looks at the Unicode
range each character falls in: Devanagari letters → Hindi, Bengali letters
→ Bengali, Latin letters → English. Whatever script has the most letters in
a word decides that word's tag. Then it walks through the tagged words in
order — every time two neighboring words have different tags, that's a
code-switch point (e.g. Hindi→English or English→Hindi). This is how it
knows a sentence is code-mixed at all, and where.

**4. Filter out unusable clips** — [`pipeline/build.py`](../src/moa_cas/pipeline/build.py)
Drops anything with no transcription text, or audio shorter than 0.3s /
longer than 30s (too short to be real speech, or too long for a single
model input).

**5. Write the manifest** — [`pipeline/manifest.py`](../src/moa_cas/pipeline/manifest.py)
Everything from steps 2–4 gets packed into one row per sentence — audio
file path, text, duration, which dataset/split it's from, the word/language
counts, how many code-switch points it has — and written as one JSON
object per line to `data/manifests/*.jsonl`. This file is the actual
deliverable; everything upstream is just how it gets built.

**6. Print a stats summary** — [`stats.py`](../src/moa_cas/stats.py)
After building a manifest, prints total hours, word counts per language,
and code-switch counts for the whole corpus — a sanity check you can
eyeball against known numbers (this is how we confirmed the pipeline
reproduces the source paper's published stats within ~2%).

## Folder map

| Path | What's in it |
|---|---|
| `configs/datasets.yaml` | List of "targets" you can build (e.g. `mucs_hi_en_test`) — which adapter to use, where the raw data is, where the output goes |
| `data/raw/` | Raw downloaded corpora (gitignored — too big for git) |
| `data/processed/` | Cut/resampled WAV files, cached (gitignored) |
| `data/manifests/` | The actual pipeline output — small JSONL files, tracked in git |
| `src/moa_cas/datasets/` | Per-corpus readers (MUCS, IndicVoices) |
| `src/moa_cas/preprocessing/` | The audio-cutting and word-tagging logic described above |
| `src/moa_cas/pipeline/` | Glue: runs a dataset through preprocessing and writes the manifest |
| `scripts/build_manifest.py` | The command you actually run |
| `scripts/benchmark.py` | Unrelated to preprocessing — runs the model against a manifest and reports WER/CER |
| `tests/` | Unit tests for the tagging logic and manifest format (no audio needed) |

## How to run it

```bash
# see what targets exist
python scripts/build_manifest.py --list

# build one
python scripts/build_manifest.py --config mucs_hi_en_test
```

Add a new corpus split by adding an entry to `configs/datasets.yaml` — no
code changes needed unless it's a genuinely new *kind* of source (not
Kaldi, not HuggingFace-streamed).

### Known gotcha: force-exit on the HuggingFace-streamed path

Running an `indicvoices_*` target streams a parquet file straight from
HuggingFace. Once, that left a background networking thread alive after the
script's own work was already done and printed — the process just sat
there instead of exiting, for over an hour, with no further output. The
script's own logic wasn't hung; a thread it doesn't control was. Rather
than chase that thread down, `scripts/build_manifest.py` now force-exits
(`os._exit(0)`) right after writing its output, once its own work is
provably done. You may see a harmless `resource_tracker: leaked semaphore`
warning on exit — that's the price of skipping normal cleanup, not a sign
anything is wrong.

## What's not built yet

All four MUCS code-switched splits (Hindi-English and Bengali-English,
train and test) and both IndicVoices monolingual baselines have been run
end-to-end — see [WHAT_WAS_DONE.md](WHAT_WAS_DONE.md) for the actual
numbers. What's still open:

- **A held-out dev split** — `pipeline/splits.py::speaker_disjoint_split`
  exists but hasn't been used yet; MUCS's own train/test split was used
  as-is. Carving a dev set out of train (for model-selection during
  training) is a one-line call away, not a missing feature.
- **No augmentation** (synthetic code-mixing, LLM/TTS-generated data) —
  intentionally out of scope for this pipeline; see the project doc's
  Track 2 for when that becomes relevant.
- **No Tamil/Telugu** — MUCS simply has no code-switched audio for those
  languages (monolingual only), so there's nothing this pipeline can build
  for that axis without a different corpus or synthetic data.

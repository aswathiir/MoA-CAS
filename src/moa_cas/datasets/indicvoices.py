"""
moa_cas/datasets/indicvoices.py
------------------------------------
Loads IndicVoices from HuggingFace — the monolingual baseline used for
degradation monitoring. Shah et al. 2020 ("Learning to Recognize
Code-switched Speech Without Forgetting Monolingual Speech Recognition")
is the reason this corpus exists in the pipeline at all: fine-tuning on
code-switched data alone quietly wrecks monolingual accuracy, so a paired
clean-speech set has to be tracked alongside every code-mixed one.

Schema (from ai4bharat/IndicVoices README):
  audio_filepath  : Audio feature
  text            : verbatim transcription
  lang            : language code string
  speaker_id, gender, age_group, state, scenario, task_name, ...
"""

from datasets import load_dataset, Audio

HF_REPO = "ai4bharat/IndicVoices"

# IndicVoices config names are full language names, not ISO codes.
_SUPPORTED = {"hindi", "bengali", "tamil", "telugu"}


def load(lang: str = "hindi", split: str = "valid", limit: int = None):
    """
    Streams IndicVoices [lang] — fetches samples on demand, never downloads
    the full dataset (~40 GB for Hindi alone). Always use --limit for
    benchmarking or profiling. Returns a HF IterableDataset with
    Audio(decode=False) applied.
    """
    if lang not in _SUPPORTED:
        raise ValueError(f"Unsupported IndicVoices language '{lang}'. Supported: {sorted(_SUPPORTED)}")

    print(f"Loading IndicVoices [{lang}] from {HF_REPO} (streaming)...")
    dataset = load_dataset(HF_REPO, lang, split=split, streaming=True)
    dataset = dataset.cast_column("audio_filepath", Audio(decode=False))

    if limit:
        dataset = dataset.take(limit)
        print(f"  Streaming first {limit} samples.\n")
    else:
        print("  No limit set — streaming the entire split.\n")

    return dataset


def to_sample(row: dict) -> dict:
    """Normalise a HF row into the unified runtime sample format."""
    return {
        "audio_type": "hf_bytes",
        "audio_data": row["audio_filepath"],   # {"bytes": ..., "path": ...}
        "text":       row["text"],
        "meta": {
            "scenario":   row.get("scenario",   "unknown"),
            "task":       row.get("task_name",  "unknown"),
            "gender":     row.get("gender",     "unknown"),
            "state":      row.get("state",      "unknown"),
            "lang":       row.get("lang",       "hi"),
        },
    }

"""
core/datasets/indicvoices.py
-----------------------------
Loads IndicVoices Hindi (valid split) from HuggingFace.

Used as the monolingual baseline — IndicConformer at its best on
clean, single-language Indian speech.

Schema (from ai4bharat/IndicVoices README):
  audio_filepath  : Audio feature
  text            : verbatim transcription
  lang            : language code string
  speaker_id, gender, age_group, state, scenario, task_name, ...
"""

from datasets import load_dataset, Audio

HF_REPO = "ai4bharat/IndicVoices"
CONFIG  = "hindi"    # config name is the full language name, NOT the ISO code
SPLIT   = "valid"    # IndicVoices has no test split — valid is the eval split


def load(limit: int = None):
    """
    Streams IndicVoices Hindi — fetches samples on demand, never downloads
    the full dataset (~40 GB). Always use --limit for benchmarking.
    Returns a HF IterableDataset with Audio(decode=False) applied.
    """
    print(f"Loading IndicVoices [{CONFIG}] from {HF_REPO} (streaming)...")
    dataset = load_dataset(HF_REPO, CONFIG, split=SPLIT, streaming=True)
    dataset = dataset.cast_column("audio_filepath", Audio(decode=False))

    if limit:
        dataset = dataset.take(limit)
        print(f"  Streaming first {limit} samples.\n")
    else:
        print("  No limit set — streaming entire valid split (3,873 samples).\n")

    return dataset


def to_sample(row: dict) -> dict:
    """Normalise a HF row into the unified benchmark sample format."""
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
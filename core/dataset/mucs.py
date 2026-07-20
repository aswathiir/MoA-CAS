"""
core/datasets/mucs.py
----------------------
Loads MUCS 2021 Hindi-English code-switched test set from a local manifest.

Download instructions
---------------------
1. Go to https://www.openslr.org/104/
2. Download the Hindi-English package
3. Extract — you'll get a folder with:
     - audio/          (WAV files)
     - hindi_test.json (or similar manifest in NeMo JSONL format)
4. Pass the manifest path via --mucs-manifest

Manifest format (one JSON object per line):
  {"audio_filepath": "path/to/audio.wav", "text": "transcription", "duration": 5.2}

Why this dataset
----------------
MUCS is the standard benchmark for Indian code-switched ASR.
Baseline WER on Hindi-English using end-to-end systems: ~30-32%.
Running IndicConformer (monolingual Hindi model) on this shows the
code-switch gap that MoA-CAS is designed to close.
"""

import json
from pathlib import Path


def load(manifest_path: str, limit: int = None) -> list:
    """
    Reads a NeMo-style JSONL manifest and returns a list of sample dicts.
    Raises FileNotFoundError with download instructions if path is missing.
    """
    path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nMUCS manifest not found: {manifest_path}\n\n"
            f"Download steps:\n"
            f"  1. Visit https://www.openslr.org/104/\n"
            f"  2. Download Hindi-English package and extract\n"
            f"  3. Re-run with: --mucs-manifest path/to/hindi_test.json\n"
        )

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if limit:
        records = records[:limit]

    print(f"  {len(records)} samples loaded from {path.name}.\n")
    return records


def to_sample(row: dict) -> dict:
    """Normalise a MUCS manifest row into the unified benchmark sample format."""
    return {
        "audio_type": "local_path",
        "audio_data": row["audio_filepath"],   # string path to WAV file
        "text":       row["text"],
        "meta": {
            "duration": row.get("duration", 0),
            "lang":     "hi-en",               # code-switched
        },
    }
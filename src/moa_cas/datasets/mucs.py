"""
moa_cas/datasets/mucs.py
----------------------------
Loads a MUCS code-switched manifest (built by scripts/build_manifest.py) for
benchmark evaluation.

Download instructions
----------------------
1. Go to https://www.openslr.org/104/
2. Download the Hindi-English or Bengali-English package
3. Extract into data/raw/mucs/<lang_pair>/<split>/ — Kaldi layout:
     transcripts/{text,wav.scp,segments,utt2spk}
     *.wav
4. Build the manifest:
     python scripts/build_manifest.py --config mucs_hi_en_test
5. Pass the resulting manifest path via --mucs-manifest, or read it directly.

Manifest format (one JSON object per line — see pipeline/manifest.py::ManifestRow):
  {"audio_filepath": "...", "text": "...", "duration": 5.2, "lang_pair": "hi-en", ...}

Why this dataset
-----------------
MUCS is the standard benchmark for Indian code-switched ASR. Running
IndicConformer (a monolingual model) on it exposes the code-switch gap
MoA-CAS is designed to close. MUCS 2021's code-switching subtask ships
Hindi-English and Bengali-English audio — Tamil/Telugu appear in MUCS only
as monolingual data, not code-switched.
"""

import json
from pathlib import Path


def load(manifest_path: str, limit: int = None) -> list:
    """Reads a ManifestRow-schema JSONL manifest and returns a list of dict rows."""
    path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nMUCS manifest not found: {manifest_path}\n\n"
            f"Build one with:\n"
            f"  python scripts/build_manifest.py --config mucs_hi_en_test\n"
            f"(raw data must be downloaded first — see https://www.openslr.org/104/)\n"
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
    """Normalise a manifest row into the unified runtime sample format."""
    return {
        "audio_type": "local_path",
        "audio_data": row["audio_filepath"],
        "text":       row["text"],
        "meta": {
            "duration": row.get("duration", 0),
            "lang":     row.get("lang_pair", "hi-en"),
        },
    }

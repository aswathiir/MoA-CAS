"""
moa_cas/pipeline/manifest.py
-------------------------------
The unified manifest schema written by pipeline/build.py and consumed by
scripts/benchmark.py (via datasets/mucs.py, datasets/indicvoices.py).

One ManifestRow per utterance carries everything downstream code needs:
audio location, reference text, split/dataset provenance, and the
language-tagging stats from preprocessing/text.py (word counts per
language, code-switch point count) — the same statistics Biswas et al.
2025 report per-corpus in their Table 1, computed here per utterance.
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class ManifestRow:
    utt_id: str
    audio_filepath: str
    text: str
    duration: float
    dataset: str            # "mucs" | "indicvoices"
    lang_pair: str           # "hi-en" | "bn-en" (code-mixed) or "hi" | "bn" (monolingual)
    split: str                # "train" | "dev" | "test" | "valid"
    n_words: int
    n_words_l1: int
    n_words_en: int
    n_cs_points: int
    is_code_mixed: bool
    speaker_id: str = ""
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def write_jsonl(rows: list, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            line = row.to_json() if isinstance(row, ManifestRow) else json.dumps(row, ensure_ascii=False)
            f.write(line + "\n")
    print(f"Manifest written → {path} ({len(rows)} rows)")


def read_jsonl(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

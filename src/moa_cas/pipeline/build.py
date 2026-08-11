"""
moa_cas/pipeline/build.py
-----------------------------
Orchestrates the preprocessing pipeline: raw corpus -> normalized,
language-tagged, filtered manifest.

    raw corpus (Kaldi dir / HF streaming)
      -> audio extraction (preprocessing/audio.py)   [MUCS only; IndicVoices audio stays HF-streamed]
      -> text normalization + word-level language tagging (preprocessing/text.py)
      -> duration / empty-text filtering
      -> ManifestRow (pipeline/manifest.py)

Called by scripts/build_manifest.py, driven by configs/datasets.yaml.
"""

from pathlib import Path

from moa_cas.preprocessing.audio import extract_kaldi_split, extract_kaldi_split_from_tar
from moa_cas.preprocessing.text import tag_utterance
from moa_cas.pipeline.manifest import ManifestRow
from moa_cas.datasets import indicvoices as indicvoices_adapter

# Segments outside this range can't be used as single-shot Conformer/Whisper
# inputs — drop them rather than let them fail silently downstream.
MIN_DURATION_S = 0.3
MAX_DURATION_S = 30.0


def _utterances_to_rows(utterances: list, lang_pair: str, split: str) -> list:
    """Shared tag + filter + ManifestRow step for both MUCS extraction paths."""
    l1 = lang_pair.split("-")[0]
    rows = []
    dropped = 0
    for utt in utterances:
        text = utt.text.strip()
        if not text or not (MIN_DURATION_S <= utt.duration <= MAX_DURATION_S):
            dropped += 1
            continue

        tags = tag_utterance(text, l1=l1)
        rows.append(ManifestRow(
            utt_id=utt.utt_id,
            audio_filepath=utt.audio_filepath,
            text=text,
            duration=utt.duration,
            dataset="mucs",
            lang_pair=lang_pair,
            split=split,
            n_words=tags.n_words,
            n_words_l1=tags.n_words_l1,
            n_words_en=tags.n_words_en,
            n_cs_points=len(tags.cs_bigrams),
            is_code_mixed=tags.is_code_mixed,
            speaker_id=utt.speaker_id or "",
        ))

    if dropped:
        print(f"  Filtered  : {dropped} utterances (empty text or duration outside "
              f"[{MIN_DURATION_S}, {MAX_DURATION_S}]s)")
    return rows


def build_mucs_manifest(raw_dir: Path, out_wav_dir: Path, lang_pair: str, split: str) -> list:
    """
    Builds a ManifestRow list from a raw MUCS Kaldi split already fully
    extracted to disk. lang_pair: "hi-en" or "bn-en".
    """
    utterances = extract_kaldi_split(Path(raw_dir), Path(out_wav_dir))
    return _utterances_to_rows(utterances, lang_pair, split)


def build_mucs_manifest_from_tar(tar_path: Path, member_root: str, raw_dir: Path,
                                  out_wav_dir: Path, lang_pair: str, split: str) -> list:
    """
    Builds a ManifestRow list from a raw MUCS Kaldi split too large to
    extract to disk in full (e.g. the 90h Hindi-English train split) —
    streams audio straight out of the downloaded .tar.gz. See
    preprocessing/audio.py::extract_kaldi_split_from_tar for why.
    """
    utterances = extract_kaldi_split_from_tar(Path(tar_path), member_root, Path(raw_dir), Path(out_wav_dir))
    return _utterances_to_rows(utterances, lang_pair, split)


def build_indicvoices_manifest(lang: str, split: str, limit: int = None) -> list:
    """
    Builds a ManifestRow list from streamed IndicVoices — the monolingual
    baseline used for degradation monitoring (Shah et al. 2020). Audio stays
    HF-streamed (never cached to disk); this manifest profiles the corpus'
    language-tagging stats (it should come back ~100% l1, ~0% code-mixed —
    a sanity check that the "clean" baseline really is clean).
    """
    lang_code = {"hindi": "hi", "bengali": "bn", "tamil": "ta", "telugu": "te"}.get(lang, lang[:2])
    dataset = indicvoices_adapter.load(lang=lang, split=split, limit=limit)

    rows = []
    for row in dataset:
        text = (row.get("text") or "").strip()
        if not text:
            continue

        tags = tag_utterance(text, l1=lang_code) if lang_code in ("hi", "bn") else None

        rows.append(ManifestRow(
            utt_id=f"iv_{lang}_{split}_{len(rows)}",
            audio_filepath="",   # HF-streamed at eval time — see datasets/indicvoices.py
            text=text,
            duration=float(row.get("duration") or 0.0),
            dataset="indicvoices",
            lang_pair=lang_code,
            split=split,
            n_words=tags.n_words if tags else 0,
            n_words_l1=tags.n_words_l1 if tags else 0,
            n_words_en=tags.n_words_en if tags else 0,
            n_cs_points=len(tags.cs_bigrams) if tags else 0,
            is_code_mixed=tags.is_code_mixed if tags else False,
            meta={"scenario": row.get("scenario", "unknown"), "note": "audio HF-streamed, not cached"},
        ))
    return rows

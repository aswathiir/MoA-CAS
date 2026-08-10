"""
moa_cas/pipeline/splits.py
-----------------------------
Speaker-disjoint splitting for corpora that don't ship a pre-defined dev split.

Not exercised on the MUCS test-only data currently on disk (MUCS already
defines train/test) — this exists for when the full train split is
downloaded and a held-out dev set needs to be carved out of it without
speaker leakage between splits.
"""

import random


def speaker_disjoint_split(utterances: list, val_fraction: float = 0.1, seed: int = 42) -> tuple:
    """
    Splits a list of objects exposing `.speaker_id` and `.utt_id` (e.g.
    preprocessing.audio.RawUtterance) into (train, val) such that no speaker
    appears in both halves.

    Utterances with no speaker_id fall back to per-utterance splitting
    (treated as their own singleton speaker).
    """
    by_speaker = {}
    for utt in utterances:
        key = utt.speaker_id or utt.utt_id
        by_speaker.setdefault(key, []).append(utt)

    speakers = list(by_speaker.keys())
    random.Random(seed).shuffle(speakers)

    n_val_speakers = max(1, int(len(speakers) * val_fraction))
    val_speakers = set(speakers[:n_val_speakers])

    train, val = [], []
    for speaker, utts in by_speaker.items():
        (val if speaker in val_speakers else train).extend(utts)

    return train, val

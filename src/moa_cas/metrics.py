"""
moa_cas/metrics.py
-------------------
Text normalization + WER / CER computation for benchmark evaluation.
"""

import re
import unicodedata
from jiwer import wer, cer


def normalize(text: str) -> str:
    """
    Lowercase, NFC-normalize, strip punctuation and symbols, collapse whitespace.

    Removes only Unicode punctuation (P*) and symbol (S*) characters. An
    earlier version used re.sub(r"[^\\w\\s]", ""), but Python's \\w does not
    match combining marks, so it silently deleted every Devanagari/Bengali
    vowel sign, anusvara and virama ("इंसर्ट" -> "इसरट", "कि" == "की").
    Every result before 2026-10-26 was scored that way; see
    docs/THESIS_REFERENCE.md, correction C6.
    """
    text = unicodedata.normalize("NFC", text.lower().strip())
    text = "".join(c for c in text if not unicodedata.category(c).startswith(("P", "S")))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_wer(reference: str, hypothesis: str) -> float:
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    return wer(ref, hyp) if ref else 0.0


def compute_cer(reference: str, hypothesis: str) -> float:
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    return cer(ref, hyp) if ref else 0.0

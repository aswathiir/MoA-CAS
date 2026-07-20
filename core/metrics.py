"""
core/metrics.py
---------------
Text normalization + WER / CER computation.
"""

import re
import unicodedata
from jiwer import wer, cer


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, NFC-normalize."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s]", "", text)
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
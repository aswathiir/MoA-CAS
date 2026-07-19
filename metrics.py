"""
metrics.py
----------
Text normalization + WER / CER computation.
Keep this separate so benchmark.py stays focused on the pipeline.
"""

import re
import unicodedata
from jiwer import wer, cer


def normalize(text: str) -> str:
    """
    Lowercase, strip punctuation, collapse whitespace, NFC-normalize unicode.
    Applied to both reference and hypothesis before scoring so minor
    formatting differences do not penalize WER/CER.
    """
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)       # collapse spaces
    return text.strip()


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate. Returns 1.0 (100%) if reference is empty."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0
    return wer(ref, hyp)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate. Returns 1.0 (100%) if reference is empty."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0
    return cer(ref, hyp)
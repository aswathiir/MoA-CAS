"""
moa_cas/preprocessing/text.py
--------------------------------
Text normalization and word-level, script-based language tagging.

Reproduces the preprocessing primitive behind Biswas et al. 2025's Table 1
("Adapting Whisper for low-resource Hindi-English Code-Mix speech..."):
Hindi/English word counts and Hindi<->English code-switch bigram counts on
MUCS. Every word is tagged by the Unicode script it's written in; two
consecutive language-bearing words with different tags mark a code-switch
point.

This is a preprocessing-time normalization, distinct from moa_cas.metrics.normalize
which runs at WER/CER evaluation time — keeping them separate means changes
here never shift benchmark numbers.
"""

import re
import unicodedata
from dataclasses import dataclass, field

# Unicode block ranges for the scripts MUCS code-switching covers.
_SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),   # Devanagari (Hindi)
    "bn": (0x0980, 0x09FF),   # Bengali
}

_WORD_RE = re.compile(r"\S+", re.UNICODE)


def normalize_text(text: str) -> str:
    """NFC-normalize and collapse whitespace. Preserves script and case —
    tagging needs the original characters, unlike eval-time normalize()."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tag_word(word: str) -> str:
    """
    Classifies a single word by the dominant Unicode script of its letters.
      "hi"    -> majority Devanagari characters
      "bn"    -> majority Bengali characters
      "en"    -> majority Latin/ASCII letters (English, per MUCS/Whisper-paper convention)
      "other" -> no alphabetic characters (pure digits/punctuation), or no letters at all
    """
    counts = {"hi": 0, "bn": 0, "en": 0}
    for ch in word:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if _SCRIPT_RANGES["hi"][0] <= cp <= _SCRIPT_RANGES["hi"][1]:
            counts["hi"] += 1
        elif _SCRIPT_RANGES["bn"][0] <= cp <= _SCRIPT_RANGES["bn"][1]:
            counts["bn"] += 1
        elif ch.isascii():
            counts["en"] += 1

    if not any(counts.values()):
        return "other"
    return max(counts, key=counts.get)


@dataclass
class UtteranceTags:
    words: list = field(default_factory=list)          # [(word, tag), ...]
    n_words: int = 0
    n_words_l1: int = 0        # native-language (hi/bn) word count
    n_words_en: int = 0
    cs_bigrams: list = field(default_factory=list)      # [(from_tag, to_tag), ...]
    n_switch_l1_en: int = 0    # e.g. HE (Hindi -> English)
    n_switch_en_l1: int = 0    # e.g. EH (English -> Hindi)
    is_code_mixed: bool = False


def tag_utterance(text: str, l1: str = "hi") -> UtteranceTags:
    """
    Tags every word in `text` and detects code-switch bigrams between `l1`
    (the native-language script, "hi" or "bn") and "en".

    "other"-tagged words (digits, punctuation-only tokens) don't count as a
    language and don't break or start a switch — they're skipped when
    looking for the previous language-bearing word.
    """
    if l1 not in _SCRIPT_RANGES:
        raise ValueError(f"Unsupported l1 '{l1}'. Supported: {sorted(_SCRIPT_RANGES)}")

    words = _WORD_RE.findall(normalize_text(text))
    tagged = [(w, tag_word(w)) for w in words]

    result = UtteranceTags(words=tagged, n_words=len(tagged))
    result.n_words_l1 = sum(1 for _, t in tagged if t == l1)
    result.n_words_en = sum(1 for _, t in tagged if t == "en")

    prev_tag = None
    for _, tag in tagged:
        if tag not in (l1, "en"):
            continue
        if prev_tag is not None and tag != prev_tag:
            result.cs_bigrams.append((prev_tag, tag))
            if prev_tag == l1 and tag == "en":
                result.n_switch_l1_en += 1
            elif prev_tag == "en" and tag == l1:
                result.n_switch_en_l1 += 1
        prev_tag = tag

    result.is_code_mixed = len(result.cs_bigrams) > 0
    return result

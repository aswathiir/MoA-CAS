"""
scripts/score_transliteration_aware.py
------------------------------------------
Standard WER counts `document` vs `डॉक्यूमेंट` as completely wrong, even
though the model heard the word correctly and simply wrote it in the
other script. On a code-switched corpus decoded by a model with no
English tokens in its output vocabulary at all, that conflates two very
different failures:

  1. the model did not understand the audio
  2. the model understood perfectly and cannot spell it in Latin

This script separates them. Both reference and hypothesis are romanised
into a coarse shared phonetic space, and WER/CER are recomputed there.
The gap between standard and transliteration-aware scores estimates how
much measured error is script mismatch rather than recognition failure.

This is an approximation. Hindi transliteration of English is phonetically
loose (`document` -> `डॉक्यूमेंट` -> `dokyument`), so the normaliser below
is deliberately aggressive: it collapses aspirates, long/short vowels,
retroflex/dental distinctions and word-final schwa. It will forgive some
genuine errors. Treat the result as an upper bound on "recognised
correctly", not an exact count.

Usage
-----
    python scripts/score_transliteration_aware.py results/adapter_eval/nemo_baseline_fixed.csv
"""

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moa_cas.metrics import compute_cer, compute_wer

DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# ITRANS leaves these vowel signs untransliterated; map them first.
PRE_MAP = {"ॉ": "ो", "ॅ": "े", "ऍ": "ए", "ऑ": "ओ"}

# Collapse distinctions Hindi transliteration of English does not preserve.
PHONETIC_RULES = [
    ("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"),
    ("gh", "g"), ("jh", "j"), ("chh", "c"), ("ch", "c"), ("sh", "s"),
    ("shh", "s"), ("ss", "s"), ("c", "k"), ("q", "k"), ("x", "ks"),
    ("z", "j"), ("w", "v"), ("y", "i"), ("aa", "a"), ("ii", "i"),
    ("ee", "i"), ("oo", "u"), ("uu", "u"), ("ai", "e"), ("au", "o"),
    ("m", "n"), ("nn", "n"),
]


def romanise_word(word: str) -> str:
    if DEVANAGARI.search(word):
        for src, dst in PRE_MAP.items():
            word = word.replace(src, dst)
        word = transliterate(word, sanscript.DEVANAGARI, sanscript.ITRANS)
    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)
    for src, dst in PHONETIC_RULES:
        word = word.replace(src, dst)
    word = re.sub(r"(.)\1+", r"\1", word)      # collapse doubled letters
    word = re.sub(r"a$", "", word)             # word-final schwa
    return word


def romanise(text: str) -> str:
    return " ".join(w for w in (romanise_word(t) for t in text.split()) if w)


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def fuzzy_recall(ref_words, hyp_words, tolerance=0.3):
    """Fraction of reference words with a phonetically close match in the
    hypothesis. Each hypothesis word is consumed at most once."""
    if not ref_words:
        return None
    available = list(hyp_words)
    matched = 0
    for rw in ref_words:
        for i, hw in enumerate(available):
            limit = max(1, int(len(rw) * tolerance))
            if _edit_distance(rw, hw) <= limit:
                matched += 1
                available.pop(i)
                break
    return matched / len(ref_words)


def score(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    std_wer, std_cer, tr_wer, tr_cer, recall = [], [], [], [], []
    for r in rows:
        std_wer.append(float(r["wer"]))
        std_cer.append(float(r["cer"]))
        ref_r, hyp_r = romanise(r["reference"]), romanise(r["hypothesis"])
        tr_wer.append(compute_wer(ref_r, hyp_r))
        tr_cer.append(compute_cer(ref_r, hyp_r))
        got = fuzzy_recall(ref_r.split(), hyp_r.split())
        if got is not None:
            recall.append(got)

    print(f"\n{csv_path.name}  (n={len(rows)})")
    print(f"  standard              WER {statistics.mean(std_wer):.4f}   CER {statistics.mean(std_cer):.4f}")
    print(f"  transliteration-aware WER {statistics.mean(tr_wer):.4f}   CER {statistics.mean(tr_cer):.4f}")
    print(f"  phonetic word recall      {statistics.mean(recall):.4f}   "
          f"(share of reference words heard, allowing spelling drift)")
    return statistics.mean(std_wer), statistics.mean(tr_wer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute WER/CER in a shared romanised space")
    parser.add_argument("csv_paths", nargs="+")
    parser.add_argument("--show-pairs", type=int, default=0, help="Print N romanised word pairs for inspection")
    args = parser.parse_args()

    if args.show_pairs:
        print("Normaliser check — English vs its Devanagari transliteration:")
        for en, dev in [("document", "डॉक्यूमेंट"), ("slide", "स्लाइड"), ("format", "फॉर्मेट"),
                        ("presentation", "प्रेजेंटेशन"), ("window", "विंडो"), ("copy", "कॉपी"),
                        ("tutorial", "ट्यूटोरियल"), ("click", "क्लिक")]:
            a, b = romanise_word(en), romanise_word(dev)
            print(f"  {en:<14} -> {a:<12} | {dev:<16} -> {b:<12} {'MATCH' if a == b else ''}")

    for p in args.csv_paths:
        score(Path(p))

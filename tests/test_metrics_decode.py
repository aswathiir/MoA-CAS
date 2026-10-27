"""
Unit tests for the scoring and decoding code every reported WER passes
through: moa_cas.metrics (normalisation, WER, CER), greedy CTC decoding,
the transliteration-aware romaniser, and Stage 2's CTC target rewriting.
"""

import sys
from pathlib import Path

import pytest
import torch

from moa_cas.adapters.decode import greedy_ctc_decode
from moa_cas.metrics import compute_cer, compute_wer, normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_normalize_is_applied_identically_to_both_sides():
    assert normalize("  Hello,   WORLD! ") == "hello world"
    assert compute_wer("नमस्ते दुनिया।", "नमस्ते दुनिया") == 0.0


def test_normalize_keeps_indic_vowel_signs_and_virama():
    # Regression: the old [^\w\s] pattern deleted combining marks (C6).
    assert normalize("इंसर्ट स्लाइड") == "इंसर्ट स्लाइड"
    assert normalize("অবশ্যই") == "অবশ্যই"
    assert normalize("नमस्ते। “ठीक” है?") == "नमस्ते ठीक है"
    assert compute_wer("कि", "की") == 1.0          # different words, must not match
    assert compute_cer("किताब", "कताब") > 0.0     # a dropped vowel sign is an error


def test_wer_counts_substitutions_insertions_deletions():
    assert compute_wer("a b c d", "a x c d") == pytest.approx(0.25)
    assert compute_wer("a b", "a b c d") == pytest.approx(1.0)  # 2 insertions / 2 words
    assert compute_wer("a b c d", "") == pytest.approx(1.0)


def test_wer_can_exceed_one_on_hallucination():
    # Doc 01 counts utterances with WER > 100% as hallucinations.
    assert compute_wer("a", "x y z") > 1.0


def test_cer_is_character_level():
    assert compute_cer("abcd", "abcx") == pytest.approx(0.25)


def test_empty_reference_scores_zero_not_crash():
    assert compute_wer("", "anything") == 0.0


class FakeSP:
    def decode(self, ids):
        return " ".join(f"t{i}" for i in ids)


def one_hot_log_probs(ids, vocab):
    lp = torch.full((1, len(ids), vocab), -10.0)
    for t, i in enumerate(ids):
        lp[0, t, i] = 0.0
    return lp


def test_greedy_ctc_collapses_repeats_and_drops_blank():
    blank = 4
    lp = one_hot_log_probs([1, 1, blank, 1, 2, 2, blank, blank, 3], vocab=5)
    assert greedy_ctc_decode(lp, FakeSP()) == "t1 t1 t2 t3"  # blank separates the repeated 1


def test_greedy_ctc_all_blank_is_empty():
    assert greedy_ctc_decode(one_hot_log_probs([4, 4, 4], vocab=5), FakeSP()) == ""


def test_romaniser_matches_english_and_its_devanagari_transliteration():
    pytest.importorskip("indic_transliteration")
    from score_transliteration_aware import fuzzy_recall, romanise_word

    assert romanise_word("click") == romanise_word("क्लिक")
    assert romanise_word("copy") == romanise_word("कॉपी")
    # Not exact, but within the 30% edit tolerance fuzzy recall allows.
    assert fuzzy_recall([romanise_word("document")], [romanise_word("डॉक्यूमेंट")]) == 1.0
    assert fuzzy_recall([romanise_word("document")], [romanise_word("स्लाइड")]) == 0.0


def test_stage2_targets_drop_only_latin_words():
    from train_router import ctc_target_text

    text = "लिबर ऑफिस impress में एक document बनाना"
    assert ctc_target_text(text, "drop") == "लिबर ऑफिस में एक बनाना"
    assert ctc_target_text(text, "keep") == text


def test_demo_marks_english_written_in_devanagari_as_other_script():
    pytest.importorskip("fastapi")
    pytest.importorskip("indic_transliteration")
    from demo_server import align_words

    tags, deletions = align_words("अब insert और slide पर click करें", "से चुने अब इंसर्ट और स्लाइड पर क्लिक करें")
    assert tags == ["ins", "ins", "ok", "script", "ok", "script", "ok", "script", "ok"]
    assert deletions == 0
    # A different word in the same slot stays a plain error.
    assert align_words("इस pane में", "इस बटन में")[0] == ["ok", "sub", "ok"]

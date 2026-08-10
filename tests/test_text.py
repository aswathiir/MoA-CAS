"""
Unit tests for moa_cas.preprocessing.text — word/script tagging and
code-switch bigram detection. Uses real reference lines from the MUCS
Hindi-English test set (see the project's benchmark results) as fixtures,
so the tests double as documentation of what the tagger is meant to catch.
"""

from moa_cas.preprocessing.text import tag_word, tag_utterance


def test_tag_word_devanagari():
    assert tag_word("धार्मिक") == "hi"


def test_tag_word_bengali():
    assert tag_word("আমি") == "bn"


def test_tag_word_english():
    assert tag_word("document") == "en"


def test_tag_word_digits_only():
    assert tag_word("123") == "other"


def test_tag_word_punctuation_only():
    assert tag_word("...") == "other"


def test_monolingual_hindi_has_no_switches():
    tags = tag_utterance("धार्मिक स्थान है", l1="hi")
    assert tags.n_words == 3
    assert tags.n_words_l1 == 3
    assert tags.n_words_en == 0
    assert tags.cs_bigrams == []
    assert tags.is_code_mixed is False


def test_code_mixed_hindi_english_detects_switch():
    tags = tag_utterance("stdlib h header file निम्न को परिभाषित करता है", l1="hi")
    assert tags.n_words_en == 4          # stdlib, h, header, file
    assert tags.n_words_l1 == 5          # निम्न, को, परिभाषित, करता, है
    assert tags.is_code_mixed is True
    assert ("en", "hi") in tags.cs_bigrams   # switch at file -> निम्न


def test_switch_counted_in_both_directions():
    tags = tag_utterance("यह हमारा main function है", l1="hi")
    # यह(hi) हमारा(hi) main(en) function(en) है(hi)
    assert tags.n_switch_l1_en == 1   # हमारा -> main
    assert tags.n_switch_en_l1 == 1   # function -> है
    assert len(tags.cs_bigrams) == 2


def test_bengali_english_code_mix():
    tags = tag_utterance("আমি school যাব", l1="bn")
    assert tags.n_words_l1 == 2
    assert tags.n_words_en == 1
    assert tags.is_code_mixed is True


def test_unsupported_l1_raises():
    import pytest
    with pytest.raises(ValueError):
        tag_utterance("hello", l1="ta")

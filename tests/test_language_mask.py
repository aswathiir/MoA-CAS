"""
Tests for moa_cas.adapters.language_mask — the component whose earlier
(string-matched) version silently put 83% of Hindi's indices on the wrong
columns. The real-checkpoint test runs only where the Hindi NeMo
checkpoint is already extracted (it is gated and ~500 MB, so CI skips it).
"""

from pathlib import Path

import pytest

from moa_cas.adapters.language_mask import build

HI_CHECKPOINT = Path.home() / ".cache" / "moa_cas" / "nemo_checkpoints" / "hi" / "model_config.yaml"


def test_rejects_vocabulary_not_divisible_into_language_blocks():
    cfg = {"aux_ctc": {"decoder": {"vocabulary": ["a"] * 7}},
           "tokenizer": {"langs": {"hi": {}, "bn": {}}, "model_path": "unused"}}
    with pytest.raises(ValueError, match="not divisible"):
        build(cfg, "hi")


def test_rejects_unknown_language():
    cfg = {"aux_ctc": {"decoder": {"vocabulary": ["a"] * 4}},
           "tokenizer": {"langs": {"hi": {}, "bn": {}}, "model_path": "unused"}}
    with pytest.raises(ValueError, match="not in checkpoint"):
        build(cfg, "en")


@pytest.mark.skipif(not HI_CHECKPOINT.exists(), reason="Hindi NeMo checkpoint not extracted locally")
def test_hindi_mask_is_its_contiguous_block_plus_blank():
    pytest.importorskip("nemo")
    from moa_cas.adapters.nemo_backbone import load_backbone

    _, encoder, ctc_decoder, cfg = load_backbone("hi")
    sp, mask = build(cfg, "hi")

    vocab = cfg["aux_ctc"]["decoder"]["vocabulary"]
    assert len(vocab) == 5632 and len(cfg["tokenizer"]["langs"]) == 22
    assert mask.tolist() == list(range(1536, 1792)) + [5632]  # Hindi = language #6, then shared blank
    assert [vocab[i] for i in mask[:-1].tolist()] == [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

    # Architecture facts the docs and thesis quote.
    assert len(encoder.layers) == 17 and encoder.d_model == 512
    assert sum(p.numel() for p in encoder.parameters()) == 115_111_424
    assert sum(p.numel() for p in ctc_decoder.parameters()) == 2_889_729
    assert not any(any(ch.isascii() and ch.isalpha() for ch in piece)
                   for piece in vocab if piece != "<unk>"), "backbone has no Latin-script output tokens"

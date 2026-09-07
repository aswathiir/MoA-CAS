"""
moa_cas/adapters/language_mask.py
------------------------------------
The CTC decoder's output is a single vocabulary shared across all 22
languages (5632 tokens + 1 blank, appended as the last class — confirmed
via the decoder's actual weight shape). A given language's own
SentencePiece tokenizer only covers a subset of it, and no mask file
ships with these checkpoints (ai4bharat's ONNX bundle has the equivalent,
assets/language_masks.json, but the NeMo checkpoints don't). This
reconstructs it by string-matching the language's own tokenizer pieces
against the shared vocabulary's token strings.

Use the language's own compiled .tokenizer.model file for this (via the
`sentencepiece` library directly), not the auxiliary vocab.txt/
tokenizer.vocab text exports bundled alongside it — those can use a
different display convention (e.g. WordPiece-style '##' prefixes) that
won't cross-reference against the SentencePiece '▁'-style master
vocabulary at all.
"""

from pathlib import Path

import sentencepiece as spm
import torch


def build(tokenizer_model_path: str, cfg_dict: dict):
    """
    Returns (SentencePieceProcessor, LongTensor). The tensor holds global
    column indices — the language's own token ids in order, followed by
    the shared blank index — for gathering language-specific columns out
    of the decoder's full (B, T, 5633) output before log_softmax + CTC
    loss. Local index i < vocab_size maps 1:1 to the tokenizer's own
    token id i; local index == vocab_size is blank.
    """
    master_vocab = cfg_dict["aux_ctc"]["decoder"]["vocabulary"]
    master_index = {tok: i for i, tok in enumerate(master_vocab)}
    blank_index = len(master_vocab)

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model_path))
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

    missing = [p for p in pieces if p not in master_index]
    if missing:
        raise ValueError(
            f"{len(missing)} tokenizer pieces not found in the shared vocabulary "
            f"(first 5: {missing[:5]}) — is tokenizer_model_path the right language's model?"
        )

    local_to_global = [master_index[p] for p in pieces] + [blank_index]
    return sp, torch.tensor(local_to_global, dtype=torch.long)

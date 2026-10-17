"""
moa_cas/adapters/language_mask.py
------------------------------------
The CTC decoder emits a single vocabulary shared across all 22 languages
(5,632 tokens + 1 blank appended as the final class — confirmed from the
decoder weight shape [5633, 512, 1]). A given language's tokenizer covers
only 256 of those, and no mask ships with these checkpoints (ai4bharat's
ONNX bundle has the equivalent in assets/language_masks.json; the NeMo
checkpoints do not).

The shared vocabulary is laid out as **contiguous per-language blocks**,
in the order the languages appear in `tokenizer.langs`: 22 languages x
256 tokens = 5,632. So a language's mask is simply its block.

Do NOT reconstruct this by matching tokenizer pieces against the
vocabulary by string. 670 token strings are duplicated across languages
(the Devanagari-script languages share many pieces), so a string->index
lookup silently resolves them to whichever language happens to appear
last — which put 83% of Hindi's indices on the wrong columns and made
the model emit near-gibberish while still producing a plausibly falling
training loss.
"""

from pathlib import Path

import sentencepiece as spm
import torch


def build(cfg_dict: dict, lang: str):
    """
    Returns (SentencePieceProcessor, LongTensor). The tensor holds the
    language's contiguous block of global column indices followed by the
    shared blank index, for gathering a language-specific view out of the
    decoder's full (B, T, 5633) output. Local index i maps 1:1 to the
    tokenizer's own token id i; the final local index is blank.

    Raises if the derived block does not match the tokenizer's pieces
    exactly — that mismatch means the layout assumption is wrong for this
    checkpoint and any training against it would be meaningless.
    """
    master_vocab = cfg_dict["aux_ctc"]["decoder"]["vocabulary"]
    langs = list(cfg_dict["tokenizer"]["langs"].keys())

    if len(master_vocab) % len(langs) != 0:
        raise ValueError(
            f"Shared vocabulary ({len(master_vocab)}) is not divisible by the "
            f"language count ({len(langs)}); per-language block layout does not hold."
        )
    block_size = len(master_vocab) // len(langs)

    if lang not in langs:
        raise ValueError(f"Language '{lang}' not in checkpoint tokenizer langs: {langs}")
    offset = langs.index(lang) * block_size
    block = master_vocab[offset:offset + block_size]

    sp = spm.SentencePieceProcessor(model_file=str(cfg_dict["tokenizer"]["model_path"]))
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

    if block != pieces:
        mismatches = sum(1 for a, b in zip(block, pieces) if a != b)
        raise ValueError(
            f"Vocabulary block for '{lang}' (indices {offset}:{offset + block_size}) does not "
            f"match its tokenizer pieces — {mismatches}/{len(pieces)} differ. "
            f"The per-language block layout assumption is wrong for this checkpoint."
        )

    blank_index = len(master_vocab)
    local_to_global = list(range(offset, offset + block_size)) + [blank_index]
    return sp, torch.tensor(local_to_global, dtype=torch.long)

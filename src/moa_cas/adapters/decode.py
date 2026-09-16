"""
moa_cas/adapters/decode.py
-----------------------------
Greedy CTC decoding over the language-masked view of the shared decoder
output (see language_mask.py). Because the mask is built in tokenizer-id
order, local index i corresponds exactly to the language tokenizer's own
token id i, so decoded ids can be handed straight back to SentencePiece.
"""

import torch


def greedy_ctc_decode(log_probs: torch.Tensor, sp, blank_id: int = None) -> str:
    """
    log_probs: (1, T, V_local) — language-masked log probabilities.
    Returns the decoded string for the single utterance in the batch.
    """
    if blank_id is None:
        blank_id = log_probs.shape[-1] - 1

    ids = log_probs.argmax(dim=-1)[0].tolist()

    collapsed = []
    previous = None
    for token_id in ids:
        if token_id != previous and token_id != blank_id:
            collapsed.append(token_id)
        previous = token_id

    return sp.decode(collapsed)

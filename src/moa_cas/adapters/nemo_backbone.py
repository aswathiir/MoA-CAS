"""
moa_cas/adapters/nemo_backbone.py
------------------------------------
Loads ai4bharat's per-language NeMo IndicConformer checkpoints
(ai4bharat/indicconformer_stt_{lang}_hybrid_ctc_rnnt_large) for adapter
training. Bypasses NeMo's own .nemo restore path, which is broken for
these checkpoints in three separate ways — see docs/ADAPTER_TRAINING.md
for the full story of each workaround below.

Requires: the target HF repo is gated. Visit
https://huggingface.co/ai4bharat/indicconformer_stt_{lang}_hybrid_ctc_rnnt_large
and accept access while logged in (huggingface-cli login) before this
will download anything.
"""

import tarfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

CACHE_ROOT = Path.home() / ".cache" / "moa_cas" / "nemo_checkpoints"


def _download_and_extract(lang: str) -> Path:
    extract_dir = CACHE_ROOT / lang
    # Skip the hub round-trip entirely once extracted — snapshot_download
    # re-validates the cached repo on every call, which costs minutes.
    if (extract_dir / "model_config.yaml").exists():
        return extract_dir

    repo_id = f"ai4bharat/indicconformer_stt_{lang}_hybrid_ctc_rnnt_large"
    snapshot_dir = Path(snapshot_download(repo_id=repo_id))
    nemo_files = list(snapshot_dir.glob("*.nemo"))
    if not nemo_files:
        raise FileNotFoundError(f"No .nemo file found in {snapshot_dir}")

    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(nemo_files[0]) as tf:
        tf.extractall(extract_dir)
    return extract_dir


def load_backbone(lang: str, extract_dir: Path = None, device=None):
    """
    Returns (preprocessor, encoder, ctc_decoder, cfg_dict), all frozen
    (requires_grad=False, eval mode) and moved to `device` — insert
    trainable adapters on top via
    moa_cas.adapters.bottleneck.insert_into_encoder.

    cfg_dict["tokenizer"] is rewritten to a flat, absolute-path form (see
    below) — read it back to build the language mask
    (moa_cas.adapters.language_mask.build) against the same paths.
    """
    extract_dir = extract_dir or _download_and_extract(lang)

    cfg_dict = OmegaConf.to_container(
        OmegaConf.load(extract_dir / "model_config.yaml"), resolve=False
    )

    # ai4bharat's checkpoints use a multilingual-style nested tokenizer
    # config (tokenizer.langs.{hi,bn,...}), but NeMo's monolingual loader
    # expects a flat top-level dir/model_path/vocab_path and breaks
    # (KeyError: 'dir') otherwise. The referenced nemo:<hash>_* files are
    # already sitting in extract_dir.
    lang_tok = cfg_dict["tokenizer"]["langs"][lang]

    def _resolve(val):
        return str(extract_dir / val.split("nemo:", 1)[1]) if val.startswith("nemo:") else val

    cfg_dict["tokenizer"] = {
        "dir": str(extract_dir),
        "type": "bpe",
        "model_path": _resolve(lang_tok["model_path"]),
        "vocab_path": _resolve(lang_tok["vocab_path"]),
        "spe_tokenizer_vocab": _resolve(lang_tok["spe_tokenizer_vocab"]),
        # Preserved because language_mask.build needs the language ordering
        # to locate this language's block in the shared CTC vocabulary.
        "langs": {name: {} for name in cfg_dict["tokenizer"]["langs"]},
    }

    # These reference ai4bharat's internal training-cluster manifest
    # paths, and train_ds is missing a 'shuffle' key that NeMo's own init
    # code tries to backfill — which fails under OmegaConf struct mode no
    # matter what set_struct(cfg, False) is called upstream, because the
    # write happens against a dataclass-backed schema that's always
    # struct-locked. Irrelevant anyway: we build our own dataloader.
    for key in ("train_ds", "validation_ds", "test_ds"):
        cfg_dict.pop(key, None)

    from nemo.collections.asr.modules import (
        AudioToMelSpectrogramPreprocessor,
        ConformerEncoder,
        ConvASRDecoder,
    )

    # ai4bharat trained this with a private patched NeMo fork that added
    # a 'multisoftmax' constructor kwarg to ConvASRDecoder/RNNTDecoder —
    # not present in public nemo_toolkit. Verified safe to drop: the
    # actual saved weights are a single plain decoder_layers.0
    # (weight+bias), not a per-language split, so the vanilla decoder
    # class loads the exact same shapes (0 missing/unexpected keys).
    ctc_decoder_cfg = dict(cfg_dict["aux_ctc"]["decoder"])
    ctc_decoder_cfg.pop("multisoftmax", None)

    preprocessor = AudioToMelSpectrogramPreprocessor.from_config_dict(cfg_dict["preprocessor"])
    encoder = ConformerEncoder.from_config_dict(cfg_dict["encoder"])
    ctc_decoder = ConvASRDecoder.from_config_dict(ctc_decoder_cfg)

    # We only ever need the CTC path (Profile Learning's stated loss
    # choice), so we never instantiate the full EncDecHybridRNNTCTCBPEModel
    # class at all — its __init__ unconditionally builds the RNNT
    # decoder/joint too, which is exactly where the multisoftmax error
    # actually bites hardest. decoder.*/joint.* keys in the checkpoint are
    # RNNT-only and simply go unused here.
    state_dict = torch.load(extract_dir / "model_weights.ckpt", map_location="cpu", weights_only=False)
    for module, prefix in (
        (preprocessor, "preprocessor"),
        (encoder, "encoder"),
        (ctc_decoder, "ctc_decoder"),
    ):
        sub_sd = {k[len(prefix) + 1:]: v for k, v in state_dict.items() if k.startswith(prefix + ".")}
        module.load_state_dict(sub_sd, strict=False)
        module.eval()
        for p in module.parameters():
            p.requires_grad = False
        if device is not None:
            module.to(device)

    return preprocessor, encoder, ctc_decoder, cfg_dict

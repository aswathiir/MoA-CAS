"""
moa_cas/model.py
-----------------
Load IndicConformer from HuggingFace and run single-utterance inference.

Supported language codes (22 scheduled Indian languages):
  hi  bn  te  mr  ta  ur  gu  kn  ml  or
  pa  as  mai kok ne  sa  sat sd  ks  doi  mni  brx

English ("en") is NOT supported — passing it returns empty string (WER = 1.0).
For code-mixed Hindi-English benchmarking, pass "hi" — the model will attempt
Hindi transcription and fail on English segments, revealing the gap.
"""

import torch
from transformers import AutoModel

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

SUPPORTED_LANGS = {
    "hi", "bn", "te", "mr", "ta", "ur", "gu", "kn", "ml", "or",
    "pa", "as", "mai", "kok", "ne", "sa", "sat", "sd", "ks", "doi", "mni", "brx"
}


def load_model(device: str):
    print(f"\nLoading IndicConformer on {device}...")
    model = (
        AutoModel
        .from_pretrained(MODEL_ID, trust_remote_code=True)
        .to(device)
    )
    model.eval()
    print("Model ready.\n")
    return model


def transcribe(model, wav: torch.Tensor, lang: str, decode: str) -> str:
    """Returns transcription string. Returns empty string on failure."""
    if lang not in SUPPORTED_LANGS:
        raise ValueError(
            f"Language '{lang}' not supported. "
            f"Supported: {sorted(SUPPORTED_LANGS)}"
        )
    try:
        with torch.no_grad():
            return model(wav, lang, decode)
    except Exception as e:
        print(f"  [transcribe error] {type(e).__name__}: {e}")
        return ""

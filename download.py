"""
download_model.py
-----------------
Run this ONCE before benchmark.py.
Downloads IndicConformer-600M to HuggingFace cache and verifies it loads.

Steps before running:
1. Go to https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
2. Click "Agree and access repository" (one-time, needs HF account)
3. Run: huggingface-cli login
4. Run: python download_model.py
"""

import torch
import torchaudio
from transformers import AutoModel

MODEL_ID  = "ai4bharat/indic-conformer-600m-multilingual"
TARGET_SR = 16000


def download_and_verify():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 52)
    print("  IndicConformer-600M — Download & Verify")
    print("=" * 52)
    print(f"  Model : {MODEL_ID}")
    print(f"  Device: {device}")
    print()
    print("Downloading model weights (cached after first run)...")

    model = (
        AutoModel
        .from_pretrained(MODEL_ID, trust_remote_code=True)
        .to(device)
    )
    model.eval()
    print("Download complete.\n")

    # ── Quick sanity check with a silent 1-second dummy audio ────────────
    print("Running sanity check with dummy audio...")
    dummy_wav = torch.zeros(1, TARGET_SR).to(device)   # 1 second of silence

    with torch.no_grad():
        result = model(dummy_wav, "hi", "ctc")

    print(f"Sanity check passed. Output: '{result}'")
    print("\nModel is ready. You can now run benchmark.py.")


if __name__ == "__main__":
    download_and_verify()
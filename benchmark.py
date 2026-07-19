"""
benchmark.py
------------
Baseline benchmark: IndicConformer-600M on Svarah (Indian-accented English).

Purpose
-------
Demonstrates the performance gap that motivates MoA-CAS.
IndicConformer is trained on 22 scheduled Indian languages, NOT English.
Passing lang="en" will show high WER — that is the gap we are documenting.

Usage
-----
    python benchmark.py                          # defaults: en, ctc, test
    python benchmark.py --lang hi --decode rnnt  # Hindi with RNNT decoding
    python benchmark.py --limit 100              # quick smoke-test on 100 samples

Output
------
    results/indicconformer_<lang>_<decode>.csv   — full per-sample results
    Console summary table: overall + per native language + per state
"""

import argparse
import csv
import torch
import torchaudio
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from transformers import AutoModel
from datasets import load_dataset

from metrics import compute_wer, compute_cer


# ── Constants ────────────────────────────────────────────────────────────────

MODEL_ID  = "ai4bharat/indic-conformer-600m-multilingual"
DATASET   = "ai4bharat/Svarah"
TARGET_SR = 16000


# ── Model ────────────────────────────────────────────────────────────────────

def load_model(device: str):
    print(f"\nLoading IndicConformer from {MODEL_ID} on {device}...")
    model = (
        AutoModel
        .from_pretrained(MODEL_ID, trust_remote_code=True)
        .to(device)
    )
    model.eval()
    print("Model ready.\n")
    return model


# ── Audio preprocessing ──────────────────────────────────────────────────────

def prepare_audio(audio_dict: dict, device: str) -> torch.Tensor:
    """
    HuggingFace Audio feature gives {"array": np.ndarray, "sampling_rate": int}.
    Returns a mono float32 tensor at 16 kHz on the target device.
    """
    wav = torch.tensor(audio_dict["array"], dtype=torch.float32)
    sr  = audio_dict["sampling_rate"]

    # Resample if needed
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    # Ensure shape is (1, T) — model expects (channels, time)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    elif wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)   # stereo → mono

    return wav.to(device)


# ── Inference ────────────────────────────────────────────────────────────────

def transcribe(model, wav: torch.Tensor, lang: str, decode: str) -> str:
    """Single-utterance inference. Returns empty string on failure."""
    try:
        with torch.no_grad():
            return model(wav, lang, decode)
    except Exception as e:
        return ""


# ── Results helpers ──────────────────────────────────────────────────────────

def avg(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def print_breakdown(label: str, groups: dict):
    """Print a per-group WER/CER table sorted by WER descending."""
    print(f"\n── Per {label} {'─' * (40 - len(label))}")
    print(f"  {'Group':<28} {'WER':>7} {'CER':>7} {'N':>5}")
    print(f"  {'─'*28} {'─'*7} {'─'*7} {'─'*5}")
    for key, s in sorted(groups.items(), key=lambda x: -avg(x[1]["wer"])):
        n = len(s["wer"])
        print(f"  {key:<28} {avg(s['wer']):>7.4f} {avg(s['cer']):>7.4f} {n:>5}")


def save_csv(rows: list, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFull results → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(device)

    print(f"Loading {DATASET} (split={args.split})...")
    dataset = load_dataset(DATASET, split=args.split, trust_remote_code=True)

    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limiting to {len(dataset)} samples (--limit flag).")

    print(f"Decoding mode : {args.decode.upper()}")
    print(f"Language code : {args.lang}")
    print(f"Samples       : {len(dataset)}\n")

    rows        = []
    by_language = defaultdict(lambda: {"wer": [], "cer": []})
    by_state    = defaultdict(lambda: {"wer": [], "cer": []})

    for sample in tqdm(dataset, desc="Evaluating"):
        # ── Svarah column names ──────────────────────────────────────────
        # audio      → sample["audio_filepath"]  (HF Audio feature)
        # transcript → sample["text"]
        # speaker L1 → sample["primary_language"]
        # location   → sample["native_place_state"]

        wav = prepare_audio(sample["audio_filepath"], device)
        ref = sample["text"]

        hyp        = transcribe(model, wav, args.lang, args.decode)
        sample_wer = compute_wer(ref, hyp)
        sample_cer = compute_cer(ref, hyp)

        lang_group  = sample.get("primary_language", "unknown")
        state_group = sample.get("native_place_state", "unknown")

        rows.append({
            "reference":        ref,
            "hypothesis":       hyp,
            "primary_language": lang_group,
            "state":            state_group,
            "wer":              round(sample_wer, 4),
            "cer":              round(sample_cer, 4),
        })

        by_language[lang_group]["wer"].append(sample_wer)
        by_language[lang_group]["cer"].append(sample_cer)
        by_state[state_group]["wer"].append(sample_wer)
        by_state[state_group]["cer"].append(sample_cer)

    # ── Summary ──────────────────────────────────────────────────────────
    all_wer = [r["wer"] for r in rows]
    all_cer = [r["cer"] for r in rows]

    print("\n" + "═" * 48)
    print("  INDICONFORMER BASELINE — SVARAH")
    print("═" * 48)
    print(f"  Samples  : {len(rows)}")
    print(f"  Model    : {MODEL_ID}")
    print(f"  Lang code: {args.lang}  |  Decode: {args.decode.upper()}")
    print(f"  Overall WER : {avg(all_wer):.4f}")
    print(f"  Overall CER : {avg(all_cer):.4f}")

    print_breakdown("Native Language", by_language)
    print_breakdown("State",           by_state)

    # ── Save ─────────────────────────────────────────────────────────────
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"indicconformer_{args.lang}_{args.decode}.csv"
    save_csv(rows, out_file)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IndicConformer baseline benchmark on Svarah"
    )
    parser.add_argument(
        "--lang", default="en",
        help="IndicConformer language code (e.g. en, hi, ta). "
             "NOTE: IndicConformer is not trained on English — "
             "high WER on en is expected and is the gap we document."
    )
    parser.add_argument(
        "--decode", default="ctc", choices=["ctc", "rnnt"],
        help="Decoding mode (default: ctc)"
    )
    parser.add_argument(
        "--split", default="test",
        help="Dataset split — Svarah only has 'test' (default: test)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of samples (useful for a quick smoke-test)"
    )
    run(parser.parse_args())
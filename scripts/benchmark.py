"""
scripts/benchmark.py
------------------------
Runs IndicConformer on IndicVoices (monolingual baseline) and/or MUCS
(code-switched) to document the gap that motivates MoA-CAS.

Usage
-----
# Monolingual baseline only (needs HF login)
poetry run python scripts/benchmark.py --dataset indicvoices --limit 100

# Code-mixed gap only (needs a MUCS manifest — see scripts/build_manifest.py)
poetry run python scripts/benchmark.py --dataset mucs --mucs-manifest data/manifests/mucs_hi-en_test.jsonl --limit 100

# Full comparison — runs both and prints the gap table (Table 1 in paper)
poetry run python scripts/benchmark.py --dataset compare --mucs-manifest data/manifests/mucs_hi-en_test.jsonl --limit 100

Notes
-----
- lang is always "hi" here — IndicConformer has no English or code-mix mode.
  On monolingual Hindi it gives its best result.
  On Hindi-English code-mix it degrades — that delta is the gap.
- MUCS manifests are built from raw data via scripts/build_manifest.py.
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moa_cas.model import load_model, transcribe
from moa_cas.audio_io import prepare_audio
from moa_cas.metrics import compute_wer, compute_cer
from moa_cas.reporting import print_summary, print_breakdown, print_comparison, save_csv
from moa_cas.datasets import load_indicvoices, iv_sample, load_mucs, mucs_sample

LANG   = "hi"    # fixed — both datasets benchmarked under Hindi decoding
DECODE = "ctc"


# ── Evaluation loop ──────────────────────────────────────────────────────────

def evaluate(model, samples, device: str, label: str) -> list:
    """
    Runs inference on a list of unified sample dicts.
    Returns list of result rows.
    """
    rows        = []
    by_scenario = defaultdict(lambda: {"wer": [], "cer": []})

    for sample in tqdm(samples, desc=label):
        wav = prepare_audio(sample, device)
        ref = sample["text"]

        try:
            hyp = transcribe(model, wav, LANG, DECODE)
        except ValueError:
            hyp = ""

        sample_wer = compute_wer(ref, hyp)
        sample_cer = compute_cer(ref, hyp)
        scenario   = sample["meta"].get("scenario") or sample["meta"].get("lang", "unknown")

        rows.append({
            "reference": ref,
            "hypothesis": hyp,
            "scenario":   scenario,
            "wer":        round(sample_wer, 4),
            "cer":        round(sample_cer, 4),
        })
        by_scenario[scenario]["wer"].append(sample_wer)
        by_scenario[scenario]["cer"].append(sample_cer)

    print_summary(label, rows, LANG, DECODE)
    print_breakdown("Scenario / Lang", by_scenario)

    return rows


# ── Dataset preparation ───────────────────────────────────────────────────────

def get_indicvoices_samples(limit):
    dataset = load_indicvoices(limit=limit)
    return [iv_sample(row) for row in dataset]


def get_mucs_samples(manifest_path, limit):
    records = load_mucs(manifest_path, limit=limit)
    return [mucs_sample(row) for row in records]


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(device)

    out_dir = Path("results")

    if args.dataset in ("indicvoices", "compare"):
        print("── Run 1: IndicVoices Hindi (monolingual baseline) ──")
        iv_samples = get_indicvoices_samples(args.limit)
        iv_rows    = evaluate(model, iv_samples, device, "IndicVoices Hindi")
        save_csv(iv_rows, out_dir / "indicvoices_hi_ctc.csv")

    if args.dataset in ("mucs", "compare"):
        if not args.mucs_manifest:
            print(
                "\n  --mucs-manifest is required for MUCS runs.\n"
                "  Build one with: python scripts/build_manifest.py --config mucs_hi_en_test\n"
            )
            return
        print("\n── Run 2: MUCS Hindi-English (code-mixed gap) ──")
        mucs_samples = get_mucs_samples(args.mucs_manifest, args.limit)
        mucs_rows    = evaluate(model, mucs_samples, device, "MUCS Hindi-English")
        save_csv(mucs_rows, out_dir / "mucs_hi_en_ctc.csv")

    if args.dataset == "compare":
        print_comparison(iv_rows, mucs_rows)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndicConformer baseline benchmark for MoA-CAS")
    parser.add_argument(
        "--dataset",
        default="compare",
        choices=["indicvoices", "mucs", "compare"],
        help=(
            "indicvoices : monolingual Hindi baseline (HuggingFace)\n"
            "mucs        : code-switched Hindi-English gap (local manifest)\n"
            "compare     : both datasets + gap table (default)"
        ),
    )
    parser.add_argument(
        "--mucs-manifest",
        default=None,
        help="Path to MUCS JSONL manifest file. Required for --dataset mucs or compare."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of samples per dataset (smoke test)."
    )
    run(parser.parse_args())

"""
core/reporting.py
-----------------
Console summary tables and CSV output.
Includes a comparison table for monolingual vs code-mixed results.
"""

import csv
from pathlib import Path


def _avg(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def print_summary(label: str, rows: list, lang: str, decode: str):
    """Overall WER/CER header for one dataset run."""
    all_wer = [r["wer"] for r in rows]
    all_cer = [r["cer"] for r in rows]

    print(f"\n{'═' * 50}")
    print(f"  {label.upper()}")
    print(f"{'═' * 50}")
    print(f"  Samples  : {len(rows)}")
    print(f"  Lang     : {lang}  |  Decode : {decode.upper()}")
    print(f"  WER      : {_avg(all_wer):.4f}")
    print(f"  CER      : {_avg(all_cer):.4f}")


def print_breakdown(label: str, groups: dict):
    """Per-group WER/CER table sorted by WER descending."""
    print(f"\n── Per {label} {'─' * (38 - len(label))}")
    print(f"  {'Group':<28} {'WER':>7} {'CER':>7} {'N':>5}")
    print(f"  {'─' * 28} {'─' * 7} {'─' * 7} {'─' * 5}")
    for key, s in sorted(groups.items(), key=lambda x: -_avg(x[1]["wer"])):
        n = len(s["wer"])
        print(
            f"  {key:<28} "
            f"{_avg(s['wer']):>7.4f} "
            f"{_avg(s['cer']):>7.4f} "
            f"{n:>5}"
        )


def print_comparison(mono_rows: list, codemix_rows: list):
    """
    Side-by-side comparison table — the core result for MoA-CAS motivation.
    The delta between monolingual and code-mixed WER is the gap we close.
    """
    mono_wer   = _avg([r["wer"] for r in mono_rows])
    mono_cer   = _avg([r["cer"] for r in mono_rows])
    codemix_wer = _avg([r["wer"] for r in codemix_rows])
    codemix_cer = _avg([r["cer"] for r in codemix_rows])

    print(f"\n{'═' * 62}")
    print("  GAP ANALYSIS — MONOLINGUAL vs CODE-MIXED")
    print(f"{'═' * 62}")
    print(f"  {'Condition':<30} {'WER':>8} {'CER':>8} {'Samples':>8}")
    print(f"  {'─' * 30} {'─' * 8} {'─' * 8} {'─' * 8}")
    print(f"  {'IndicVoices Hindi (monolingual)':<30} {mono_wer:>8.4f} {mono_cer:>8.4f} {len(mono_rows):>8}")
    print(f"  {'MUCS Hindi-English (code-mixed)':<30} {codemix_wer:>8.4f} {codemix_cer:>8.4f} {len(codemix_rows):>8}")
    print(f"  {'─' * 30} {'─' * 8} {'─' * 8} {'─' * 8}")
    print(f"  {'WER degradation (Δ)':<30} {codemix_wer - mono_wer:>+8.4f}")
    print(f"\n  → This Δ is the gap MoA-CAS targets.\n")


def save_csv(rows: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved → {path}")
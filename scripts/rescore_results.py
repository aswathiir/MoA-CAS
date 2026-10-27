"""
scripts/rescore_results.py
-----------------------------
Recomputes the wer / cer (and wer_l1, where present) columns of result
CSVs from their stored reference and hypothesis text, using the current
moa_cas.metrics. Decoding is deterministic, so this gives exactly the
numbers a fresh evaluation run would, without re-running any model.

Written for correction C6 (docs/THESIS_REFERENCE.md): the original
normaliser deleted Devanagari vowel signs, anusvara and virama before
scoring, so every CSV produced before 2026-10-26 under-counted errors.

Usage
-----
    python scripts/rescore_results.py results/**/*.csv
    python scripts/rescore_results.py --check results/**/*.csv   # exit 1 if any CSV is stale
"""

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moa_cas.metrics import compute_cer, compute_wer

LATIN = re.compile(r"[A-Za-z]")


def l1_only(text: str) -> str:
    # Same rule as scripts/train_router.py::ctc_target_text(..., "drop").
    return " ".join(w for w in text.split() if not LATIN.search(w))


def rescore_rows(rows):
    changed = 0
    for r in rows:
        new = {"wer": round(compute_wer(r["reference"], r["hypothesis"]), 4),
               "cer": round(compute_cer(r["reference"], r["hypothesis"]), 4)}
        if "wer_l1" in r:
            new["wer_l1"] = round(compute_wer(l1_only(r["reference"]), r["hypothesis"]), 4)
        for k, v in new.items():
            if float(r[k]) != v:
                changed += 1
            r[k] = v
    return changed


def main(paths, check):
    stale = 0
    for path in map(Path, paths):
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames, list(reader)
        if not {"reference", "hypothesis", "wer", "cer"} <= set(fields):
            print(f"skip   {path} (no reference/hypothesis/wer/cer columns)")
            continue
        changed = rescore_rows(rows)
        stale += bool(changed)
        if check:
            print(f"{'STALE' if changed else 'ok   '}  {path}  ({changed} values differ)")
            continue
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote  {path}  ({changed} values changed)")
    return 1 if (check and stale) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute WER/CER columns of result CSVs from stored text")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--check", action="store_true", help="Only report stale CSVs; write nothing")
    args = parser.parse_args()
    sys.exit(main(args.paths, args.check))

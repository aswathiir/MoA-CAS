"""
Pins every headline number quoted in the README and docs/THESIS_REFERENCE.md
to the result CSVs it comes from. If a CSV is regenerated and the numbers
move, this fails, and the documents must be updated with it — the docs can
never silently drift away from the evidence.
"""

import csv
import json
import math
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def rows(name):
    return list(csv.DictReader(open(RESULTS / name, encoding="utf-8")))


def mean_pct(name, col="wer"):
    return round(100 * statistics.mean(float(r[col]) for r in rows(name)), 2)


@pytest.mark.parametrize("name, wer, cer", [
    ("indicvoices_hi_ctc.csv", 11.72, 4.57),                         # ONNX, monolingual Hindi
    ("mucs_hi_en_ctc.csv", 59.69, 59.12),                           # ONNX, code-mixed
    ("adapter_eval/nemo_baseline_fixed.csv", 58.06, 55.30),         # NeMo baseline, mixed
    ("adapter_eval/eval_base_pure.csv", 72.28, 59.17),              # NeMo baseline, pure Hindi
    ("adapter_eval/eval_adapt_pure.csv", 59.45, 44.22),             # adapter (Sep 16 ckpt, overwritten)
    ("adapter_eval/eval_adapt_mixed.csv", 47.89, 48.50),
    ("adapter_eval/eval_adapt_pure_ckpt0917.csv", 60.35, 44.97),    # adapter (checkpoint on disk)
    ("adapter_eval/eval_adapt_mixed_ckpt0917.csv", 48.72, 48.26),
    ("router_eval/router_hi_id_mixed.csv", 52.94, 50.03),           # trained router
    ("router_eval/router_hi_id_pure.csv", 64.78, 50.85),
])
def test_headline_wer_cer(name, wer, cer):
    assert len(rows(name)) == 100
    assert mean_pct(name, "wer") == pytest.approx(wer, abs=0.01)
    assert mean_pct(name, "cer") == pytest.approx(cer, abs=0.01)


def paired(adapter, baseline):
    a, b = rows(adapter), rows(baseline)
    assert [r["utt_id"] for r in a] == [r["utt_id"] for r in b], "paired test needs identical utterances"
    d = [float(x["wer"]) - float(y["wer"]) for x, y in zip(a, b)]
    m, se = statistics.mean(d), statistics.stdev(d) / math.sqrt(len(d))
    return {"better": sum(x < 0 for x in d), "worse": sum(x > 0 for x in d),
            "same": sum(x == 0 for x in d), "t": round(m / se, 2)}


def test_adapter_improvement_is_paired_and_significant():
    assert paired("adapter_eval/eval_adapt_pure.csv", "adapter_eval/eval_base_pure.csv") == \
        {"better": 50, "worse": 27, "same": 23, "t": -4.62}
    assert paired("adapter_eval/eval_adapt_pure_ckpt0917.csv", "adapter_eval/eval_base_pure.csv") == \
        {"better": 51, "worse": 25, "same": 24, "t": -4.45}
    assert paired("adapter_eval/eval_adapt_mixed_ckpt0917.csv", "adapter_eval/nemo_baseline_fixed.csv")["t"] == -3.83


def test_router_beats_base_but_not_adapter_alone():
    mixed = "router_eval/router_hi_id_mixed.csv"
    assert paired(mixed, "adapter_eval/nemo_baseline_fixed.csv")["t"] == -2.59
    assert paired(mixed, "adapter_eval/eval_adapt_mixed_ckpt0917.csv")["t"] == 2.68
    pure = "router_eval/router_hi_id_pure.csv"
    assert paired(pure, "adapter_eval/eval_base_pure.csv")["t"] == -3.57
    assert paired(pure, "adapter_eval/eval_adapt_pure_ckpt0917.csv")["t"] == 2.85


def test_stored_scores_match_current_metric():
    """Guards against C6 recurring: every stored WER/CER must equal what
    moa_cas.metrics computes from the stored reference and hypothesis."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from rescore_results import rescore_rows

    for path in sorted(RESULTS.rglob("*.csv")):
        data = list(csv.DictReader(open(path, encoding="utf-8")))
        if data and {"reference", "hypothesis", "wer", "cer"} <= set(data[0]):
            assert rescore_rows(data) == 0, f"{path.relative_to(ROOT)} has stale scores; run scripts/rescore_results.py"


def test_train_and_test_speakers_are_disjoint():
    def speakers(split):
        path = ROOT / f"data/manifests/mucs_hi-en_{split}.jsonl"
        return {json.loads(line)["speaker_id"] for line in open(path, encoding="utf-8")}
    train, test = speakers("train"), speakers("test")
    assert (len(train), len(test), len(train & test)) == (520, 30, 0)


def test_pure_hindi_training_pool_size():
    path = ROOT / "data/manifests/mucs_hi-en_train.jsonl"
    pool = [r for r in map(json.loads, open(path, encoding="utf-8"))
            if r["n_words_en"] == 0 and 1.0 <= r["duration"] <= 10.0]
    assert len(pool) == 7083

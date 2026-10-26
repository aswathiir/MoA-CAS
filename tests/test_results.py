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
    ("indicvoices_hi_ctc.csv", 7.93, 4.16),                         # ONNX, monolingual Hindi
    ("mucs_hi_en_ctc.csv", 57.14, 56.13),                           # ONNX, code-mixed
    ("adapter_eval/nemo_baseline_fixed.csv", 55.37, 54.40),         # NeMo baseline, mixed
    ("adapter_eval/eval_base_pure.csv", 67.18, 56.05),              # NeMo baseline, pure Hindi
    ("adapter_eval/eval_adapt_pure.csv", 51.81, 40.57),             # adapter (Sep 16 ckpt, overwritten)
    ("adapter_eval/eval_adapt_mixed.csv", 45.44, 50.37),
    ("adapter_eval/eval_adapt_pure_ckpt0917.csv", 52.84, 41.58),    # adapter (checkpoint on disk)
    ("adapter_eval/eval_adapt_mixed_ckpt0917.csv", 46.60, 49.01),
    ("router_eval/router_hi_id_mixed.csv", 50.50, 51.07),           # trained router
    ("router_eval/router_hi_id_pure.csv", 58.25, 47.88),
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
        {"better": 53, "worse": 25, "same": 22, "t": -4.84}
    assert paired("adapter_eval/eval_adapt_pure_ckpt0917.csv", "adapter_eval/eval_base_pure.csv") == \
        {"better": 55, "worse": 23, "same": 22, "t": -4.81}
    assert paired("adapter_eval/eval_adapt_mixed_ckpt0917.csv", "adapter_eval/nemo_baseline_fixed.csv")["t"] == -3.56


def test_router_beats_base_but_not_adapter_alone():
    mixed = "router_eval/router_hi_id_mixed.csv"
    assert paired(mixed, "adapter_eval/nemo_baseline_fixed.csv")["t"] == -2.30
    assert paired(mixed, "adapter_eval/eval_adapt_mixed_ckpt0917.csv")["t"] == 2.75
    pure = "router_eval/router_hi_id_pure.csv"
    assert paired(pure, "adapter_eval/eval_base_pure.csv")["t"] == -3.69
    assert paired(pure, "adapter_eval/eval_adapt_pure_ckpt0917.csv")["t"] == 3.36


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

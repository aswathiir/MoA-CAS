"""
scripts/evaluate_adapter.py
-------------------------------
Measures WER/CER of the NeMo IndicConformer CTC path, with or without a
trained adapter, so the two can be compared directly on identical data.

Run with no --adapter-checkpoint to get the untouched backbone's baseline.
That baseline doubles as a correctness gate on the hand-reconstructed
language mask (see moa_cas/adapters/language_mask.py): if it lands far
from the independently measured ONNX benchmark on the same utterances,
the mask or the decode path is wrong and any training against it is
meaningless.

Usage
-----
    # baseline, same 100 utterances the ONNX benchmark used
    poetry run python scripts/evaluate_adapter.py --lang hi \
        --manifest data/manifests/mucs_hi-en_test.jsonl --limit 100

    # pure-monolingual subset, with a trained adapter
    poetry run python scripts/evaluate_adapter.py --lang hi \
        --manifest data/manifests/mucs_hi-en_test.jsonl --limit 100 \
        --pure-l1-only --adapter-checkpoint checkpoints/hi_adapter.pt
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moa_cas.adapters.bottleneck import insert_into_encoder
from moa_cas.adapters.decode import greedy_ctc_decode
from moa_cas.adapters.language_mask import build as build_language_mask
from moa_cas.adapters.nemo_backbone import load_backbone
from moa_cas.metrics import compute_cer, compute_wer


def load_rows(manifest: str, limit: int, pure_l1_only: bool, skip: int):
    rows = [json.loads(line) for line in open(manifest, encoding="utf-8")]
    if pure_l1_only:
        rows = [r for r in rows if r["n_words_en"] == 0]
    rows = rows[skip:]
    return rows[:limit] if limit else rows


def run(lang, manifest, limit, pure_l1_only, skip, adapter_checkpoint, bottleneck, out_csv, device):
    device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                          else ("cpu" if device == "auto" else device))
    preprocessor, encoder, ctc_decoder, cfg_dict = load_backbone(lang, device=device)
    sp, lang_mask = build_language_mask(cfg_dict, lang)
    lang_mask = lang_mask.to(device)

    label = "baseline (no adapter)"
    if adapter_checkpoint:
        adapters = insert_into_encoder(encoder, bottleneck=bottleneck)
        state = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
        adapters.load_state_dict(state["adapters"] if "adapters" in state else state)
        adapters.to(device).eval()
        label = f"adapter ({Path(adapter_checkpoint).name})"

    rows = load_rows(manifest, limit, pure_l1_only, skip)
    print(f"Evaluating {label} on {len(rows)} utterances from {Path(manifest).name}"
          f"{' [pure-L1 only]' if pure_l1_only else ''}\n")

    results = []
    start = time.time()
    with torch.no_grad():
        for i, row in enumerate(rows, 1):
            wav_np, _ = sf.read(row["audio_filepath"])
            wav = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0).to(device)
            wav_len = torch.tensor([wav.shape[-1]], device=device)

            feats, feat_lens = preprocessor(input_signal=wav, length=wav_len)
            enc_out, enc_lens = encoder(audio_signal=feats, length=feat_lens)
            logits = ctc_decoder(encoder_output=enc_out)
            log_probs = torch.log_softmax(logits[:, :, lang_mask], dim=-1)

            hyp = greedy_ctc_decode(log_probs, sp)
            ref = row["text"]
            results.append({
                "utt_id": row["utt_id"], "reference": ref, "hypothesis": hyp,
                "wer": round(compute_wer(ref, hyp), 4),
                "cer": round(compute_cer(ref, hyp), 4),
                "n_words_en": row["n_words_en"], "duration": row["duration"],
            })
            if i % 20 == 0:
                print(f"  {i}/{len(rows)}  running WER="
                      f"{statistics.mean(r['wer'] for r in results):.4f}", flush=True)

    wers = [r["wer"] for r in results]
    cers = [r["cer"] for r in results]
    print(f"\n{'=' * 56}\n  {label}\n{'=' * 56}")
    print(f"  Utterances : {len(results)}")
    print(f"  WER        : {statistics.mean(wers):.4f}")
    print(f"  CER        : {statistics.mean(cers):.4f}")
    print(f"  Median WER : {statistics.median(wers):.4f}")
    print(f"  Elapsed    : {time.time() - start:.1f}s")

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"  Saved      : {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the NeMo CTC path, with or without a trained adapter")
    parser.add_argument("--lang", required=True, choices=["hi", "bn"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--skip", type=int, default=0, help="Skip the first N rows (e.g. to reach a held-out tail)")
    parser.add_argument("--pure-l1-only", action="store_true", help="Keep only utterances with zero English words")
    parser.add_argument("--adapter-checkpoint", default=None)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = parser.parse_args()
    run(args.lang, args.manifest, args.limit, args.pure_l1_only, args.skip,
        args.adapter_checkpoint, args.bottleneck, args.out_csv, args.device)

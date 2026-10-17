"""
scripts/evaluate_router.py
-----------------------------
Measures the Stage 2 mixture (frozen backbone + frozen Stage 1 experts +
router) on a manifest, and reports what the router actually decided.

Run without --router-checkpoint to evaluate the untrained router, whose
zero-initialised output layer gives exactly uniform weights. That is the
router's own baseline: a trained router has to beat "average the experts".

Besides WER/CER against the full reference, it reports:

  wer_l1       WER against the reference with Latin-script words removed —
               the part of the transcript the backbone can actually write
  share_<name> mean routing weight on each expert over the utterance's
               real frames
  corr         Pearson correlation, across utterances, between each
               expert's share and the utterance's English-word fraction.
               A router that has learned *language* (not just "always pick
               the better adapter") should show a clear non-zero value.

--trace UTT_ID prints the frame-by-frame top-1 expert next to the greedy
CTC token emitted on that frame, to see mid-utterance switching directly.

Usage
-----
    python scripts/evaluate_router.py --lang hi \
        --expert hi=checkpoints/hi_adapter.pt --expert identity \
        --manifest data/manifests/mucs_hi-en_test.jsonl --limit 100 \
        --router-checkpoint checkpoints/router_hi_id.pt \
        --out-csv results/router_eval/router_mixed.csv
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from moa_cas.adapters.decode import greedy_ctc_decode
from moa_cas.adapters.language_mask import build as build_language_mask
from moa_cas.adapters.nemo_backbone import load_backbone
from moa_cas.adapters.router import MixtureOfAdapters, attach, build_experts
from moa_cas.metrics import compute_cer, compute_wer
from train_router import ctc_target_text


def pearson(xs, ys):
    if len(xs) < 3 or statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    return cov / (statistics.pstdev(xs) * statistics.pstdev(ys))


def print_trace(row, log_probs, weights, names, sp, blank_id):
    ids = log_probs.argmax(dim=-1)[0].tolist()
    top = weights[0].argmax(dim=-1).tolist()
    print(f"\nTrace {row['utt_id']}\n  ref: {row['text']}")
    print(f"  {'frame':>5}  {'time':>6}  {'expert':<10} {'w':>5}  token")
    previous = None
    for t, (tok, k) in enumerate(zip(ids, top)):
        piece = "" if tok == blank_id or tok == previous else sp.id_to_piece(tok)
        previous = tok
        print(f"  {t:5d}  {t / 25:6.2f}  {names[k]:<10} {weights[0, t, k].item():5.2f}  {piece}")


def run(args):
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    preprocessor, encoder, ctc_decoder, cfg_dict = load_backbone(args.lang, device=device)
    n_layers, dim = len(encoder.layers), getattr(encoder, "d_model", 512)

    router_cfg = {}
    if args.router_checkpoint:
        router_cfg = torch.load(args.router_checkpoint, map_location="cpu", weights_only=False)
        if router_cfg["experts"] != args.expert:
            raise ValueError(f"Router was trained with experts {router_cfg['experts']}, "
                             f"but --expert gives {args.expert}; order and identity must match.")

    stacks, names = build_experts(args.expert, n_layers, dim, backbone_lang=args.lang)
    moa = MixtureOfAdapters(stacks, dim, names,
                            router_hidden=router_cfg.get("router_hidden", 128),
                            router_context=router_cfg.get("router_context", 5),
                            temperature=router_cfg.get("temperature", 1.0),
                            routing=args.routing)
    if router_cfg:
        moa.routers.load_state_dict(router_cfg["routers"])
    attach(encoder, moa).to(device).eval()

    sp, lang_mask = build_language_mask(cfg_dict, args.lang)
    lang_mask = lang_mask.to(device)
    blank_id = len(lang_mask) - 1

    rows = [json.loads(line) for line in open(args.manifest, encoding="utf-8")]
    if args.code_mixed_only:
        rows = [r for r in rows if r["is_code_mixed"]]
    rows = rows[args.skip:]
    rows = rows[:args.limit] if args.limit else rows

    label = (f"router ({Path(args.router_checkpoint).name}, {args.routing})"
             if args.router_checkpoint else "untrained router (uniform weights)")
    print(f"Evaluating {label} over experts {names} on {len(rows)} utterances "
          f"from {Path(args.manifest).name}\n")

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
            weights = moa.routing_weights()

            hyp = greedy_ctc_decode(log_probs, sp)
            ref = row["text"]
            share = weights[0, :int(enc_lens[0])].mean(dim=0).tolist()
            result = {
                "utt_id": row["utt_id"], "reference": ref, "hypothesis": hyp,
                "wer": round(compute_wer(ref, hyp), 4),
                "cer": round(compute_cer(ref, hyp), 4),
                "wer_l1": round(compute_wer(ctc_target_text(ref, "drop"), hyp), 4),
                "en_fraction": round(row["n_words_en"] / max(row["n_words"], 1), 4),
                "n_cs_points": row["n_cs_points"], "duration": row["duration"],
            }
            result.update({f"share_{n}": round(s, 4) for n, s in zip(names, share)})
            results.append(result)

            if args.trace and row["utt_id"] == args.trace:
                print_trace(row, log_probs, weights, names, sp, blank_id)
            if i % 20 == 0:
                print(f"  {i}/{len(rows)}  running WER="
                      f"{statistics.mean(r['wer'] for r in results):.4f}", flush=True)

    print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
    print(f"  Utterances : {len(results)}")
    for key in ("wer", "cer", "wer_l1"):
        print(f"  {key.upper():<11}: {statistics.mean(r[key] for r in results):.4f}")
    en = [r["en_fraction"] for r in results]
    for n in names:
        s = [r[f"share_{n}"] for r in results]
        print(f"  share {n:<10}: mean {statistics.mean(s):.3f}   corr(en_fraction) {pearson(s, en):+.3f}")
    print(f"  Elapsed    : {time.time() - start:.1f}s")

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"  Saved      : {args.out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the MoA-CAS router over frozen Stage 1 experts")
    parser.add_argument("--lang", required=True, choices=["hi", "bn"])
    parser.add_argument("--expert", action="append", required=True,
                        help="Same --expert list, in the same order, as used for training")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--router-checkpoint", default=None)
    parser.add_argument("--routing", choices=["soft", "hard"], default="soft")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--code-mixed-only", action="store_true")
    parser.add_argument("--trace", default=None, help="utt_id to print a frame-level routing trace for")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    run(parser.parse_args())

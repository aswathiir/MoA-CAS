"""
scripts/train_router.py
--------------------------
Stage 2 "Fluidity Learning": train the frame-level router
(moa_cas/adapters/router.py) on code-mixed MUCS utterances, with the
backbone AND every Stage 1 adapter frozen. Loss = CTC + alpha * load
balancing, per the MoA-CAS design (deck slide 10).

Experts are given as repeated --expert flags, each either a Stage 1
checkpoint trained on the SAME backbone (`name=path.pt`) or `identity`
(the raw frozen backbone — "no adaptation").

Usage
-----
    # Hindi adapter vs. untouched backbone, CPU smoke run
    python scripts/train_router.py --lang hi \
        --expert hi=checkpoints/hi_adapter.pt --expert identity \
        --max-seconds 600 --n-samples 200 --save-to checkpoints/router_hi_id.pt

    # GPU
    python scripts/train_router.py --lang hi \
        --expert hi=checkpoints/hi_adapter.pt --expert identity \
        --device cuda --batch-size 16 --max-seconds 10800 \
        --save-to checkpoints/router_hi_id.pt

Then measure it with scripts/evaluate_router.py. Without a checkpoint
that script evaluates the untrained (uniform-weight) mixture, which is
the router's own baseline.

English words in the targets
----------------------------
The backbone's CTC vocabulary has no Latin script at all (all 5,632
tokens are Indian-language pieces), so English words in a MUCS reference
cannot be written by any expert. --english-targets controls what the
CTC target does with them:

  drop (default)  remove Latin-script words; the frames that carry them
                  are then free to emit blank, so no expert is rewarded
                  for hallucinating Devanagari there
  keep            leave them in; SentencePiece maps them to <unk>, which
                  teaches the router to favour whichever expert best
                  produces <unk> on English speech (an English detector,
                  but it pollutes the transcript)

Neither is a substitute for an English-capable decoder — see
docs/05_ROUTER.md.
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from moa_cas.adapters.language_mask import build as build_language_mask
from moa_cas.adapters.nemo_backbone import load_backbone
from moa_cas.adapters.router import MixtureOfAdapters, attach, build_experts
from train_profile_adapter import ENCODER_FRAMES_PER_SEC, collate, make_batches, resolve_device

MANIFESTS = {
    "hi": "data/manifests/mucs_hi-en_train.jsonl",
    "bn": "data/manifests/mucs_bn-en_train.jsonl",
}

LATIN = re.compile(r"[A-Za-z]")


def ctc_target_text(text: str, english_targets: str) -> str:
    if english_targets == "keep":
        return text
    return " ".join(w for w in text.split() if not LATIN.search(w))


def load_rows(manifest_path, n, sp, english_targets, include_monolingual=False,
              min_dur=1.0, max_dur=10.0, seed=42):
    """Code-mixed utterances (optionally plus pure-L1 ones), CTC-feasible after target rewriting."""
    rows = [json.loads(line) for line in open(manifest_path, encoding="utf-8")]
    pool = [r for r in rows
            if (r["is_code_mixed"] or include_monolingual) and min_dur <= r["duration"] <= max_dur]

    keep = []
    for r in pool:
        text = ctc_target_text(r["text"], english_targets)
        target_len = len(sp.encode(text, out_type=int))
        if target_len == 0:
            continue
        # 2x margin covers blanks required between repeated tokens.
        if r["duration"] * ENCODER_FRAMES_PER_SEC < 2 * target_len:
            continue
        keep.append({**r, "text": text})

    random.Random(seed).shuffle(keep)
    return keep[:n], len(pool) - len(keep)


def run(args):
    device = resolve_device(args.device)
    print(f"Device: {device}   batch size: {args.batch_size}")

    preprocessor, encoder, ctc_decoder, cfg_dict = load_backbone(args.lang, device=device)
    n_layers, dim = len(encoder.layers), getattr(encoder, "d_model", 512)

    stacks, names = build_experts(args.expert, n_layers, dim, backbone_lang=args.lang)
    moa = MixtureOfAdapters(stacks, dim, names, router_hidden=args.router_hidden,
                            router_context=args.router_context, temperature=args.temperature)
    attach(encoder, moa).to(device).train()
    print(f"Experts: {names}   trainable router params: "
          f"{sum(p.numel() for p in moa.routers.parameters()):,}")

    sp, lang_mask = build_language_mask(cfg_dict, args.lang)
    lang_mask = lang_mask.to(device)
    blank_id = len(lang_mask) - 1

    optimizer = torch.optim.Adam(moa.routers.parameters(), lr=args.lr)

    rows, prefiltered = load_rows(MANIFESTS[args.lang], args.n_samples, sp,
                                  args.english_targets, args.include_monolingual)
    print(f"Training set: {len(rows)} utterances "
          f"({prefiltered} dropped as empty/CTC-infeasible), english targets: {args.english_targets}")

    log_f = open(args.log_path, "w", encoding="utf-8")
    start = time.time()
    step = epoch = utterances = skipped = 0

    try:
        while (time.time() - start) < args.max_seconds:
            epoch += 1
            for batch in make_batches(rows, args.batch_size, seed=epoch):
                if (time.time() - start) >= args.max_seconds:
                    break

                waves, wave_lens, targets, target_lens = collate(batch, sp, device)

                feats, feat_lens = preprocessor(input_signal=waves, length=wave_lens)
                enc_out, enc_lens = encoder(audio_signal=feats, length=feat_lens)
                logits = ctc_decoder(encoder_output=enc_out)
                log_probs = torch.log_softmax(logits[:, :, lang_mask], dim=-1)

                per_sample = F.ctc_loss(
                    log_probs.transpose(0, 1), targets, enc_lens, target_lens,
                    blank=blank_id, reduction="none", zero_infinity=True,
                )
                valid = torch.isfinite(per_sample) & (per_sample > 0)
                skipped += int((~valid).sum().item())
                if not valid.any():
                    continue
                ctc = per_sample[valid].mean()
                balance = moa.load_balancing_loss()
                loss = ctc + args.balance_weight * balance

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(moa.routers.parameters(), max_norm=5.0)
                optimizer.step()

                step += 1
                utterances += int(valid.sum().item())
                elapsed = time.time() - start
                with torch.no_grad():
                    w = moa.routing_weights()
                    real = torch.arange(w.shape[1], device=w.device)[None, :] < enc_lens[:, None]
                    share = w[real].mean(dim=0).tolist()
                log_f.write(json.dumps({
                    "step": step, "epoch": epoch, "batch_size": len(batch),
                    "loss": loss.item(), "ctc": ctc.item(), "balance": balance.item(),
                    "expert_share": dict(zip(names, share)), "elapsed_seconds": elapsed,
                }) + "\n")
                log_f.flush()

                if step % 20 == 0:
                    shares = " ".join(f"{n}={s:.2f}" for n, s in zip(names, share))
                    print(f"step {step:6d} | epoch {epoch} | ctc {ctc.item():8.3f} "
                          f"| bal {balance.item():.3f} | {shares} | elapsed {elapsed / 60:.1f}min", flush=True)
    finally:
        elapsed = time.time() - start
        print(f"\nStopped after {elapsed:.1f}s, {step} steps, {utterances} utterances, "
              f"{epoch} epoch(s), {skipped} skipped.")
        log_f.close()

        if args.save_to and step:
            Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "routers": moa.router_state_dict(),
                "lang": args.lang, "experts": args.expert, "expert_names": names,
                "router_hidden": args.router_hidden, "router_context": args.router_context,
                "temperature": args.temperature, "balance_weight": args.balance_weight,
                "english_targets": args.english_targets, "lr": args.lr,
                "steps": step, "utterances": utterances, "epochs": epoch,
            }, args.save_to)
            print(f"Router saved -> {args.save_to}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fluidity Learning: train the frame-level router over frozen Stage 1 adapters")
    parser.add_argument("--lang", required=True, choices=["hi", "bn"], help="Backbone (and MUCS pair) to use")
    parser.add_argument("--expert", action="append", required=True,
                        help="'name=stage1_checkpoint.pt' or 'identity'; repeat for each expert (>= 2)")
    parser.add_argument("--max-seconds", type=float, default=10800)
    parser.add_argument("--n-samples", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balance-weight", type=float, default=0.01, help="alpha on the load-balancing loss")
    parser.add_argument("--router-hidden", type=int, default=128)
    parser.add_argument("--router-context", type=int, default=5, help="Frames of temporal context (odd)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--english-targets", choices=["drop", "keep"], default="drop")
    parser.add_argument("--include-monolingual", action="store_true",
                        help="Also train on pure-L1 utterances (guards monolingual accuracy)")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--log-path", default="/tmp/router_training_log.jsonl")
    parser.add_argument("--save-to", default=None, help="Path to write the trained router weights")
    run(parser.parse_args())

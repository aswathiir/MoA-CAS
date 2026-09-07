"""
scripts/train_profile_adapter.py
------------------------------------
Stage 1 "Profile Learning": fine-tune a bottleneck adapter inserted into
a frozen ai4bharat NeMo IndicConformer encoder, on monolingual-language
utterances already sitting in a preprocessing-pipeline manifest (no new
data prep needed — filters on the n_words_en/duration fields the
pipeline already computed).

Usage
-----
    poetry run python scripts/train_profile_adapter.py --lang hi --max-seconds 18000
    poetry run python scripts/train_profile_adapter.py --lang bn --max-seconds 3600

Safety notes (learned from a wasted 5-hour run)
------------------------------------------------
A single too-short utterance can produce infinite CTC loss (not enough
encoder frames to align its target text), which then corrupts every
adapter weight via optimizer.step() and silently wastes every step after
it computing nan forever — no crash, no warning. This script guards
against that: skips CTC-infeasible samples before computing loss, skips
any non-finite loss as defense in depth, and clips gradients.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moa_cas.adapters.bottleneck import insert_into_encoder
from moa_cas.adapters.language_mask import build as build_language_mask
from moa_cas.adapters.nemo_backbone import load_backbone

MANIFESTS = {
    "hi": "data/manifests/mucs_hi-en_train.jsonl",
    "bn": "data/manifests/mucs_bn-en_train.jsonl",
}


def load_pure_language_rows(manifest_path, n, min_dur=1.0, max_dur=10.0, seed=42):
    rows = [json.loads(line) for line in open(manifest_path, encoding="utf-8")]
    pure = [r for r in rows if r["n_words_en"] == 0 and min_dur <= r["duration"] <= max_dur]
    random.Random(seed).shuffle(pure)
    return pure[:n]


def run(lang: str, max_seconds: float, n_samples: int, lr: float, bottleneck: int, log_path: Path):
    print(f"Loading {lang} NeMo backbone...")
    preprocessor, encoder, ctc_decoder, cfg_dict = load_backbone(lang)
    adapters = insert_into_encoder(encoder, bottleneck=bottleneck)
    sp, lang_mask = build_language_mask(cfg_dict["tokenizer"]["model_path"], cfg_dict)

    from nemo.collections.asr.losses.ctc import CTCLoss
    ctc_loss_fn = CTCLoss(num_classes=len(lang_mask) - 1, reduction="mean_batch")
    optimizer = torch.optim.Adam(adapters.parameters(), lr=lr)

    rows = load_pure_language_rows(MANIFESTS[lang], n=n_samples)
    print(f"Training set: {len(rows)} pure-{lang} utterances (real audio already on disk)")

    log_f = open(log_path, "w", encoding="utf-8")
    start = time.time()
    step = epoch = skipped = 0

    try:
        while (time.time() - start) < max_seconds:
            epoch += 1
            for row in rows:
                if (time.time() - start) >= max_seconds:
                    break

                wav_np, sr = sf.read(row["audio_filepath"])
                wav = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)
                wav_len = torch.tensor([wav.shape[-1]])

                target_ids = sp.encode(row["text"], out_type=int)
                if not target_ids:
                    continue
                targets = torch.tensor([target_ids], dtype=torch.long)
                target_lens = torch.tensor([len(target_ids)])

                feats, feat_lens = preprocessor(input_signal=wav, length=wav_len)
                enc_out, enc_lens = encoder(audio_signal=feats, length=feat_lens)

                if enc_lens.item() < 2 * target_lens.item():
                    skipped += 1
                    continue

                logits = ctc_decoder(encoder_output=enc_out)
                log_probs = torch.log_softmax(logits[:, :, lang_mask], dim=-1)
                loss = ctc_loss_fn(
                    log_probs=log_probs, targets=targets,
                    input_lengths=enc_lens, target_lengths=target_lens,
                )

                if not torch.isfinite(loss):
                    skipped += 1
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapters.parameters(), max_norm=5.0)
                optimizer.step()

                step += 1
                elapsed = time.time() - start
                log_f.write(json.dumps({
                    "step": step, "epoch": epoch, "utt_id": row["utt_id"],
                    "loss": loss.item(), "elapsed_seconds": elapsed,
                }) + "\n")
                log_f.flush()

                if step % 20 == 0:
                    print(f"step {step:6d} | epoch {epoch} | loss {loss.item():8.3f} "
                          f"| elapsed {elapsed / 60:.1f}min", flush=True)
    finally:
        elapsed = time.time() - start
        print(f"\nStopped after {elapsed:.1f}s, {step} steps, {epoch} epoch(s), {skipped} skipped.")
        if step:
            print(f"Throughput: {step / elapsed:.3f} utt/sec")
        log_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Learning: adapter fine-tuning on a frozen NeMo IndicConformer encoder")
    parser.add_argument("--lang", required=True, choices=["hi", "bn"])
    parser.add_argument("--max-seconds", type=float, default=18000)
    parser.add_argument("--n-samples", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--log-path", default="/tmp/profile_adapter_training_log.jsonl")
    args = parser.parse_args()
    run(args.lang, args.max_seconds, args.n_samples, args.lr, args.bottleneck, Path(args.log_path))

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
    # CPU, small run
    poetry run python scripts/train_profile_adapter.py --lang hi \
        --max-seconds 3600 --save-to checkpoints/hi_adapter.pt

    # GPU, batched
    poetry run python scripts/train_profile_adapter.py --lang hi \
        --device cuda --batch-size 16 --max-seconds 18000 \
        --save-to checkpoints/hi_adapter.pt

Then measure what it actually did with scripts/evaluate_adapter.py —
a falling loss curve is not evidence of better transcription. On this
project a smoothly converging loss once masked an 83%-wrong target
mapping for hours; only WER exposed it.

CTC degenerate cases
--------------------
An utterance whose transcript needs more encoder frames than its audio
provides cannot be aligned, and yields infinite loss. Left unchecked
that propagates through optimizer.step() and silently corrupts every
adapter weight, turning every subsequent step into nan. Three guards:
a pre-filter on estimated frames, `zero_infinity=True` on the loss, and
gradient clipping.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moa_cas.adapters.bottleneck import insert_into_encoder
from moa_cas.adapters.language_mask import build as build_language_mask
from moa_cas.adapters.nemo_backbone import load_backbone

MANIFESTS = {
    "hi": "data/manifests/mucs_hi-en_train.jsonl",
    "bn": "data/manifests/mucs_bn-en_train.jsonl",
}

# Conformer encoder emits 25 frames/sec (10ms window stride, 4x subsampling).
ENCODER_FRAMES_PER_SEC = 25.0


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_rows(manifest_path, n, sp, min_dur=1.0, max_dur=10.0, seed=42):
    """Pure-L1 utterances, pre-filtered to those CTC can actually align."""
    rows = [json.loads(line) for line in open(manifest_path, encoding="utf-8")]
    pure = [r for r in rows if r["n_words_en"] == 0 and min_dur <= r["duration"] <= max_dur]

    keep = []
    for r in pure:
        target_len = len(sp.encode(r["text"], out_type=int))
        if target_len == 0:
            continue
        # 2x margin covers blanks required between repeated tokens.
        if r["duration"] * ENCODER_FRAMES_PER_SEC < 2 * target_len:
            continue
        keep.append(r)

    random.Random(seed).shuffle(keep)
    return keep[:n], len(pure) - len(keep)


def make_batches(rows, batch_size, seed=42):
    """Bucket by duration so padding within a batch stays small, then
    shuffle batch order so training still sees varied lengths."""
    ordered = sorted(rows, key=lambda r: r["duration"])
    batches = [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]
    random.Random(seed).shuffle(batches)
    return batches


def collate(batch, sp, device):
    waves, wave_lens, targets, target_lens = [], [], [], []
    for row in batch:
        wav_np, _ = sf.read(row["audio_filepath"])
        wav = torch.tensor(wav_np, dtype=torch.float32)
        waves.append(wav)
        wave_lens.append(wav.shape[-1])
        ids = sp.encode(row["text"], out_type=int)
        targets.append(torch.tensor(ids, dtype=torch.long))
        target_lens.append(len(ids))

    max_len = max(wave_lens)
    padded = torch.zeros(len(waves), max_len, dtype=torch.float32)
    for i, w in enumerate(waves):
        padded[i, :w.shape[-1]] = w

    return (
        padded.to(device),
        torch.tensor(wave_lens, dtype=torch.long, device=device),
        torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(device),
        torch.tensor(target_lens, dtype=torch.long, device=device),
    )


def run(lang, max_seconds, n_samples, lr, bottleneck, log_path, save_to, device, batch_size):
    device = resolve_device(device)
    print(f"Device: {device}   batch size: {batch_size}")

    preprocessor, encoder, ctc_decoder, cfg_dict = load_backbone(lang, device=device)
    adapters = insert_into_encoder(encoder, bottleneck=bottleneck).to(device)
    sp, lang_mask = build_language_mask(cfg_dict, lang)
    lang_mask = lang_mask.to(device)
    blank_id = len(lang_mask) - 1

    optimizer = torch.optim.Adam(adapters.parameters(), lr=lr)

    rows, prefiltered = load_rows(MANIFESTS[lang], n_samples, sp)
    print(f"Training set: {len(rows)} pure-{lang} utterances "
          f"({prefiltered} dropped as CTC-infeasible)")

    log_f = open(log_path, "w", encoding="utf-8")
    start = time.time()
    step = epoch = utterances = skipped = 0

    try:
        while (time.time() - start) < max_seconds:
            epoch += 1
            for batch in make_batches(rows, batch_size, seed=epoch):
                if (time.time() - start) >= max_seconds:
                    break

                waves, wave_lens, targets, target_lens = collate(batch, sp, device)

                feats, feat_lens = preprocessor(input_signal=waves, length=wave_lens)
                enc_out, enc_lens = encoder(audio_signal=feats, length=feat_lens)
                logits = ctc_decoder(encoder_output=enc_out)
                log_probs = torch.log_softmax(logits[:, :, lang_mask], dim=-1)

                # torch expects (T, B, C); zero_infinity neutralises any
                # unalignable sample that slipped past the pre-filter.
                per_sample = F.ctc_loss(
                    log_probs.transpose(0, 1), targets,
                    enc_lens, target_lens,
                    blank=blank_id, reduction="none", zero_infinity=True,
                )
                valid = torch.isfinite(per_sample) & (per_sample > 0)
                skipped += int((~valid).sum().item())
                if not valid.any():
                    continue
                loss = per_sample[valid].mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapters.parameters(), max_norm=5.0)
                optimizer.step()

                step += 1
                utterances += int(valid.sum().item())
                elapsed = time.time() - start
                log_f.write(json.dumps({
                    "step": step, "epoch": epoch, "batch_size": len(batch),
                    "loss": loss.item(), "elapsed_seconds": elapsed,
                }) + "\n")
                log_f.flush()

                if step % 20 == 0:
                    print(f"step {step:6d} | epoch {epoch} | loss {loss.item():8.3f} "
                          f"| {utterances} utts | elapsed {elapsed / 60:.1f}min", flush=True)
    finally:
        elapsed = time.time() - start
        print(f"\nStopped after {elapsed:.1f}s, {step} steps, {utterances} utterances, "
              f"{epoch} epoch(s), {skipped} skipped.")
        if utterances:
            print(f"Throughput: {utterances / elapsed:.3f} utt/sec")
        log_f.close()

        if save_to and step:
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "adapters": {k: v.cpu() for k, v in adapters.state_dict().items()},
                "lang": lang, "bottleneck": bottleneck, "lr": lr,
                "steps": step, "utterances": utterances, "epochs": epoch,
            }, save_to)
            print(f"Adapter saved -> {save_to}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Learning: adapter fine-tuning on a frozen NeMo IndicConformer encoder")
    parser.add_argument("--lang", required=True, choices=["hi", "bn"])
    parser.add_argument("--max-seconds", type=float, default=18000)
    parser.add_argument("--n-samples", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--log-path", default="/tmp/profile_adapter_training_log.jsonl")
    parser.add_argument("--save-to", default=None, help="Path to write the trained adapter weights")
    args = parser.parse_args()
    run(args.lang, args.max_seconds, args.n_samples, args.lr, args.bottleneck,
        Path(args.log_path), args.save_to, args.device, args.batch_size)

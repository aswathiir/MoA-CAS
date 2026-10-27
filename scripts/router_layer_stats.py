"""
scripts/router_layer_stats.py
--------------------------------
Per-layer routing statistics for a trained router. The demo and
evaluate_router.py report the adapter share averaged over the 17 encoder
layers; this script looks inside that average and records, per layer,
how often the router sends a frame through the adapter.

Writes one row per layer: mean adapter weight, share of frames above 0.9
("on") and below 0.1 ("off"). Prints how many decisions are hard and how
many layers are on per frame.

Usage
-----
    python scripts/router_layer_stats.py --out results/router_eval/router_hi_id_layers.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from demo_server import Demo


def main(args):
    demo = Demo(str(ROOT / args.router_checkpoint), args.manifest, args.limit, "cpu")
    demo.moa.fixed_weights = None
    per_utt = []
    for r in demo.samples.values():
        wav, _ = sf.read(ROOT / r["audio_filepath"], dtype="float32")
        x = torch.tensor(wav).unsqueeze(0)
        with torch.no_grad():
            feats, feat_lens = demo.preprocessor(input_signal=x, length=torch.tensor([x.shape[-1]]))
            _, enc_lens = demo.encoder(audio_signal=feats, length=feat_lens)
        per_utt.append(demo.moa.routing_weights(per_layer=True)[:, 0, :int(enc_lens[0]), 0].numpy())
    w = np.concatenate(per_utt, axis=1)  # (layers, frames)

    with open(ROOT / args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "mean_adapter_weight", "share_on", "share_off", "n_frames"])
        for layer, row in enumerate(w, 1):
            writer.writerow([layer, round(float(row.mean()), 4), round(float(np.mean(row > 0.9)), 4),
                             round(float(np.mean(row < 0.1)), 4), row.size])

    on_per_frame = (w > 0.5).sum(axis=0)
    print(f"{len(per_utt)} utterances, {w.shape[1]} frames, overall adapter share {w.mean():.3f}")
    print(f"hard decisions (>0.9 or <0.1): {np.mean((w > 0.9) | (w < 0.1)):.3f}")
    print(f"layers on per frame: mean {on_per_frame.mean():.2f}, min {on_per_frame.min()}, max {on_per_frame.max()}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-layer routing statistics of a trained router")
    parser.add_argument("--router-checkpoint", default="checkpoints/router_hi_id.pt")
    parser.add_argument("--manifest", default="data/manifests/mucs_hi-en_test.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", default="results/router_eval/router_hi_id_layers.csv")
    main(parser.parse_args())

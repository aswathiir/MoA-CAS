"""
scripts/demo_server.py
-------------------------
Local demo: transcribe one utterance three ways on the same frozen
backbone and show the difference.

  base     the untouched NeMo IndicConformer          (weights [0, 1])
  adapter  backbone + Stage 1 Hindi adapter only      (weights [1, 0])
  router   backbone + MoA-CAS router choosing per frame (learned weights)

All three share one loaded model: MixtureOfAdapters.fixed_weights pins
the mixture for the first two, so the only thing that differs between
columns is the routing. Input is either a MUCS test utterance (with its
reference, so WER is shown) or audio uploaded/recorded in the browser
(decoded to 16 kHz mono client-side, sent as raw float32).

Usage
-----
    python scripts/demo_server.py
    # then open http://127.0.0.1:8000

    python scripts/demo_server.py --router-checkpoint checkpoints/router_hi_id.pt --port 8000
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from moa_cas.adapters.decode import greedy_ctc_decode
from moa_cas.adapters.language_mask import build as build_language_mask
from moa_cas.adapters.nemo_backbone import load_backbone
from moa_cas.adapters.router import MixtureOfAdapters, attach, build_experts
from moa_cas.metrics import compute_cer, compute_wer, normalize
import re

from score_transliteration_aware import _edit_distance, romanise, romanise_word

SAMPLE_RATE = 16000
FRAMES_PER_SEC = 25.0
MODES = {"base": [0.0, 1.0], "adapter": [1.0, 0.0], "router": None}


LATIN = re.compile(r"[a-z]")
SCRIPT_TOLERANCE = 0.4  # edit distance allowed, as a share of the romanised reference word


def same_word_other_script(ref_word: str, hyp_word: str) -> bool:
    """
    True when an English (Latin-script) reference word was written in
    Devanagari by the model and the two sound alike: `insert` vs `इंसर्ट`.
    Display-only. WER still counts these as errors, because the backbone
    has no Latin tokens and exact-match scoring can't credit them.
    Tolerance 0.4 was picked by inspecting all Latin-reference
    substitutions in the 100 demo samples: it accepts slide/स्लाइड and
    download/डाउनलोड but leaves short words like pane/पेन red.
    """
    if not LATIN.search(ref_word) or LATIN.search(hyp_word):
        return False
    a, b = romanise_word(ref_word), romanise_word(hyp_word)
    return bool(a) and _edit_distance(a, b) <= max(1, int(len(a) * SCRIPT_TOLERANCE))


def align_words(ref: str, hyp: str):
    """
    Word-level Levenshtein alignment -> per-hypothesis-word tags plus
    deletion count. Tags: ok, sub, ins, and script (a substitution where
    the hypothesis is the reference's English word written in Devanagari).
    """
    r, h = normalize(ref).split(), normalize(hyp).split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    tags, deletions, i, j = [], 0, len(r), len(h)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            if r[i - 1] == h[j - 1]:
                tags.append("ok")
            else:
                tags.append("script" if same_word_other_script(r[i - 1], h[j - 1]) else "sub")
            i, j = i - 1, j - 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            tags.append("ins")
            j -= 1
        else:
            deletions += 1
            i -= 1
    return list(reversed(tags)), deletions


class Demo:
    def __init__(self, router_checkpoint: str, manifest: str, n_samples: int, device: str):
        self.device = torch.device(device)
        cfg = torch.load(router_checkpoint, map_location="cpu", weights_only=False)
        self.router_meta = {k: v for k, v in cfg.items() if k != "routers"}

        self.preprocessor, self.encoder, self.ctc_decoder, cfg_dict = load_backbone(cfg["lang"], device=self.device)
        n_layers, dim = len(self.encoder.layers), self.encoder.d_model
        stacks, names = build_experts(cfg["experts"], n_layers, dim, backbone_lang=cfg["lang"])
        if names[-1] != "identity" or len(names) != 2:
            raise ValueError(f"Demo expects experts [adapter, identity], got {names}")
        self.names = names
        self.moa = MixtureOfAdapters(stacks, dim, names, router_hidden=cfg["router_hidden"],
                                     router_context=cfg["router_context"], temperature=cfg["temperature"])
        self.moa.routers.load_state_dict(cfg["routers"])
        attach(self.encoder, self.moa).to(self.device).eval()

        self.sp, lang_mask = build_language_mask(cfg_dict, cfg["lang"])
        self.lang_mask = lang_mask.to(self.device)
        self.blank_id = len(lang_mask) - 1

        rows = [json.loads(line) for line in open(ROOT / manifest, encoding="utf-8")][:n_samples]
        self.samples = {r["utt_id"]: r for r in rows if (ROOT / r["audio_filepath"]).exists()}
        self.params = {
            "backbone": sum(p.numel() for m in (self.encoder, self.ctc_decoder) for p in m.parameters()),
            "adapter": sum(p.numel() for p in self.moa.experts.parameters()),
            "router": sum(p.numel() for p in self.moa.routers.parameters()),
        }

    @torch.no_grad()
    def run(self, wav: np.ndarray, mode: str):
        self.moa.fixed_weights = MODES[mode]
        x = torch.tensor(wav, dtype=torch.float32, device=self.device).unsqueeze(0)
        start = time.time()
        feats, feat_lens = self.preprocessor(input_signal=x, length=torch.tensor([x.shape[-1]], device=self.device))
        enc_out, enc_lens = self.encoder(audio_signal=feats, length=feat_lens)
        log_probs = torch.log_softmax(self.ctc_decoder(encoder_output=enc_out)[:, :, self.lang_mask], dim=-1)
        latency = time.time() - start

        n = int(enc_lens[0])
        layer_weights = self.moa.routing_weights(per_layer=True)[:, 0, :n, 0]  # (L, T), weight on the adapter
        weights = layer_weights.mean(dim=0).tolist()
        ids = log_probs[0, :n].argmax(dim=-1).tolist()

        # Emitted tokens with the frame they were emitted on (greedy CTC collapse).
        tokens, previous = [], None
        for t, tok in enumerate(ids):
            if tok != self.blank_id and tok != previous:
                tokens.append((t, self.sp.id_to_piece(tok)))
            previous = tok

        # Group tokens into words; a word's routing weight is the mean over its frames.
        words = []
        for idx, (t, piece) in enumerate(tokens):
            if piece.startswith("▁") or not words:
                words.append({"start": t, "text": piece.lstrip("▁")})
            else:
                words[-1]["text"] += piece
        for k, w in enumerate(words):
            end = words[k + 1]["start"] if k + 1 < len(words) else n
            span = weights[w["start"]:max(end, w["start"] + 1)]
            w["adapter_weight"] = round(statistics.mean(span), 3)
            w["time"] = round(w["start"] / FRAMES_PER_SEC, 2)
        words = [w for w in words if w["text"]]

        return {
            "hypothesis": greedy_ctc_decode(log_probs, self.sp, self.blank_id),
            "words": words,
            "frame_weights": [round(v, 3) for v in weights],
            "layer_weights": [[round(v, 2) for v in row] for row in layer_weights.tolist()],
            "mean_adapter_weight": round(statistics.mean(weights), 3) if weights else None,
            "latency_ms": round(1000 * latency),
        }

    def transcribe(self, wav: np.ndarray, reference: str = None, modes=("base", "adapter", "router")):
        out = {"duration": round(len(wav) / SAMPLE_RATE, 2), "results": {}}
        for mode in modes:
            r = self.run(wav, mode)
            if reference:
                r["wer"] = round(compute_wer(reference, r["hypothesis"]), 4)
                r["cer"] = round(compute_cer(reference, r["hypothesis"]), 4)
                r["wer_translit"] = round(compute_wer(romanise(reference), romanise(r["hypothesis"])), 4)
                r["tags"], r["deletions"] = align_words(reference, r["hypothesis"])
            out["results"][mode] = r
        return out


def summarise_csv(path: Path):
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return {"n": len(rows), "wer": round(100 * statistics.mean(float(r["wer"]) for r in rows), 2),
            "cer": round(100 * statistics.mean(float(r["cer"]) for r in rows), 2), "file": str(path.relative_to(ROOT))}


def build_info(demo: Demo) -> dict:
    res = ROOT / "results"
    return {
        "router": demo.router_meta, "experts": demo.names, "params": demo.params,
        "benchmarks": {
            "base": summarise_csv(res / "adapter_eval" / "nemo_baseline_fixed.csv"),
            "adapter": summarise_csv(res / "adapter_eval" / "eval_adapt_mixed_ckpt0917.csv"),
            "router": summarise_csv(res / "router_eval" / "router_hi_id_mixed.csv"),
        },
    }


def build_samples(demo: Demo) -> list:
    return [{
        "utt_id": r["utt_id"], "text": r["text"], "duration": r["duration"],
        "en_fraction": round(r["n_words_en"] / max(r["n_words"], 1), 2),
        "is_code_mixed": r["is_code_mixed"], "n_cs_points": r["n_cs_points"],
    } for r in demo.samples.values()]


def transcribe_sample(demo: Demo, utt_id: str) -> dict:
    row = demo.samples[utt_id]
    wav, sr = sf.read(ROOT / row["audio_filepath"], dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"manifest audio is {sr} Hz, expected {SAMPLE_RATE}")
    return {"utt_id": utt_id, "reference": row["text"], **demo.transcribe(wav, row["text"])}


def create_app(demo: Demo) -> FastAPI:
    app = FastAPI(title="MoA-CAS demo")

    @app.get("/")
    def index():
        return FileResponse(ROOT / "demo" / "index.html")

    @app.get("/api/info")
    def info():
        return build_info(demo)

    @app.get("/api/samples")
    def samples():
        return build_samples(demo)

    @app.get("/api/audio/{utt_id}")
    def audio(utt_id: str):
        row = demo.samples.get(utt_id)
        if not row:
            raise HTTPException(404, "unknown utterance")
        return FileResponse(ROOT / row["audio_filepath"], media_type="audio/wav")

    @app.post("/api/transcribe/{utt_id}")
    def transcribe(utt_id: str):
        if utt_id not in demo.samples:
            raise HTTPException(404, "unknown utterance")
        return transcribe_sample(demo, utt_id)

    @app.post("/api/transcribe_audio")
    async def transcribe_audio(request: Request):
        """Body: raw little-endian float32 mono samples at 16 kHz (the page resamples)."""
        body = await request.body()
        if not body or len(body) % 4:
            raise HTTPException(400, "expected raw float32 samples")
        wav = np.frombuffer(body, dtype="<f4").copy()
        if not 0.3 * SAMPLE_RATE <= len(wav) <= 30 * SAMPLE_RATE:
            raise HTTPException(400, "audio must be between 0.3 and 30 seconds")
        reference = request.headers.get("x-reference")
        reference = reference.encode("latin-1").decode("utf-8") if reference else None
        return {"utt_id": None, "reference": reference, **demo.transcribe(wav, reference or None)}

    @app.exception_handler(ValueError)
    def value_error(_, exc):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="MoA-CAS local demo: base vs adapter vs router")
    parser.add_argument("--router-checkpoint", default="checkpoints/router_hi_id.pt")
    parser.add_argument("--manifest", default="data/manifests/mucs_hi-en_test.jsonl")
    parser.add_argument("--n-samples", type=int, default=100, help="First N test utterances offered as samples")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    demo = Demo(str(ROOT / args.router_checkpoint), args.manifest, args.n_samples, args.device)
    print(f"Loaded {len(demo.samples)} samples; experts {demo.names}; open http://{args.host}:{args.port}")
    uvicorn.run(create_app(demo), host=args.host, port=args.port)

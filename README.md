# MoA-CAS — Mixture-of-Adapters for Code-Switched ASR

Speech recognition for people who switch between Hindi and English inside one sentence:

> "लिबर ऑफिस **impress** में एक प्रस्तुति **document** बनाना..."

Team C-15. Supervisor: Dr. Ganga Gowri.

---

## The idea in one minute

1. **Start with a strong Indian-language speech model and never change it.** The model is IndicConformer, and it stays *frozen*: none of its weights ever move. That way it can't forget what it already knows.
2. **Add small "adapter" layers inside it.** Each one is a specialist, for example a Hindi adapter. Only these small layers are trained, about 1% of the model's size.
3. **Add a "router" that picks the right specialist for every 40 ms of audio.** When the speaker switches language mid-sentence, the router can switch adapters at that moment.

Training happens in two stages:

| | Stage 1 — Profile Learning | Stage 2 — Fluidity Learning |
|---|---|---|
| Backbone | frozen | frozen |
| Adapters | **trained**, one language at a time | frozen |
| Router | off | **trained** on code-mixed speech |
| Loss | CTC | CTC + load balancing |

```
audio ─► frozen encoder layer ─┬─► Hindi adapter ─┐
                               ├─► other adapter ─┼─ weighted by router ─► next layer ... ─► frozen decoder ─► text
                               └─► router (w_t) ──┘
```

---

## Where the project stands

| Part | Status | Evidence |
|---|---|---|
| Data pipeline (MUCS 2021 Hindi-English and Bengali-English, IndicVoices) | ✅ Done | [docs/02](docs/02_DATA_PREPROCESSING.md), `data/manifests/` |
| Baseline benchmark | ✅ Done | [docs/01](docs/01_BASELINE_AND_PROBLEM_ANALYSIS.md), `results/*.csv` |
| Trainable backbone (NeMo IndicConformer, 5 loading bugs fixed) | ✅ Done | [docs/04](docs/04_HOW_THE_ADAPTER_SYSTEM_WORKS.md) |
| Hindi adapter (Stage 1) | ✅ Trained and measured | [docs/03](docs/03_PROFILE_LEARNING_ADAPTERS.md), `results/adapter_eval/` |
| Router (Stage 2) | ✅ Trained (38,531 steps, 5 epochs) and measured. It beats the base model but **not yet the adapter alone**. | [docs/05](docs/05_ROUTER.md), `results/router_eval/` |
| Bengali adapter | ⏳ Code ready, never run | — |
| English adapter | ⛔ Blocked: no English speech data, and the model can't write Latin letters | [docs/05 §3](docs/05_ROUTER.md) |
| Accent adapters | ⏳ Planned for next semester | — |

### Key results so far (each recomputed from the CSVs by `tests/test_results.py`)

> **Corrected on 2026-10-26.** The scoring code used to delete Hindi vowel signs (ि ी ं ्) before comparing words, which made every error rate 2–7.5 points too low. All results were rescored from the saved transcripts. Every comparison still points the same way and is still significant. Earlier figures (for example 7.93% and 57.14% in the review deck) are the old, under-counted values. Details: [THESIS_REFERENCE §5, C6](docs/THESIS_REFERENCE.md).

| What was measured (100 utterances each) | Word error rate |
|---|---|
| Original model on clean Hindi (IndicVoices) | **11.72%** |
| Original model on Hindi-English tutorials (MUCS) | **59.69%** |
| NeMo backbone, no adapter, Hindi-only part of MUCS | 72.28% |
| Same, **with the Hindi adapter** | **60.35%**: −11.9 points, 51 better / 25 worse, t = −4.45 |
| NeMo backbone, no adapter, full mixed MUCS | 58.06% |
| Same, **with the Hindi adapter** | **48.72%**: −9.3 points, t = −3.83 |
| Same, **with the MoA-CAS router** | **52.94%**: −5.1 points vs base (t = −2.59), but +4.2 vs adapter alone (t = +2.68) |
| Router on the Hindi-only part | 64.78%: −7.5 vs base (t = −3.57), +4.4 vs adapter alone (t = +2.85) |

What these numbers mean, in plain words:
- **Most of the 12% → 60% gap is not code-switching.** Even pure-Hindi sentences from the tutorial corpus score ~59%. Most of the gap comes from the domain (software tutorials, recording conditions) and from how the corpus is cut into segments ([docs/01 §4](docs/01_BASELINE_AND_PROBLEM_ANALYSIS.md)).
- **The adapter genuinely helps,** but mainly by adapting to this corpus and its vocabulary. It doesn't yet handle English ([docs/03](docs/03_PROFILE_LEARNING_ADAPTERS.md)).
- **The router works, but it hasn't learned *language* routing yet.** It splits the signal roughly 46/54 between the adapter and the base model on almost every frame. Its choices don't follow where the English words are (r = 0.02), so it loses to simply applying the adapter everywhere.
- **About 25% of reference words are English in Latin script, and the model has no Latin letters to write them with.** It writes `डॉक्यूमेंट` for `document`. That puts a floor of roughly 25% under word error rate that no adapter or router can remove.

The full list of claims, numbers, sources and caveats is in **[docs/THESIS_REFERENCE.md](docs/THESIS_REFERENCE.md)**.

---

## Setup

Python 3.10+.

```bash
python -m venv venv
```

```bash
source venv/bin/activate
```

```bash
pip install -e ".[adapters]" pytest
```

The NeMo checkpoint is gated. Accept access at
`huggingface.co/ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large`, then log in:

```bash
huggingface-cli login
```

## Running things

**Tests.** Fast, no GPU needed. The checkpoint test is skipped if the model isn't downloaded.

```bash
python -m pytest -q tests
```

**Rebuild a dataset manifest.** Audio is not in git; see [docs/PIPELINE.md](docs/PIPELINE.md).

```bash
python scripts/build_manifest.py --config mucs_hi_en_test
```

**Stage 1: train and evaluate the Hindi adapter.**

```bash
python scripts/train_profile_adapter.py --lang hi --max-seconds 3600 --save-to checkpoints/hi_adapter.pt
```

```bash
python scripts/evaluate_adapter.py --lang hi --manifest data/manifests/mucs_hi-en_test.jsonl --limit 100 --pure-l1-only --adapter-checkpoint checkpoints/hi_adapter.pt
```

**Stage 2: train and evaluate the router.**

```bash
python scripts/train_router.py --lang hi --expert hi=checkpoints/hi_adapter.pt --expert identity --save-to checkpoints/router_hi_id.pt
```

```bash
python scripts/evaluate_router.py --lang hi --expert hi=checkpoints/hi_adapter.pt --expert identity --manifest data/manifests/mucs_hi-en_test.jsonl --limit 100 --router-checkpoint checkpoints/router_hi_id.pt
```

**Live demo: base vs adapter vs router, side by side.**

```bash
python scripts/demo_server.py
```

Then open http://127.0.0.1:8000. Pick one of the 100 MUCS test utterances, or upload or record your own audio. The demo shows:
- what each system transcribes, with every word marked correct, wrong or extra against the reference;
- WER and CER, plus an optional transliteration-aware score;
- a frame-by-frame chart of how much the router sent through the Hindi adapter.

All three systems run on one loaded model. They differ only in the routing weights: base = [0, 1], adapter = [1, 0], router = learned. The demo reads `checkpoints/router_hi_id.pt` and `checkpoints/hi_adapter.pt`.

**Hosted (static) copy of the demo.** Vercel and other static hosts can't run the model, so export the precomputed results for the 100 samples (about 18 MB, including audio) and deploy that folder:

```bash
python scripts/export_static_demo.py --out demo_static
```

```bash
cd demo_static && npx vercel deploy --prod
```

The hosted page shows exactly what the live server shows for those samples. Uploading or recording your own audio works only with the local server.

**Score while forgiving script mismatch** (`document` ≈ `डॉक्यूमेंट`):

```bash
python scripts/score_transliteration_aware.py results/adapter_eval/eval_adapt_mixed_ckpt0917.csv
```

---

## Repository map

```
src/moa_cas/
  preprocessing/     text tagging (which word is Hindi/English), audio resampling
  pipeline/          builds the JSONL manifests from raw corpora
  datasets/          MUCS and IndicVoices loaders
  adapters/
    nemo_backbone.py   loads the frozen NeMo model (works around 5 bugs)
    language_mask.py   picks Hindi's 256 tokens out of the shared 5,632
    bottleneck.py      the adapter (Stage 1)
    router.py          the router + mixture of adapters (Stage 2)
    decode.py          greedy CTC decoding
  metrics.py         WER / CER
scripts/             runnable entry points (benchmark, train, evaluate, score, demo_server)
demo/index.html      the demo page served by scripts/demo_server.py
data/manifests/      one JSON line per utterance: audio path, text, language counts, speaker
results/             every CSV a reported number comes from
tests/               52 tests, including ones that pin documented numbers to the CSVs
docs/                the write-ups below
```

## Documents — read in this order

| # | Document | What it answers |
|---|---|---|
| 1 | [01_BASELINE_AND_PROBLEM_ANALYSIS](docs/01_BASELINE_AND_PROBLEM_ANALYSIS.md) | How bad is the problem, and what actually causes it? |
| 2 | [02_DATA_PREPROCESSING](docs/02_DATA_PREPROCESSING.md) | What data we use and how it was prepared |
| 3 | [03_PROFILE_LEARNING_ADAPTERS](docs/03_PROFILE_LEARNING_ADAPTERS.md) | Stage 1 results |
| 4 | [04_HOW_THE_ADAPTER_SYSTEM_WORKS](docs/04_HOW_THE_ADAPTER_SYSTEM_WORKS.md) | Stage 1 mechanism, end to end |
| 5 | [05_ROUTER](docs/05_ROUTER.md) | Stage 2: the router, its research basis, how to run it |
| ★ | [THESIS_REFERENCE](docs/THESIS_REFERENCE.md) | Every claim, with its number, source file, test and caveat |

Shorter, earlier notes: [PIPELINE](docs/PIPELINE.md), [WHAT_WAS_DONE](docs/WHAT_WAS_DONE.md), [ADAPTER_TRAINING](docs/ADAPTER_TRAINING.md), [PROGRESS_REPORT](docs/PROGRESS_REPORT.md).

## Main references

1. Gulati et al., *Conformer*, Interspeech 2020.
2. Winata et al., *Learning to Recognize Code-Switched Speech Without Forgetting Monolingual Speech Recognition*, Interspeech 2020.
3. Kulkarni et al., *Adapting the Adapters for Code-Switching in Multilingual ASR*, arXiv:2310.07423, 2023.
4. Zhao et al., *CAMEL: Cross-Attention Enhanced Mixture-of-Experts and Language Bias for Code-Switching Speech Recognition*, arXiv:2412.12760, 2024.
5. Fedus et al., *Switch Transformers*, JMLR 2022.
6. Datasets: MUCS 2021 (OpenSLR); IndicVoices (AI4Bharat, CC BY 4.0).

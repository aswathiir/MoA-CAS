# Thesis Reference — Claims, Evidence, and Caveats

One place to look up every number or claim the thesis might use: where it comes from, how to reproduce it, what checks it, and what it does *not* show.

**Verification status** (re-checked on 2026-10-17):

| Mark | Meaning |
|---|---|
| ✅ **Tested** | Asserted by an automated test; it fails if the evidence changes. |
| ✔️ **Recomputed** | Re-derived from the repository data for this document, but not in a test. |
| 📝 **Recorded** | Comes from a training log or manual analysis that isn't in the repository. Cite it with that qualification. |

Run every check with:

```bash
python -m pytest -q tests
```

Status on 2026-10-26: 49 passed, 0 skipped.

---

## 1. System facts

| # | Claim | Value | Source | Status |
|---|---|---|---|---|
| S1 | Baseline model (benchmark only) | `ai4bharat/indic-conformer-600m-multilingual`, ONNX, **not trainable** | docs/03 §1 | 📝 |
| S2 | Trainable backbone actually used | `ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large` (NeMo), CTC path only | `adapters/nemo_backbone.py` | ✅ `test_language_mask.py` |
| S3 | Encoder size | 17 Conformer layers, d = 512, **115,111,424** parameters; CTC head adds 2,889,729 (118.0 M total) | real checkpoint | ✅ `test_language_mask.py` |
| S4 | Shared CTC vocabulary | 22 languages × 256 = 5,632 tokens + 1 blank; Hindi = indices 1536–1791 | `language_mask.py` | ✅ `test_language_mask.py` |
| S5 | No English output tokens | The only token containing a Latin letter is `<unk>` | real checkpoint | ✅ `test_language_mask.py` |
| S6 | Why string-matching the mask failed | 670 token strings repeat across languages; string matching puts 213 of 256 Hindi indices (83%) in the wrong columns | real checkpoint | ✔️ |
| S7 | Stage 1 trainable parameters | 1,123,904: one 512→64→512 adapter per layer, 0.98% of the encoder | `bottleneck.py` | ✔️ |
| S8 | Stage 2 trainable parameters | 1,190,306 (17 routers) | `router.py`, training log | ✔️ |
| S9 | Total added parameters (adapter + router) | 2,314,210 = **1.96%** of the 118.0 M backbone. This meets the deck's `< 2%` target, only just. | S3, S7, S8 | ✔️ |
| S10 | The backbone is never modified | Encoder and expert parameters get no gradients; only adapters (Stage 1) or routers (Stage 2) do | | ✅ `test_adapters.py`, `test_router.py` |
| S11 | An untrained adapter or router changes nothing | Zero-initialised adapter = identity. Zero-initialised router = exactly uniform weights. Identity expert at full weight = original backbone. | | ✅ |

> **Thesis wording.** The deck describes the backbone as "IndicConformer – 600M". That is the ONNX model used for the *baseline benchmark*. The adapters and router sit on the **118 M-parameter per-language NeMo model** (S2, S3), because the 600 M model can't be trained. State both, and say which one each result uses.

## 2. Data facts

| # | Claim | Value | Source | Status |
|---|---|---|---|---|
| D1 | MUCS Hindi-English train/test sizes | 52,769 / 3,132 utterances | `data/manifests/` | ✔️ |
| D2 | Train and test speakers don't overlap | 520 train vs 30 test speakers, overlap 0 | manifests | ✅ `test_results.py` |
| D3 | Pure-Hindi training pool (0 English words, 1–10 s) | 7,083 utterances | manifests | ✅ `test_results.py` |
| D4 | Share of reference words in Latin script | 25.0% (train), 24.4% (test), 25.4% (the 100-utterance evaluation sample) | manifests | ✔️ |
| D5 | Word-level language tagging is correct | Cross-checked against Biswas et al. (2025) | docs/02 §"Correctness validation" | 📝 + ✅ `test_text.py` |
| D6 | MUCS segments contain parts of neighbouring utterances | Segments tile recordings at whole seconds; 39% of hypotheses start with the previous utterance's words | docs/01 §4c | 📝 |

## 3. Results

All evaluations use the **first 100 utterances** (in manifest order) of the MUCS Hindi-English **test** split, filtered to pure Hindi where stated. Greedy CTC decoding, no language model.

### 3.1 Baseline and problem analysis (docs/01)

| # | Claim | Value | File | Status |
|---|---|---|---|---|
| R1 | ONNX model, IndicVoices Hindi | WER **7.93%**, CER 4.16% | `results/indicvoices_hi_ctc.csv` | ✅ |
| R2 | ONNX model, MUCS Hindi-English | WER **57.14%**, CER 56.13%; 11/100 with WER > 100% | `results/mucs_hi_en_ctc.csv` | ✅ / ✔️ |
| R3 | WER by English share of each utterance (0 / 1–20 / 20–40 / >40%) | 53.90 / 45.77 / 51.54 / 79.71%; Pearson r = 0.333 | R2 joined to the manifest. Share = `n_words_en / (n_words_l1 + n_words_en)`, bin edges inclusive. | ✔️ |
| R4 | Most of the gap is domain, not code-switching | Pure-Hindi MUCS already sits at 53.9% | R3 | ✔️ |

### 3.2 Stage 1 — Hindi adapter (docs/03)

| # | Claim | Value | File | Status |
|---|---|---|---|---|
| R5 | NeMo baseline, mixed (correctness check for the mask) | WER **55.37%**, within 1.8 points of the ONNX 57.14% | `adapter_eval/nemo_baseline_fixed.csv` | ✅ |
| R6 | NeMo baseline, pure Hindi | WER **67.18%**, CER 56.05%. Re-run on 2026-10-17: all 100 hypotheses byte-identical. | `adapter_eval/eval_base_pure.csv` | ✅ |
| R7 | **Adapter (checkpoint on disk), pure Hindi** | WER **52.84%** (−14.34 points), CER 41.58%; 55 better / 23 worse / 22 same; paired t = −4.81; 95% CI [−20.3, −8.4] points | `adapter_eval/eval_adapt_pure_ckpt0917.csv` | ✅ |
| R8 | **Adapter (checkpoint on disk), mixed** | WER **46.60%** (−8.78 points), CER 49.01%; t = −3.56; 95% CI [−13.7, −3.9] points | `adapter_eval/eval_adapt_mixed_ckpt0917.csv` | ✅ |
| R9 | Adapter (16 Sep checkpoint, **no longer on disk**), pure / mixed | 51.81% (53/25/22, t = −4.84) / 45.44% | `adapter_eval/eval_adapt_pure.csv`, `eval_adapt_mixed.csv` | ✅ CSV only. **The model can't be regenerated** (see §5 C1). |
| R10 | How it helps | The baseline emits 0.75 words per reference word; the adapter raises this to 0.85–0.86. Most of the gain is recovered words. | R6, R7, R9 | ✔️ |
| R11 | Script mismatch vs. recognition failure (transliteration-aware CER, mixed) | Baseline 54.40 → 33.88%; adapter (on disk) 49.01 → 24.65%. Phonetic word recall 66.9% → 74.7%. | `scripts/score_transliteration_aware.py` | ✔️ (romaniser ✅ `test_metrics_decode.py`) |
| R12 | Gain on words seen in training vs. unseen | +12.2 points (significant) vs. +6.1 points (n = 182, not significant) | docs/03; analysis script not in repo; computed on the R9 checkpoint | 📝 |

**What can be claimed from R5–R12:** a frozen 118 M-parameter Indic backbone plus 1.1 M adapter parameters gives a significant, held-out, in-domain WER reduction (−14 points on pure Hindi, −9 points on mixed). The gain is concentrated in vocabulary seen during training.

**What can't be claimed:** that this improves *code-switching*. The adapter was trained on monolingual Hindi, and the model can't write English.

### 3.3 Stage 2 — router (docs/05)

| # | Claim | Value | Status |
|---|---|---|---|
| R13 | The router implements the deck's `Σ w_t · A_k(h_t)` at every layer and every frame. It generalises Kulkarni et al.'s two-adapter gate. | Mixture output equals the hand-computed weighted sum | ✅ `test_router.py` |
| R14 | Load-balancing loss behaves correctly | 1.0 when balanced, K when collapsed, ignores padding | ✅ |
| R15 | Runs end to end on the real model and audio | 150 s CPU smoke run: 89 steps, no NaN; evaluation and frame trace produced | ✔️ |
| R16 | Router (`checkpoints/router_hi_id.pt`: 38,531 steps, 5 epochs, experts hi + identity), mixed | WER **50.50%**, CER 51.07%. vs base −4.87 points (27 better / 23 worse, t = −2.30); vs adapter alone **+3.91** (13 / 29, t = +2.75) | `results/router_eval/router_hi_id_mixed.csv` | ✅ |
| R17 | Router, pure Hindi (degradation monitoring) | WER **58.25%**, CER 47.88%. vs base −8.94 (t = −3.69); vs adapter alone **+5.40** (t = +3.36) | `results/router_eval/router_hi_id_pure.csv` | ✅ |
| R18 | Does routing follow language? | Mean adapter share 0.47 (mixed) / 0.46 (pure). Correlation with English-word fraction across utterances: r = +0.02, i.e. none. | R16 CSV | ✔️ |

**What can be claimed from R16–R18:** the router is trained end to end with only its own 1.19 M parameters, and it significantly beats the frozen base model on both sets.

**What can't be claimed:** that it beats the adapter alone, or that it routes by language. It learned a nearly constant ~46% blend.

**Plausible causes, not yet tested:**
- The second expert is the unadapted backbone, not an English expert.
- With `--english-targets drop`, English frames are trained toward blank.
- The router was trained outside this machine (it arrived as an unzipped `router_hi_id/` folder), so there is no record of which `hi_adapter.pt` it was trained against. If that adapter differs from the one on disk, the router is mismatched to its expert.

Record the adapter hash with every router run.

## 4. Evaluation protocol (state this in the thesis method chapter)

- **Metric.** WER and CER via `jiwer`, after the same normalisation on both sides: NFC, lowercase, punctuation stripped, whitespace collapsed (`moa_cas/metrics.py`, ✅ `test_metrics_decode.py`).
- **Decoding.** Greedy CTC, no beam search or language model (✅ `test_metrics_decode.py`). Deterministic: re-running reproduces the CSVs exactly (R6).
- **Comparisons are paired.** Every system is scored on the same utterance IDs (asserted in `test_results.py`). Report win/loss counts and paired t / 95% CI, not just means.
- **Sample size.** n = 100 per condition out of 3,132 test utterances. CIs are wide, about ±6 points.
- **Held out.** Train and test speakers are disjoint (D2). There is **no dev split**, so no hyperparameter tuning or checkpoint selection has been done.

## 5. Corrections and integrity notes

| # | Issue | Effect | What to do in the thesis |
|---|---|---|---|
| C1 | `checkpoints/hi_adapter.pt` was **overwritten on 17 Sep** by a shorter run: 1,933 steps, 1 epoch, 1,933 utterances. The 16 Sep checkpoint behind the 51.81% figure (doc 03: 3,960 steps, 3 epochs) is gone. `checkpoints/` is git-ignored, so nothing recorded the change. | R9 can't be reproduced; R7/R8 can. | Report **R7/R8** as the main result. Mention R9 only as an earlier run. Current checkpoint SHA-256: `0387975f62995508b7f8561e61b05d04d9d86ba1bffa81599efdf06725827a59`. Record a hash for every future checkpoint. |
| C2 | The per-epoch loss table in docs/03 ("clean 5-hour run", 7 epochs) comes from a run doc 03 itself **retracts** (broken mask). | That loss curve describes training toward a scrambled target. | Don't cite it. |
| C3 | All training runs before the mask fix (88.56% WER mask) are retracted. | | Cite only post-fix numbers. Throughput figures (1.5–3.6 utterances/s on CPU) remain valid. |
| C5 | The trained router arrived as an unzipped PyTorch archive folder (`router_hi_id/`). It was repacked byte-for-byte into `checkpoints/router_hi_id.pt` (all 136 router tensors load). The adapter used during its training isn't recorded. | See R18 | Keep the `.pt` file intact when copying it from Colab |
| C4 | The deck lists the backbone as 600 M. | See the S9 note. | Name both models. |

## 6. Known limitations (for the discussion chapter)

1. **English can't be written.** The decoder has no Latin tokens, so roughly 25% of reference words can never be correct under exact-match WER. Adapters and routers change what the encoder *hears*, not what the decoder can *write*.
2. **No English expert.** There's no English speech data in the pipeline, so the router currently chooses between *Hindi adapter* and *original backbone*, not between languages.
3. **Bengali** is a separate backbone and has never been trained or evaluated.
4. **Domain and segmentation confounds** dominate the MUCS error (R4, D6).
5. **Small evaluation** (n = 100), and no dev split.
6. **CPU only.** The GPU code path exists but has never been run.
7. **Tamil, Telugu and Gujarati** have no code-switched audio in MUCS, so the equity axis has no data.

## 7. Work needed before the thesis can claim the full MoA-CAS objective

| Needed for | Work |
|---|---|
| Any router result | GPU training run; four-way comparison (baseline / adapter / uniform mix / trained router) on the same utterances, mixed and pure (docs/05 §4) |
| A trustworthy evaluation | Speaker-disjoint dev split (`pipeline/splits.py`); evaluate more than 100 utterances, ideally all 3,132 |
| The code-switching claim | English speech data **and** a way to output Latin script (vocabulary extension or transliterated targets) |
| Switch-point accuracy (CS-WER / CBA) | Word timestamps via forced alignment |
| Degradation monitoring | IndicVoices Hindi WER with adapter and router (target ≤ 8%) |

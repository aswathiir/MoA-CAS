# 05 — The Frame-Level Router (Stage 2, Fluidity Learning)

Stage 1 produced a specialist: one bottleneck adapter per encoder layer,
trained on monolingual Hindi over a frozen IndicConformer (see
`03_PROFILE_LEARNING_ADAPTERS.md`, `04_HOW_THE_ADAPTER_SYSTEM_WORKS.md`).
Stage 2 adds the part that makes MoA-CAS a *mixture*: a router that
decides, frame by frame, how much of each specialist to use.

Code: `src/moa_cas/adapters/router.py`, `scripts/train_router.py`,
`scripts/evaluate_router.py`, `tests/test_router.py`.

---

## 1. What the design is based on

| Source | What it contributes here |
|---|---|
| MoA-CAS Zeroth Review deck, slides 8–10 | Frozen backbone. Frame-level router producing `w_t`. Specialists fused as `Σ w_t · A_k(h_t)`. Stage 2 trains **only** the router; adapters are locked. Loss = CTC + load balancing. |
| Kulkarni et al. 2023, *Adapting the Adapters for Code-Switching in Multilingual ASR* (arXiv:2310.07423) | Two language adapters combined at **every** transformer block by a per-frame gate: `O = (1 − g_t)·A₁ + g_t·A₂`, with backbone frozen and only the switching network trained on a small amount of code-switched data. Our router is the K-expert softmax generalisation; `--routing hard` reproduces their thresholded gate. |
| Zhao et al. 2024, *CAMEL* (arXiv:2412.12760) | Language-specific experts after each layer whose representations are fused, plus a language-diarization signal. Motivates looking at whether routing tracks language (the `corr(en_fraction)` diagnostic) — not implemented as an extra loss. |
| Fedus et al. 2022 (JMLR), *Switch Transformers* | The load-balancing loss `K · Σ_k f_k · P_k`. |
| Winata et al. 2020 | The "don't forget monolingual" constraint: here met structurally, because the base model is never changed and an `identity` expert keeps it reachable per frame. |

## 2. Architecture

At every Conformer layer `l` (all 17), for every encoder frame `t` (25 per second):

```
h_t ─┬─► FrameRouter_l ─► w_t = softmax(logits / τ)        (K weights)
     ├─► A_1(h_t)  (frozen Stage 1 adapter, e.g. Hindi)
     ├─► A_2(h_t)  (frozen; or Identity = raw backbone)
     └─► ...
out_t = h_t + Σ_k w_t,k · (A_k(h_t) − h_t)        ≡ Σ_k w_t,k · A_k(h_t)
```

**FrameRouter:** LayerNorm, then a depthwise temporal conv (5 frames ≈ 200 ms of context, padding zeroed first), then Linear 512→128, ReLU, Linear 128→K.
- The last layer is **zero-initialised**, so an untrained router gives exactly uniform weights. This is the same "start from a known state" principle as the zero-init adapters.
- The router is attached with a forward hook that reads NeMo's `pad_mask`. Padded frames never enter the load-balancing statistics.

**Trainable parameters:** 1,190,306 (routers only), on top of 1,123,904 for the Hindi adapter. Together that is 1.96% of the 118.0 M-parameter NeMo backbone (encoder + CTC head), just inside the deck's `< 2%` target.

**Routing modes:**
- `soft` (training default): full weighted mix.
- `hard`: top-1 expert per frame, with straight-through gradients.

**Loss:** `CTC + α · LB`, with α = 0.01 by default. LB is 1.0 when routing is perfectly balanced and K when it collapses onto one expert.

## 3. Which experts can exist today

Experts must share one frozen backbone, because an adapter only means something on the encoder whose hidden states it was trained to reshape. `load_stage1_expert` refuses a checkpoint whose `lang` doesn't match the backbone. That leaves:

| Expert | Status |
|---|---|
| `hi` — `checkpoints/hi_adapter.pt` | Trained (Stage 1) |
| `identity` — the untouched backbone | Always available |
| English (`Een`) | **Blocked.** There is no English speech corpus, and the backbone's CTC vocabulary has no Latin tokens. |
| Bengali | Lives on a *different* backbone (`indicconformer_stt_bn_…`), so it can't share a router with `hi` |
| Accent experts | Not started (deck timeline: next semester) |

So the configuration that runs now is **`hi` vs `identity`**. The router learns per frame whether domain-adapted Hindi or the original backbone serves the transcript better. That is a real Stage 2 experiment, and it is also the catastrophic-forgetting guard: if the adapter hurts on some frames, the router can switch it off there.

It is **not** yet Hindi-vs-English routing. The code is K-expert generic: adding an English adapter is one more `--expert en=…` flag, with no code change.

### English words in the CTC targets

No expert can write Latin script. `--english-targets` decides what the target does with English words:

- `drop` (default): remove them, so those frames are free to emit blank.
- `keep`: leave them in, which maps them to `<unk>`.

Evaluation reports both `wer` (full reference) and `wer_l1` (Latin words removed). `wer_l1` can come out *higher* than `wer`: when the model writes `डॉक्यूमेंट` for `document`, that word is an insertion against the L1-only reference.

## 4. Running it

```bash
# train (CPU); add --device cuda --batch-size 16 on GPU
python scripts/train_router.py --lang hi \
    --expert hi=checkpoints/hi_adapter.pt --expert identity \
    --max-seconds 10800 --save-to checkpoints/router_hi_id.pt

# baseline: untrained router = uniform 50/50 mix
python scripts/evaluate_router.py --lang hi \
    --expert hi=checkpoints/hi_adapter.pt --expert identity \
    --manifest data/manifests/mucs_hi-en_test.jsonl --limit 100

# trained router, with a frame-by-frame trace of one utterance
python scripts/evaluate_router.py --lang hi \
    --expert hi=checkpoints/hi_adapter.pt --expert identity \
    --manifest data/manifests/mucs_hi-en_test.jsonl --limit 100 \
    --router-checkpoint checkpoints/router_hi_id.pt \
    --trace 103085_w5Jyq3XMbb3WwiKQ_0000 \
    --out-csv results/router_eval/router_mixed.csv
```

The full comparison set for a report is these four on the **same** utterances:
1. Baseline: `evaluate_adapter.py`, no adapter.
2. Hindi adapter alone: `evaluate_adapter.py --adapter-checkpoint`.
3. Uniform mixture: `evaluate_router.py`, no router checkpoint.
4. Trained router: `evaluate_router.py --router-checkpoint`.

Run each on both the mixed and pure-Hindi subsets. The pure-Hindi run is the deck's "Degradation Monitoring".

## 5. Verification done

- **Unit tests** (`tests/test_router.py`, 9 tests). They cover:
  - the router is uniform at init;
  - the LB loss equals 1 when balanced, K when collapsed, and ignores padding;
  - the mixture output equals the weighted expert sum;
  - an identity expert at full weight recovers the frozen backbone exactly;
  - only router parameters receive gradients;
  - hard routing picks one expert per frame;
  - the Stage 1 checkpoint loads, and a backbone mismatch is rejected.

  Full suite: 49/49 pass.
- **End-to-end smoke run** on the real Hindi NeMo backbone, the real `hi_adapter.pt`, and MUCS audio: 150 s on CPU, 60 code-mixed utterances, 89 steps, no NaNs. Evaluation, routing shares, correlation, and the per-frame trace all produce output.

  The smoke numbers (20 test utterances: uniform WER 0.424 vs smoke-trained 0.451) are **not a result**. That is 2.5 minutes of training, and n = 20 is far too small to read either way.

## 5b. First full training run (router_hi_id)

Run settings: 38,531 steps, 5 epochs, `hi` + `identity`, α = 0.01, `--english-targets drop`.

Evaluated on the same 100 utterances as every other system:

| Set | Base | Adapter alone | Router |
|---|---|---|---|
| Mixed | 55.37% | **46.60%** | 50.50% |
| Pure Hindi | 67.18% | **52.84%** | 58.25% |

- The router significantly beats base on both sets (t = −2.30 / −3.69).
- It is significantly *worse* than the adapter alone (t = +2.75 / +3.36).
- Routing is close to a constant ~46% blend, and it doesn't track English density (r = 0.02).

See THESIS_REFERENCE R16–R18 for causes to test next. Try it live with `python scripts/demo_server.py`.

## 6. What is still missing

1. **A real training run** (GPU, hours) and the four-way comparison above on ≥ 100 utterances, with paired win/loss counts.
2. **A dev split** for checkpoint selection. `pipeline/splits.py::speaker_disjoint_split` still goes unused.
3. **An English expert.** This needs both an English speech corpus *and* a way to emit Latin script: vocabulary extension, a transliteration target, or an English-capable decoder head. Until then, routing cannot be language routing in the deck's sense.
4. **Frame-level language labels.** MUCS only has word-level tags with no timestamps. A forced alignment would allow an auxiliary LID loss on the router (as CAMEL's diarization signal does) and direct measurement of switch-point accuracy (CBA).
5. **Hyperparameters** (α, τ, router width/context, soft vs hard) are first-guess values, never varied.

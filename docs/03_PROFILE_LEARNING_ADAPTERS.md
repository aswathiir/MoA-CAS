# 03 — Profile Learning (Adapter Training)

Stage 1 of the MoA-CAS two-stage plan: train per-language expert adapters
on top of a frozen pretrained encoder, before the router is introduced.

---

## 1. Work done

### Backbone selection — and why the benchmarked model could not be used

The model used for the baseline benchmark
(`ai4bharat/indic-conformer-600m-multilingual`) turned out to be
**untrainable**. Its encoder, CTC decoder and RNNT joint are all shipped
as `onnxruntime.InferenceSession` objects — a frozen, inference-only
computation graph with no autograd, no accessible layer structure, and no
way to insert a trainable module mid-network. The entire "freeze the
backbone, train adapters on top" premise is impossible against it.

ai4bharat separately publishes **per-language NeMo checkpoints**
(`ai4bharat/indicconformer_stt_{lang}_hybrid_ctc_rnnt_large`) which are
genuine PyTorch. Those became the backbone for this stage.

### Making the NeMo checkpoint load at all

The public `nemo_toolkit` cannot load these checkpoints out of the box.
Five distinct failures had to be worked through, each a different root
cause:

| # | Failure | Resolution |
|---|---|---|
| 1 | HF repo is gated (separately from the ONNX model) | One-time manual access approval while authenticated |
| 2 | NeMo's `.nemo` auto-restore looks for a pre-extracted `model_config.yaml` that is never created | Bypass it: `snapshot_download`, then extract the archive directly (it is a tar) |
| 3 | Tokenizer config is multilingual-style (`langs.{hi,bn,…}`); NeMo's monolingual loader expects flat `dir`/`model_path` and raises `KeyError: 'dir'` | Promote the target language's sub-entry to the top level, resolving `nemo:` paths to absolute ones |
| 4 | `train_ds` lacks a `shuffle` key that NeMo's init tries to backfill — fails under OmegaConf struct mode regardless of `set_struct(False)`, because the write targets a dataclass-backed schema that is permanently struct-locked | Delete `train_ds`/`validation_ds`/`test_ds` entirely (they point at ai4bharat's internal cluster paths and are unused — we supply our own data loading) |
| 5 | A `multisoftmax` constructor kwarg from ai4bharat's private NeMo fork does not exist in the public package | Drop the kwarg; verified safe because the saved decoder weights are a single plain layer, not a per-language split |

Additionally, NeMo's full `EncDecHybridRNNTCTCBPEModel` class is never
instantiated — its constructor unconditionally builds the RNNT
decoder/joint (where failure #5 bites hardest), and Profile Learning only
needs the CTC path. Only the preprocessor, encoder and auxiliary CTC
decoder are built, directly.

### Reconstructing the language mask

The CTC decoder emits a **single 5,633-wide output shared across all 22
languages** (5,632 vocabulary tokens plus one shared blank appended as
the final class — confirmed from the decoder weight shape
`[5633, 512, 1]` against a 5,632-entry vocabulary list). Hindi's own
tokenizer knows only 256 of those tokens, and — unlike ai4bharat's ONNX
bundle, which ships `language_masks.json` — the NeMo checkpoints include
no mask to say which 256.

It was reconstructed by string-matching the language's SentencePiece
pieces against the shared vocabulary's token strings: **all 256 Hindi
pieces matched exactly**. The resulting index list (256 language tokens
+ the shared blank) gathers a 257-wide local view out of the decoder
output, in which the tokenizer's own IDs line up directly as CTC targets.

One trap worth recording: this must use the compiled `.tokenizer.model`
file via the `sentencepiece` library. The auxiliary `vocab.txt` bundled
alongside it uses a WordPiece-style `##` display convention that will not
cross-reference against the SentencePiece `▁`-style master vocabulary at
all.

---

## 2. What was achieved

### Model assembly, verified

| Property | Value |
|---|---|
| Encoder layers (Conformer blocks) | 17 |
| Encoder hidden dimension | 512 |
| Frozen backbone parameters | 115,111,424 |
| Trainable adapter parameters | 1,123,904 (0.98% of backbone) |
| Adapter shape | 512 → 64 → 512, one per layer |
| State-dict load | **0 missing, 0 unexpected keys** across preprocessor, encoder and CTC decoder |

A zero missing/unexpected key count is the strongest available evidence
that the reconstructed architecture matches what was actually trained —
every workaround above preserved the real model rather than approximating
it.

### Gradient isolation, verified

A dedicated forward/backward test confirmed that **all 68 adapter
parameters receive gradients while zero backbone parameters do** — the
freezing is real, not assumed. (At initialisation the down-projection
sees zero gradient because the up-projection is zero-initialised; this is
the expected behaviour of an identity-initialised adapter, not a fault.)

### Training runs

| Run | Duration | Steps | Epochs | Throughput | Non-finite losses | Outcome |
|---|---|---|---|---|---|---|
| First 5-hour attempt | 5 h | 39,236 | 6 | 2.18 utt/s | Corrupted at step 289 | **Failed** — see below |
| 1-hour validation | 1 h | 6,707 | 1 | 1.86 utt/s | 0 | Loss trend 109 → 57 |
| Full 5-hour run | 5 h | 46,394 | 7 | 2.58 utt/s | 0 | Loss 68.07 → 43.65 |
| 1-hour epoch run (current code) | 1 h | 5,394 | 6 | 1.50 utt/s | 0 | Loss 97.89 → 38.38 |

The final row is the important one for reproducibility: the three earlier
runs used an exploratory script that no longer exists, so their results
cannot be re-derived from the repository. The 1-hour epoch run was
executed against **the committed code** (`scripts/train_profile_adapter.py`)
over a fixed 1,000-utterance subset, completing five full epochs plus a
partial sixth:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 (partial) |
|---|---|---|---|---|---|---|
| Mean loss | 97.89 | 58.72 | 51.73 | 46.58 | 43.91 | 38.38 |
| Median loss | 72.73 | 23.65 | 15.44 | 13.08 | 10.24 | 10.87 |

A **60.8% reduction in mean loss** and **85% in median** across the run,
with zero non-finite losses and 152 utterances (2.7%) skipped by the
safety guard. The median falling far faster than the mean indicates the
adapter is fitting the bulk of the data well while a minority of hard
utterances continue to dominate the average — consistent with the
segmentation artifacts described in
`01_BASELINE_AND_PROBLEM_ANALYSIS.md`, where a subset of utterances
contain audio whose words are absent from the reference.

Per-epoch mean loss on the clean 5-hour run, over the full 7,083-utterance
pure-Hindi pool:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Mean loss | 68.07 | 52.07 | 49.01 | 47.22 | 45.96 | 45.34 | 43.65 |

Monotonic improvement across every epoch, with zero non-finite losses and
~3% of utterances skipped by the safety guard.

### The failed run, and what it taught

The first 5-hour run reported success (exit code 0) but had actually gone
to `NaN` at **step 289 of 39,236**. A 1.56-second utterance did not have
enough encoder frames to align its target text — violating CTC's
`input_length ≥ target_length` precondition — producing infinite loss.
That `inf` propagated through `optimizer.step()`, permanently corrupting
every adapter weight, after which all **38,946 remaining steps (4.95 of
the 5 hours) computed `nan`** silently. No crash, no warning, and the
process exit code gave no indication.

Three safeguards were added in response, and the subsequent runs were
clean:

1. Skip samples where `encoder_frames < 2 × target_length` (the 2× margin
   covers blanks required between repeated tokens).
2. Skip any non-finite loss before `backward()`, as defence in depth.
3. Clip adapter gradients (`max_norm=5.0`).

---

## 3. Method followed

- **Frozen backbone, trainable adapters.** All preprocessor, encoder and
  CTC decoder parameters are frozen. Bottleneck adapters are attached to
  each Conformer layer via **forward hooks**, so the pretrained module
  tree and state dict remain untouched — the adapters are not submodules
  of the encoder and can be saved, swapped or discarded independently.
- **Identity initialisation.** Each adapter's up-projection is
  zero-initialised, so at step 0 it is an exact identity passthrough and
  the backbone behaves exactly as pretrained. Training moves it away from
  identity gradually rather than perturbing a working model immediately.
- **Loss.** CTC over the language-masked 257-wide view of the shared
  decoder output. Because the CTC head itself is frozen, the adapters
  must learn to shift the encoder's internal representations such that
  the *unchanged* head decodes them better.
- **Data.** Utterances from the existing preprocessing manifests filtered
  to `n_words_en == 0` and 1–10 s duration — monolingual Hindi with real
  audio already on disk, requiring no new data preparation.
- **Optimiser.** Adam, lr 1e-3, over adapter parameters only, batch size
  1 (CPU-bound, no GPU available on this machine).

---

## 4. Possible outcome

Projections, explicitly labelled — none of these are measured results.

**Compute required for a real training run**, extrapolated from the
measured 2.58 utterances/sec. Measured throughput across runs ranged
1.50–3.63 utt/s depending on concurrent load on the machine, so treat
these figures as a central estimate with roughly ±40% spread:

| Scope | Utterances | Per epoch (CPU) | ~10 epochs (CPU) |
|---|---|---|---|
| Pure-Hindi pool (as piloted) | 7,083 | 46 min | ~7.6 h |
| Full Hindi-English | 55,901 | 6.0 h | ~60 h |
| Full Bengali-English | 30,838 | 3.3 h | ~33 h |
| **Combined Hindi + Bengali** | **86,739** | **9.4 h** | **~94 h (≈3.9 days)** |

On a single modest GPU, with real batching and parallelism (a
conservative 15–30× speedup over batch-size-1 CPU), the combined run
projects to roughly **3–6 hours**. That is the actionable conclusion:
Profile Learning needs a GPU, but not a cluster.

**On expected quality** — the falling loss curve demonstrates the adapter
is learning *something* the frozen head can decode better. It does **not**
demonstrate improved transcription, because WER has never been measured
on a trained adapter (Section 5). The honest position is that
feasibility is established and effectiveness is entirely unmeasured.

---

## 5. Final works left

Ordered by how much they block progress.

1. **The training script never saves the adapter weights.** There is no
   `torch.save` anywhere in it. Every run — including the successful
   5-hour one — trains adapters and discards them entirely on exit; only
   the loss log persists. As a *measurement instrument* for throughput
   and feasibility this was fine, and that is what the pilot was for, but
   **no trained adapter currently exists**. Checkpointing must be added
   before any run can produce a usable artifact. This is the single most
   important fix.
2. **No WER evaluation of a trained adapter.** Falling CTC loss is not
   evidence of better transcription. Until an adapter is trained, saved,
   and benchmarked with the same WER harness used in
   `01_BASELINE_AND_PROBLEM_ANALYSIS.md`, the project has no evidence
   that Profile Learning improves anything. This is the decisive
   experiment and it has not been run.
3. **No held-out validation set.** Training loss on the training data
   cannot distinguish learning from memorisation. With 520 speakers
   available, a speaker-disjoint dev split is straightforward — the
   helper already exists (`pipeline/splits.py`) and is unused.
4. **Bengali has never been run.** HF access is granted and the code path
   is language-parameterised, but `--lang bn` has not been executed once.
   `Eia` is supposed to cover both languages; currently it covers one.
5. **No GPU run.** The 3–6 hour projection is extrapolated from CPU
   throughput and has never been validated against real hardware. Data
   loading may become the bottleneck rather than compute.
6. **Hyperparameters entirely unexplored.** Bottleneck width (64),
   learning rate (1e-3), which layers receive adapters (all 17), and
   adapter placement within the Conformer block were chosen as
   reasonable defaults and never varied or ablated.
7. **Training data may be unrepresentative.** The pure-Hindi filter
   yields 7,083 of 52,769 Hindi-English utterances (13.4%). These are the
   monolingual *remainder* of a code-switched tutorial corpus, which is
   not the same distribution as genuine monolingual Hindi speech — and,
   per `01_BASELINE_AND_PROBLEM_ANALYSIS.md`, they carry the same domain
   shift and segmentation artifacts as the rest of MUCS.
8. **`Een` (English) remains blocked.** No English speech corpus exists
   anywhere in this pipeline. MUCS's non-code-mixed utterances are still
   Hindi/Bengali narration, and IndicVoices covers only Indian languages.
   An English corpus must be sourced before the English expert can be
   trained at all.
9. **Stage 2 (the router) has not been started.** Nothing in this work
   touches the frame-level gating network, the load-balancing loss, or
   mid-utterance transition handling.

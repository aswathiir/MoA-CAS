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

> ### ⚠️ Correction — earlier training results retracted
>
> Every training run reported in earlier revisions of this document was
> computed against a **broken language mask**. The mask was originally
> built by matching the language's tokenizer pieces against the shared
> vocabulary *by string*. But 670 token strings are duplicated across
> the 22 languages (the Devanagari-script languages share many pieces),
> so that lookup silently resolved them to whichever language appeared
> last — putting **213 of 256 indices (83%) on the wrong columns**.
>
> The loss curves still fell, because the adapter was successfully
> fitting whatever scrambled target space it was given. This is a useful
> cautionary result in itself: **a falling loss curve validated nothing**,
> and the error was only caught by decoding the model and comparing WER
> against an independent benchmark.
>
> The mask is now derived from the vocabulary's actual layout
> (contiguous per-language blocks) and verified against the tokenizer at
> load time. Results below are from the corrected implementation.

### Reconstructing the language mask

The CTC decoder emits a **single 5,633-wide output shared across all 22
languages** (5,632 vocabulary tokens plus one shared blank appended as
the final class — confirmed from the decoder weight shape
`[5633, 512, 1]` against a 5,632-entry vocabulary list). Hindi's own
tokenizer knows only 256 of those tokens, and — unlike ai4bharat's ONNX
bundle, which ships `language_masks.json` — the NeMo checkpoints include
no mask to say which 256.

The shared vocabulary turns out to be laid out as **contiguous
per-language blocks**, in the order the languages appear in
`tokenizer.langs`: 22 languages × 256 tokens = 5,632 exactly. Hindi is
language #6, so its block is indices `[1536:1792]` — and that block
equals its tokenizer's pieces exactly, in order. The mask is that block
plus the shared blank index, giving a 257-wide local view in which the
tokenizer's own IDs line up directly as CTC targets.

Two traps worth recording:

- **Do not match by string.** 670 token strings are duplicated across
  languages, so a string→index lookup resolves them to the wrong
  language's column (this was the original bug — see the correction
  notice above). The block layout is the correct derivation, and the
  implementation now asserts the derived block matches the tokenizer
  exactly, failing loudly if the assumption ever breaks.
- Use the compiled `.tokenizer.model` file via the `sentencepiece`
  library. The auxiliary `vocab.txt` bundled alongside it uses a
  WordPiece-style `##` display convention that will not cross-reference
  against the SentencePiece `▁`-style master vocabulary at all.

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

### Decode-path validation (the correctness gate)

Before any training result can mean anything, the CTC decode path and
the reconstructed language mask have to be shown correct. The test:
decode the NeMo model with **no adapter at all** over the same 100
utterances the ONNX model was independently benchmarked on, and compare.

| Configuration | WER on MUCS Hindi-English test (n=100) |
|---|---|
| ONNX IndicConformer (independent benchmark) | 57.14% |
| NeMo backbone, broken string-matched mask | **88.56%** ❌ |
| NeMo backbone, corrected block mask | **55.37%** ✅ |

The corrected path lands within 1.8 points of an independently measured
benchmark — expected for two different checkpoint variants of the same
model family — and produces fluent Hindi that tracks the reference, with
English transliterated into Devanagari (`impress` → `इम्प्रेस`,
`document` → `डॉक्यूमेंट`). The broken mask produced sparse near-gibberish
while *still yielding a smoothly falling training loss*.

**This gate is the single most valuable experiment run on this project so
far.** It invalidated several hours of apparently successful training and
cost about a minute to run.

### Training runs

Earlier runs (a first 5-hour attempt destroyed by NaN corruption, a
1-hour validation, a clean 5-hour run, and a 1-hour six-epoch run) are
**retracted** — all used the broken mask, and their loss curves describe
convergence toward a scrambled target. They are not reproducible from
the repository in any case, as they predate the current code.

Throughput measurements from those runs remain valid, since throughput
does not depend on the mask being correct: **1.5–3.6 utterances/sec** on
CPU at batch size 1, centring around 2.5 utt/s.

The run that produced the result below used the corrected mask:
**1 hour CPU, 3,960 steps, 3 epochs over 2,000 pure-Hindi training
utterances**, 111 skipped, zero non-finite losses, median loss 12.36 →
7.14 across epochs. The adapter was saved and evaluated.

### Headline result — measured WER improvement

Trained on the MUCS Hindi-English **train** split, evaluated on the
**test** split. The two are genuinely held out from each other: 520 vs
30 speakers with **zero overlap**, verified.

| Evaluation set | Baseline | With adapter | Change |
|---|---|---|---|
| Pure-Hindi held-out (n=100) — **WER** | 67.18% | **51.81%** | **−15.37 pts (−22.9% relative)** |
| Pure-Hindi held-out — CER | 56.05% | **40.57%** | −15.48 pts |
| Pure-Hindi held-out — median WER | 79.29% | **39.57%** | −39.72 pts |
| Full mixed test set (n=100) — WER | 55.37% | **45.44%** | −9.93 pts (−17.9% relative) |

**Paired statistics** on the identical 100 held-out utterances:
53 improved, 25 worse, 22 unchanged; mean change −0.1537 with a 95%
confidence interval of **[−0.216, −0.092]**; paired *t* = −4.84. The
improvement is statistically significant, not noise.

### Why it improves — the mechanism

The frozen backbone systematically **under-emits** on this domain: it
produces only **75%** as many words as the reference contains, truncating
longer utterances. The adapter raises that ratio to **86%**, recovering
words the baseline dropped. The gain holds across every duration bucket
(−0.10 WER for 1–5 s, −0.20 for 5–10 s, −0.17 for >10 s).

Illustrative (baseline → adapter, same utterance):

```
REF : अगर स्पेशल फ्लेग का चुनाव न हो तो देखते है आगे क्या होता है चलो हम यहाँ आते है
BASE: तो स् च चमते हैं                                              (WER 0.89)
ADAP: अगर स्पेशल फ्लैग का चुनाव न हो तो देखते हैं क्या होता है चलो हम यहाँ आते हैं  (WER 0.05)
```

Note the adapter also improves the **mixed** code-switched set (−9.93
pts) despite being trained only on monolingual Hindi. That is consistent
with the finding in `01_BASELINE_AND_PROBLEM_ANALYSIS.md` that most of
the degradation on this corpus is **domain shift rather than
code-switching** — the adapter is primarily closing the domain gap, which
is exactly what a monolingual in-domain training signal should teach it.

### Scope of the result — where the gain actually comes from

Splitting the held-out test words by whether the adapter encountered that
word type during training:

| Word type | Baseline recall | With adapter | Change | n | Significance |
|---|---|---|---|---|---|
| Seen in training | 56.8% | 69.0% | **+12.2 pts** | 1,339 | z = +6.5, significant |
| Never seen | 41.2% | 47.3% | +6.1 pts | 182 | z = +1.2, **not significant** |

The improvement on familiar vocabulary is **twice** that on unfamiliar
vocabulary, and the unfamiliar gain cannot be distinguished from noise at
this sample size. A substantial share of the headline −15.37 points is
the adapter becoming fluent in *this corpus's particular vocabulary* —
the Devanagari spellings of `presentation`, `slide`, `format`,
`dialog box` that saturate these transcripts.

That vocabulary-specific component would **not** transfer to a different
code-switched corpus. On new material with different vocabulary, the
expected gain is closer to the unseen-word figure than the headline one.

Two qualifications in both directions. The unseen improvement is +6.1
points rather than zero, so some genuinely transferable adaptation
plausibly occurred — it simply cannot be proven with 182 words. Against
that, the "seen" bucket includes ordinary Hindi function words that would
appear in any Hindi text, so the truly corpus-specific concentration may
be higher than this split shows.

**The defensible claim is therefore narrower than the headline:** adapters
on a frozen Indic backbone measurably improve *in-domain* ASR, with gains
concentrated in vocabulary observed during training. Not "adapters improve
Hindi ASR" in general, and not "adapters improve code-switched ASR".

### How to read this result

It is a genuine, held-out, statistically significant improvement from one
hour of CPU training with 1.12M trainable parameters (0.98% of the
backbone). It is **not** yet evidence that MoA-CAS solves code-switching:
the adapter was trained on monolingual data and is mostly performing
domain adaptation, concentrated in observed vocabulary.

There is also a hard ceiling that no adapter can lift. The backbone's
output vocabulary contains **no English tokens at all** — the only
Latin-containing entry across all 5,632 tokens is `<unk>` — so the 24.6%
of reference words written in Latin script can never be produced
correctly. That is a **~25% WER floor** on this corpus, and it means the
code-switching objective cannot be met by adapters alone: they change what
the encoder hears, while the constraint lives in what the decoder can
write. See `04_HOW_THE_ADAPTER_SYSTEM_WORKS.md` for the analysis and the
three available routes around it.

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

1. **Scale the evaluation.** The result rests on n=100 per condition.
   The full Hindi-English test set has 3,132 utterances; running all of
   them would tighten the confidence interval substantially. Bengali is
   entirely unevaluated.
2. **Train properly on GPU.** The current adapter had one hour of CPU
   time, three epochs, 2,000 utterances. The full pure-Hindi pool is
   7,083 utterances and the model was still improving when the clock ran
   out — there is clear headroom left unexploited.
3. **No dev split for model selection.** Training ran for a fixed wall
   clock with no early stopping and no validation-based checkpoint
   selection. The helper exists (`pipeline/splits.py`) and is unused; the
   current result uses the official test split directly, which is fine
   for a single measurement but invites overfitting the moment
   hyperparameters start being tuned against it.
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

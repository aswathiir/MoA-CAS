# 04 — How the Adapter System Works, End to End

A complete walkthrough of the NeMo checkpoint work: where the model came
from, why it was chosen, every obstacle between the published checkpoint
and a working training loop, how the adapter is constructed, what the
frozen backbone actually does, why any of this should be expected to
help, and where it can go wrong.

This is the explanatory companion to `03_PROFILE_LEARNING_ADAPTERS.md`,
which reports results. This document explains mechanism.

---

## Part 1 — The goal, in one paragraph

A large speech model has already learned, from thousands of hours of
audio, how to turn sound into Hindi text. We do not want to retrain it:
that is expensive, needs the original training data we do not have, and
risks destroying what it already knows. Instead we **freeze** it
completely and insert small trainable layers — *adapters* — at intervals
inside it. Only those small layers learn. The large model's own weights
never change by even one value. If the adapters help, we keep them; if
they do not, we discard them and the original model is untouched. The
eventual MoA-CAS design has several such adapters — one per language
family — with a router choosing between them mid-utterance. This
document covers building and training a single one.

---

## Part 2 — Choosing the backbone

### The model we started with could not be trained

The baseline benchmark used `ai4bharat/indic-conformer-600m-multilingual`
from HuggingFace. It scores well and is easy to run, but inspecting how
it is packaged revealed a hard blocker: its encoder, CTC decoder and
RNNT joint are all shipped as **ONNX Runtime inference sessions**.

ONNX is a format for *running* a finished model fast. It is a frozen
computation graph. There is no autograd, no exposed layer structure, and
no way to insert a trainable module between layers, because from the
outside the whole network is a single opaque function. You can feed audio
in and read text out, and that is all.

Every part of the MoA-CAS plan — freeze the backbone, insert adapters,
train them — is impossible against an ONNX artifact. Not difficult:
impossible.

### The model we switched to

ai4bharat also publishes **per-language NeMo checkpoints**:

```
ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large
ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large
```

These are genuine PyTorch models — real layers, real weights, full
gradient support. NeMo is NVIDIA's speech toolkit, and a `.nemo` file is
just a tar archive holding a config file and a weights file.

Reading the name: `hybrid_ctc_rnnt` means the model has **two** ways of
turning encoder output into text — a CTC head and an RNNT head — sharing
one encoder. We use the CTC path only, for reasons in Part 5.

---

## Part 3 — Getting the checkpoint to load

The published checkpoint does not load with the public `nemo_toolkit`.
Five separate obstacles, each with a different cause. They are documented
here because none of them is findable in any documentation, and anyone
repeating this work will hit all five.

### Obstacle 1 — The repository is gated

The model page requires accepting terms while signed in before any
download works, and this gate is **separate** from the one on the ONNX
model. Downloads fail with HTTP 403 until a human clicks accept. There is
no programmatic bypass, nor should there be.

### Obstacle 2 — NeMo's own loader is broken for this file

The natural call, `ASRModel.from_pretrained(...)`, downloads the file
successfully and then fails looking for an extracted `model_config.yaml`
that nothing ever created. The download and the extraction steps disagree
about who is responsible for unpacking.

Resolution: skip that path entirely. Download with
`huggingface_hub.snapshot_download`, then open the `.nemo` with Python's
`tarfile` and extract it. It is an ordinary tar archive containing
`model_config.yaml`, `model_weights.ckpt`, and tokenizer files for all 22
languages.

### Obstacle 3 — The tokenizer config has an unexpected shape

A tokenizer splits text into sub-word pieces. NeMo's loader expects the
config to name one tokenizer directly. This checkpoint instead carries a
nested map of all 22 languages:

```yaml
tokenizer:
  type: multilingual
  langs:
    hi: {model_path: nemo:890a...tokenizer.model, ...}
    bn: {model_path: nemo:8369...tokenizer.model, ...}
    ...
```

NeMo reaches for a top-level `dir` key, finds nothing, and raises
`KeyError: 'dir'`. Resolution: lift the target language's entry up to the
top level before handing the config over, and rewrite the `nemo:` path
prefixes to real absolute paths inside the extraction directory.

### Obstacle 4 — Stale training config that cannot be repaired in place

The config still contains the dataset sections from ai4bharat's own
training run, pointing at paths on their internal cluster. Worse,
`train_ds` is missing a `shuffle` key that NeMo's initialiser tries to
fill in with a default.

That write fails. OmegaConf has a "struct mode" that forbids adding keys
that were not already present, and the usual escape hatch —
`OmegaConf.set_struct(cfg, False)` — does not help here, because the
write targets a schema backed by a Python dataclass, which is permanently
struct-locked regardless of the surrounding config's setting.

Resolution: delete `train_ds`, `validation_ds` and `test_ds` outright. We
supply our own data loading, so they were never going to be used.

### Obstacle 5 — A constructor argument that exists only in a private fork

Instantiating the decoder raises:

```
TypeError: ConvASRDecoder.__init__() got an unexpected keyword argument 'multisoftmax'
```

ai4bharat trained this with a patched NeMo containing a `multisoftmax`
option that never shipped publicly. The config references it; the public
library has never heard of it.

Dropping an unknown argument is normally risky — it might change the
architecture, producing a model whose shapes no longer match the saved
weights. Here it was verified safe by inspection: the saved CTC decoder
weights are a single plain layer (`decoder_layers.0.weight` and `.bias`),
not the per-language split that flag might have implied. Loading confirms
it — zero missing and zero unexpected keys.

### A structural decision: build the parts, not the model

NeMo's full `EncDecHybridRNNTCTCBPEModel` class always constructs the
RNNT decoder and joint network, which is exactly where Obstacle 5 bites
hardest — and we never use RNNT.

So the class is never instantiated. Instead, three components are built
directly from the config and loaded from the matching prefixes of the
checkpoint's weights:

| Component | Role | Weight prefix |
|---|---|---|
| `AudioToMelSpectrogramPreprocessor` | waveform → mel spectrogram features | `preprocessor.*` |
| `ConformerEncoder` | features → 17 layers of learned representation | `encoder.*` |
| `ConvASRDecoder` | representation → per-frame character scores | `ctc_decoder.*` |

The `decoder.*` and `joint.*` weights in the file are RNNT-only and go
deliberately unused. This is simpler, avoids the failure, and gives
direct control of the forward pass, which adapter insertion needs.

---

## Part 4 — The vocabulary problem

This is the subtlest part of the system, and the part that produced its
most serious bug.

### One vocabulary for 22 languages

The CTC decoder does not output Hindi characters. It outputs scores over
a **single vocabulary shared by all 22 languages**: 5,632 tokens, plus
one "blank" symbol appended at the end, for 5,633 outputs per frame.
(Confirmed by the decoder's weight shape, `[5633, 512, 1]`, against a
5,632-entry vocabulary list.)

Hindi's own tokenizer knows only 256 pieces. So to work in Hindi, we must
know **which 256 of those 5,633 columns are Hindi's**, ignore the rest,
and present the model with a 257-wide view (256 Hindi tokens + blank).

ai4bharat's ONNX bundle ships exactly this mapping in a file called
`language_masks.json`. The NeMo checkpoints do not include it. It has to
be reconstructed.

### How it was reconstructed wrongly, and why nothing noticed

The first attempt matched Hindi's tokenizer pieces against the shared
vocabulary **by string**: for each Hindi piece, find where that text
appears in the big list.

This is wrong, because Hindi, Marathi, Nepali, Sanskrit, Konkani,
Maithili and Dogri all use the Devanagari script and therefore **share
many identical pieces**. 670 token strings appear more than once across
the vocabulary. A plain string lookup returns only one of them —
whichever happened to be stored last. **213 of Hindi's 256 indices (83%)
ended up pointing at another language's columns.**

The failure mode is what makes this worth recording. The model still
trained. The loss still fell smoothly, epoch after epoch, for hours. Of
course it did: the adapter was being asked to optimise toward a scrambled
target space, and it obligingly learned to do so. Nothing in the training
signal indicated a problem.

### The correct derivation

The vocabulary is not arbitrary. 5,632 = 22 languages × 256 tokens, laid
out as **contiguous per-language blocks**, in the same order the
languages appear in `tokenizer.langs`. Hindi is language #6, so its
tokens are simply indices **1536 to 1791**, and that block matches its
tokenizer's pieces exactly, in order.

The implementation now derives the block from that layout and **asserts**
it matches the tokenizer piece for piece, raising rather than proceeding
if the assumption ever fails for another checkpoint.

### How it was caught

Not by inspection — by measurement. Decode the backbone with **no adapter
at all** and compare its WER against the independently measured ONNX
benchmark on identical audio:

| Configuration | WER |
|---|---|
| ONNX benchmark (independent reference) | 57.14% |
| NeMo backbone, string-matched mask | 88.56% |
| NeMo backbone, corrected block mask | 55.37% |

The corrected figure lands within 1.8 points of the reference. The
broken one is 31 points adrift. The check cost about a minute of compute
and invalidated several hours of apparently successful training.

**The operating rule this produces: a falling loss curve proves nothing.
Measure WER.**

---

## Part 5 — Why CTC, and how the loss works

The encoder turns roughly one second of audio into 25 output frames. The
CTC decoder scores every possible token at every frame. But nobody has
labelled which frame corresponds to which letter — the transcript is just
a sentence.

**CTC (Connectionist Temporal Classification)** solves this. It allows a
special "blank" symbol and permits repeats, then considers *every*
possible way the frame-by-frame outputs could collapse into the target
sentence, and sums their probabilities. Training maximises that sum. No
frame-level alignment is ever needed.

The hybrid checkpoint also offers RNNT, which is generally more accurate
but more complex to train and is where the loading obstacle lives. CTC is
also what the project's baseline benchmark used, so staying with CTC
keeps results comparable.

### The degenerate case that destroyed a five-hour run

CTC requires enough frames to fit the target. A one-second clip yields 25
frames; if its transcript needs 30 tokens, no valid alignment exists, and
the loss is **infinity**.

Infinity propagates: through the backward pass, into the optimiser, into
every adapter weight. From then on every weight is `nan`, every
subsequent loss is `nan`, and training silently accomplishes nothing. It
does not crash. The process exits successfully.

This happened at step 289 of 39,236. The remaining 38,947 steps — just
under five hours — computed `nan`.

Three defences now, layered:

1. **Pre-filter.** Before training, drop any utterance where
   `duration × 25 < 2 × target_length`. The factor of 2 leaves room for
   the blanks CTC must insert between repeated tokens.
2. **`zero_infinity=True`** on the loss, which zeroes out any infinite
   value and its gradients — PyTorch's native handling for exactly this.
3. **Gradient clipping** at norm 5.0, bounding the damage from any
   extreme-but-finite gradient.

---

## Part 6 — The adapter itself

### Shape

Each adapter is three operations:

```
input (512 numbers)
  → compress to 64          (down-projection)
  → ReLU                    (non-linearity)
  → expand back to 512      (up-projection)
  → add the original input  (residual connection)
```

The squeeze to 64 is what keeps it small: 512→64→512 is about 66,000
weights, against the ~6.8 million in a full Conformer layer. One adapter
per layer, 17 layers: **1,123,904 trainable weights, 0.98% of the
backbone's 115,111,424.**

### Identity initialisation

The up-projection's weights and bias start at **exactly zero**. So at
step 0 the adapter computes `input + 0 = input` — a perfect passthrough.
The model behaves precisely as it did before the adapters existed.

This matters. A randomly initialised adapter would immediately corrupt a
working model's internal representations, and training would first have
to repair that damage. Starting at identity means training can only
depart from a known-good state deliberately.

A side effect worth understanding: at step 0 the *down*-projection
receives zero gradient, because the gradient reaching it is multiplied by
the up-projection's zero weights. This looks alarming and is correct —
the up-projection moves first, and once it is non-zero the
down-projection begins learning too.

### Insertion by forward hook

The adapters are **not** inserted into the encoder's structure. They are
attached with PyTorch **forward hooks** — callbacks that fire after a
layer computes, receive its output, and return a modified version.

```
layer 1 output → adapter 1 → layer 2 → adapter 2 → ... → layer 17 → adapter 17
```

Why hooks rather than rebuilding the model:

- The encoder's own module tree and state dict stay **untouched**, so the
  pretrained weights can still be saved, reloaded or compared normally.
- Adapters live in a separate `ModuleList`, so they can be saved on their
  own — the checkpoint is 4.5 MB, not 500 MB.
- Swapping adapters (the whole point of MoA-CAS) becomes detaching one
  set and attaching another, with no surgery on the backbone.

---

## Part 7 — What "frozen" means, concretely

Freezing is two separate things, both required:

1. **`requires_grad = False`** on every backbone parameter. Gradients are
   never computed for them, so the optimiser cannot move them.
2. **`.eval()` mode**, which disables dropout and stops normalisation
   layers updating their running statistics. Without this, the backbone
   would drift during training even with gradients disabled.

The optimiser is constructed over `adapters.parameters()` **only** — it
is not merely discouraged from touching the backbone, it has no reference
to it.

This was verified rather than assumed: a forward/backward test confirmed
**all 68 adapter parameters receive gradients and zero backbone
parameters do**.

### The consequence for what the adapter must learn

Because the CTC decoder is *also* frozen, the adapters cannot change how
representations are turned into text. They can only change the
representations themselves, such that the **unchanged** decoder reads
them better. The adapter is not learning to write Hindi; it is learning
to reshape the encoder's internal description of the audio into something
the existing decoder already knows how to read.

---

## Part 8 — Why this should work, and what it actually did

### The expectation

The backbone knows Hindi well — roughly 8% WER on clean, elicited Hindi
speech. On this corpus of software tutorials it scores far worse. The gap
is mostly **domain**: different recording conditions, dense technical
vocabulary, spontaneous lecture delivery. The knowledge is present but
mismatched to the material. That is precisely the situation adapters are
designed for.

### What was measured

Trained on the training split, evaluated on the test split — genuinely
held out, 520 speakers versus 30, with **zero overlap** (verified).

| Evaluation (n = 100) | Baseline | With adapter | Change |
|---|---|---|---|
| Pure-Hindi held-out — WER | 67.18% | **51.81%** | −15.37 points |
| Pure-Hindi held-out — CER | 56.05% | **40.57%** | −15.48 points |
| Full mixed test — WER | 55.37% | **45.44%** | −9.93 points |

Paired over identical utterances: 53 improved, 25 worsened, 22 unchanged;
95% confidence interval [−0.216, −0.092]; t = −4.84. Significant.

### The mechanism, identified

The frozen backbone **truncates**. It emits only 75 words for every 100
in the reference, cutting utterances short. The adapter raises that to
86. Most of the gain is recovered words rather than corrected ones, and
it holds across every utterance-length bucket.

### The honest caveat

The adapter was trained on **monolingual Hindi**, so what it learned is
domain adaptation, not code-switching. It improves the code-mixed set too
(−9.93 points) precisely because the domain problem dominates there as
well. This is a real result and it validates the adapter machinery — but
it is **not yet evidence about code-switching**, which needs the English
expert and the router, neither of which exists.

Absolute quality also remains poor: a 52% word error rate is not a usable
transcription system. The honest summary is that the method moved a bad
number meaningfully in the right direction under a very small compute
budget.

---

## Part 9 — Running it

```bash
# train (CPU)
python scripts/train_profile_adapter.py --lang hi \
    --max-seconds 3600 --save-to checkpoints/hi_adapter.pt

# train (GPU, batched)
python scripts/train_profile_adapter.py --lang hi \
    --device cuda --batch-size 16 --max-seconds 18000 \
    --save-to checkpoints/hi_adapter.pt

# measure the baseline
python scripts/evaluate_adapter.py --lang hi \
    --manifest data/manifests/mucs_hi-en_test.jsonl --limit 100

# measure the adapter on the same data
python scripts/evaluate_adapter.py --lang hi \
    --manifest data/manifests/mucs_hi-en_test.jsonl --limit 100 \
    --adapter-checkpoint checkpoints/hi_adapter.pt
```

Batching exists for GPU. On CPU it is slower than batch size 1, because
padding wastes work that parallelism would otherwise absorb. The GPU path
is implemented but **has not been run** — no CUDA device was available.

---

## Part 10 — Where this can go wrong

Failure modes to watch for, most of which have already occurred once:

| Risk | Symptom | Guard |
|---|---|---|
| Wrong language mask | Falling loss, terrible WER | Block derivation asserted against tokenizer; always run the no-adapter baseline first |
| CTC-infeasible utterance | Loss becomes `nan` forever, no crash | Pre-filter, `zero_infinity`, gradient clipping |
| Overfitting | Training loss falls, held-out WER worsens | Currently only guarded by using a speaker-disjoint test set; no dev split yet |
| Trusting loss | Everything looks fine, nothing works | Measure WER; loss is diagnostic only |
| Adapter helps on average, hurts specific cases | Net gain masks 25% regressions | Report paired win/loss counts, not just means |

---

## Part 11 — What is still missing

1. **A GPU run.** The code supports it; no GPU was available to verify.
2. **Batching validated at scale.** Correct at batch size 4 on CPU;
   untested at the batch sizes that make GPUs worthwhile.
3. **Bengali.** Access granted, code is language-parameterised, never run.
4. **A dev split.** Model selection and early stopping currently do not
   exist; the run stops when the clock does.
5. **Hyperparameter search.** Bottleneck width, learning rate, and which
   layers get adapters are all first-guess values.
6. **The English expert.** Blocked — there is no English speech corpus in
   this pipeline.
7. **The router.** Stage 2 has not been started. Everything here is a
   single adapter; MoA-CAS needs several plus frame-level gating between
   them.

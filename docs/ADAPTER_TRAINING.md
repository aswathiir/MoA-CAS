# Profile Learning (adapter training) — plain-English guide

Stage 1 of MoA-CAS's two-stage training plan: train small "expert"
adapters on top of a frozen, pretrained encoder — one per language
family — before the router (Stage 2) ever gets involved. This covers
`Eia` (Hindi/Bengali); `Een` (English) is intentionally out of scope for
now — there's no English speech corpus in this pipeline yet.

## Why NeMo, not the ONNX model the benchmark used

`ai4bharat/indic-conformer-600m-multilingual` (used for the baseline
benchmark) ships its encoder as a frozen `onnxruntime.InferenceSession`
— inference-only, no autograd graph, no way to attach a trainable
adapter. ai4bharat separately ships per-language NeMo checkpoints
(`ai4bharat/indicconformer_stt_{lang}_hybrid_ctc_rnnt_large`) that ARE
real trainable PyTorch — that's what this code uses instead.

## Five bugs, and how the code works around each

Loading these NeMo checkpoints with the public `nemo_toolkit` doesn't
work out of the box. Getting `moa_cas/adapters/nemo_backbone.py` right
meant working around all five of these, in order — each is called out
at the point it's handled in the code, this is just the narrative:

1. **The HF repo is gated** — separately from the ONNX model's gate.
   Needs a one-time manual "agree and access" click on the repo page
   while logged in (`huggingface-cli login`).
2. **NeMo's own `.nemo` auto-restore is broken for this repo layout** —
   `from_pretrained` downloads fine but then looks for a pre-extracted
   `model_config.yaml` that was never created. `load_backbone` bypasses
   this: downloads via `huggingface_hub.snapshot_download`, then
   extracts the `.nemo` (it's just a tar archive) itself.
3. **The tokenizer config is multilingual-style** (nested
   `langs: {hi: {...}, bn: {...}}`), but NeMo's monolingual loader
   expects a flat `dir`/`model_path`/`vocab_path`. Fixed by promoting the
   target language's sub-entry to the top level before instantiation.
4. **`train_ds`/`validation_ds`/`test_ds` are incomplete stubs**
   referencing ai4bharat's internal cluster paths, and `train_ds` is
   missing a `shuffle` key NeMo's own init code tries to backfill —
   which fails under OmegaConf struct mode no matter what you do
   upstream. Fixed by just deleting these three keys; irrelevant anyway
   since we build our own dataloader.
5. **A `multisoftmax` constructor kwarg** from ai4bharat's private,
   patched NeMo fork doesn't exist in the public `nemo_toolkit`. Verified
   safe to drop: the actual saved decoder weights are a single plain
   layer (not a per-language split), so the vanilla decoder class loads
   the exact same shapes — `load_state_dict` reports 0 missing/0
   unexpected keys.

**Also:** we never instantiate NeMo's full `EncDecHybridRNNTCTCBPEModel`
class at all — its `__init__` unconditionally builds the RNNT
decoder/joint too (where bug #5 actually bites hardest), and Profile
Learning only needs the CTC path anyway. `nemo_backbone.py` builds just
the preprocessor, encoder, and CTC decoder directly.

## The shared-vocabulary alignment problem

The CTC decoder's output is one vocabulary shared across all 22
languages (5632 tokens + 1 blank, appended as the last class). A single
language's own tokenizer only covers a subset of it, and no mask file
ships with the checkpoint to say which subset (ai4bharat's ONNX bundle
has the equivalent, `language_masks.json` — the NeMo checkpoints don't).
`moa_cas/adapters/language_mask.py` reconstructs it by string-matching
the language's own SentencePiece pieces against the shared vocabulary's
token strings — using the compiled `.tokenizer.model` file directly
(via the `sentencepiece` library), not the auxiliary `vocab.txt` export
bundled alongside it, which can use a different display convention
(WordPiece-style `##` prefixes) that won't match at all.

## A wasted 5 hours, and the fix

The first real training run "completed" (exit code 0) but had gone to
`NaN` after 289 of 39,236 steps: one 1.56-second utterance didn't have
enough encoder frames to align its target text (the standard CTC
precondition `input_length >= target_length`), producing infinite loss
that corrupted every adapter weight via `optimizer.step()`. Every
subsequent step then computed `nan` forever, silently — no crash, no
warning in the exit code. `scripts/train_profile_adapter.py` guards
against this: skips any sample where `enc_lens < 2 * target_lens`
(2x margin for repeated-token blanks), skips non-finite losses as
defense in depth, and clips gradients.

## What's actually been verified

- **Hindi**: loaded with 0 missing/0 unexpected keys across all three
  pieces (686 encoder keys, 2 preprocessor, 2 CTC decoder). A real
  5-hour training run (7 epochs, 46,394 steps, 2.58 utterances/sec on
  CPU) over the full 7,083-utterance pure-Hindi pool showed zero
  non-finite losses and a clean per-epoch loss drop (68.07 → 43.65).
- **Bengali**: HF gate access granted, architecturally identical code
  path, but never actually run end-to-end — untested.

## Running it

```bash
poetry run python scripts/train_profile_adapter.py --lang hi --max-seconds 18000
poetry run python scripts/train_profile_adapter.py --lang bn --max-seconds 3600
```

Requires `nemo_toolkit[asr]` and `sentencepiece` installed, and gated HF
access granted to the target language's repo (see bug #1 above).

## What's not built yet

- Extrapolated compute estimate (from the Hindi pilot's throughput): a
  full Hindi+Bengali training run (~86,739 utterances, ~10 epochs) is
  roughly 93.5 hours on CPU, or an estimated 3-6 hours on a single
  modest GPU. Not yet run on an actual GPU.
- Stage 2 (the router) — nothing here touches it.
- `Een` (English) — no corpus, no backbone decision made yet.

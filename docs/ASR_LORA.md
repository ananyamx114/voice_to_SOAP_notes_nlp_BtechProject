# Whisper Large-v3 LoRA experiment

The requested larger-data, five-epoch experiment is documented in
[ASR_LORA_EXPANDED.md](ASR_LORA_EXPANDED.md). Its verified cohort contains at least
two hours of English and one hour each of Hindi/Marathi from the same datasets.
The original three-epoch results below remain a separate control.

Publication was authorized on 15 November 2026 after results review. The shared
entry point is [results/asr](../results/asr/README.md); source execution artifacts
stay under ignored data/outputs or private Kaggle storage. Software tests alone
do not establish full-model training fit or an ASR accuracy improvement.

## Model and experiment contract

The base is the complete `openai/whisper-large-v3`, pinned to
`06f233fe06e710322aca913c1bc4249a0d71fce1`, about 1.55 billion parameters and
3,087,130,976 checkpoint bytes. No quantization or smaller Whisper substitution.
`configs/asr-lora.yaml` places rank-16 LoRA on query/value projections in both
encoder and decoder, including decoder cross-attention. All original parameters
are frozen. Trainable adapter weights stay float32; accelerator base weights use
float16/autocast with gradient scaling. CPU uses float32. Non-reentrant gradient
checkpointing, batch size one and accumulation 16 reduce memory use without
dropping model layers. These initial hyperparameters need validation; they are
not an already optimized recipe or a claim of state-of-the-art accuracy.

Normal LoRA introduces trainable low-rank weight updates. It does not add more
Transformer blocks. The trained adapter is task- and base-revision-specific and
cannot be exchanged with the repository's SOAP adapters. At ASR inference it is
merged into the full base in memory; the raw base cache remains unchanged.

The starting machine has 8 GB shared RAM/M1 Metal. That established inference
only. A subsequent full-model optimizer resource check passed on English segments
(16 accumulated microbatches, about 142 seconds; Metal driver allocation about
4.47 GB at the update boundary). The subsequent multilingual run completed
three epochs and 90 optimizer updates after the memory correction below.
Each teammate must inspect their own
machine. Keep normal memory protections enabled. An OOM is a failed resource
check; preserve its artifacts, investigate avoidable retention, and measure again.
If the full model still does not fit, use a computer with enough memory.
There is no automatic smaller-model, CPU or quantization fallback after failure.
An explicit local config can select decoder-only adapters, but that is a separate
experiment which leaves the acoustic encoder unadapted; record and compare it.

### Recorded long-run memory correction

The first sustained attempt completed 13 updates and hit the configured 5.33 GiB
Metal ceiling during update 14. It retained a valid optimizer checkpoint at
update 10. The error, original source snapshot and session metadata are retained.
No trained accuracy result was produced by that incomplete attempt.

The correction keeps the complete model, all supervised tokens and the same
hyperparameters. Decoder target lengths are bucketed to multiples of 32 using
`-100` loss padding; regression tests verify that valid-token logits/loss are
preserved by causal attention. Generation receives no target labels or decoder
masks derived from them. Each MPS batch/inference completes before its temporary
tensors are released and unused allocator memory is returned. This bounds graph
shape variety and avoids carrying completed batch allocations forward.

Resume is explicit: add `--resume-source-patch "reason"` to `asr-train` or the
experiment coordinator only for a reviewed compatible correction. Model/config,
data, device and dependency checks remain mandatory. Previous session metadata
is archived under `training/session-history`; old/new source hashes and the
reason are recorded. Optimizer and RNG states restore from the saved update.
Padding can change subsequent dropout draws, so a source-corrected continuation
must not be called bit-identical to the earlier implementation.

The resumed update recorded about 3.40 GB Metal driver allocation between updates,
versus about 4.85 GB at the earlier checkpoint. These are boundary observations,
not measured peak system memory or proof of a complete run. Memory protection
remains enabled at the same limit. Completed results must be assessed separately.

Paired production decoding also preserves CPU/accelerator RNG state around model
loading so adapter initialization cannot change the decoding random stream.
Decoder implementation hashes enter provenance/resume checks; comparisons reject
different decoder implementations. Use fresh output directories for older
decoding implementations instead of overwriting their results.

Long-form evaluation exposed separate MPS allocation failures. The decoder now
synchronizes and releases unused buffers between completed windows. Timestamp
extraction also projects the configured alignment heads before concatenation,
rather than concatenating every head from all decoder layers first. It delegates
the remaining timestamp computation to the unchanged Transformers 4.57.6 method.
A regression test checks exact timestamp equality for scalar and per-example
frame counts, preserving the real model config and original attention outputs.
These changes preserve the full model and timestamp output; they do not relax
the memory limit. Real failing-clip and complete-run verification are recorded
in the local experiment artifacts. Failed evaluation attempts are retained in
`evaluation-attempt-*`; each decoder change requires fresh matched base/adapter
runs. Do not combine different implementation hashes into one comparison.

## Current local experiment

The current recipe prepares 12 randomly sampled BeTraC training consultations
(selection seed 42), four different validation consultations (seed 43, excluding
the two original BeTraC diagnostic encounters), plus FLEURS Hindi/Marathi clips.
BeTraC yielded 345 accepted training segments / 445 rejected units and 175
accepted validation segments / 171 rejected units. Rejections remain auditable.
No rejected units are silently turned into training labels.

FLEURS train/validation audio is pinned at converted dataset revision
`168de341b3db6859a9bac1c50a2ef5e3b47647e0`; its CC BY 4.0 source card is pinned
at `70bb2e84b976b7e960aa89f1c648e09c59f894dd`. Read the retained attribution and
[original dataset card](https://huggingface.co/datasets/google/fleurs).
The selected training split contributes 64 Hindi and 63 Marathi whole clips;
one selected Marathi training clip exceeded 30 seconds and was rejected, not
truncated. Validation adds 12 clips per language. These are general read speech,
not clinical-domain Hindi/Marathi recordings and not a replacement for them.

Total supervised training: 472 segments, 3,893.74 seconds (about 65 minutes).
Validation: 199 segments, 1,469.41 seconds. All original splits are retained;
encounter/group/source hash checks passed. The second full-model resource probe
also passed on the 16 largest decoder targets (maximum 308 target tokens), with
one optimizer update taking about 183 seconds. That establishes this update,
not a guarantee of an entire training run or peak total-system memory.

Separate FLEURS test files supply 32 Hindi and 32 Marathi clips (seed 44), checked
for source-group overlap with training/validation before model evaluation.
The paired comparison also reruns the seven recordings in the earlier result
table: four Eka clips, two complete BeTraC consultations and one complete Sully
consultation. This is **71 selected recordings**, not full Eka/BeTraC benchmarks.
No Marathi clinical test data has been supplied. FLEURS test scores concern
general speech. Prior Eka clips remain diagnostic because they were inspected.

Training completed three epochs / 90 optimizer updates. Epoch 3 was selected
using macro language validation WER. Its validation WER changed from 8.48% to
7.99% (English), 34.03% to 31.60% (Hindi), and 77.94% to 71.08% (Marathi).
These are validation results, not independent test results. The initial paired
evaluation showed improved diagnostic Hindi, unchanged diagnostic English and
regressions on the two BeTraC and one Sully consultations; final tables must
include those outcomes. A separate epoch-1 English candidate is selected using
English validation WER alone and evaluated separately, retaining both sets of
results. Do not select checkpoints individually using test-recording scores.

Checkpoint selection currently uses greedy generation on short validation
segments. Production comparisons use native sequential decoding with temperature
fallback and word timestamps. Validation improvements therefore need confirmation
under the production decoding policy and on full consultations before promotion.

The completed shared-adapter comparison is recorded in
[the aggregate results](../results/asr-lora-run1/README.md): WER improved in
three of six dataset/language groups, regressed in two and stayed unchanged in
one. Marathi CER increased despite its WER improvement. This adapter is not an
across-the-board quality upgrade.

One later Marathi evaluation clip failed even in a fresh worker under the
originating machine's deliberately strict 5.33 GiB MPS cap. The retry used a
protected 5.87 GiB cap (local high-watermark ratio 1.1, low ratio 0.9), and the
complete shared-adapter evaluation then finished. Memory protection was never
disabled. These are historical local execution settings, not shared defaults
or requirements for teammates. Recovery records remain under the ignored
experiment's `evaluation-recovery/` directory.

Portable additions:

```text
python scripts/project.py download-data --dataset fleurs-training
python scripts/project.py prepare-fleurs --outdir data/processed/asr-lora-fleurs-run1 --train-per-language 64 --validation-per-language 12 --seed 42
python scripts/project.py download-data --dataset fleurs-test
python scripts/project.py prepare-fleurs --dataset-dir data/raw/fleurs-test --outdir data/processed/asr-lora-fleurs-test-run1 --test-per-language 32 --seed 44
```

Training/validation download is about 4.8 GB; the separate test selection adds
about 1.2 GB. Both are explicit selections, excluded from `download-data --dataset
all`. Downloading test audio does not permit using it for training or checkpoint
selection. The scripts pin and verify bytes, retain the license card, and never
execute dataset code. Prepared whole-clip pairs have `dataset_clip_pair`
provenance, not invented human review; arbitrary crops are rejected.

The sequential experiment coordinator combines verified prepared manifests,
trains and selects by macro language validation WER, then evaluates the base and
adapter with identical production settings and compiles `RESULTS.md`:

```text
python scripts/project.py asr-experiment --clinical-train data/processed/asr-lora-clinical-train-run1-v2 --clinical-validation data/processed/asr-lora-clinical-validation-run1 --fleurs data/processed/asr-lora-fleurs-run1 --fleurs-test data/processed/asr-lora-fleurs-test-run1 --outdir data/outputs/asr-lora-experiment-run1 --device auto --rerun-base
```

This is an hours-long real-model experiment. The originating machine uses an
ignored local shell environment with stricter MPS allocator watermarks and an
idle-sleep inhibitor while the process runs; these are not shared hardware
requirements. All progress and predictions stay under ignored `.local/` and
`data/outputs/`. `--prepare-only` audits without loading models; `--resume`
continues a matching local experiment. `--full-eka` explicitly expands evaluation
to all 3,939 Eka test clips for both models, substantially increasing compute.
Source audio for all Eka clips totals about ten hours; the selected 71-recording
experiment must never be described as that full benchmark.

## Dependency setup

Use `python3` before activation on macOS/Linux, or `py -3.12` on Windows. After
activation the commands below use `python`. The launcher handles local paths and
the ignored model cache. Commands are single lines for PowerShell portability.

```text
python scripts/project.py system
python scripts/project.py setup --profile asr-training
python scripts/project.py test
python scripts/project.py hf-check
```

PEFT and number normalization are in `requirements-asr-training.txt`. There is no
bitsandbytes/CUDA-only requirement. Existing HF login can be reused; the proposed
English data, Whisper and aligner are public. No extra paid API token is needed.
If authentication is required locally, use `python scripts/project.py hf-login`
and enter a personal Read token in the hidden prompt. Never commit it or share
accounts. New gated Hindi/Marathi resources require the user's accepted terms.

## Training audio and correct alignment

BeTraC has 7,200 synthetic English training consultations and 400 validation
consultations, released under CC BY 4.0. The existing downloaded `dev` archive is
**validation**, not training. Eka is **test only**. Sully/PriMock audio is for
external diagnostic evaluation; its ACI overlap must not become an independence
claim. Hindi/Marathi supervised training requires separate authorized audio and
accurate transcripts. English fine-tuning does not establish gains in them.

BeTraC's turns have no audio boundaries. Its event Unix timestamps describe text
generation, not the subsequent TTS waveform. Giving a 30-second arbitrary crop a
whole consultation transcript would corrupt supervision. `asr-align` instead:

1. Reads the selected pinned train/dev archive without executing dataset code.
2. Preserves raw audio/transcript and metadata, source SHA-256, source IDs and
   parent group. Removes speaker prefixes, bracketed stage directions, Markdown
   emphasis and escaped newlines from a separate spoken-target view.
3. Uses independent English `facebook/wav2vec2-base-960h` at pinned revision
   `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` to compute CTC emissions. This is
   an alignment tool; Whisper remains the ASR model being trained/evaluated.
4. Processes CTC audio in 20-second cores with 1-second context on the exact
   convolution frame grid, then aligns the complete supplied text with an exact
   monotonic CTC path. Repeated letters require separating blank frames.
5. Aligns roughly 20-word text units, keeping original text as Whisper labels.
   Only the aligner's view expands numbers to English words. Unknown CTC tokens
   cause failure; they are not silently deleted.
6. Rejects overlapping/invalid intervals, durations outside 0.5–29.5 seconds,
   mean token log probability below -1.5, or worst token score below -5. These
   are initial screening thresholds, **not validated gold boundary accuracy**.
   Keeps every candidate/rejection and raw reference for review.

Listen to aligned audio and inspect its matching text, especially boundaries,
drug names and doses, before a substantive training experiment. Automatic
confidence can miss errors. The trainer accepts explicit human-reviewed segment
manifests too. Do not invent review status. Text cleanup is not medical fact
normalization. Different number pronunciations or non-spoken annotations can
make automatic alignment unsuitable; review/re-align those examples.

Run these locally under the current authorization:

```text
python scripts/project.py download-data --dataset betrac-train
python scripts/project.py asr-align --archive data/raw/betrac-train/data/train-00000.tar --split train --outdir data/processed/asr-lora-train-shard0
python scripts/project.py asr-align --archive data/raw/betrac/data/dev-00000.tar --split validation --outdir data/processed/asr-lora-validation
```

`betrac-train` explicitly downloads the first full training shard, about 1 GB;
the ordinary `all` benchmark selection does not download training data. The full
training collection is about 8.7 GB across nine shards. The first-shard experiment
is not the full collection. To expand, add a reviewed/pinned download selection
and combine disjoint segment manifests with correctly resolved audio paths.
No archive split is reassigned. The importer refuses existing output directories;
failed/partial exports retain `preparation.json` with failure state. Explicit
`--resume` reuses completed alignments only when archive/options, prepared audio
hash and target units match; the accepted manifest is rebuilt atomically.
The exact backtracking table packs four predecessor decisions per byte to keep
long consultations within a bounded memory footprint. The aligner adds about 360 MB of model
weights. Source audio copies, exports, adapter/optimizer files and all predictions
stay under ignored local data directories. Preserve at least 12 GiB free before
training-data downloads; exact runtime requirements must be measured locally.

For an initial alignment integration check, use explicit `--limit 2` with new
train/validation output names. This runs the actual aligner, but two encounters
are not sufficient to claim a trained final system. Keep integration artifacts
distinct from substantive experiments.

## Audit, resource check, then training

`segments.jsonl` stores one object per aligned interval with fields `id`,
`encounter_id`, `group_id`, `dataset`, `dataset_revision`, `split`, `audio_path`,
`source_sha256`, `start`, `end`, `text`, `language`, and `alignment`. Paths resolve
relative to the JSONL file. Alignment has `method` (`ctc_forced_alignment` or
`human_reviewed`) and `status: accepted`. CTC exports additionally keep complete
encounter text hashes. Manual corpora must provide genuine provenance/review.

The trainer rejects train/validation encounter, group, source-audio and full
encounter-text leakage; reports repeated short texts without dropping ordinary
shared phrases; checks every audio hash/boundary; rejects non-training examples
in train; rejects non-validation examples in validation; and refuses oversized
targets instead of truncating or silently skipping them. Original datasets may
still share synthesis voices/personas; this does not prove speaker-independent
or clinically independent evaluation.

```text
python scripts/project.py asr-train --train data/processed/asr-lora-train-shard0/segments.jsonl --validation data/processed/asr-lora-validation/segments.jsonl --outdir data/outputs/asr-lora-audit --audit-only
python scripts/project.py asr-train --train data/processed/asr-lora-train-shard0/segments.jsonl --validation data/processed/asr-lora-validation/segments.jsonl --outdir data/outputs/asr-lora-resource-check --probe
```

The resource check performs **one real optimizer update on full Large-v3**, using
the configured gradient accumulation, verifies an adapter changed and records
loss, gradient norm, time and available process/accelerator memory counters.
It does not evaluate WER or save a deployable adapter. It does not guarantee the
largest segment, validation generation or a complete run will fit. All later
failures are recorded explicitly; there is no silent substitution.

If it passes and alignment review supports training:

```text
python scripts/project.py asr-train --train data/processed/asr-lora-train-shard0/segments.jsonl --validation data/processed/asr-lora-validation/segments.jsonl --outdir data/outputs/asr-lora-shard0-run1
```

No default sample limit applies. This trains all accepted segments in the input
manifests for up to three epochs, with warmup/linear decay, norm clipping,
macro language WER checkpoint selection (each language has equal weight) and
early stopping. Hyperparameters are
shared in `configs/asr-lora.yaml`; local alternatives belong in ignored
`configs/*.local.yaml` and must be passed explicitly with `--config`.

Before training, the unmodified base generates on the exact validation segments.
Every epoch uses those same segments and deterministic decoding settings, with
no reference prompt. Keep baseline/adapter predictions and metrics. Validation
selects the checkpoint; it is not an independent final test. A gain on aligned
clips also does not establish better long-form consultation accuracy.

The run saves an atomically replaced adapter/AdamW/scaler/RNG checkpoint every
10 optimizer updates and at epoch boundaries. It never checkpoints the frozen
base. Each completed epoch also retains `epoch-N/checkpoint.pt`, so later
language-validation selection does not depend on a monitoring helper. A crash
between the main checkpoint save and epoch archiving is repaired on resume;
different existing epoch archives are never silently overwritten. Plan disk
space for one additional adapter/optimizer checkpoint per completed epoch.
This storage-only follow-up was added after the measured run and verified with
the interrupted-training contract test; reported weights/metrics are unchanged.
Resume using the same arguments/output directory plus `--resume`; data,
config, device, implementation and dependency versions must match. An interrupted
partial accumulation restarts from the last completed saved optimizer update.
Progress logs can therefore contain repeated update numbers after recovery;
the restored checkpoint and run status define the authoritative state. Never
load optimizer checkpoints from untrusted sources.

Outputs include `run.json`, `training.jsonl`, local optimizer state, per-epoch
predictions/metrics and `best-adapter/{adapter_model.safetensors,adapter_config.json,
asr_adapter.json}`. A best checkpoint can still regress against the base:
`validation_wer_change` and `improved_validation_wer` explicitly record that.
Do not promote an adapter simply because it trained or loss decreased.

## Evaluate the real long-form ASR section

Keep the unmodified ASR config. Choose a fixed held-out manifest and use separate
outputs. These examples reuse the existing four-clip Eka **diagnostic** subset;
use the full held-out data for substantive evaluation. Do not repeatedly select
hyperparameters on test scores. Run model stages sequentially.

```text
python scripts/project.py asr-run --manifest data/manifests/eka-smoke.csv --outdir data/outputs/eka-lora-paired-base
python scripts/project.py asr-run --manifest data/manifests/eka-smoke.csv --adapter data/outputs/asr-lora-shard0-run1/best-adapter --outdir data/outputs/eka-lora-paired-adapter
python scripts/project.py asr-compare --base data/outputs/eka-lora-paired-base --adapted data/outputs/eka-lora-paired-adapter --output data/outputs/eka-lora-comparison.json
```

The loader checks ASR task, exact base revision, completion and adapter checksum.
Resource-check outputs and SOAP adapters cannot be used. Native sequential
Large-v3 decoding and raw word timestamps remain intact. `--resume` checks the
adapter checksum too. The paired comparer refuses missing/failed encounters,
different audio/reference/regions/seeds, or changed decoding/base settings, and
reports WER/CER changes separately by language. It never drops failures to make
scores look better. Inspect dose/negation/omission errors and timestamps directly;
all downstream generated SOAP notes still require clinician approval.

## Reproduce the English validation checkpoint comparison

The same jointly trained model has its minimum English validation WER at epoch 1,
while Hindi/Marathi minima occur at epoch 3. The original shared-checkpoint report
is retained. The [separate English comparison](../results/asr-lora-run1/english-validation-checkpoint.md)
matches base WER on Eka/BeTraC and lowers Sully from 22.21% to 22.01%.
The [language-validation summary](../results/asr-lora-run1/validation-selected.md)
has four WER gains and two ties, with Marathi CER still worse. This remains an
exploratory diagnostic analysis, not an improvement on every group or clinical
validation. No checkpoint is chosen per test recording/dataset.

After completing the shared run, use fresh output paths:

```text
python scripts/export_asr_epoch_adapter.py --run-dir data/outputs/asr-lora-experiment-run1 --language en --outdir data/models/asr-adapters/large-v3-english-validation-best
python scripts/evaluate_asr_english_checkpoint.py --run-dir data/outputs/asr-lora-experiment-run1 --adapter data/models/asr-adapters/large-v3-english-validation-best --outdir data/outputs/asr-lora-english-validation-checkpoint-run1 --device auto
python scripts/report_asr_lora_experiment.py --run-dir data/outputs/asr-lora-experiment-run1
python scripts/report_asr_checkpoint_selection.py --shared-run data/outputs/asr-lora-experiment-run1 --english-run data/outputs/asr-lora-english-validation-checkpoint-run1 --english-adapter data/models/asr-adapters/large-v3-english-validation-best --outdir data/outputs/asr-validation-selected-checkpoints
```

The exporter uses the trainer's completed-epoch checkpoint, with compatibility
for the first run's retained `epoch-archives/` layout. The report checks language
validation minima, adapter hashes, complete matching coverage and equivalent
English controls. It compiles existing verified predictions; automatic language
detection/adapter routing is not implemented by this report. Run ASR explicitly
with the selected `--adapter` on each known-language manifest, sequentially.

## Attribution and technical references

BeTraC: Labrak et al., *Generating Synthetic Doctor-Patient Conversations for
Long-form Audio Summarization*, CC BY 4.0; retain the citation and license in the
[dataset card](https://huggingface.co/datasets/BeTraC/betrac-2026).
The aligner is Apache 2.0; retain its cached
[model card](https://huggingface.co/facebook/wav2vec2-base-960h).
Whisper preprocessing and label handling follow the
[Hugging Face Whisper training guide](https://huggingface.co/blog/fine-tune-whisper).
Adapter integration uses [PEFT](https://huggingface.co/docs/peft/v0.17.0/en/index);
this implementation uses ordinary LoRA, not the quantization in its int8 tutorial.

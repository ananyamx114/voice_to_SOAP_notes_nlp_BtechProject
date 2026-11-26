# Hindi/Marathi ASR improvement work — 26 November 2026

The previous 12-hour experiment remains paused at update4100 and recoverable.
Changes below define new scoring and experiments; they do not change archived
metrics, raw transcripts, audio, dataset permissions or clinician-approval rules.

## Scoring correction

New inference/comparison metrics default to **numeric_indic_v2**. It applies
NFC and digit-script normalization, keeps Indic vowel signs/virama/nukta, retains
decimal separators, signs, numeric fractions/ranges/times and percent signs,
and separates ordinary word punctuation instead of concatenating words.
Parenthesized text and negation remain present. Numeric punctuation is retained
conservatively: no locale-dependent decimal/thousands interpretation is guessed.

`2.5 mg` differs from `25 mg`; `-5` differs from `5`; `1/2` differs from `12`.
The metric also reports count-only numeric-inventory and negation-marker flags.
These flags are not dose validation or clinical assertion understanding.

Historical training configs without an explicit protocol keep
**punctuation_delete_v1** for reproducible checkpoint selection. New experiments
must specify the protocol. Existing published files are not overwritten or
relabelled. Raw references/hypotheses are immutable. To compare both scorers on
saved predictions without loading any model:

```bash
python scripts/rescore_asr.py --predictions PRIVATE_PREDICTIONS.json --output .local/rescored.json
```

Input accepts the private prediction array or JSONL; exported JSON contains
aggregate scores and a source hash, no transcripts or encounter identifiers.
For complete paired full-encounter results:

```bash
python scripts/project.py asr-compare --base BASE_RUN --adapted ADAPTER_RUN \
  --output .local/comparison-v2.json --scoring-protocol numeric_indic_v2
```

Changing scoring when resuming a completed-output directory is rejected by run
provenance. Use a new output or rescore saved text explicitly.

## Diagnose before spending another long training run

The frozen epoch12 adapter's actual training-subset WER and validation decoder
ablation distinguish memorization/generalization from decoder/capacity problems:

```bash
python scripts/diagnose_asr.py --train PRIVATE_TRAIN.jsonl \
  --validation PRIVATE_VALIDATION.jsonl --adapter PRIVATE_SELECTED_EPOCH12 \
  --outdir .local/asr-diagnostics --device cuda:0 --beams 1 5 --seconds 4500
```

The real Large-v3 checkpoint is used with known language/transcribe hints,
FP16 on CUDA, no quantization and identical raw audio/references. A seeded
32-group-per-language training subset is diagnostic only; all frozen validation
segments are evaluated. Labels are removed before generation. Both scorers are
reported for each stage. Every encounter is saved atomically and reruns reuse
the same protocol/results. Budget exhaustion pauses without dropping completed
predictions. No optimizer or training state is changed, and no test data is used
to choose the decoder. Private raw predictions remain outside Git.

Native Hindi/Marathi review of high-error recordings remains necessary. Model
outputs cannot certify that source transcripts/pronunciations are correct.

## Expanded cohort and balanced sampling

The existing four-hour-per-language cohort has2,096 English /1,287 Hindi /1,118
Marathi segments. Uniform segment sampling consequently gives46.6%/28.6%/24.8%
exposure. `language_balanced` uses a deterministic equal-count schedule at the
same total examples/updates per epoch, cycling each language's examples before
repeating. The original uniform-segment order is unchanged when that policy is
selected. Repetition adds exposure, not new linguistic information.

Already-downloaded FLEURS training data contains additional eligible candidates.
Build a separate cohort, retaining the exact239 frozen validation identities,
text, boundaries and audio hashes, and existing clinical English:

```bash
python scripts/prepare_asr_expansion.py --prepared-archive PRIVATE_BASE005.tar \
  --fleurs-source data/raw/fleurs-training --tokenizer LOCAL_WHISPER_TOKENIZER \
  --output data/processed/asr-expanded-v4
```

This does not download models/data. It verifies the archive and pinned Parquet
hashes/license card, decodes train audio, checks mono16kHz/duration/finite values,
rejects oversized targets and duplicate audio, excludes validation sentence
groups and audits train/validation hashes. FLEURS test is never read. References
remain the dataset's original text; no model-generated pseudo-labels are added.
Attribution cards and rejection counts are retained. Output is private/ignored.

First compare the new balanced policy on the original cohort, then add expanded
data in a separately named experiment. Do not change decoder, rank and data
simultaneously and attribute the result to one cause. No improvement is claimed
until measured. A full expanded-data run uses a new experiment directory:

```bash
python scripts/project.py asr-train --config configs/asr-lora-balanced.yaml \
  --train data/processed/asr-expanded-v4/train.jsonl \
  --validation data/processed/asr-expanded-v4/validation.jsonl \
  --initial-adapter PRIVATE_SELECTED_EPOCH12 --outdir data/outputs/asr-balanced-v4 \
  --device cuda:0
```

Warm-start validates base/revision, adapter hash, projections and LoRA settings;
optimizer/scaler/step count reset for the new schedule. Initial-adapter validation
is scored under the new protocol and retained as a selectable epoch0 checkpoint,
so a worse new run cannot discard its starting candidate. This is explicitly
new training, not a bitwise-identical continuation of4100. Resume an existing
new run with `--resume`, without `--initial-adapter`. Source/config/manifest/device
guards remain in force. Five epochs with patience3 are an initial controlled
budget, not a guarantee or a replacement for the original20-epoch run.

Changing sampler, cohort or scoring in an old run is rejected. Keep recovery4100
and the selected epoch12 adapter immutable. Teammates inspect their own hardware;
GPU IDs, credentials, model/data caches and local execution flags stay outside Git.

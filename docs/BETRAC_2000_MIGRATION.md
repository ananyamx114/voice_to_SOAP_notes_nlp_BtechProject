# BeTraC-2000 dataset migration

Current training default: **3 epochs**, explicitly requested after dataset
preparation. This applies equally to DIRECT and HYBRID through both CLI and
Python API defaults. Other training settings remain unchanged. The historical
statements below about unchanged hyperparameters describe the migration itself.

## Current execution: approved removal of the standalone facts cap

The revised selection succeeded: **2,000 encounters**, split into **1,600 train,
200 validation and 200 internal test**, seed 42. It inspected 6,642 candidates
from eight cached official training shards and stopped after finding 2,000.
The remaining candidates/shard were not needed. Source split is still official
`train`; neither ACI nor official BeTraC dev is included.

Only the separate facts rejection was disabled (`max_hybrid_facts_tokens: null`).
All original facts are retained. The transcript cap remains 2,200, original and
serialized SOAP caps remain 700, and both complete training sequences remain
limited to 4,096 tokens with no truncation. The pipeline algorithms, model and
training hyperparameters are unchanged.

| Tokens in the selected 2,000 | Min | Median | p95 | Max |
| --- | ---: | ---: | ---: | ---: |
| Original transcript | 500 | 1,445 | 2,027 | 2,199 |
| Original SOAP | 88 | 429 | 613 | 698 |
| Serialized facts | 1 | 883.5 | 1,800 | 2,714 |
| Full DIRECT training sequence | 887 | 2,017 | 2,679 | 2,996 |
| Full HYBRID training sequence | 939 | 3,044 | 3,990 | 4,096 |

The 4,642 rejected candidates had these first-failing reasons: 3,005 transcript
length, 1,479 full HYBRID sequence length, 143 original SOAP length, and 15
serialized SOAP target length. Counts describe the scanned prefix, not all 7,200
official training encounters. First-eligible/prefix selection bias remains.

The actual files are under `data/processed/betrac_2000/`, including split IDs,
hashes, rejection reasons, and provenance. `independent_validation.json` confirms
all 2,000 existing DIRECT/HYBRID encodings were recomputed, normalized dialogue
lengths also pass, seed-42 ordering is reproducible, split IDs/hashes match, no
duplicate IDs or normalized transcripts exist, and all files load through the
existing dataset loader. **109 tests pass**. This is dataset
preparation, not a model training or quality result. No Kaggle/GPU training was
executed, no model settings were changed, and nothing was committed or pushed.

Current approved selection removes the standalone facts cap. All facts are kept;
the 4,096-token full DIRECT/HYBRID sequence limit still applies. The original
facts-700 attempt below is retained as historical evidence, not current status.

## Historical facts-700 execution: 14 November 2026

**Blocked: the requested 2,000-example dataset could not be created.** All nine
pinned training shards and all 7,200 encounters were inspected. Only **797** meet
all configured limits with the existing pipeline. No train/validation/test split
files were published, no partial-size substitute was used, and no training ran.
The new default paths intentionally fail closed rather than silently using legacy
data. The complete dataset migration infrastructure is implemented, but DIRECT
and HYBRID are **not ready for this 2,000-example experiment**.

First-failing rejection reasons (mutually exclusive, not an analysis of all
possible violations per encounter):

| Reason | Rejected |
| --- | ---: |
| Original transcript over 2,200 tokens | 3,271 |
| Existing serialized HYBRID facts over the attachment's 700-token budget | 2,960 |
| Original SOAP over 700 tokens | 155 |
| Serialized existing SOAP target over 700 tokens | 17 |
| Total | 6,403 |

Statistics for the 797 eligible candidates, using the actual pinned tokenizer:

| Tokens | Min | Median | p95 (nearest rank) | Max |
| --- | ---: | ---: | ---: | ---: |
| Original transcript | 500 | 1,360 | 2,046 | 2,198 |
| Original SOAP | 167 | 411 | 607 | 680 |
| Serialized HYBRID facts | 1 | 373 | 666 | 700 |
| Complete DIRECT training sequence | 887 | 1,912 | 2,693 | 2,975 |
| Complete HYBRID training sequence | 939 | 2,331 | 3,225 | 3,677 |

Both complete encoders succeeded for these candidates; the 700-token facts cap
and transcript cap are major selection bottlenecks. This does not show that the
4,096-token sequence budget itself is impossible. Any revised selection budget
requires explicit approval and a new measured run; none was silently changed.

Historical artifacts: `data/processed/betrac_2000_facts700/selection_report.json`,
`rejections.jsonl`, and `eligible_incomplete.jsonl`. These are ignored, not a ready
training dataset. Training archives remain in the ignored Hugging Face cache;
audio was not extracted. ACI and BeTraC dev remain intact.

Verification: **8 migration tests and 108 full-suite tests passed**. The production
split sizes are demonstrated by synthetic integrity tests; they were not achieved
on the real source. Source comparison verified that the only existing pipeline
source edits are the six CLI dataset-default paths.

This migration changes dataset selection and CLI dataset defaults only. It does
not change ASR, diarization, clinical facts, assertion/temporal rules, prompts,
SOAP mapping, generation, model, LoRA settings, training functions, verification,
metrics, UI or orchestration. The attachment's suggested hyperparameter changes
are intentionally NOT applied because the controlling request forbids them.

## Selection contract

`configs/betrac_2000.json` pins the BeTraC repository revision and the existing
Qwen2.5-0.5B-Instruct tokenizer revision. Source: official **train** only.
Limits: original transcript 2,200 tokens; original SOAP and serialized four-field
target 700 tokens; no standalone HYBRID facts cap; full existing
DIRECT and HYBRID encoded training sequences 4,096 tokens including target/EOS.
No truncation or facts removal is allowed. Existing SOAP heading conversion is
reused unchanged; the original transcript and reference are retained verbatim in
`transcript` and `reference_soap_raw`, alongside loader-compatible `dialogue` and
four-string `soap`. Mapping warnings remain visible.

The builder walks shards and encounter keys deterministically, stops only after
2,000 eligible records, then shuffles using seed 42 into 1,600/200/200. This is
first-eligible prefix sampling, not an unbiased sample of the full corpus.
Exact/whitespace-and-case-normalized transcripts, source IDs and metadata IDs
are deduplicated before splitting. Semantic/persona relatedness cannot be
established from these checks and is not claimed. Any known related-ID grouping
must be audited before substantive experiments.

Official development data and ACI remain untouched and are not included. These
200 test records constitute an **internal split of official training data**, not
an official BeTraC test split. Synthetic reference notes are not clinician labels.
Retain the BeTraC dataset-card CC BY 4.0 attribution and citation. This migration
does not establish clinical factuality or citation accuracy.

## Preparation (Windows PowerShell)

Use the existing verified tokenizer environment; do not upgrade dependencies.
The following command downloads pinned training archives one at a time into the
ignored cache, verifies LFS SHA-256 and size, and skips audio extraction. Tar
archives themselves include audio bytes; the source does not offer separate text
shards. Downloads are reused on reruns. Preserve at least 12 GiB free space.

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m voice_to_soap.betrac_dataset
```

Existing output directories are refused. For a fresh rerun, pass
`--output data/processed/betrac_2000_rerun`, retaining the same config/cache.
Never delete legacy data to make room. If insufficient examples qualify, the
output contains `selection_report.json`, `rejections.jsonl` and
`eligible_incomplete.jsonl`; no train/validation/test files are published. Report
this as incompatibility with the current limits/unchanged extractor, not a
completed 2,000-example dataset. Do not silently use partial candidates.

A successful output contains `all_2000.jsonl`, `train.jsonl`, `validation.jsonl`,
`test.jsonl`, `split_manifest.json` and `selection_report.json`. DIRECT and HYBRID
load the exact same files; no loader change is needed. Training/evaluation CLI
defaults now point to this directory and fail rather than falling back to ACI.

## Kaggle reproduction (only after preparation succeeds)

On a GPU host with the already compatible environment and pinned tokenizer cache:

```bash
export PYTHONPATH=src
python -m voice_to_soap.betrac_dataset
```

Alternatively transfer the complete prepared directory intact to Kaggle with the
source and config. Verify the report says `complete` and counts equal 2,000;
inspect split_manifest.json IDs/hashes. A missing or incomplete dataset is a stop
condition. Do not use the historical 338/13/87 bundle as a substitute.

Using the existing matched orchestration (training/generation settings unchanged):

```bash
PYTHONPATH=src python -m voice_to_soap.aci_soap compare \
  --train data/processed/betrac_2000/train.jsonl \
  --validation data/processed/betrac_2000/validation.jsonl \
  --test data/processed/betrac_2000/test.jsonl \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --model-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --max-seq-length 4096 --seed 42 --epochs 3 \
  --output-dir data/outputs/betrac-2000-comparison
```

This command trains and evaluates both methods using the existing implementation;
it was not executed on the CPU laptop. It uses the approved three-epoch default;
learning rate, batch size and LoRA defaults remain unchanged. Do not launch until CUDA/VRAM and a
separate smoke run are verified. Two T4 cards do not merge memory automatically.

For separate stages, the existing Python API supports `approach` (the individual
CLI train/evaluate commands do not expose that option). Run the following only
after the dataset and GPU checks succeed, with `src` on Python's import path:

```python
from pathlib import Path
from voice_to_soap.aci_soap import train_aci_soap_lora, evaluate_aci_soap_lora

data = Path('data/processed/betrac_2000')
revision = '7ae557604adf67be50417f59c2c2f167def9a775'
base = 'Qwen/Qwen2.5-0.5B-Instruct'
for approach in ('direct', 'hybrid'):
    train_aci_soap_lora(
        data/'train.jsonl', data/'validation.jsonl', base,
        f'data/models/betrac_2000/{approach}', approach=approach,
        model_revision=revision, seed=42, max_seq_length=4096, epochs=3,
    )
for approach in ('direct', 'hybrid'):
    evaluate_aci_soap_lora(
        data/'test.jsonl', base, f'data/models/betrac_2000/{approach}',
        f'data/outputs/betrac_2000/{approach}', approach=approach,
        model_revision=revision, device='cuda', max_new_tokens=1024,
    )
```

Both adapters start from the original base, never the other adapter. Unspecified
settings retain the repository's existing defaults; no training setting is changed.
Keep clinical metrics unavailable without proper reference labels.

Legacy benchmarks remain accessible by explicitly passing paths under
`data/processed/aci-bench-soap` or the old mixed preparation. BeTraC dev remains
validation material; do not mix it into this training selection.

## Verification

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_betrac_dataset.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use synthetic fixtures, not downloads or clinical validation. Actual source
counts and length min/median/p95/max are recorded by the real selection report.
No training completion or GPU fit is inferred from passing these tests.

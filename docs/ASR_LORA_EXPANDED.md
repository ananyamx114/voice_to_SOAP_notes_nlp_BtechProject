# Expanded existing-dataset ASR experiment

Completed: five epochs / 515 updates and all 71 selected paired CUDA recordings.
See [the final report and tables](../results/asr-lora-run2/README.md) and
[team integration handoff](ASR_TEAM_HANDOFF.md). WER improved in three
dataset/language groups and regressed in three; this is not universal improvement.
The original local source is paused at update 90; do not resume that obsolete
checkpoint after the completed CUDA continuation. Final weights remain private.

This is a new five-epoch experiment, preserving the original three-epoch run
and its results. The complete pinned Whisper Large-v3 model, rank-16 LoRA,
learning rate and precision policy are unchanged. New adapters start from the
pretrained base; the old three-epoch optimizer state is not resumed on new data.
After reviewing the completed results, the user authorized publication of the
source, setup and aggregate evidence on 15 November 2026. Data and weights stay
private. Start with [the ASR results index](../results/asr/README.md).

## Available data and measured preparation

The existing cached BeTraC training shard contains 946 consultations / about
125.88 raw hours. The first run used 12 consultations; 934 were unused.
FLEURS training contains 2,120 Hindi recordings / 6.66 raw hours and 3,269 Marathi
recordings / 11.95 raw hours. Filtering to nonempty targets and <=30-second
audio leaves about 6.61 and 11.80 hours respectively, before decoder-target
screening. These datasets are not exhausted. No additional dataset/model-weight
download was needed on the originating machine.

The prepared expanded cohort has:

| Language | Samples | Accepted training audio |
|---|---:|---:|
| English | 1,045 segments from 36 consultations | 7,222.37 seconds |
| Hindi | 321 whole clips | 3,641.28 seconds |
| Marathi | 278 whole clips | 3,631.14 seconds |

Total: 1,644 samples / about 4.03 hours. English reuses 345 previously accepted
segments and adds 700 from 24 previously unused consultations. Validation has
239 separate segments: the existing 175 English segments and 32 clips per Indic
language. Full train/validation and selected-test group/hash checks passed.
English training excludes all BeTraC dev parent groups, including diagnostic
test consultations. Eka remains evaluation-only; Sully is not added to training.

All corpora already provide reference transcripts. Whisper predictions are not
used as gold training labels. BeTraC remains synthetic English speech with
automatic CTC alignment; FLEURS remains general read speech, not clinical
Hindi/Marathi. These limitations do not disappear with additional hours.

## Prepare prerequisites on each machine

Read [TEAM_SETUP](TEAM_SETUP.md) and inspect the local system first. Activate
the local environment, then use the portable launcher for model stages:

```text
python scripts/project.py system
python scripts/project.py setup --profile asr-training
python scripts/project.py download-data --dataset all
python scripts/project.py download-data --dataset betrac-train
python scripts/project.py download-data --dataset fleurs-training
python scripts/project.py download-data --dataset fleurs-test
python scripts/project.py prepare-asr-data
```

Prepare the BeTraC VAD artifacts described in [COMPONENT_01_VAD](COMPONENT_01_VAD.md)
and [COMPONENT_02_ASR](COMPONENT_02_ASR.md) before running the complete evaluation.
Prepare separate validation and selected-test audio if they do not already exist:

```text
python scripts/project.py asr-align --archive data/raw/betrac/data/dev-00000.tar --outdir data/processed/asr-lora-clinical-validation-run1 --split validation --limit 4 --selection-seed 43 --exclude-encounter dialog_0060_0120 --exclude-encounter dialog_0060_0121
python scripts/project.py prepare-fleurs --dataset-dir data/raw/fleurs-test --outdir data/processed/asr-lora-fleurs-test-run1 --test-per-language 32 --seed 44
```

Use existing complete preparations when available; preparation commands refuse
to overwrite them. Downloads preserve source pins/licenses and disk headroom.
Each machine must independently establish its full-model memory fit.

## Start the expanded run

From the repository root with the environment activated:

```text
python scripts/run_expanded_asr.py --outdir data/outputs/asr-lora-experiment-run2 --device auto
```

If the original run exists locally, reuse its training segments and exact base
controls to avoid repeated work:

```text
python scripts/run_expanded_asr.py --outdir data/outputs/asr-lora-experiment-run2 --device auto --existing-clinical-train data/processed/asr-lora-clinical-train-run1-v2 --baseline-run data/outputs/asr-lora-experiment-run1
```

The originating launch uses explicit `mps`; automatic resolution selects the
same device on that machine. Controls from a different device/config/source
signature are rejected during preflight. Omit `--baseline-run` on a machine
without matching local controls; it then executes the real base model.

`configs/asr-lora-expanded.yaml` specifies five epochs and patience five, so
validation early stopping cannot shorten this run below five epochs. The
original config remains unchanged as a historical control. At 1,644 examples
and accumulation 16, this schedules 103 updates per epoch / 515 updates total.
Best-checkpoint selection still uses validation macro language WER; every
completed epoch is archived for later language-validation selection.

Stages run sequentially: verify controls, prepare Hindi/Marathi, align additional
English until its accepted-duration target is reached, audit all data, probe
one full-model update on the 16 largest targets, train, evaluate the selected
adapter, and compile results. Raw source audio, original text and alignment
rejections are preserved. No target is truncated to meet a limit. A failed
duration/resource check stops the job instead of training a smaller cohort or
silently substituting another model.

This is a substantially longer local job than run 1. CPU-thread settings should
be measured locally; hardware settings and sleep inhibitors stay in ignored
local launchers. The originating process uses Metal, measured CPU threading,
and enabled memory limits. Tiny CPU preprocessing times did not justify a large
feature cache. Reusing signature-verified controls saves repeated inference.
Do not disable memory protection or run multiple large model stages together.

## Monitor and resume

The local job controller can attach to an already running process using an
ignored, machine-specific `.local/asr-expanded-run2.job.json` manifest. The
originating process has been registered; no second trainer was started.
Teammates do not inherit that manifest. On a fresh clone, use the foreground
experiment command and its `--resume` option; job-controller status reports
`not_registered` until this machine has a registered job. Never copy another
machine's PID, paths or environment into your runtime settings.

```text
python scripts/project.py asr-job status
python scripts/project.py asr-job pause
python scripts/project.py asr-job resume
```

Without activation, use `python3` on macOS/Linux or `py -3.12` on Windows. The
launcher selects the environment. Pause is a request to finish at the next
atomic optimizer checkpoint: at most ten optimizer updates during training,
currently about half an hour per saved batch. During final ASR evaluation it
waits for a completed recording. Wait until status says `paused` before shutting
down. The controller then stops only the registered process tree and frees its
model memory. Resume launches the same command with `--resume`, restoring the
saved adapter, optimizer, scaler, epoch/offset and RNG state. Completed training
is preserved when pausing during evaluation. PID creation times prevent stopping
an unrelated process after PID reuse. A small detached local watcher handles
requests without ChatGPT/API polling or continuous conversation tokens.

These controls were verified with disposable test processes; the actual trainer
and its numerical configuration were not modified to add the controller.

The run directory contains `expanded-run.json`, `duration-audit.json`, source
snapshots, `resource-probe/`, `training/run.json`, `training/training.jsonl`,
completed-epoch checkpoints and final evaluation artifacts. Models, prepared
audio, logs and these outputs remain ignored. No new accuracy result exists
until the actual training and paired evaluations finish.

Repeat the same command with `--resume` after interruption. Completed English
alignments, verified controls and matching optimizer/RNG checkpoints are reused.
A partial FLEURS preparation currently requires a fresh run directory; retain
the partial output for inspection. Source/config/data changes require a new
experiment or an explicitly reviewed compatible source-patch continuation.

The fixed 71-recording diagnostic evaluation is unchanged from run 1. It is not
a full-corpus or clinical validation claim. Training validation still uses
greedy short segments; production evaluation uses temperature fallback and
word timestamps. Report WER and CER independently, including regressions, and
keep every downstream generated note pending clinician approval.

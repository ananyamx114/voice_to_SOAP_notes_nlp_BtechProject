# Team setup: start here

Completed testing: [27 November ASR paired benchmark](../results/asr-run3-test-2026-11-27/README.md) and [all three Qwen7B test stages](../results/qwen7b-2026-11-27/README.md). These supersede earlier pending-test statuses.


Latest verified handoff: [27 November report-ready results](LATEST_RESULTS.md), including completed Qwen training/tests and Whisper decoder diagnostics.


This is the reproducible handoff for teammates and their Codex sessions. The
deliverable is the complete local Voice-to-SOAP pipeline. The repository is
still under development; use the status below to distinguish completed work
from remaining integration. Everyone uses their own accounts and local clone,
and the team has explicitly chosen to collaborate on `main`.

Latest report handoff: [26 November interim metrics](../results/interim-2026-11-26/README.md)
and [what to quote in the report](REPORT_METRICS.md). This dated snapshot does not
replace older completed experiments or establish end-to-end clinical quality.

New ASR work: [numeric-safe scoring, diagnostics and balanced-data experiments](ASR_IMPROVEMENT_WORK.md).
Published scores keep their original protocol. New inference scores identify
`numeric_indic_v2`; historical trainer configs remain reproducible under v1.

The user reviewed the completed ASR work and authorized publication on
15 November 2026. Publish only code, portable setup and aggregate results;
preserve local artifacts and teammate updates. Future user-requested publication
holds must still be honored.

The latest local execution record is documented in
[PROGRESS_SO_FAR.md](PROGRESS_SO_FAR.md).
For the prepared direct Qwen LoRA transcript-to-SOAP run on Kaggle, see
[KAGGLE_TEXT_TO_SOAP.md](KAGGLE_TEXT_TO_SOAP.md).

The expanded ASR run completed five epochs and all 71 selected paired CUDA
recordings. Its [report and measured results](../results/asr-lora-run2/README.md)
show three WER gains and three regressions; read the
[ASR-to-SOAP handoff](ASR_TEAM_HANDOFF.md) before integrating with the Qwen stage.
The shared [ASR results index](../results/asr/README.md) matches the team's stage
folders; [ASR_RESOURCES.md](ASR_RESOURCES.md) inventories model/data pins,
dependencies, downloaded resources, compute, and reproduction commands.

Kaggle CLI authentication and checkpoint-preserving ASR GPU migration are
documented in [KAGGLE_ASR.md](KAGGLE_ASR.md). Local training must be confirmed
paused before exporting its optimizer checkpoint; keep all transfers private.
Before resuming local ASR, check for `.local/kaggle-asr/migration.local.json`:
an active cloud continuation must not be duplicated by restarting the old source.

## Instructions for a teammate's Codex

Read `AGENTS.md`, this file, `docs/REPOSITORY_AUDIT.md`, and
`docs/LOCAL_BENCHMARKS.md`. Preserve working code, the separate evidence-cited
and ACI-Bench tracks, original data splits, and existing experiment attribution.
Inspect local hardware/free space before installing models. Complete the setup
and tests, then report actual results. Do not silently download all datasets or
the full training dependency stack. Never display, commit or share credentials.
Only the human account holder can accept gated-repository terms. Do not claim
model access on this laptop establishes access on another teammate's account.

## 1. Get the code and create an environment

Clone once, or fetch/update your existing clone. For an existing clean `main`:

```bash
git status --short
git fetch origin main
git pull --ff-only origin main
```

If there are local edits or diverging commits, preserve and review them before
integrating. Do not reset/clean away teammate work, force-push, or blindly pull
over local changes. Coordinate file ownership before simultaneous edits to main.
If a push is rejected, fetch again and integrate/retest the new commits normally.

For a new clone:

```bash
git clone https://github.com/ananyamx114/voice_to_SOAP_notes_nlp_BtechProject.git
cd voice_to_SOAP_notes_nlp_BtechProject
```

Run commands from the repository root. Each person installs Python 3.12 and
Git locally; no virtual environment or machine-specific dependency snapshot is
shared. Inspect the machine before installing model dependencies.

macOS/Linux:

```bash
python3 scripts/project.py system
python3 scripts/project.py setup
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.12 scripts/project.py system
py -3.12 scripts/project.py setup
.\.venv\Scripts\Activate.ps1
```

After activation, use the same commands on all three operating systems:

```text
python scripts/project.py system
python scripts/project.py check
python scripts/project.py test
```

If PowerShell activation is restricted, use `py -3.12` instead of `python` for
the launcher; it selects the local environment without requiring activation.
On macOS/Linux without activation, use `python3` for the launcher.

`system` checks OS, architecture, Python, physical RAM, free disk, FFmpeg/Ollama,
and CUDA/MPS when PyTorch is installed. Its report goes to ignored
`.local/system.json`. A restricted sandbox may deny hardware queries; retry
with the environment's normal permissions. Unknown RAM/GPU remains unknown,
never an invented value. Before installing torch, GPU availability is unchecked.
These facts are inputs to model selection, not a guarantee that a model fits.

The launcher uses argument arrays, the repository working directory, the
appropriate Windows/POSIX .venv executable, and a local HF cache. VS Code tasks
now provide Windows overrides and a portable .venv folder setting. Select the
interpreter manually if a previous workspace selection persists.

`setup` installs development/HF tooling only; it does not run a model. The
`--profile asr` option additionally installs real Whisper inference dependencies.
The `--profile vad` option installs the released Silero JIT model dependencies.
All profiles use the same source pipeline; none substitutes a mock model.
Model weights are downloaded only by a requested real inference/download step.
Different OS/GPU driver combinations still require their own execution checks.

## 2. Connect your own Hugging Face account

Sign into [Hugging Face](https://huggingface.co/). Review/accept the conditions at
[pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
Create a **Read** token named for your laptop at
[Access Tokens](https://huggingface.co/settings/tokens).

```bash
python scripts/project.py hf-login
python scripts/project.py hf-check
```

Paste the token only into the hidden CLI prompt. Answer `n` if asked to add it
as a Git credential. Do not paste it into chat, a command argument, source code,
or a tracked file. The launcher selects the same ignored cache for login and access checks on
all supported systems; no shell-specific export is needed.

The checker uses the project cache unless HF_HOME is already configured. It
verifies authentication, downloads tiny configs and checks weight-file access
for Whisper Small, pyannote Community-1 and Qwen2.5-3B-Instruct. It does not
download weights or run inference. It prints safe status/revision fields only.
403 usually means missing gated access or insufficient token permissions; 401
means authentication needs attention. Review the actual status before retrying.

The pipeline/diarization CLI now accepts either the explicit configured token
environment variable (`HF_TOKEN` by default) or the CLI's saved login. A saved
login removes the need to copy your token into .env. HF_HOME must select the
same cache used during login. `.env` is not auto-loaded by the CLI.

The originating laptop verified access on 13 November 2026. See the safe record
at `results/local-benchmark-setup/huggingface-access.json`; this is not a
guarantee that a teammate has accepted the terms or can run the models.

## 3. Download the selected audio benchmarks

These public selections need no HF login. The downloader pins revisions and
checks sizes/SHA-256; it preserves 12 GiB free and downloads roughly 732 MiB:

```bash
python scripts/project.py download-data --dataset all
```

Selections: full BeTraC dev (400 synthetic audio/transcript/SOAP encounters),
full Eka test (3,619 English + 320 Hindi clips), and one Sully/PriMock57 audio
consultation with metadata and attribution. Use `--dataset betrac`, `eka` or
`sully-smoke` to download individually. Re-running reuses hash-verified large
files. No training shards or entire Sully audio corpus are selected.

Prepare a bounded validation subset first using the real importer. These commands
use POSIX line continuations; in PowerShell enter each command on one line:

```bash
python -m voice_to_soap.benchmark_datasets \
  --archive data/raw/betrac/data/dev-00000.tar \
  --output-dir data/processed/betrac-smoke --limit 2
python scripts/project.py cli import-eka-asr \
  --dataset-dir data/raw/eka --outdir data/processed/eka-smoke \
  --manifest-out data/manifests/eka-smoke.csv --limit-per-language 2
```

To export all Eka clips, choose distinct output paths and omit the limit:

```bash
python scripts/project.py cli import-eka-asr \
  --dataset-dir data/raw/eka \
  --outdir data/processed/eka-medical-asr-evaluation \
  --manifest-out data/manifests/eka-medical-asr-evaluation.csv
```

Preparation refuses existing outputs: use an existing valid manifest or a new
output name; do not delete local data automatically. BeTraC's audio manifest
keeps the input transcript empty, and the separate gold manifest includes
speaker-labeled text. Do not feed reference transcripts to ASR or treat synthetic
reference notes as clinically reviewed evidence labels. Sully includes ACI-Bench
records already present here; prevent overlap in future experiments.

IndicVoices/Svarah need individual gated access; DISPLACE-M requires challenge
authorization. Follow `docs/LOCAL_BENCHMARKS.md`, not a blanket download command.

## 4. Real local ASR execution

The new full-checkpoint fine-tuning implementation is documented in
[ASR_LORA.md](ASR_LORA.md). Read its publication gate, data-alignment requirements,
resource probe and validation procedure before training. A local full Large-v3
LoRA run completed three epochs / 90 updates; validation improved in all three
languages. The [71-recording paired evaluation](../results/asr-lora-run1/README.md)
improved WER in three of six dataset/language groups, regressed in two and left
one unchanged. Marathi CER regressed despite lower WER.
The [English checkpoint follow-up](../results/asr-lora-run1/validation-selected.md)
uses per-language validation minima and yields four WER gains/two ties, while
CER regressions remain. It does not implement automatic adapter routing.
Do not treat validation gains as universal test gains. Do not train on Eka test data.
The [expanded five-epoch experiment](ASR_LORA_EXPANDED.md) uses a verified
four-hour training cohort from the same datasets. Five-epoch training and its
71-recording CUDA evaluation are complete, including the reported English
regressions. Do not attribute the original run's measured scores to this model.

The current full-checkpoint component is [COMPONENT_02_ASR.md](COMPONENT_02_ASR.md),
with sequential Large-v3 decoding and [actual measured results](../results/component-02-asr/README.md).
The Small command below is retained only as the earlier baseline, not the new
full-checkpoint component recipe.

```bash
python scripts/project.py setup --profile asr
python scripts/project.py system
python scripts/project.py cli evaluate-asr \
  --manifest data/manifests/eka-smoke.csv --model openai/whisper-small \
  --device cpu --limit-per-language 2 \
  --outdir data/outputs/eka-whisper-small-local-smoke
```

This downloads about 1 GB of Whisper weights and uses actual CPU inference.
The originating M1/8 GB run completed but had poor Hindi/repetition results;
read `results/local-benchmark-setup/README.md`. Four clips are not a representative
benchmark. Latency/peak-memory measurement and decoding investigation remain.

Computer-specific dependency snapshots were removed from shared setup. Each
machine installs from the shared dependency profiles and may record its exact
resolved packages under ignored `.local/`. Historical result files retain the
hardware on which they were measured; they are experiment provenance, never
requirements imposed on other computers. No Windows/Linux model execution is
claimed solely from the Mac run or from mocked portability tests.

## 5. Current status and next implementation

Available: repository audit, VS Code setup, public dataset downloader, BeTraC
manifest exporter, Eka importer/evaluator, local Whisper execution, safe HF
access checker, cached authentication support, regression tests and aggregate
results. Model-file access was verified for the three named repositories.
The earlier publication check ran 79 passing unit tests, including cached-authentication
regressions; newer portability tests also exercise Windows/POSIX command selection
and Windows/macOS/Linux memory-query paths without model execution. Unit tests do not contact Hugging Face or validate clinical accuracy.

Latest pulled update 751da5a fixes approval status: every generated note stays
pending clinician review and automated issues are recorded separately.

Remaining: local SOAP-serving runtime/model, measured end-to-end cited generation,
actual diarization environment and execution, VAD boundary-quality evaluation,
clinician-review application/export, terminology population, protected
translation, and complete research evaluation. HF login does not install these
components. Keep diarization disabled until its dependencies and a real smoke
run work. Do not claim the complete pipeline is finished from passing tests.

## 6. What belongs in Git

Commit source, tests, setup scripts, guides, config templates, dependency profiles
and small non-sensitive result summaries. Raw datasets, new prepared audio,
local manifests, .venv, model weights, tokens and ordinary output predictions
stay local and ignored. Each teammate downloads/authenticates separately.
Existing attributed ACI-Bench files are already tracked; preserve their license
and experiment context. Never add ignored datasets with `git add -f`.

## 7. Run sections and advance to the complete pipeline

Use [PIPELINE_EXECUTION.md](PIPELINE_EXECUTION.md) for exact existing commands,
current blockers, completion evidence and the next implementation order. Before
large execution, publish the tested implementation and run instructions to main.
Do not enable absent/untested modules just to label a run complete.

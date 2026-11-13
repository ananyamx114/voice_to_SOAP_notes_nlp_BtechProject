# Team setup: start here

This is the reproducible handoff for teammates and their Codex sessions. The
deliverable is the complete local Voice-to-SOAP pipeline. The repository is
still under development; use the status below to distinguish completed work
from remaining integration. Everyone uses their own accounts and local clone,
and the team has explicitly chosen to collaborate on `main`.

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

Run all subsequent commands from the repository root. macOS/Linux setup:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e '.[hub]'
.venv/bin/python scripts/check_setup.py
.venv/bin/python -m unittest discover -s tests -v
```

Python 3.12.4 on macOS arm64 was exercised. The base package declares >=3.10,
but other OS/Python combinations need their own dependency verification. Windows
uses `.venv\Scripts\python.exe` and `.venv\Scripts\hf.exe` instead of bin paths;
the supplied VS Code interpreter/tasks currently target macOS/Linux. System
FFmpeg should be available for broader audio tooling; inspect rather than assume
it is installed. SoundFile/Opus decoding was tested on the originating Mac.

## 2. Connect your own Hugging Face account

Sign into [Hugging Face](https://huggingface.co/). Review/accept the conditions at
[pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
Create a **Read** token named for your laptop at
[Access Tokens](https://huggingface.co/settings/tokens).

```bash
export HF_HOME="$PWD/data/models/huggingface"
.venv/bin/hf auth login
.venv/bin/hf auth whoami
.venv/bin/python scripts/check_huggingface.py
```

Paste the token only into the hidden CLI prompt. Answer `n` if asked to add it
as a Git credential. Do not paste it into chat, a command argument, source code,
or a tracked file. Repeat the HF_HOME export in new terminals; the VS Code
macOS/Linux terminal settings also set it for newly opened terminals.

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
.venv/bin/python scripts/download_benchmarks.py --dataset all
```

Selections: full BeTraC dev (400 synthetic audio/transcript/SOAP encounters),
full Eka test (3,619 English + 320 Hindi clips), and one Sully/PriMock57 audio
consultation with metadata and attribution. Use `--dataset betrac`, `eka` or
`sully-smoke` to download individually. Re-running reuses hash-verified large
files. No training shards or entire Sully audio corpus are selected.

Prepare small manifests first:

```bash
.venv/bin/python -m voice_to_soap.benchmark_datasets \
  --archive data/raw/betrac/data/dev-00000.tar \
  --output-dir data/processed/betrac-smoke --limit 2
.venv/bin/python -m voice_to_soap.cli import-eka-asr \
  --dataset-dir data/raw/eka --outdir data/processed/eka-smoke \
  --manifest-out data/manifests/eka-smoke.csv --limit-per-language 2
```

To export all Eka clips, choose distinct output paths and omit the limit:

```bash
.venv/bin/python -m voice_to_soap.cli import-eka-asr \
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

## 4. Optional real local ASR smoke run

```bash
.venv/bin/python -m pip install -r requirements-local-asr.txt
export HF_HOME="$PWD/data/models/huggingface"
.venv/bin/python -m voice_to_soap.cli evaluate-asr \
  --manifest data/manifests/eka-smoke.csv --model openai/whisper-small \
  --device cpu --limit-per-language 2 \
  --outdir data/outputs/eka-whisper-small-local-smoke
```

This downloads about 1 GB of Whisper weights and uses actual CPU inference.
The originating M1/8 GB run completed but had poor Hindi/repetition results;
read `results/local-benchmark-setup/README.md`. Four clips are not a representative
benchmark. Latency/peak-memory measurement and decoding investigation remain.

Dependency snapshots `requirements-*-macos-py312.lock.txt` record the originating
environments. They exclude the editable repository and are platform-specific;
prefer the small requirements profiles above on a different platform. Do not
install CUDA/bitsandbytes or the entire requirements.txt blindly on an 8 GB Mac.

## 5. Current status and next implementation

Available: repository audit, VS Code setup, public dataset downloader, BeTraC
manifest exporter, Eka importer/evaluator, local Whisper execution, safe HF
access checker, cached authentication support, regression tests and aggregate
results. Model-file access was verified for the three named repositories.
The publication check ran 79 passing unit tests, including cached-authentication
regressions. Unit tests do not contact Hugging Face or validate clinical accuracy.

Remaining: local SOAP-serving runtime/model, end-to-end cited generation and
approval-status fixes, actual diarization/VAD environment and execution,
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

# Voice-to-SOAP: Progress So Far

## Interim ASR continuation and Qwen7B report handoff (26 November 2026)

The [dated snapshot](../results/interim-2026-11-26/README.md) is the latest report
entry point. The SOAP export cutoff was 14:42 IST; ASR progress was observed at
14:42 IST. Jobs continued during export. Nothing in this snapshot stops jobs.

Qwen2.5-7B-Instruct DIRECT completed 4,800 updates / three epochs; selected
validation loss was 0.736586. The pretrained baseline completed 200 internal-test
encounters, with 196 valid outputs and four invalid; failure-aware mean
section ROUGE-L was 0.229839. DIRECT generation was partial (186/200;
173 valid, 13 invalid; subset ROUGE-L 0.426978), so no completed paired score
improvement is claimed. HYBRID training was at saved update 1200 / epoch 0.75,
best validation loss 0.779724; HYBRID generation had not started.

The 12.03-hour ASR continuation was observed at update 3384, with the last fully
verified local archive at 3330. Epoch 10 remained selected: English/Hindi/Marathi
validation WER 1.49%/23.99%/44.83%. Epoch 11 regressed in macro-language WER;
all 11 available epoch tables are included. These are not new held-out tests.

Use [REPORT_METRICS.md](REPORT_METRICS.md) for distinctions, limitations and refresh
commands. Earlier sections below are historical; their completed test results
and experiment provenance remain unchanged.

## Expanded ASR final evaluation and report ready (15 November 2026)

The complete base and validation-selected epoch-5 adapter finished matched CUDA
evaluation on all 71 selected recordings. The final private recovery archive
passed its SHA-256 and all 358 file checks. The report exporter additionally
checked exact per-language coverage, adapter identity, model/decoder pairing
and validation-only checkpoint selection. There were no failed/skipped test
recordings. All 137 local tests passed for the handoff.

| Dataset/language | Base WER | LoRA WER |
|---|---:|---:|
| Eka English (2) | 13.33% | 16.67% |
| Eka Hindi (2) | 50.00% | 39.13% |
| BeTraC English (2) | 2.98% | 4.56% |
| Sully English (1) | 22.21% | 22.63% |
| FLEURS Hindi (32) | 32.80% | 26.02% |
| FLEURS Marathi (32) | 79.87% | 61.55% |

WER and CER improved in the three Hindi/Marathi groups and worsened in all
three English groups. Do not hide those regressions or describe the adapter as
universally better. Current data expansion is deferred to a separate experiment.
The report section, Word document, tables, JSON provenance and PNG/SVG/PDF figures
are in [results/asr-lora-run2](../results/asr-lora-run2/README.md). The
[team handoff](ASR_TEAM_HANDOFF.md) explains real ASR prediction inputs to Qwen
and the remaining speaker/evidence/end-to-end validation boundaries.

The selected adapter was installed and checksum-validated under ignored
`data/models/asr-adapters/large-v3-expanded-run2/`. Models, audio, transcripts,
credentials and runtime files remain outside Git. No commit/push was made;
the existing resolved merge and newer SOAP/BeTraC remote commits still require
normal integration when the user approves publication.

## Expanded ASR training complete; evaluation recovery (15 November 2026)

The expanded run finished all five epochs and 515 optimizer updates on Kaggle.
The downloaded recovery archive passed its SHA-256 and all 67 file checks.
Its final checkpoint was also loaded on CPU to verify update 515, next epoch 5
and offset 0, all 384 adapter tensors/optimizer states, scaler step 515 and CUDA
RNG state. The epoch-5 adapter was selected by validation macro-language WER.

| Validation language | Historical base WER | Epoch-5 WER | Historical base CER | Epoch-5 CER |
|---|---:|---:|---:|---:|
| English | 8.48% | 1.32% | 6.32% | 0.54% |
| Hindi | 30.10% | 24.13% | 13.44% | 9.99% |
| Marathi | 81.55% | 59.59% | 23.55% | 15.87% |

The historical validation control ran on MPS, and the trained adapter validation
ran on CUDA. These are diagnostic validation results, not completed paired CUDA
test results. Marathi error remains high. All five epochs' metrics and the
adapter weights are retained locally; no commit or push has been made.

The first evaluation session completed base and adapter transcription for the
four Eka diagnostic clips, then failed in metric comparison because its parent
coordinator used system Python without `jiwer`. The initial English result
regressed (13.33% to 16.67% WER on two clips), while Hindi improved (50.00% to
39.13% on two clips). The remaining dataset comparisons are unfinished.
The launcher now uses the ASR environment for the coordinator, and evaluation
recovery requires the completed update-515 archive before loading any model.

The local live log stream stopped updating at step 187 while the actual GPU
training continued. The saved session log and verified archive are authoritative.
The monitor now reconnects stale streams, flags unconfirmed live progress,
ignores backward replay of older updates and reads persisted logs at termination.
The attempted evaluation restart could not attach the failed kernel as a source;
the verified completed archive was transferred through a private dataset instead.
The user cancelled the source-less restart, and corrected evaluation session 002
version 2 was submitted with all required inputs. It has a separate five-minute
monitor, a mandatory update-515 guard and automatic verified output download.

## ASR GPU migration update (15 November 2026)

The expanded five-epoch experiment has moved from local MPS to a private
Kaggle CUDA session. Local training stopped at an atomic checkpoint at update
90, with the original adapter, AdamW state, scaler and sample offset retained.
A separate verified recovery archive remains local. No Git commit or push was
made; the working source and selected audio were transferred privately.

The cloud verified 2,393 asset files and six checkpoint files, restored update
90/sample offset 1,440, and resumed the complete FP16 Whisper Large-v3 model
with the same 7,864,320 trainable LoRA parameters. New CUDA optimizer checkpoints
at updates 100 and 103 confirm continuation. At this recorded handoff, epoch 1
training had finished and validation was in progress; the schedule has 515
updates in total. There are no new expanded-run WER results yet.

Transformers 4.57.6 and PEFT 0.17.1 remain pinned. The backend migration records
the CUDA PyTorch/NumPy version changes and the necessary fresh CUDA RNG seed;
it preserves learned and optimizer state but does not claim bitwise identical
continuation across GPU backends. The current trainer uses one T4. Observed
GPU microbatches were mostly about 0.9 seconds; validation/evaluation time is
additional and has not yet been measured for a completion estimate.

See [KAGGLE_ASR.md](KAGGLE_ASR.md) for authentication, verified transfer, session
checkpointing and continuation instructions. Query the ignored local migration
handoff record and remote job before attempting to resume the old local source.
The full local suite passed 125 tests after the input-loader correction.

## ASR experiment update (14 November 2026)

The local ASR experiment trained 7,864,320 LoRA parameters on the complete
Whisper Large-v3 model. Base weights stayed frozen; no quantization or smaller
model was used.
Training completed three epochs / 90 optimizer updates on 472 segments
(64.90 minutes); 199 separate segments were used for validation. The frozen base
and trained adapter were both evaluated on 71 selected recordings.

[Measured results](../results/asr-lora-run1/README.md): WER decreased in three of
six dataset/language groups, increased in two, and stayed unchanged in one.
Marathi CER increased despite lower WER. This is a mixed experiment result,
not completion of the whole pipeline or a universal ASR improvement. The separate
English validation-selected epoch-1 checkpoint matches base WER on Eka/BeTraC
and lowers Sully WER from 22.21% to 22.01%. Using epoch 1 for English and epoch 3
for Hindi/Marathi gives four WER gains and two ties; the original shared-adapter
regressions remain reported. Marathi CER still worsens. These are exploratory
results using known language hints, not automatic adapter routing.

The run recovered from recorded MPS allocation failures using memory corrections
and a protected local resource budget. Full predictions, training checkpoints,
source snapshots and recovery histories remain ignored. The shared results
contain aggregate metrics and portable experiment settings.

The teammate update `9d14ea0` was fetched and integrated in the working tree.
It changes the direct/hybrid Qwen SOAP experiment, with no ASR runtime changes.
All 117 repository tests pass after integration; dependency checks also pass.
The merge and local ASR work remain uncommitted and unpushed under the user's
explicit publication restriction. Local backup patches and a recovery stash
were retained before integration.

## Earlier Windows execution record (13 November 2026)

Updated: 13 November 2026

This document records the setup and execution completed on the Windows
development computer. It is an execution record, not a claim that the complete
clinical pipeline is finished.

### Repository state

- Shared branch: `main`
- Latest synchronized commit: `3f78366`
- Worktree: clean
- Root project guidance and the separate evidence-cited and ACI-Bench tracks
  were preserved.
- Local datasets, model weights, outputs, hardware reports, and credentials
  remain outside Git and under ignored paths.

### Computer and environment

The local system report recorded:

- Windows 10, AMD64
- Python 3.11.1
- Physical RAM: approximately 16 GiB
- Free disk at the final check: approximately 130 GiB
- CPU-only PyTorch; no CUDA accelerator detected
- FFmpeg installed locally
- Ollama installed locally

Machine-specific details are written to the ignored `.local/system.json`.

### Installed components

The lightweight development environment was created in `.venv` and extended
with the following local profiles:

- Development and Hugging Face tooling
- Whisper ASR: PyTorch and Transformers
- Silero VAD
- Pyannote diarization: `torchaudio` and `pyannote.audio`
- FFmpeg shared build for audio decoding
- Ollama local runtime
- Ollama model: `qwen2.5:3b` (approximately 1.9 GiB)

The full training stack, CUDA, and a 7B local model were not installed.

### Hugging Face access

The project cache is authenticated as the local account and access checks
passed for:

- `openai/whisper-small`
- `pyannote/speaker-diarization-community-1`
- `Qwen/Qwen2.5-3B-Instruct`

The check only downloads small configuration/metadata files and does not run
inference. Tokens were entered through the hidden Hugging Face CLI prompt and
were not stored in the repository.

### Dataset preparation

The selected public benchmark files were downloaded with pinned revisions and
hash checks:

- BeTraC development archive
- Eka English/Hindi ASR test shards
- One Sully/PriMock57 smoke audio sample and attribution metadata

Bounded local validation data was prepared:

- Two BeTraC samples in `data/processed/betrac-smoke`
- Four Eka samples in `data/manifests/eka-smoke.csv`
- BeTraC audio converted to local 16 kHz validation files

The raw datasets and generated audio remain ignored.

### Executed stages and results

#### Setup and regression tests

All **84 unit tests pass**, including Windows portability tests.

```powershell
py scripts\project.py system
py scripts\project.py setup
py scripts\project.py setup --profile asr
py scripts\project.py test
.venv\Scripts\python.exe -m pip check
```

#### Public benchmark download and preparation

```powershell
py scripts\project.py download-data --dataset all

.venv\Scripts\python.exe -m voice_to_soap.benchmark_datasets `
  --archive data\raw\betrac\data\dev-00000.tar `
  --output-dir data\processed\betrac-smoke --limit 2

py scripts\project.py cli import-eka-asr `
  --dataset-dir data\raw\eka `
  --outdir data\processed\eka-smoke `
  --manifest-out data\manifests\eka-smoke.csv `
  --limit-per-language 2
```

#### ASR smoke evaluation

```powershell
py scripts\project.py cli evaluate-asr `
  --manifest data\manifests\eka-smoke.csv `
  --config configs\base.yaml `
  --device cpu `
  --outdir data\outputs\asr-validation
```

Measured results on four clips:

- English WER: `0.20`
- Hindi WER: `3.3261`
- Overall WER: `2.0921`

The Hindi result contains substantial repetition/error and must not be treated
as acceptable clinical performance. Full results are in the ignored local
[ASR metrics](../data/outputs/asr-validation/metrics.json).

#### VAD

```powershell
py scripts\project.py cli prepare-audio `
  --manifest data\processed\betrac-smoke\audio_manifest.csv `
  --outdir data\processed\audio-validation `
  --sample-rate 16000

py scripts\project.py cli detect-speech `
  --manifest data\processed\betrac-smoke\audio_manifest.csv `
  --audio-dir data\processed\audio-validation `
  --outdir data\processed\vad-validation
```

Silero VAD executed on both BeTraC samples and exported 180 speech regions.
This validates execution and timestamps, not speech recall or clinical-word
preservation.

#### Diarization

FFmpeg was installed through Windows Package Manager. The downloaded Sully
MP3 had a codec-duration mismatch under the pyannote decoder, so it was
converted to a 16 kHz mono WAV excerpt for a bounded validation:

```powershell
$ffmpeg = "C:\Users\<user>\AppData\Local\Microsoft\WinGet\Packages\..."
& $ffmpeg -y -i data\raw\sully-smoke\audio\pm57-d1c01.mp3 `
  -t 30 -ac 1 -ar 16000 data\processed\sully-smoke-30s.wav

$env:HF_HOME = (Resolve-Path .\data\models\huggingface).Path
py scripts\project.py cli diarize-audio `
  --audio data\processed\sully-smoke-30s.wav `
  --out data\outputs\diarization-validation-30s.json `
  --config configs\base.yaml
```

The 30-second pyannote run succeeded and wrote speaker segments to the ignored
local [diarization result](../data/outputs/diarization-validation-30s.json).
The full 7.6-minute CPU run was stopped after roughly 12 minutes without
producing a result; no full-recording quality claim is made.

#### Local LLM

Ollama was installed and the small local model was downloaded:

```powershell
ollama list
ollama pull qwen2.5:3b

$env:LLM_BASE_URL = "http://localhost:11434/v1"
$env:LLM_MODEL = "qwen2.5:3b"
py scripts\project.py check --check-llm
```

The project check confirmed that the model is listed. A direct JSON response
smoke test also succeeded.

#### Evidence-cited SOAP pipeline

The bounded real-model run was started with:

```powershell
$env:LLM_BASE_URL = "http://localhost:11434/v1"
$env:LLM_MODEL = "qwen2.5:3b"
py scripts\project.py cli run `
  --manifest data\processed\betrac-smoke\gold_manifest.csv `
  --config configs\base.yaml `
  --outdir data\outputs\soap-transcript-validation
```

The local model responded, but both encounters failed schema validation:

1. One generated fact assigned `patient` to evidence from a different speaker.
2. One generated fact used unsupported entity type `finding`.

The pipeline correctly recorded failed encounter entries in the ignored local
[run log](../data/outputs/soap-transcript-validation/run.jsonl) rather than
presenting invalid facts or an unsafe SOAP note as successful output.

### Current limitations

- The real SOAP path needs stronger bounded correction/retry handling for
  malformed fact extraction output.
- Pyannote was validated only on a short WAV excerpt; full CPU diarization is
  too slow for the tested recording.
- ASR Hindi quality is poor on the four-clip smoke subset.
- No local model run establishes clinical accuracy or factual correctness.
- Ollama is local and CPU-backed; latency and memory have not been fully
  benchmarked.
- The terminology table and dose-range table remain empty templates.
- Clinician review/export UI and API are not implemented.
- Translation, training, research evaluation, and full audio-to-SOAP integration
  remain incomplete.

### Recommended next steps

1. Add bounded fact-extraction retries or deterministic normalization for
   invalid `entity_type` and speaker/evidence mismatches without weakening
   evidence validation.
2. Measure CPU memory and latency for short diarization and ASR runs.
3. Investigate Hindi ASR repetition and long-form decoding behavior.
4. Run the corrected transcript-to-facts-to-cited-SOAP path on synthetic data,
   retaining all raw transcript, evidence IDs, attempts, and pending approval.
5. Keep generated notes pending clinician approval even when automated checks
   pass.
6. Implement the review/export workflow before claiming an end-to-end complete
   clinical documentation system.

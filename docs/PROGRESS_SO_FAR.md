# Voice-to-SOAP: Progress So Far

Updated: 13 November 2026

This document records the setup and execution completed on the Windows
development computer. It is an execution record, not a claim that the complete
clinical pipeline is finished.

## Repository state

- Shared branch: `main`
- Latest synchronized commit: `3f78366`
- Worktree: clean
- Root project guidance and the separate evidence-cited and ACI-Bench tracks
  were preserved.
- Local datasets, model weights, outputs, hardware reports, and credentials
  remain outside Git and under ignored paths.

## Computer and environment

The local system report recorded:

- Windows 10, AMD64
- Python 3.11.1
- Physical RAM: approximately 16 GiB
- Free disk at the final check: approximately 130 GiB
- CPU-only PyTorch; no CUDA accelerator detected
- FFmpeg installed locally
- Ollama installed locally

Machine-specific details are written to the ignored `.local/system.json`.

## Installed components

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

## Hugging Face access

The project cache is authenticated as the local account and access checks
passed for:

- `openai/whisper-small`
- `pyannote/speaker-diarization-community-1`
- `Qwen/Qwen2.5-3B-Instruct`

The check only downloads small configuration/metadata files and does not run
inference. Tokens were entered through the hidden Hugging Face CLI prompt and
were not stored in the repository.

## Dataset preparation

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

## Executed stages and results

### Setup and regression tests

All **84 unit tests pass**, including Windows portability tests.

```powershell
py scripts\project.py system
py scripts\project.py setup
py scripts\project.py setup --profile asr
py scripts\project.py test
.venv\Scripts\python.exe -m pip check
```

### Public benchmark download and preparation

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

### ASR smoke evaluation

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

### VAD

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

### Diarization

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

### Local LLM

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

### Evidence-cited SOAP pipeline

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

## Current limitations

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

## Recommended next steps

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

# First model component run: voice activity detection

Executed results are in [results/component-01-vad](../results/component-01-vad/README.md):
two complete encounters, 180 regions, all exported WAV/timestamp consistency
checks passed. Manual speech-boundary accuracy still needs assessment.

Audio preparation already ran on actual recordings. Next is Silero VAD:
waveform -> speech regions with original recording timestamps. The standalone
command calls the same detect_speech function used by the integrated audio
pipeline, with the full released JIT model. It does not transcribe or assign
speakers. No model substitution is made for this component run.

## Size and authentication

Pinned package: `silero-vad==6.2.3`. Its PyPI wheel is 11,317,527 bytes (10.8 MiB),
including model assets. Upstream describes the JIT model as about 2 MB. This is
the released model's design, not a reduced project implementation. No token or
login is required. First-time PyTorch installation is larger and depends on OS
and accelerator builds; PyTorch is already installed on the originating laptop.

The project decodes with SoundFile and resamples with SciPy, passing a mono
float32 tensor to Silero. Version 6.2.3 supports this without the torchaudio,
torchcodec or ONNX extras. CPU inference is supported. Inspect every computer
individually using the launcher before installing dependencies.

Sources: [Silero](https://github.com/snakers4/silero-vad),
[pinned package](https://pypi.org/project/silero-vad/6.2.3/).

## Exact execution

Pull clean main and follow TEAM_SETUP to create/activate your own environment.
Commands below work after activation on all supported operating systems. Without
activation use `python3` on macOS/Linux or `py -3.12` on Windows for the launcher.
Publish code, dependencies and these instructions before long execution.

```text
python scripts/project.py system
python scripts/project.py setup --profile vad
python scripts/project.py test
```

If the BeTraC manifest is absent, download and prepare it:

```text
python scripts/project.py download-data --dataset betrac
python -m voice_to_soap.benchmark_datasets --archive data/raw/betrac/data/dev-00000.tar --output-dir data/processed/betrac-smoke --limit 2
```

The two complete consultations total about 9.7 minutes; neither is cropped for
this execution. Reuse an existing valid manifest; preparation refuses overwrites.
The archive contains 400 dev encounters. A separate larger run can prepare all
400 into a new directory with `--limit 400`, after publishing its intended command.
The bounded first run does not replace a full quality benchmark.

```text
python scripts/project.py cli prepare-audio --manifest data/processed/betrac-smoke/audio_manifest.csv --outdir data/processed/audio-validation --sample-rate 16000
python scripts/project.py cli detect-speech --manifest data/processed/betrac-smoke/audio_manifest.csv --audio-dir data/processed/audio-validation --outdir data/outputs/vad-component-01 --sample-rate 16000 --threshold 0.5 --min-speech-ms 250 --min-silence-ms 100 --padding-ms 100
```

Each encounter directory under `data/outputs/vad-component-01/` contains:

- `speech_segments.json`: source/speech durations, speech fraction, segment count,
  processing time, model/package version, parameters and original timestamps.
- `segments/speech_*.wav`: actual audio corresponding to detected regions.

Original audio remains intact. Processing time includes loading, model
initialization, detection and segment WAV writing, up to metadata writing; it
is not pure neural-model latency. Empty speech is a possible outcome requiring
inspection, not automatic proof of correctly identified silence.

## Validation

Check that all encounters finish, intervals are finite/increasing and inside
the recording, merged regions do not overlap, WAVs decode, and saved durations
match timestamp intervals and their summary totals.

Quality assessment also requires listening around source region starts/ends,
especially short answers, negations, medication names, dosages and interruptions.
The 250 ms minimum can affect brief utterances. Record errors on development
examples; never tune against a held-out final benchmark. Execution checks do
not establish speech recall, clinical word preservation or diarization quality.

Audio, weights and local outputs remain ignored. Only small aggregate summaries
belong in Git. After VAD, run ASR on preserved timestamps, investigate the earlier
Hindi repetition failure, and compare with gold transcripts. Final settings
require measured quality; passing software tests is not clinical validation.

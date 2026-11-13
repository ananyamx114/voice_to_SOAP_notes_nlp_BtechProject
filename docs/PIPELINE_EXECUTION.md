# Run real pipeline sections, then integrate the complete system

The target is the complete local pipeline on each teammate's computer. Run the
same committed implementation with real weights and actual audio. A bounded
first encounter checks execution; it is not a replacement for full evaluation,
missing features, clinician review, or the complete deliverable.

Latest reviewed teammate change: `751da5a` separates automated issues from
clinician approval. Generated notes always remain pending clinician review.
This fixes the earlier flag ambiguity; the actual review application remains
to be implemented.

## Shared code, local resources

Every person runs `python scripts/project.py system` after following TEAM_SETUP.
OS, RAM, disk and available accelerators are checked independently; no one must
know the other three computers' specifications in advance. Reports stay in
ignored `.local/system.json`. Sandbox-denied hardware queries must be repeated
with normal local permissions; unknown values never establish model capacity.

Commit code, schemas, dependency specifications, run configurations and commands
**before** large downloads, benchmarks, training or long inference. Others then
pull the same main revision. Each has independent local data, weights, login,
outputs and optional `configs/runtime.local.yaml`. Do not commit that local
configuration or hardware inventory. Historical benchmark hardware in results/
explains previous measurements and cannot select settings for another machine.

Choose a common reference checkpoint, language settings, precision/quantization,
context length and dataset split before comparing scores. Different hardware
may use CPU/CUDA/MPS where compatible, but record that distinction. Never
silently change model size or precision and call the results comparable.
Waiting longer cannot make an arbitrary model/context fit RAM or GPU memory.

## Current stage readiness

| Section | Implemented real behavior | What still needs to happen |
| --- | --- | --- |
| Dataset import | BeTraC manifests, Eka audio export and labels; revision/hash checks | Independently download/validate on each system |
| Audio preparation | Decode, finite-sample checks, mono/16 kHz resampling and metadata | Test each OS's codecs with actual target recordings |
| ASR | Real Transformers Whisper transcription, timestamps and en/hi WER/CER; executed on CPU | Investigate poor Hindi repetition, long-form decoding, model choice, language-specific evaluation and measured latency/memory |
| VAD | Silero adapter and timestamp preservation | Install compatible torch/torchaudio/Silero; execute and inspect actual regions/short clinical turns |
| Diarization | pyannote turns and ASR timestamp alignment | Install compatible pyannote/audio stack; download approved weights; run actual speaker tests and measure DER on annotated data |
| Speaker roles | Local-LLM role assignment with source-ID checks | Configure the local LLM and evaluate roles against reviewed labels |
| Normalization | Raw/normalized separation and auditable lookup substitutions | Review language/abbreviation coverage; avoid unsupported corrections |
| Fact extraction | LLM prompt, schema/evidence/speaker validation | Real local LLM execution and reviewed assertion/medication/number checks |
| Terminology | Exact CSV linker with ambiguous/not-found statuses | Populate a licensed, curated vocabulary; current table is empty |
| Cited SOAP and verification | Claim schema, citation validation, bounded retries, omission/dose hooks, pending clinician approval | Real LLM serving and end-to-end evaluation; dose table is empty and LLM verification is not ground truth |
| Plan translation | Optional local-LLM translation prompt exists | Implement protected numbers/drug names, backend integration and bilingual evaluation; not a completed translation subsystem |
| Clinician review/export | Approval state exists | Implement review UI/API, versioned edits/approval and JSON/PDF exports |
| Training/research | Separate ACI pilot and cited-LoRA utilities exist | Reviewed evidence labels, matched experiments/ablations, provenance and real measured results |

## Commands that run existing sections

Activate .venv using TEAM_SETUP, then the following one-line commands work with
the portable launcher on Windows/macOS/Linux. The paths are repository-relative.
Prepare the documented dataset manifests first. Each command executes actual
stage code; model commands fail if the required model/dependency is unavailable.

Audio preparation from the actual BeTraC audio manifest:

```text
python scripts/project.py cli prepare-audio --manifest data/processed/betrac-smoke/audio_manifest.csv --outdir data/processed/audio-validation --sample-rate 16000
```

Real ASR on the existing four-clip validation manifest, once the ASR dependencies
are installed with `python scripts/project.py setup --profile asr`:

```text
python scripts/project.py cli evaluate-asr --manifest data/manifests/eka-smoke.csv --config configs/base.yaml --device cpu --outdir data/outputs/asr-validation
```

This uses the explicit configured checkpoint (currently Whisper Small). The
earlier CPU run's Hindi failure must be investigated before declaring it the
final ASR choice. `transcribe-audio --model CHECKPOINT --device DEVICE` supports
an explicit alternative without rewriting shared defaults. We have not run a
large checkpoint on all four machines and do not claim it fits them all.

Once a common ASR configuration is frozen, complete Eka evaluation uses the
full exported manifest and no sample limit:

```text
python scripts/project.py cli evaluate-asr --manifest data/manifests/eka-medical-asr-evaluation.csv --config configs/base.yaml --device cpu --outdir data/outputs/asr-full
```

This can take substantial time. Publish its validated dependency/config changes
first, then each teammate runs locally. Outputs include predictions and WER/CER;
they are not SOAP, clinical-factuality or diarization scores.

After installing and validating the real VAD environment:

```text
python scripts/project.py cli detect-speech --manifest data/processed/betrac-smoke/audio_manifest.csv --audio-dir data/processed/audio-validation --outdir data/processed/vad-validation
```

After installing and validating pyannote with the approved individual HF login:

```text
python scripts/project.py cli diarize-audio --audio data/raw/sully-smoke/audio/pm57-d1c01.mp3 --out data/outputs/diarization-validation.json --config configs/base.yaml
```

These two dependency environments are not validated yet. A successful HF file
access check does not prove pyannote loads or decodes audio on every OS.

Once a real local OpenAI-compatible LLM server and the configured model are
running, check connectivity and run the supplied transcript through the actual
fact -> cited SOAP -> verification path:

```text
python scripts/project.py check --check-llm
python scripts/project.py cli run --manifest data/processed/betrac-smoke/gold_manifest.csv --config configs/base.yaml --outdir data/outputs/soap-transcript-validation
```

Inspect facts, evidence, all claims, omissions, warnings, attempts and pending
approval. The current run CLI may finish with failed encounter entries, so
`run.jsonl` must be checked; fixing failure exit status remains necessary.

After VAD/ASR/diarization/role stages and the LLM path work, use the audio
manifest with a reviewed configuration to exercise the integrated audio path:

```text
python scripts/project.py cli run --manifest data/processed/betrac-smoke/audio_manifest.csv --config configs/runtime.local.yaml --outdir data/outputs/soap-audio-validation
```

Create that local config from base.yaml and enable diarization only when its
real environment works. If it is disabled, the run cannot be reported as the
complete diarization-enabled pipeline. Translation and clinician review also
need their implemented stages before claiming the complete intended system.

## Next implementation order

1. All four machines inspect themselves, recreate their own environments,
   authenticate, and run the shared tests. Record failures locally with secrets
   removed; do not change shared main to match one machine's paths.
2. Publish a compatible real audio dependency setup and decoding configuration;
   run audio preparation/VAD/ASR on actual recordings and investigate Hindi
   repetition. Add per-stage timing, failure records and memory measurements.
3. Publish a local LLM-serving setup with a jointly chosen model/context; prove
   transcript -> facts -> cited SOAP -> verification with saved artifacts.
4. Validate pyannote and roles, then connect the actual audio path to the same
   SOAP code. Add end-to-end regressions, input/output containment, full attempt
   provenance and reliable failure statuses.
5. Finish terminology, protected translation and the clinician review/export
   application. Measure language quality, citations, omissions and factuality
   with reviewed labels. Run controlled baselines and training only with suitable
   data/hardware and saved evidence; never invent clinical or research results.

For four contributors, ownership can be split into audio/ASR, diarization/roles,
cited SOAP/verification, and review/export/evaluation. This divides development
responsibility; it does not split deployment across computers. Everyone must
eventually be able to run the entire integrated application locally. Coordinate
shared schemas and same-file changes before pushing main.

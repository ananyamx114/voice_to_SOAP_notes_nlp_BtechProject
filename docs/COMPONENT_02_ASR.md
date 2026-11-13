# ASR component: full Whisper Large-v3, sequential transcription

This run uses the complete pretrained multilingual `openai/whisper-large-v3`
checkpoint, revision `06f233fe06e710322aca913c1bc4249a0d71fce1`, without quantization
or substitution by small/base/turbo/distilled checkpoints. It is a strong
multilingual reference; no unmeasured claim of best clinical Hindi/Marathi
accuracy is made. This stage transcribes speech, not translation or SOAP.

## Model and local requirements

The selected safetensors weight file is **3,087,130,976 bytes** (about 2.88 GiB).
The downloader selects that file plus processor/config/attribution files, not
the separate FP32 weight shards. Full checkpoint access was verified using the
existing HF login; no additional paid service or account is needed.

The runtime explicitly records precision: auto selects FP16 on CUDA/Apple Metal
and FP32 on CPU. This is floating-point precision, not weight quantization. The
entire model architecture is retained. CPU FP32 uses more memory than the stored
FP16 checkpoint; actual capacity must be checked on each machine. Batch size is
one, and model/device errors are reported rather than silently using another
checkpoint or device. Do not disable framework memory protections to force a run.

The first Metal attempt showed a 7.4 GiB process footprint and stalled before
finishing its first clip. Transformers' timestamp generation requests encoder
attention outputs as well as decoder cross-attention; only the decoder weights
are used for token alignment. The backend now leaves encoder computations and
the complete checkpoint intact but does not retain unused per-layer encoder
attention matrices. Decoder attention/timestamps are preserved. This memory
optimization is explicit in output metadata; it is not quantization or pruning.

The earlier Whisper Small run produced Hindi repetition. This implementation
uses native sequential long-form decoding, attention masks supplied by the
Transformers pipeline, and the model card's previous-token/repetition/silence
fallback settings. These are published settings, not parameters fitted to the
Eka test references. A changed model and changed decoding do not isolate the
cause of any improvement; controlled comparisons are still required.

Sources: [Large-v3 model card](https://huggingface.co/openai/whisper-large-v3),
[Whisper generation documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/whisper).

## Publish first, then execute

Pull shared main with a clean checkout. Follow TEAM_SETUP, activate .venv, and
run the following one-line commands. Without activation, the launcher can be
invoked with python3 on macOS/Linux or py -3.12 on Windows.

```text
python scripts/project.py system
python scripts/project.py setup --profile asr
python scripts/project.py test
python scripts/project.py download-asr
python scripts/project.py download-data --dataset all
python scripts/project.py prepare-asr-data
```

HF files and all input/output audio stay local and ignored. The download checks
space before starting and retains at least 8 GiB free after the selected weight
download. Its default cache is this clone's data/models/huggingface.

## Execute real ASR and scored validation

Choose the device explicitly, using `mps`, `cuda:0` or `cpu` as supported by that
machine. Example commands use `mps` for the originating machine; this is not a
requirement on teammates. The configuration is shared and independent of OS.

Four complete short English/Hindi clips, with reference text used only to score:

```text
python scripts/project.py asr-run --manifest data/manifests/eka-smoke.csv --outdir data/outputs/asr-large-v3-eka --device mps
```

Two complete BeTraC consultations, using the previous VAD artifacts (see
COMPONENT_01_VAD). Speech regions separated by at most two seconds are grouped
to retain context and interior pauses; larger gaps are skipped. Original region
offsets are restored for every timestamp. No speech text is taken from gold data.

```text
python scripts/project.py asr-run --manifest data/processed/betrac-smoke/audio_manifest.csv --vad-dir data/outputs/vad-component-01 --outdir data/outputs/asr-large-v3-betrac --device mps
```

One full human role-played Sully/PriMock57 consultation, direct sequential path:

```text
python scripts/project.py asr-run --manifest data/manifests/sully-asr-selection.csv --outdir data/outputs/asr-large-v3-sully --device mps
```

These selections exercise full recordings and both available languages; they
are not the full 3,939-row Eka benchmark or the full BeTraC development split.
After validation, the same runner accepts the full exported Eka manifest with
no code changes or example limit. Publish the intended command before that long
run. Do not claim Marathi or code-mixed accuracy without scored examples.

## Artifacts, progress and failure handling

Every completed encounter writes `asr.json` and `transcript_raw.json` with raw
text, word timestamps, source hash, decoded source intervals, model revision,
precision, generation settings, seed, timings and warnings. Speaker/role remain
unknown until diarization/role attribution; model confidence is not fabricated.
`progress.jsonl` is written after each encounter. `summary.json` includes WER/CER
by language and encounter, completion/failure counts and processing/memory data.

Timing includes model initialization/loading in the first inference call.
Process peak RSS is a process-lifetime high-water mark, not peak total GPU/Metal
memory. Compare performance only with those scopes and actual precision/device
reported. A saved input hash does not prove dataset provenance or consent.

On failure the runner records details, stops rather than repeatedly retrying a
resource error, and exits nonzero. Completed encounters are retained. Re-run
with `--resume` to reuse matching completed records; source/config/reference
changes require a new output directory. References never enter decoding.
Temperature fallback is seeded per encounter, but cross-device bitwise equality
is not promised. Do not edit the saved transcript to improve automated scores.

Words with missing, reversed or out-of-source timestamps are flagged for review,
not assigned invented times. Keep original transcripts and input audio, inspect
repetition and medication/number/negation errors, and listen against source audio.
WER/CER and timestamp checks do not establish clinical correctness.

The old chunked path and previous results remain reproducible. The new decoder
is also wired into normal transcribe/evaluate and integrated audio processing
when explicit asr_options are configured. The component config is ASR-only; a
full SOAP run needs the other required configuration sections as well.

# ASR results and SOAP-stage handoff

Completed follow-up: [27 November epoch 12 paired benchmark](../results/asr-run3-test-2026-11-27/README.md), with four WER gains and two regressions. Preserve the older run2 report below.


Latest handoff: [27 November results](LATEST_RESULTS.md), [decoder comparison](../results/asr-diagnostics-2026-11-26/README.md) and [14-epoch history](../results/asr/run3-validation.csv). These later validation results do not replace the older completed run2 independent-test claims below.


Latest report addition: [26 November continuation snapshot](../results/interim-2026-11-26/README.md)
and [report metric guidance](REPORT_METRICS.md). The 12-hour continuation has
validation results only; the completed paired-test scores and adapter setup
below refer to the older run2 experiment. Do not substitute the new validation
scores into that test table or assume the old run2 adapter is the run3 adapter.

The shared entry point is [`results/asr`](../results/asr/README.md).
The expanded ASR experiment finished five epochs / 515 optimizer updates.
Publishable aggregates and the report section are under
[`results/asr-lora-run2`](../results/asr-lora-run2/README.md).
Check `summary.json` before quoting test results: only `status: complete` means
all four matched CUDA evaluation datasets finished. Pending rows are not scores.

## What to use in the report

- `ASR_REPORT_SECTION.md` / `.docx`: methods, data, results and limitations.
- `validation-epochs.csv`: all five epochs; these are validation measurements.
- `paired-test-results.csv`: matched base-versus-adapter CUDA test measurements.
- `validation-wer.{png,svg,pdf}` and, once complete,
  `paired-test-wer.{png,svg,pdf}`: report figures.
- `summary.json`: configuration, model/adapter pins, software versions, archive
  checksums and aggregate evidence. No raw audio or transcripts are published.

Validation base and adapter ran on different backends (MPS/CUDA). Use the
separate paired CUDA evaluation for the final base-versus-adapter test comparison.
There are 71 selected recordings, not a full-dataset benchmark. Include all
regressions; a lower WER can coexist with a worse CER.

## Run the trained ASR component

Set up each machine independently using `TEAM_SETUP.md` and
`python scripts/project.py system`. The complete base model and ASR adapter are
required; the repository contains their configuration and results, not weights.
Obtain the validated adapter through an approved private channel or reproduce
the training run under your own account. Keep it under ignored
`data/models/asr-adapters/large-v3-expanded-run2/`.

The selected adapter SHA-256 is
`2557815bc1214fd48fbc4364efe6b1fb074c4598a66309d57a50f661da97012f`.
The base model is `openai/whisper-large-v3` at revision
`06f233fe06e710322aca913c1bc4249a0d71fce1`. `validate_adapter` checks task,
checkpoint pin and weights; an ASR adapter is not interchangeable with a Qwen
SOAP adapter. Keep `adapter_config.json`, `adapter_model.safetensors`,
`asr_adapter.json` and the saved processor/tokenizer files together.

With the appropriate environment installed and actual input audio manifested:

```text
python scripts/project.py asr-run --manifest YOUR_AUDIO_MANIFEST.csv --adapter data/models/asr-adapters/large-v3-expanded-run2 --device auto --outdir data/outputs/asr-team-run
```

Use `python3` on macOS/Linux or `py` on Windows before environment activation.
The launcher selects this machine's environment and working directory. Do not
assume the originating Mac's memory settings or Kaggle's CUDA device exist on
another machine. Read the completed metrics before selecting the base or LoRA
model for an application; LoRA has known regressions on some diagnostic audio.

## Input to the Qwen stage

Each successful ASR encounter saves:

- `asr.json`: `prediction` is the recognized transcript; `chunks` contain
  timestamps; model pin, adapter hash, warnings and audio hash record provenance.
- `transcript_raw.json`: utterances retain IDs, raw text and timestamp bounds.
  `speaker_id` and `role` are currently `unknown` in this ASR-only experiment.

The direct Qwen experiment's `dialogue` input should receive the ASR **prediction**
when measuring a real voice-to-SOAP path. Its `reference_transcript` field is
evaluation gold and must not be used as recognized input. Preserve the original
ASR artifacts when passing text onward. Do not invent doctor/patient labels;
speaker diarization and role assignment require their own implementation and
validation. The hybrid Qwen path relies on speaker/clinical heuristics and must
be evaluated with the actual information available from audio.

The friends' Qwen gold-transcript-to-SOAP scores are a separate experiment.
They cannot be reported as measured voice-to-SOAP quality without running the
ASR predictions through that stage. The heuristic four-string SOAP format also
remains separate from the evidence-cited claim schema. Hindi/Marathi ASR scores
do not establish multilingual SOAP generation or translation quality. All
generated notes remain pending clinician approval.

## Rebuild the report

For optional figures, install `requirements-reports.txt` in a separate report
environment. Markdown/CSV/JSON and Word export use the Python standard library.
The inputs are verified private `resume-state.tar` archives with their matching
`.inventory.json` files; they stay outside Git.

```text
python scripts/build_asr_report.py --training-archive PATH_TO_TRAINING/resume-state.tar --evaluation-archive PATH_TO_COMPLETED_EVALUATION/resume-state.tar --require-complete --plots --docx
```

`--require-complete` rejects incomplete evaluation. The exporter checks input
checksums, validation-based epoch selection, exact paired coverage and evaluated
adapter identity. Do not edit generated metrics to change a conclusion.

## Git handoff

Keep models, source/prepared audio, prediction text, tokens, local job manifests
and machine settings ignored. Publish code, tests, portable config/dependencies,
aggregate results and the report section only. Fetch and review teammate updates
before integrating. Preserve local edits and resolve merges normally; never
force-push shared `main`. The user authorized publication of this completed ASR
package on 15 November 2026.

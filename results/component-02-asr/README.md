# Full Whisper Large-v3: actual local ASR execution

Executed 13 November 2026 using the complete `openai/whisper-large-v3` checkpoint,
revision `06f233fe06e710322aca913c1bc4249a0d71fce1`, FP16 on Apple Metal, without
quantization, pruning or a smaller-model fallback. Inference implementation:
`59cd70f`. See [reproduction instructions](../../docs/COMPONENT_02_ASR.md).

Seven inputs total **18.10 minutes of source audio**. Both BeTraC consultations
used VAD-derived context spans with original offsets, skipping only the configured
long silence gaps. The Sully recording and Eka clips used direct sequential
decoding on their entire waveforms. The seven inputs are a validation selection,
not the full 3,939-row Eka benchmark or the entire BeTraC development set.

| Source/language | Complete inputs | WER | CER |
| --- | ---: | ---: | ---: |
| Eka English | 2 | 0.1333 | 0.1083 |
| Eka Hindi | 2 | 0.5000 | 0.4932 |
| BeTraC English, synthetic consultations | 2 | 0.0298 | 0.0262 |
| Sully/PriMock57 English, human role-play | 1 | 0.2221 | 0.1704 |

Do not present these mixed results as clinical readiness or state-of-the-art
performance for this project's domain/languages. The low BeTraC error does not
establish real consultation accuracy. Small language selections do not support
general conclusions or a statistically meaningful superiority claim.

## Execution and artifact checks

- All seven completed. Total recorded inference time was about 26.78 minutes,
  including first model loading within each of three separate runs. It is not a
  hardware-independent latency estimate. Process RSS does not account for all
  GPU/Metal memory.
- Saved 3,029 word-timestamp chunks and corresponding unique utterance IDs.
  Speaker/role remain unknown pending the diarization/role stages.
- Independent timestamp checks found one 40 ms backward word-start step in
  Sully (chunk 745). All other checked chunks passed bounds/order/presence
  checks. Raw timings and text were preserved, not silently clamped/reordered.
  The validator was extended to flag this case for future runs. Original run
  summaries predate that additional warning; run-validation.json records the
  independent check.
- A real `--resume` invocation reused all four matching completed Eka artifacts
  without model inference. Config/source/reference changes require a new run.
- The initial Metal attempt retained encoder attention outputs and stalled
  with approximately 7.4 GB native-observed process footprint before producing
  a prediction. It was stopped. Encoder computations were retained but unused
  encoder attention outputs were omitted; all seven subsequent runs completed.
  Decoder cross-attention needed for word alignment was preserved.

## Errors and outstanding evaluation

Hindi references mix Latin-script medical terms and Devanagari; predictions use
Devanagari, so strict WER includes script differences. Those differences do not
explain away all errors. One reference numeric medication dose was not reproduced
reliably. This needs bilingual/clinical review, not score normalization chosen
after seeing test outputs. The earlier Small run also changed decoding settings,
so score differences do not isolate the effect of model size.

Sully scoring contained no residual line-start speaker labels. Its alignment
counts were 85 substitutions, 208 deletions and 26 insertions. Raw source and
predictions remain local for error review; these counts do not classify clinical
severity. No Marathi recordings/reference transcripts were available for scoring.
Word-timestamp consistency is not gold boundary accuracy or medical correctness.

Dataset provenance/attribution: [BeTraC](https://huggingface.co/datasets/BeTraC/betrac-2026),
[Eka](https://huggingface.co/datasets/ekacare/eka-medical-asr-evaluation-dataset),
[Sully](https://huggingface.co/datasets/SullyHQ/open-clinical-conversations) and
its [PriMock57 source](https://github.com/babylonhealth/primock57). Cards and
attribution files remain beside the ignored downloads. This directory contains
only aggregate metadata/results, no audio, transcripts, tokens or model weights.

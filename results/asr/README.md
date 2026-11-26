# ASR experiment results

## Latest interim continuation — 26 November 2026

The [report snapshot](../interim-2026-11-26/README.md) adds the separate 12.03-hour,
4,501-segment Whisper Large-v3 LoRA continuation. Selected epoch-10 validation
WER is English **1.49%**, Hindi **23.99%**, Marathi **44.83%** (macro **23.44%**).
Validation CER is **0.57% / 9.77% / 11.62%**, respectively, on 239 segments from
68 encounters. Live progress was update 3384; locally verified archive: 3330.

These are validation results, not a new paired independent test. Epoch 11
regressed in macro-language WER; see [all available epochs](../interim-2026-11-26/asr-validation.csv)
and [exact aggregate/provenance](../interim-2026-11-26/asr-snapshot.json).
The completed run2 paired test below remains unchanged.

This is the ASR entry point alongside the team's `results/hybrid/` and
`results/baseline/` directories. It links measured summaries and report files;
source audio, datasets, private predictions, checkpoints and credentials are
not distributed with the ASR results.

## Completed experiment

Full Whisper Large-v3 with rank-16 LoRA; five epochs / 515 optimizer updates.
Training: 1,644 aligned samples, about 4.03 hours (English 2.01 h, Hindi 1.01 h,
Marathi 1.01 h). Validation: 239 separate samples. The shared adapter was
selected at epoch 5 using validation macro-language WER, before test evaluation.

| Epoch | English validation WER | Hindi validation WER | Marathi validation WER | Macro-language WER |
|---|---:|---:|---:|---:|
| 1 | 4.45% | 26.07% | 69.19% | 33.24% |
| 2 | 1.44% | 24.69% | 64.21% | 30.11% |
| 3 | 1.38% | 24.13% | 60.33% | 28.61% |
| 4 | 1.32% | 24.13% | 59.96% | 28.47% |
| 5 | 1.32% | 24.13% | 59.59% | 28.35% |

These are validation WER measurements, not Qwen token losses or percentages
of clinically correct notes. The selected checkpoint is fixed for all final
test groups, including those on which it regresses.

The base and adapter were evaluated on the same 71 selected recordings on CUDA.
These are selected-recording benchmarks, not complete-corpus or clinical tests.

| Dataset | Language | Recordings | Base WER | LoRA WER | Base CER | LoRA CER |
|---|---|---:|---:|---:|---:|---:|
| Eka | English | 2 | 13.33% | 16.67% | 10.83% | 14.01% |
| Eka | Hindi | 2 | 50.00% | 39.13% | 49.32% | 42.53% |
| BeTraC | English | 2 | 2.98% | 4.56% | 2.62% | 2.63% |
| Sully/PriMock57 | English | 1 | 22.21% | 22.63% | 17.04% | 17.07% |
| FLEURS | Hindi | 32 | 32.80% | 26.02% | 15.42% | 10.28% |
| FLEURS | Marathi | 32 | 79.87% | 61.55% | 27.27% | 21.04% |

WER and CER improved in the three Hindi/Marathi groups and regressed in all
three English groups. Do not describe this adapter as universally better.
Validation improvements and full-encounter test improvements are different
claims. Historical validation used MPS for the base and CUDA for the adapter;
the final test table above uses matched CUDA runs for both models.

## Report and machine-readable evidence

- [Complete report section](../asr-lora-run2/ASR_REPORT_SECTION.md)
- [Word report section](../asr-lora-run2/ASR_REPORT_SECTION.docx)
- [Paired test CSV](../asr-lora-run2/paired-test-results.csv)
- [All five validation epochs](../asr-lora-run2/validation-epochs.csv)
- [Aggregate JSON, model/config pins and archive checksums](../asr-lora-run2/summary.json)
- [Publication verification and test scope](verification.json)
- [Paired test figure](../asr-lora-run2/paired-test-wer.png)
  ([SVG](../asr-lora-run2/paired-test-wer.svg), [PDF](../asr-lora-run2/paired-test-wer.pdf))
- [Validation figure](../asr-lora-run2/validation-wer.png)
  ([SVG](../asr-lora-run2/validation-wer.svg), [PDF](../asr-lora-run2/validation-wer.pdf))
- [Earlier three-epoch experiment and checkpoint-selection follow-up](../asr-lora-run1/README.md)
- [Original full-model base evaluation](../component-02-asr/README.md)
- [Preceding Silero VAD execution](../component-01-vad/README.md)

## Reproduce or integrate

Start with [team setup](../../docs/TEAM_SETUP.md), then read
[resources and commands](../../docs/ASR_RESOURCES.md),
[expanded experiment](../../docs/ASR_LORA_EXPANDED.md),
[Kaggle continuation](../../docs/KAGGLE_ASR.md), and
[ASR-to-Qwen handoff](../../docs/ASR_TEAM_HANDOFF.md).
Each person downloads/authenticates independently. The full model is required;
the scripts do not substitute a smaller model or silently truncate targets.

The ASR scores measure transcription overlap, not SOAP quality or clinical
correctness. Feeding gold transcripts to a Qwen experiment does not measure
voice-to-SOAP performance. Every generated SOAP note still needs clinician
approval. Preserve the supplied BeTraC/FLEURS attribution and the Sully overlap
limitations documented in the detailed report.

# Latest report metrics — completed testing,27 November2026

Use [the completed Qwen7B comparison](../results/qwen7b-2026-11-27/README.md) and [the selected epoch 12 ASR paired benchmark](../results/asr-run3-test-2026-11-27/README.md). Earlier [26 November interim snapshots](../results/interim-2026-11-26/README.md) remain historical; their pending/partial statuses no longer describe the latest tests.

## Minimal report update

1. Add the three-model Qwen7B table below to Chapter6 and update the conclusion. HYBRID is now evaluated; remove current-tense statements that its test is pending.
2. Add the new71-recording ASR table as a separate experiment in Section6.9. Keep the older run2 table unchanged and clearly labelled.
3. Keep the Qwen7B experimental setup: three epochs/4800 updates,1600/200/200 synthetic internal split, full FP16 base, rank 8 LoRA, deterministic1024-new-token test generation.
4. State both improvements and regressions. These are component overlap/structure results, not clinical accuracy or a completed end-to-end system.

| Qwen7B model | Test examples | Valid / invalid | Validity | Failure-aware mean section ROUGE-L |
|---|---:|---:|---:|---:|
| Pretrained control | 200 | 196 /4 | 98.0% | 0.229839 |
| DIRECT LoRA | 200 | 186 /14 | 93.0% | **0.429454** |
| HYBRID LoRA | 200 | 167 /33 | 83.5% | **0.378346** |

All three stages cover the exact same 200 dialogue/reference pairs. Invalid outputs count as zero. HYBRID minus DIRECT is-0.051108, paired bootstrap95% interval[-0.078376,-0.024121] over 10000 encounter resamples, seed 42. This is sample uncertainty conditional on these trained models, not repeated-seed or clinical evidence. On160 common-valid encounters, DIRECT/HYBRID scores are0.457607/0.454027. Both adapters improve overlap over the pretrained control while reducing structural validity.

> On200 synthetic BeTraC internal-test encounters, Qwen2.5-7B DIRECT and HYBRID LoRA achieved failure-aware mean section ROUGE-L of 0.4295 and0.3783, compared with 0.2298 for the pretrained control. Valid-JSON rates were93.0%,83.5% and98.0%, respectively. The rule-augmented HYBRID configuration did not outperform transcript-only DIRECT. Clinical correctness was not evaluated.

## Latest ASR table: numeric_indic_v2

| Dataset | Language | N | Base WER% | Selected epoch 12 WER% | Base CER% | Selected epoch 12 CER% |
|---|---|---:|---:|---:|---:|---:|
| Eka | English | 2 | 13.33 | 16.67 | 10.83 | 14.01 |
| Eka | Hindi | 2 | 50.00 | 41.30 | 49.32 | 42.99 |
| BeTraC | English | 2 | 3.88 | 4.75 | 2.61 | 2.46 |
| Sully/PriMock57 | English | 1 | 21.97 | 19.75 | 17.09 | 14.69 |
| FLEURS | Hindi | 32 | 34.80 | 24.54 | 18.44 | 9.63 |
| FLEURS | Marathi | 32 | 81.44 | 50.41 | 25.73 | 14.15 |

> The selected epoch 12 Whisper Large-v3 adapter improved WER in four of six repeated dataset/language groups and regressed in two. Hindi and Marathi FLEURS WER decreased from 34.80% to 24.54% and81.44% to 50.41%, respectively. These previously inspected recordings include general read speech and very small English clinical groups; results do not establish independent clinical generalization. Hindi/Marathi remain above the5% objective.

This evaluation uses sequential beam 1 decoding. The prior beam 5 Marathi42.60% figure is a validation result and must not replace the repeated-benchmark Marathi50.41%. Both legacy and numeric-safe scores are available in [CSV](../results/asr-run3-test-2026-11-27/paired-results.csv). Scoring corrections alone are not model improvements. Base predictions differ from older experiments; do not attribute cross-run differences entirely to more training data/epochs.

The22.42-hour expanded cohort is prepared but untrained. Native reference review remains pending. Clinical extraction F1, DER, role accuracy, clinical factuality, translation preservation and end-to-end audio-to-SOAP performance remain unmeasured. Two SOAP test encounters occur in ASR training; handle those separately for any independent audio-to-SOAP claim. Every generated note requires clinician approval.

## Files for teammates

- [Complete handoff](LATEST_RESULTS.md).
- [Qwen table CSV](../results/qwen7b-2026-11-27/metrics.csv), [aggregate JSON](../results/qwen7b-2026-11-27/summary.json), [paired analysis](../results/qwen7b-2026-11-27/paired-analysis.json), [invalid-output diagnostics](../results/qwen7b-2026-11-27/invalid-output-analysis.json).
- [HYBRID stage metrics](../results/hybrid/qwen7b-2026-11-27/metrics.json).
- [ASR aggregate comparisons](../results/asr-run3-test-2026-11-27/comparisons.json), [provenance](../results/asr-run3-test-2026-11-27/provenance.json), [older-run context](../results/asr-run3-test-2026-11-27/historical-context-v1.csv).

Pull shared main while preserving local edits. No datasets/models are needed to read these aggregates. Raw transcripts, predictions, audio, models, optimizer state, credentials and server logs remain private. Server resources were released after checksum verification.

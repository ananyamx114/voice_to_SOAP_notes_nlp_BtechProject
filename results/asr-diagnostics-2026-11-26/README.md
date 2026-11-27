# Completed Whisper Large-v3 decoder diagnostics

Published 27 November 2026. Real Kaggle CUDA inference used the selected epoch12 rank-16 LoRA adapter, full FP16 Whisper Large-v3 and unchanged frozen validation audio/references. No training occurred. Greedy decoding exactly reproduced archived epoch12 WER under the legacy scorer; backups were verified before releasing the GPU session.

| Language | Greedy validation WER | Beam5 validation WER | Greedy training-subset WER |
|---|---:|---:|---:|
| English | 1.37% | 1.34% | 1.41% |
| Hindi | 22.90% | 21.66% | 14.81% |
| Marathi | 44.06% | 42.60% | 33.72% |

All table values use `numeric_indic_v2`. Validation comprises 175 English, 32 Hindi and 32 Marathi segments (239 segments, 68 encounters); training diagnostics use 32 distinct groups per language, 96 segments total. Macro-language validation WER falls from 22.78% to 21.86%, an absolute 0.91 percentage-point reduction. This is a validation decoder comparison, not an independent test gain or fine-tuning gain. Native review is pending; dataset references have not been relabelled.

[Machine-readable CSV](metrics.csv), [greedy validation](validation-beam1.json), [beam5 validation](validation-beam5.json), [training subset](train-subset-beam1.json), [checkpoint/archive provenance](provenance.json). Both legacy and numeric-safe scorer results are retained in JSON/CSV; do not mix protocols.

Beam search provides a measured but modest improvement; it does not meet the below-5% Hindi/Marathi objective. High training-subset error means a simple overfitting explanation is insufficient. The original training run remains paused at update4100; epoch12 remains its selected adapter. [Run3 state and validation history](../asr/README.md) and [balanced-data work](../../docs/ASR_IMPROVEMENT_WORK.md) remain separate.

The [22.42-hour expanded cohort](../asr-balanced-v4/README.md) is prepared, not trained. Next compare balanced sampling on the original cohort before adding expanded data. Keep optimizer state and selected checkpoints recoverable; do not overwrite the old experiment. Source: [diagnostic runner](../../scripts/diagnose_asr.py). Raw audio, references, predictions, private connection URLs and model weights are excluded.

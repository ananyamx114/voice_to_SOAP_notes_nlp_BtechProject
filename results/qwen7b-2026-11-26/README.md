# Qwen7B completed training and available test results

Update: [HYBRID test evaluation completed on 27 November](../qwen7b-2026-11-27/README.md). The pending status below records the26 November cutoff and is preserved as historical evidence.


Published 27 November 2026 from the checksum-verified 26 November DGX handoff.
This supersedes the older interim snapshot for these experiments; older snapshots remain unchanged.

Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28` used full FP16 base weights and rank-8 LoRA (alpha 16, dropout 0.05, q/k/v/o projections; 5,046,272 trainable parameters). Both approaches completed three epochs / 4,800 optimizer updates on 1,600 synthetic BeTraC examples, with 200 validation examples. Checkpoint selection used validation loss only. DIRECT consumes dialogue; HYBRID adds rule-extracted facts. These four-string SOAP adapters are separate from the evidence-cited pipeline.

| Training | Selected update | Best validation loss |
|---|---:|---:|
| DIRECT | 4800 | 0.7365856767 |
| HYBRID | 4800 | 0.7362419963 |

The small loss difference does not establish a HYBRID quality gain. HYBRID validation moved to a separate GPU after update 1401; parity with the synchronous calculation passed before continuing. Its resumed training-loss average has a different denominator and must not be compared with the uninterrupted DIRECT average.

| Internal test | Examples | Valid | Invalid | Failure-aware mean section ROUGE-L |
|---|---:|---:|---:|---:|
| Pretrained baseline | 200 | 196 | 4 | 0.229839 |
| DIRECT LoRA | 200 | 186 | 14 | 0.429454 |
| HYBRID LoRA | not evaluated | — | — | — |

Invalid outputs score zero. The baseline and DIRECT used identical test data, deterministic decoding and a 1,024-new-token budget. DIRECT improves mean section ROUGE-L by 0.199614 (19.96 points on a 0–100 scale), while output validity regresses from 98% to 93%. Paired bootstrap 95% interval for the score difference: [0.178812, 0.219893], 10,000 resamples, seed 42. There were 185 wins, 14 losses and one tie. This interval reflects this internal synthetic sample, not clinical generalization.

Thirteen of the 14 invalid DIRECT outputs reached the 1,024-token budget and had unterminated JSON strings, consistent with truncation. One shorter output had a JSON syntax error. The baseline's four failures were syntax errors. Budget/format changes must be selected using validation data and reported as a new protocol; do not repair only failed test cases and call that an independent gain.

[Exact snapshot](soap-snapshot.json), [paired diagnostics](paired-analysis.json), [validation curve](validation.csv), [DIRECT/baseline stage files](../baseline/qwen7b-2026-11-26/), [HYBRID stage files](../hybrid/qwen7b-2026-11-26/), [source and reproduction](../../scripts/qwen7b/README.md).

BeTraC-2000 is an internal split selected from official synthetic training data, with heuristic four-section reference mapping. Two SOAP internal-test encounters occur in ASR training; handle them separately for any audio-to-SOAP claim. No clinical factuality, clinician-reviewed quality or end-to-end performance has been measured. Every generated note requires clinician approval. Preserve [dataset attribution](../../datasets/betrac_2000/SOURCE_DATASET_CARD.md) and [preparation methodology](../../docs/BETRAC_2000_MIGRATION.md). Models, raw predictions, optimizer files, data and credentials are excluded from this publication; verified recovery archives remain private.

# Completed Qwen7B control, DIRECT and HYBRID evaluation — 27 November 2026

HYBRID test generation is now complete. This report supersedes the earlier pending/partial status in dated snapshots; the earlier files remain historical evidence. Training was not restarted. All three stages cover exactly the same 200 dialogue/reference pairs; all 600 saved per-example ROUGE-L calculations were independently recomputed after transfer.

## Main results

| Model | Valid JSON | Invalid | Validity | Mean section ROUGE-L, all 200 | Valid-only ROUGE-L |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B pretrained control | 196 | 4 | 98.0% | 0.229839 | 0.234530 |
| DIRECT LoRA | 186 | 14 | 93.0% | **0.429454** | 0.461778 |
| HYBRID LoRA | 167 | 33 | 83.5% | **0.378346** | 0.453109 |

All-example scores count invalid outputs as zero. Valid-only scores use different successful subsets and are supplementary. Both adapters improve reference overlap relative to the pretrained control, while producing more invalid JSON. DIRECT is the strongest completed configuration for the measured overlap and validity outcomes; clinical quality has not been measured.

| SOAP section, all 200 | Control | DIRECT | HYBRID |
|---|---:|---:|---:|
| Subjective | 0.199785 | 0.354367 | 0.309416 |
| Objective | 0.249792 | 0.581748 | 0.522186 |
| Assessment | 0.198787 | 0.420034 | 0.357932 |
| Plan | 0.270994 | 0.361666 | 0.323848 |

## Paired comparison and interpretation

| Difference | Mean ROUGE-L difference | Paired bootstrap95% interval | Wins / losses / ties |
|---|---:|---:|---:|
| DIRECT minus control | +0.199614 | [+0.178812,+0.219893] | 185 /14 /1 |
| HYBRID minus control | +0.148507 | [+0.121035,+0.175295] | 165 /34 /1 |
| HYBRID minus DIRECT | **-0.051108** | **[-0.078376,-0.024121]** | 83 /110 /7 |

Bootstrap uses10,000 paired encounter resamples, seed 42, with failures retained. It reflects variation within this synthetic internal test sample, not model-training seed uncertainty or clinical generalization.

HYBRID adds rule-extracted candidate facts to the dialogue. It creates19 more invalid outputs than DIRECT. On the160 common-valid encounters, DIRECT and HYBRID score0.457607 and0.454027 respectively: their conditional overlap is close. DIRECT alone is valid on 26 cases, HYBRID alone on 7, and both fail on 7. Output reliability therefore accounts for much of the overall separation; this does not establish which individual facts caused failures. It also does not justify evaluating only the common-valid subset as the headline metric.

HYBRID's slightly lower validation loss (0.736242 versus DIRECT 0.736586) did not predict a test-time generation advantage. Validation loss measures reference-token prediction; it does not capture every autoregressive formatting, output-length or extraction failure.

## Invalid output investigation

All33 HYBRID failures were recorded as invalid JSON. Independent parsing found32 unterminated-string errors;28 decoded invalid outputs retokenized to exactly1,024 tokens. This is consistent with frequent truncation under the fixed1,024-new-token budget, but does not prove a single cause for every failure. DIRECT had14 failures,13 of which retokenized to 1,024 tokens and had unterminated strings. No failed output was repaired or excluded from the primary score.

Next study output budgets/structured decoding on validation data before a separately identified evaluation. Do not tune using these200 test cases and present the revised result as untouched evaluation. Review the facts extractor separately; candidate counts are not correctness labels.

Recorded generation time totals: control 2,528s, DIRECT 6,537s, HYBRID 8,143s (approximately12.64/32.69/40.72s per encounter on the recorded setup). These are observed generation costs, not a universal or isolated latency benchmark.

## Protocol, provenance and report wording

Full FP16 `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`; no quantization. Both adapters used rank 8/alpha 16/dropout 0.05, q/k/v/o projections, three epochs and4,800 updates; checkpoint selection used validation loss. All test stages used deterministic generation, seed 42 and1,024 new tokens. [Training records](../qwen7b-2026-11-26/soap-snapshot.json), [execution source](../../scripts/qwen7b/README.md).

[Complete aggregate snapshot](summary.json), [spreadsheet table](metrics.csv), [paired results](paired-analysis.json), [failure diagnostics](invalid-output-analysis.json), [HYBRID stage files](../hybrid/qwen7b-2026-11-27/).

> On200 internal synthetic BeTraC test encounters, full Qwen2.5-7B DIRECT and HYBRID LoRA achieved failure-aware mean section ROUGE-L of 0.4295 and0.3783, compared with 0.2298 for the pretrained control. Their valid-JSON rates were93.0%,83.5% and98.0%, respectively. Rule-augmented HYBRID did not outperform DIRECT in this configuration. Scores measure reference overlap and structure, not clinical correctness.

BeTraC-2000 is an internal1,600/200/200 split selected from official synthetic training data with heuristic four-string SOAP mapping. Two SOAP test encounters occur in ASR training; this text-only comparison is not independent audio-to-SOAP validation. No clinician-reviewed factuality or end-to-end results exist. Every note requires clinician approval. Preserve [BeTraC attribution](../../datasets/betrac_2000/SOURCE_DATASET_CARD.md). Raw outputs, dialogues, credentials, weights and execution logs remain private. The returned archive was checksum-verified and the college workspace removed afterward.

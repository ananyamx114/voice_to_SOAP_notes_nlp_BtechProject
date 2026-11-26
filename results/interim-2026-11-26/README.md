# Interim report metrics — 2026-11-26

This is a frozen interim snapshot, not final pipeline results. Jobs continued running during export.
SOAP snapshot: `2026-11-26T09:12:17.592471+00:00`. ASR progress observed: `2026-11-26T09:12:44.477709+00:00`.

## ASR: larger-cohort Whisper Large-v3 LoRA continuation

Training uses 4,501 segments / 12.03 hours, roughly four hours per language.
Validation uses 239 segments from 68 encounters.
Live progress was update 3384; the latest fully verified local continuation archive was update 3330.
The selected checkpoint is epoch **10**, chosen using macro-language validation WER, not the latest update.

| Language | Selected validation WER | Selected validation CER |
|---|---:|---:|
| en | 1.49% | 0.57% |
| hi | 23.99% | 9.77% |
| mr | 44.83% | 11.62% |

Selected macro-language WER: **23.44%**.
These are diagnostic validation scores; no new independent test evaluation has completed for this continuation.
[All available validation epochs, including regressions](asr-validation.csv) and [configuration/provenance](asr-snapshot.json).

The completed [run2 paired test](../asr-lora-run2/README.md) remains a different experiment. Do not replace its test table with these validation scores.

## SOAP: Qwen2.5-7B-Instruct with rank-8 LoRA

BeTraC-2000 contains 1,600 training, 200 validation and 200 internal-test encounters, selected from official training data.
Conversations/references are synthetic; four-string SOAP targets use heuristic mapping. This is not the evidence-cited claim schema.
DIRECT uses the transcript. HYBRID adds the existing rule-based clinical facts; it is not a trained entity-extraction model.

| Approach | Status | Saved updates | Epoch | Best validation loss |
|---|---|---:|---:|---:|
| direct | complete | 4800/4800 | 3.0 | 0.736586 |
| hybrid | incomplete_checkpoint_snapshot | 1200/4800 | 0.75 | 0.779724 |

Validation loss is token prediction loss, not WER, ROUGE-L or clinical accuracy. Different inputs and training positions preclude claiming a winner from these losses.

| Model | Evaluation status | Encounters | Valid / invalid | Mean section ROUGE-L |
|---|---|---:|---:|---:|
| base | complete | 200/200 | 196 / 4 | 0.229839 |
| direct | partial | 186/200 | 173 / 13 | 0.426978 |
| hybrid | not_started | 0/200 | 0 / 0 | not measured |

Invalid generations count as zero in the all-example ROUGE-L score. The valid-only baseline score is retained separately in JSON.
**DIRECT evaluation is partial in this snapshot. Do not subtract its subset score from the full baseline score or describe the difference as a completed paired improvement.**
Output validity and factual correctness are different outcomes; regressions in validity must also be reported.
HYBRID generation evaluation status: not_started. Test cases must not be used for checkpoint selection or hyperparameter tuning.

[Training summary](soap-training.csv), [validation curves](soap-validation.csv), [evaluation table](soap-evaluation.csv), [exact exported SOAP metrics](soap-snapshot.json).

## Report wording and limits

> Full-precision Qwen2.5-7B-Instruct was adapted with rank-8 LoRA on 1,600 synthetic BeTraC encounters. DIRECT training status was complete (4800 updates), with validation-selected loss 0.736586. The pretrained baseline evaluated 200 internal-test encounters (196 valid, 4 invalid; failure-aware mean section ROUGE-L 0.229839). DIRECT generation evaluation status was partial; HYBRID training status was incomplete_checkpoint_snapshot. Incomplete outcomes must remain explicitly labelled in the report.

> Whisper Large-v3 rank-16 LoRA training continued on a roughly 12-hour multilingual cohort. The validation-selected epoch-10 checkpoint achieved English/Hindi/Marathi WER of 1.49%/23.99%/44.83%. These are validation results, not a new held-out ASR benchmark. The below-5% target is an unmet research objective for languages whose reported WER exceeds it.

Two SOAP internal-test encounters also occur in ASR training. This does not contaminate the text-only SOAP training split, but those encounters cannot support independent audio-to-SOAP test claims.
No end-to-end audio-to-SOAP evaluation, clinical factuality validation, multilingual SOAP evaluation or clinician approval is established by these tables.
Every generated note still requires clinician approval. Model weights, optimizer state, transcripts, predictions, raw datasets and credentials are excluded from Git.

## Refresh without stopping jobs

On the GPU host, run the aggregate-only exporter:

```bash
python scripts/export_soap_interim.py --output .local/interim-soap-results.json
```

Transfer only that JSON through your own authenticated connection. Keep the ASR aggregate snapshot separate from private controller records.
Rebuild a new dated snapshot using aggregate inputs; preserve older snapshots and completed experiments:

```bash
python scripts/build_interim_report.py --asr ASR_AGGREGATE.json --soap SOAP_AGGREGATE.json --outdir results/interim-2026-11-26
```

This builder exports tables only; it does not start, pause or otherwise control model jobs.

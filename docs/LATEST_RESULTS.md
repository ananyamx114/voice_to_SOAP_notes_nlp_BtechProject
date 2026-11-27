# Team report handoff — 27 November 2026

All results below are available as aggregate JSON/CSV in Git. Read these files without downloading models. Earlier dated snapshots remain historical records. Teammates with clean working trees can run `git pull --ff-only origin main`; preserve local edits before integrating.

| Component | Latest evidence | What can be reported |
|---|---|---|
| Latest repeated ASR benchmark | [27 November paired results](../results/asr-run3-test-2026-11-27/README.md) | All71 pairs complete; WER gains in four groups, regressions in two; FLEURS HI24.54%, MR50.41% under numeric-safe scoring |
| Whisper training continuation | [state](../results/asr/run3-2026-11-26.json), [14-epoch curve](../results/asr/run3-validation.csv) | Paused at 4100 updates; epoch 12 selected; legacy validation WER EN1.67%, HI23.72%, MR44.65% |
| Whisper decoder diagnostics | [report, CSV and both scoring protocols](../results/asr-diagnostics-2026-11-26/README.md) | Completed 239-segment validation; beam 5 numeric-safe WER EN1.34%, HI21.66%, MR42.60%; no new training |
| ASR scoring fix | [protocol and tests](ASR_IMPROVEMENT_WORK.md), [rescoring](../results/asr-scoring-v2/README.md) | Decimals/signs/fractions and Indic marks preserved; scoring changes alone are not recognition gains |
| Additional ASR data | [preparation audit](../results/asr-balanced-v4/README.md) | 7,465 accepted training segments, 22.42 hours; not yet trained; frozen validation retained |
| Qwen7B DIRECT | [complete results](../results/qwen7b-2026-11-26/README.md), [stage JSON](../results/baseline/qwen7b-2026-11-26/) | 4,800 updates/3 epochs; validation loss0.736586; 200-example test ROUGE-L0.429454 versus base0.229839; validity regresses186/200 versus196/200 |
| Qwen7B HYBRID | [completed metrics](../results/hybrid/qwen7b-2026-11-27/metrics.json), [three-model comparison](../results/qwen7b-2026-11-27/README.md) | 200/200 test examples;167 valid/33 invalid; failure-aware ROUGE-L0.378346, below DIRECT 0.429454 |
| Older completed ASR test | [run2 paired results](../results/asr-lora-run2/README.md) | 71 selected recordings; separate experiment, with improvements and regressions disclosed |

The ASR decoder improvement is measured on validation, with the same selected adapter; it is not a fine-tuning or independent-test result. Below5% Hindi/Marathi WER remains unmet. Native-speaker reference review is pending because no reviewer is available; references remain unchanged. Balanced sampling/warm-start source is implemented and tested, but its new full training experiment has not run.

Qwen DIRECT improves text overlap while producing more invalid JSON. Report both findings. Thirteen DIRECT failures are consistent with the 1,024-token output budget truncating a JSON string; this motivates validation-selected output-budget/format experiments. HYBRID now has a complete test result:0.378346 failure-aware ROUGE-L and83.5% validity. It improves overlap over the pretrained control but trails DIRECT. On160 common-valid pairs their overlap scores are close; excess HYBRID invalid outputs account for much of the full-set gap. Lower validation loss did not predict a test-generation advantage.

BeTraC notes are synthetic and heuristically mapped. Two SOAP internal-test encounters occur in ASR training; exclude or separate them in future independent audio-to-SOAP claims. No end-to-end evaluation or clinical factuality/clinician validation has completed. Every note requires clinician approval.

## Source and reproducibility

- [Portable setup, accounts and dependencies](TEAM_SETUP.md), [dataset/model resources](ASR_RESOURCES.md), [dataset preparation](BETRAC_2000_MIGRATION.md).
- [Qwen7B DGX experiment scripts and commands](../scripts/qwen7b/README.md), including resumable single-GPU training, two-GPU asynchronous validation, memory probe, length audit and resumable test evaluation.
- [Whisper diagnostic runner](../scripts/diagnose_asr.py), [cohort preparation](../scripts/prepare_asr_expansion.py), [balanced config](../configs/asr-lora-balanced.yaml), [ASR implementation](../src/voice_to_soap/whisper_lora.py).
- [Aggregate progress exporter](../scripts/export_soap_interim.py) reads saved artifacts without interrupting jobs.

Qwen execution scripts are shared and GPU selection is configurable. The HYBRID evaluation launcher completed on the V100 on 27 November; the portable training variants have not been GPU-rerun. The latest142 ASR records and200 HYBRID outputs were backed up and rechecked locally before publication. Raw predictions remain private.

Only code, documentation and aggregate evidence are published. Recovery backups, selected adapters, optimizer checkpoints, datasets, audio, private predictions, local hardware reports, tokens and Jupyter URLs remain private. College server workspace was removed after verified backup. The diagnostic Kaggle allocation was released after verified backup; the original ASR continuation remains paused.

The latest paired ASR benchmark uses sequential beam 1 full-recording decoding; the older beam 5 figures are validation diagnostics. Do not present them as the same evaluation. [CSV of both scoring protocols](../results/asr-run3-test-2026-11-27/paired-results.csv).

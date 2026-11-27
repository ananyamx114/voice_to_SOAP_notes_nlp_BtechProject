# Whisper selected epoch 12: completed paired evaluation — 27 November 2026

Full Whisper Large-v3 and its selected epoch 12 rank 16 LoRA adapter were evaluated on the same 71 previously selected recordings across six dataset/language groups. This is a repeated benchmark with inspected examples, not a new untouched clinical test. No training occurred during this session; the original continuation remains paused at update 4100.

## Numeric-safe scoring: primary current results

| Dataset | Language | Recordings | Base WER% | Adapter WER% | Change, percentage points | Base CER% | Adapter CER% |
|---|---|---:|---:|---:|---:|---:|---:|
| Eka | English | 2 | 13.33 | 16.67 | +3.33 | 10.83 | 14.01 |
| Eka | Hindi | 2 | 50.00 | 41.30 | -8.70 | 49.32 | 42.99 |
| BeTraC | English | 2 | 3.88 | 4.75 | +0.87 | 2.61 | 2.46 |
| Sully/PriMock57 | English | 1 | 21.97 | 19.75 | -2.22 | 17.09 | 14.69 |
| FLEURS | Hindi | 32 | 34.80 | 24.54 | -10.26 | 18.44 | 9.63 |
| FLEURS | Marathi | 32 | 81.44 | 50.41 | -31.03 | 25.73 | 14.15 |

The table uses `numeric_indic_v2`; negative change means improvement. Adapter WER improves in four groups and regresses in two. CER improves in five and regresses in one. BeTraC demonstrates why WER and CER can disagree: fewer character errors do not necessarily produce fewer whole-word errors.

Hindi and Marathi remain substantially above the below 5% objective. The low English result on two synthetic BeTraC recordings does not establish general clinical English accuracy: Eka and PriMock57 remain much worse and have tiny sample counts. FLEURS is general read speech, not clinical conversation. Do not pool these non-equivalent groups into one headline accuracy figure. Native review remains pending.

## Legacy scoring and older experiment context

Both protocols are retained in [paired CSV](paired-results.csv) and [aggregate JSON](comparisons.json). Under legacy `punctuation_delete_v1`, selected-adapter WER is: Eka EN16.67%, Eka HI41.30%, BeTraC EN4.51%, Sully EN20.75%, FLEURS HI25.03%, FLEURS MR50.17%.

For context, the older run2 selected adapter scored FLEURS HI26.02% and MR61.55% under legacy scoring. The later values are descriptively lower; however, this is not a controlled data-size/epoch ablation: base predictions also changed across runs, and execution/environment/implementation differ. The valid controlled contrast is the base-versus-adapter pair within this new run. [Historical CSV](historical-context-v1.csv) keeps both old and new baselines visible; [old run2](../asr-lora-run2/README.md) remains unchanged.

The [earlier greedy/beam 5 experiment](../asr-diagnostics-2026-11-26/README.md) measured validation segments with another decoding setup. Its Marathi42.60% is not this repeated-test result of 50.41%. This new full-recording paired evaluation uses the published sequential decoder with beam 1, word timestamps, fixed fallback settings and the same settings for base and adapter. No decoder settings were chosen using these test outcomes.

## Verification and reproduction

Every comparison was recomputed after transfer using the published comparison implementation. Checks cover identical audio hashes, references, languages, regions, per-encounter seed, model revision/precision/device/generation/timestamp implementation and explicit adapter identity. All142 ASR records report complete. The paired IDs match the older71-recording selection, but raw IDs and predictions are not republished in this aggregate handoff.

Model: `openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1`, FP16 without quantization, V10032GB, PyTorch2.11.0+cu126, Transformers4.57.6. Selected adapter SHA256 `0be58d5beb9c0f70f781cede004a7213abcf472c4313f1b9666293b35a397dbb`. [Provenance](provenance.json).

Use `scripts/project.py asr-run` for complete encounters, with `configs/asr-large-v3.yaml`, the privately restored per-dataset manifest, an explicit CUDA device and a new output directory. Repeat with the explicit selected adapter, then use `scripts/project.py asr-compare` separately for each dataset and scoring protocol. Run `--resume` only with unchanged inputs/settings. Sources: [runner](../../src/voice_to_soap/asr_component.py), [comparison](../../src/voice_to_soap/asr_comparison.py), [scorer](../../src/voice_to_soap/asr_scoring.py), [resource/dataset instructions](../../docs/ASR_RESOURCES.md).

Raw references and hypotheses remain unchanged. Numeric/negation inventory flags are lexical diagnostics, not dose, negation-scope or clinical correctness validation. No end-to-end audio-to-SOAP, speaker-role or clinician-reviewed claim is established. The22.42-hour expanded cohort remains prepared but untrained. Private recordings, manifests, predictions, weights and local connection/config files remain outside Git.

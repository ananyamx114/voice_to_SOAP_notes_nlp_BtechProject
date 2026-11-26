# Post-run results review

**Review date:** 26 November 2026
**Scope:** Results and claims in the attached report, compared with experiment
summaries currently in this repository. This is an analysis of reported evidence,
not a code change or a new model evaluation.

## Executive summary

The attached report accurately presents several earlier component experiments,
but it predates the repository's 26 November Qwen7B and expanded ASR interim
results. Keep those newer experiments separate from the report's Qwen2.5-0.5B
and completed Whisper run2 results; they have different models, checkpoints,
and completion states.

The strongest supported conclusions are:

- Whisper LoRA has a language/domain tradeoff, not a universal improvement.
- In the 0.5B synthetic BeTraC comparison, DIRECT produced valid structured
  outputs more often than HYBRID. This is not evidence of clinical accuracy.
- The ACI-Bench 0.5B LoRA pilot did not improve its reported metrics over base.
- VAD and fact-extraction results establish execution and output counts, not
  accuracy.
- The 7B DIRECT result is promising but partial; no completed paired gain is
  established. HYBRID has no generation result yet.
- No experiment establishes end-to-end audio-to-SOAP quality, clinical
  factuality, safety, or clinician-approved performance.

## Results in the attached report

### Whisper Large-v3 LoRA: completed run2 test

The report's paired CUDA evaluation covers 71 selected recordings across six
dataset-language groups. Negative WER change means lower error after adaptation.

| Dataset / language | N | Base WER | LoRA WER | Change (percentage points) | Base CER | LoRA CER |
|---|---:|---:|---:|---:|---:|---:|
| Eka English | 2 | 13.33% | 16.67% | +3.34 | 10.83% | 14.01% |
| Eka Hindi | 2 | 50.00% | 39.13% | -10.87 | 49.32% | 42.53% |
| BeTraC English | 2 | 2.98% | 4.56% | +1.58 | 2.62% | 2.63% |
| Sully/PriMock57 English | 1 | 22.21% | 22.63% | +0.42 | 17.04% | 17.07% |
| FLEURS Hindi | 32 | 32.80% | 26.02% | -6.78 | 15.42% | 10.28% |
| FLEURS Marathi | 32 | 79.87% | 61.55% | -18.32 | 27.27% | 21.04% |

WER improved in three groups and regressed in three. CER also regressed in
three. The two FLEURS groups provide the largest sample sizes, but FLEURS is
general speech, not clinical conversation. The English clinical groups have
only one or two recordings each. Treat them as diagnostic evidence, not stable
clinical accuracy estimates. The report correctly cautions that the validation
base and adapter ran on different backends; the matched CUDA test is the
stronger base-versus-adapter comparison. Full provenance is in
[run2 results](results/asr-lora-run2/README.md).

### BeTraC-2000 Qwen2.5-0.5B DIRECT versus HYBRID

On 200 matched synthetic encounters:

| Approach | Valid outputs | Invalid outputs | Valid-only mean section ROUGE-L | All-example ROUGE-L, invalid = 0 |
|---|---:|---:|---:|---:|
| DIRECT | 123/200 (61.5%) | 77 | 0.3718 | 0.2287 |
| HYBRID | 71/200 (35.5%) | 129 | 0.3532 | 0.1254 |

Valid-only scores use different subsets and should not be compared as if they
were paired. The report's common-valid analysis is more controlled: among the
55 encounters where both approaches were valid, DIRECT averaged 0.3753 and
HYBRID 0.3471. DIRECT scored higher on 37 pairs and HYBRID on 18, with no ties.
State the difference's direction explicitly: DIRECT minus HYBRID is +0.0282.

The validity cross-tab has 68 encounters valid only under DIRECT and 16 valid
only under HYBRID. An exact two-sided McNemar calculation from those counts is
approximately 8.5e-9. This supports a validity-rate difference on this fixed
synthetic test set, not clinical benefit or generalization. A sign test on the
55 common-valid win/loss counts is approximately 0.014; it is conditional on
both systems succeeding and should remain exploratory. A paired uncertainty
analysis of all encounters using the failure-aware score would be more useful
for an overall text-overlap comparison.

Invalid generations are a major outcome, not a minor formatting issue. The
report records 77 DIRECT failures and 129 HYBRID failures; most were not
directly parseable. It also notes longer average transcripts among failures,
but correctly does not infer that input length caused the failures. A failure
taxonomy and token-length analysis would be needed before making that claim.

### ACI-Bench 0.5B pilot

The separate ACI-Bench test covers 120 English encounters:

| Model | Valid | Invalid | Valid-only mean section ROUGE-L |
|---|---:|---:|---:|
| Base Qwen2.5-0.5B-Instruct | 112/120 (93.3%) | 8 | 0.10452 |
| Qwen2.5-0.5B-Instruct + LoRA | 110/120 (91.7%) | 10 | 0.09888 |

The pilot therefore does not show a LoRA improvement. On the 103 common-valid
pairs, the reported LoRA-minus-base mean difference is -0.00469. Only 26/67
training and 8/20 validation examples were used because longer examples were
skipped. These heuristic ACI-Bench targets and results must remain separate
from the BeTraC experiment. See the [ACI-Bench pilot record](results/aci-bench-pilot/README.md).

### Other reported components

- **VAD:** Two synthetic recordings yielded 97 and 83 regions. All 180 exported
  WAVs passed format and interval checks. Speech-labelled fractions were about
  91.7% and 88.4%; without manually reviewed boundaries, these are not recall
  or accuracy scores. See [VAD results](results/component-01-vad/README.md).
- **Candidate facts:** 1,347 candidates were saved over 200 encounters; 10
  encounters had none. Unknown section was assigned to 697/1,347 (about 51.7%);
  416 assertions were uncertain, 278 absent, and 14 candidates were medication.
  These counts identify audit priorities, not extraction precision or recall.
- **Evidence-cited generation:** Two Qwen2.5-3B smoke cases did not pass schema
  checks. They do not establish factuality or end-to-end performance.
- **Diarization and translation:** The report contains no DER or bilingual
  translation-quality result.

## Comparison with previous and newer repository results

### ASR experiment progression

The earlier [run1 experiment](results/asr-lora-run1/README.md) trained for
three epochs / 90 updates on 472 segments. Its paired test improved WER in
three of six groups, regressed in two, and tied in one; Marathi CER regressed
despite a WER improvement. The report's run2 was a larger experiment: 1,644
training segments, about 4.03 hours, five epochs / 515 updates. It retained
three WER gains but had three regressions, including all three English groups.
These are separate experiments, not evidence that more training uniformly
improves results; their provenance and evaluation conditions should stay
attached to each table.

The repository's [26 November interim snapshot](results/interim-2026-11-26/README.md)
adds a separate 4,501-segment / 12.03-hour ASR continuation. It selected epoch
10 on macro-language validation WER:

| Language | Run2 selected validation WER (epoch 5) | Interim continuation WER (epoch 10) | Descriptive change |
|---|---:|---:|---:|
| English | 1.32% | 1.49% | +0.17 pp |
| Hindi | 24.13% | 23.99% | -0.14 pp |
| Marathi | 59.59% | 44.83% | -14.76 pp |
| Macro-language WER | 28.35% | 23.44% | -4.91 pp |

This is a validation comparison, not a new held-out test. The latest run has
239 validation segments, with 175 English and 32 each Hindi and Marathi.
English dominates example-weighted overall WER, so macro-language WER is the
more balanced selection measure. Epoch 11 rose to 23.86% macro WER, so epoch 10
remains selected. The below-5% WER target is still unmet for Hindi and Marathi.
At the snapshot, live progress was update 3384 and the last fully verified
local archive was update 3330. Do not replace the completed run2 paired-test
table with these validation results.

### New Qwen2.5-7B interim experiment

The repository also reports a different BeTraC-2000 experiment using
Qwen2.5-7B-Instruct and rank-8 LoRA. DIRECT training completed 4,800 updates
with selected validation loss 0.736586. HYBRID was incomplete at 1,200/4,800
updates with validation loss 0.779724; these losses are at different training
positions and use different inputs, so they do not rank the approaches.

The pretrained 7B baseline was evaluated on 200 cases: 196 valid, 4 invalid,
all-example ROUGE-L 0.229839. DIRECT generation was partial at 186/200: 173
valid, 13 invalid, all-example ROUGE-L 0.426978. Its observed valid rate is
93.0%, versus 98.0% for the complete baseline, but the coverage differs. Do
not subtract these scores or claim a completed paired gain. HYBRID generation
had not started.

The 7B DIRECT score is a promising observation, but it is not directly
comparable to the PDF's 0.5B results: the model and experiment differ, and the
7B evaluation is incomplete. Finish and compare the same encounters under
matched conditions before making an improvement claim. Preserve this
distinction in the [current interim report](results/interim-2026-11-26/README.md).

## Report changes recommended from the evidence

1. Add the 26 November ASR continuation and Qwen7B results as a dated interim
   update. Preserve the earlier run2 paired test and 0.5B results as distinct
   experiments.
2. Expand the methods for the BeTraC 0.5B comparison: identify its base-model
   revision, split, training/selection setup, DIRECT/HYBRID inputs and
   generation settings. The report currently gives more detail for the
   ACI-Bench pilot than for the separate 200-case BeTraC result.
3. Keep valid-only, common-valid paired, and all-example failure-aware ROUGE-L
   results distinct, with their denominators. For the 7B run, wait for full
   paired coverage. Confidence intervals or paired resampling would help
   communicate uncertainty.
4. Mark metrics listed as objectives (DER, extraction F1, BERTScore, clinical
   concept accuracy, hallucination/omission rates and translation scores) as
   not measured where appropriate.
5. Align capability claims with implementation evidence. The report says
   UMLS/RxNorm resources were used and IndicTrans2 is used; current repository
   status says the terminology table is empty and translation is not a
   completed, evaluated subsystem. Either cite measured implementation
   evidence or describe them as planned/prototype work. See
   [pipeline execution status](docs/PIPELINE_EXECUTION.md).
6. Describe transcript citations as provenance for what was said, not proof
   that a statement is clinically true. Make clear there is no clinician-
   reviewed quality result, and every generated note remains pending clinician
   approval.
7. Carry forward the overlap caveat for audio-to-SOAP claims: two SOAP
   internal-test encounters also appear in ASR training. This does not
   contaminate text-only SOAP training, but it prevents treating those
   encounters as independent end-to-end test evidence.

## Overall assessment

The report supports a narrow finding that DIRECT had higher structured-output
validity than HYBRID in the 0.5B synthetic BeTraC experiment, with higher
ROUGE-L on the common-valid subset. It does not show that HYBRID facts improve
notes, that either approach is clinically correct, or that the full pipeline
works end to end. ASR adaptation produced mixed test results; the newer ASR
continuation is validation-only. The 7B DIRECT result may be promising but
needs complete paired evaluation. No code or experiment results were changed
for this review.

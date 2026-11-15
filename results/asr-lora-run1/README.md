# Measured local Whisper LoRA results

WER decreased in 3/6 dataset/language groups, increased in 2, and was unchanged in 1.

Evaluated 71 selected recordings with both full base and LoRA models.

| Dataset | Language | Clips | Base WER | LoRA WER | Change (pp) | Base CER | LoRA CER |
|---|---|---:|---:|---:|---:|---:|---:|
| eka | en | 2 | 13.33% | 13.33% | +0.00 | 10.83% | 10.83% |
| eka | hi | 2 | 50.00% | 34.78% | -15.22 | 49.32% | 39.82% |
| betrac | en | 2 | 2.98% | 3.80% | +0.82 | 2.62% | 3.60% |
| sully | en | 1 | 22.21% | 23.05% | +0.84 | 17.04% | 17.59% |
| fleurs | hi | 32 | 39.21% | 34.16% | -5.06 | 20.99% | 16.33% |
| fleurs | mr | 32 | 78.71% | 73.76% | -4.95 | 24.96% | 27.12% |

## Training and selection

Full openai/whisper-large-v3 at `06f233fe06e710322aca913c1bc4249a0d71fce1`, torch.float16, no quantization.
7,864,320 trainable LoRA parameters; 1,551,354,880 total parameters.
472 training segments / 64.90 minutes of speech; 199 separate validation segments.
Selected epoch 3 using macro_language_wer; 90 optimizer updates executed.

| Validation language | Base WER | Selected LoRA WER |
|---|---:|---:|
| en | 8.48% | 7.99% |
| hi | 34.03% | 31.60% |
| mr | 77.94% | 71.08% |

## Review and limits

- Validation selects the adapter; final comparisons are separate.
- Previously inspected Eka examples are diagnostic, not untouched final test data.
- Automatic alignment is not human-reviewed gold; general FLEURS is not clinical speech.
- A measured improvement on these recordings does not establish clinical readiness.
- Checkpoint selection used greedy short-segment validation; production evaluation uses sequential decoding with temperature fallback and word timestamps.
- Numeric/negation inventory differences are lexical review flags, not clinical error diagnoses.
- Matching inventories do not prove correct dosage, assertion, speaker, or clinical meaning.
- Timestamp consistency warnings are not gold word-boundary accuracy.
- MPS driver samples at update boundaries are not measured peak total-system memory.
- Every downstream generated note still requires clinician approval.

This directory contains aggregate scores and model/data provenance only. Detailed predictions, lexical review flags and execution/resource histories remain in the ignored local experiment output.
WER/CER use the same scorer for both models. This selected sample is not a full-corpus or clinical validation claim.

Marathi CER worsened from 24.96% to 27.12% despite lower WER. The shared adapter is not an across-the-board quality upgrade.

Dataset attribution and pinned sources: [ASR_LORA.md](../../docs/ASR_LORA.md) and [LOCAL_BENCHMARKS.md](../../docs/LOCAL_BENCHMARKS.md).

## Follow-up: English validation checkpoint

The English validation minimum selected epoch 1 from the same joint multilingual
training run. [Its separate comparison](english-validation-checkpoint.md) gives
English Eka 13.33% (unchanged), BeTraC 2.98% (unchanged from base), and Sully
22.01% (base 22.21%). This removes the shared epoch-3 English WER regressions;
the original table above is retained.

[Language-validation checkpoint selection](validation-selected.md) consistently
uses epoch 1 for English and epoch 3 for Hindi/Marathi, producing four WER gains
and two ties across the six groups. Checkpoint choices are checked against the
per-language validation minima, not chosen per test encounter or dataset.
The compiler verifies the English control IDs, audio/reference/seed/model
provenance, predictions and timestamps match the original controls. It also
checks adapter checksums. This is an exploratory analysis on 71 selected
recordings, not an untouched clinical validation or an automatic routing feature.
Marathi CER remains worse (24.96% to 27.12%); BeTraC CER also increases slightly
with epoch 1 despite unchanged WER.

## Reproduce and refine

Use the export, evaluation and report commands in
[ASR_LORA.md](../../docs/ASR_LORA.md). Completed-epoch checkpoints are now retained
by the trainer itself; the first measured run used preserved local archives.
The storage-only follow-up passed the interrupted-training regression, including
recovery from a crash immediately before epoch archiving. Reported full-model
weights and metrics are unchanged.

The next quality experiment should expand/review clinical supervision and
select checkpoints using the production decoding policy on encounter-disjoint
validation. Track WER, CER, omissions, repetitions and timestamp quality
separately. Keep test audio/transcripts out of training; this run does not
establish clinical Marathi quality or an improvement on every dataset.

# Exploratory language-validation checkpoint selection

Minimum language validation WER across completed epochs; ties choose earlier epoch

| Dataset | Language | Clips | Epoch | Base WER | Shared WER | Selected WER | Base CER | Selected CER |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eka | en | 2 | 1 | 13.33% | 13.33% | 13.33% | 10.83% | 10.83% |
| eka | hi | 2 | 3 | 50.00% | 34.78% | 34.78% | 49.32% | 39.82% |
| betrac | en | 2 | 1 | 2.98% | 3.80% | 2.98% | 2.62% | 2.72% |
| sully | en | 1 | 1 | 22.21% | 23.05% | 22.01% | 17.04% | 16.89% |
| fleurs | hi | 32 | 3 | 39.21% | 34.16% | 34.16% | 20.99% | 16.33% |
| fleurs | mr | 32 | 3 | 78.71% | 73.76% | 73.76% | 24.96% | 27.12% |

- Validation selects the adapter; final comparisons are separate.
- Previously inspected Eka examples are diagnostic, not untouched final test data.
- Automatic alignment is not human-reviewed gold; general FLEURS is not clinical speech.
- A measured improvement on these recordings does not establish clinical readiness.
- Exploratory diagnostic comparison on previously inspected clinical samples.
- Do not choose a winner per test encounter; retain every candidate result.
- Greedy short-segment validation differs from production decoding with temperature fallback and word timestamps.
- These are separate verified runs using known language hints; automatic adapter routing is not implemented.
- Retain the original shared-checkpoint results, including its English regressions.
- WER gains do not imply CER gains or clinical correctness.

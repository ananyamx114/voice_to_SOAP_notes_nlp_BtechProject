# Local dataset setup and real CPU smoke run

Executed 13 November 2026 on a MacBook Air M1 with 8 GB RAM. Public datasets
were downloaded with revision/file checksum records in `dataset-downloads.json`.
No consultation text or audio is redistributed in this results directory.
Original dataset cards and attribution are kept beside ignored local downloads;
see [the compatibility audit and commands](../../docs/LOCAL_BENCHMARKS.md).

Whisper Small ran on CPU using the existing ASR evaluator on the first two
English and first two Hindi Eka test clips. This is an executed smoke test,
not a representative full-dataset evaluation or clinical validation.

| Language | Clips | WER | CER |
| --- | ---: | ---: | ---: |
| English | 2 | 0.2000 | 0.1720 |
| Hindi | 2 | 3.3261 | 3.4389 |

WER is a ratio; insertions can make it exceed 1 (100%). The very poor Hindi
result includes substantial excess generated text and must not be described as
acceptable multilingual accuracy. Diagnose repetition, decoding and timestamp
behavior on development examples before freezing a larger benchmark setup.
Do not tune prompts/model settings against an untouched final test set.

Transformers emitted chunked-seq2seq and timestamp warnings. Latency and peak
memory were not instrumented. No generation/clinical grounding metrics were
measured. Exact aggregate metrics and model revision are in the adjacent JSONs;
raw predictions remain under ignored `data/outputs/eka-whisper-small-local-smoke`.

The BeTraC exporter was also exercised on two actual Opus samples. Full Eka
preparation exported 3,939 clips and all audio paths were validated. The test
suite passed 75 tests, including three new manifest/archive regression tests.

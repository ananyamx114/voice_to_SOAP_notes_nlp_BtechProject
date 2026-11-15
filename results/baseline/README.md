# Actual base-Qwen CPU smoke test

One synthetic transcript was passed to the pinned Qwen2.5-0.5B-Instruct base
model on this Windows laptop. This was real model inference, with no adapter,
fine-tuning or DIRECT-versus-HYBRID experiment.

- Pinned revision: 7ae557604adf67be50417f59c2c2f167def9a775.
- Downloaded model file: 988,097,824 bytes; matched upstream LFS SHA-256.
- CPU-only PyTorch 2.15.0+cpu, Transformers 4.57.6; bfloat16; four CPU threads.
- Input: 215 tokens. Output: 120 tokens, ending with EOS.
- Model/tokenizer load: 1.99 seconds after imports.
- Generation: 28.87 seconds. Measured load/generation total: 30.91 seconds.
- Peak process working set: 1,459,552,256 bytes (about 1.36 GiB).
  This is a process working-set measurement, not total system memory or a
  representative performance benchmark.

The model produced four SOAP fields surrounded by Markdown code fences. The
strict parser rejected that output; no automatic repair was applied and
prediction_soap remains null in run_report.json. raw_model_output.txt preserves
the response verbatim.

Qualitative source checks, not clinician review or measured factuality rates:

- Output adds the blood-pressure unit "mmHg", not explicitly supplied in input.
- The requested chest X-ray appears under Assessment rather than Plan.
- Input "Possible pneumonia" becomes "symptoms consistent with pneumonia";
  uncertainty wording changed and needs review.

The output is a synthetic research draft pending clinician approval. Successful
loading/generation establishes only that this small model can execute under this
specific short-input CPU configuration. It does not establish clinical quality,
full-consultation performance, training feasibility or HYBRID improvement.

Adapter weights and raw prediction outputs remain local and ignored; this report
is the published result summary.

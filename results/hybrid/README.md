# HYBRID SOAP results

**HYBRID test evaluation is now complete:** [27 November metrics](qwen7b-2026-11-27/metrics.json) and [full three-model comparison](../qwen7b-2026-11-27/README.md). Valid JSON167/200 (83.5%); failure-aware mean section ROUGE-L0.378346, below DIRECT 0.429454. Pending/partial statuses below describe older snapshots only.


Latest full 7B experiment: [completed HYBRID training](qwen7b-2026-11-26/training.json) and [shared report](../qwen7b-2026-11-26/README.md). HYBRID test evaluation remains not started. Older experiments below remain separate.


Latest: [26 November Qwen7B interim report](../interim-2026-11-26/README.md),
[training CSV](../interim-2026-11-26/soap-training.csv),
[validation curve](../interim-2026-11-26/soap-validation.csv),
[exact SOAP snapshot](../interim-2026-11-26/soap-snapshot.json).

The new Qwen2.5-7B-Instruct HYBRID experiment uses transcripts plus the existing
lexicon/pattern clinical fact extractor. It is a separate rank-8 LoRA adapter;
it is not a trained clinical NER model or the evidence-cited claim schema.

At the frozen 14:42 IST export cutoff on 26 November, training was **incomplete**:
saved update **1200/4800**, epoch **0.75/3**, best validation loss **0.779724**.
The same BeTraC-2000 1,600/200/200 split is used as DIRECT. HYBRID test generation
had **not started**, so there is no new Qwen7B HYBRID ROUGE-L result to quote.

Older archives/report text in this directory are historical experiments. Preserve
their settings and dataset identity rather than presenting their scores as the
current Qwen7B model's results. Do not rank approaches using different training
positions or claim clinical accuracy from token validation loss.

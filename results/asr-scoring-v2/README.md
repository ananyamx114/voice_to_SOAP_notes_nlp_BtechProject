# Versioned ASR scoring correction

The legacy scorer deleted punctuation and could collapse `2.5` into `25` or
`-5` into `5`. **numeric_indic_v2** fixes that failure, retains Indic marks and
numeric punctuation, and separates ordinary word punctuation. It also emits
count-only number/negation inventory warnings, which are not clinical validation.

The [exact aggregate JSON](epoch12-recomputed.json) rescores the same saved
epoch12 predictions from the paused run3 experiment, without loading a model:

| Language | Legacy WER | numeric_indic_v2 WER |
|---|---:|---:|
| English | 1.67% | 1.37% |
| Hindi | 23.72% | 22.90% |
| Marathi | 44.65% | 44.06% |

**These differences are scoring changes, not improvements in recognition.**
The predictions, references, old checkpoint selection and dated reports remain
unchanged. Epoch12 was selected under the legacy metric; this table does not
reselect it under the new protocol. No new independent test result is claimed.
The scorer is deliberately conservative about ambiguous numeric formats and
does not equate written number words with digits or infer clinical meaning.

Reproduce privately with `scripts/rescore_asr.py`; see
[the improvement runbook](../../docs/ASR_IMPROVEMENT_WORK.md) for protocols,
balanced sampling, warm-start semantics, diagnostics and data preparation.
New inference/comparison metrics identify their protocol explicitly. Historical
trainer configs without an explicit protocol retain legacy scoring.

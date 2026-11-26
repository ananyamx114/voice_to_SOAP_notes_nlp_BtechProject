# Expanded ASR cohort / balanced experiment preparation

**Data preparation complete; new-model training has not started.** No lower WER
is claimed from these counts. The paused run3 checkpoint4100 is unchanged.

| Language | Prior selected segments | Expanded accepted segments | Expanded hours | Distinct reference strings |
|---|---:|---:|---:|---:|
| English |2096|2096|4.0124|2096|
| Hindi |1287|2115|6.6107|1309|
| Marathi |1118|3254|11.8016|1489|

Total: **7,465 segments, 22.4247 hours**. Existing clinical English was retained;
all eligible whole-clip Hindi/Marathi examples were decoded from the already
downloaded, pinned FLEURS training shards. Twenty clips were rejected by the
audio contract. Different recordings of a sentence are retained; distinct text
is reported separately rather than treating all recordings as new wording.

All239 frozen validation segments retain exactly the same IDs, groups, raw
references, boundaries and audio hashes. Accepted train/validation sets have
no shared segment IDs, encounter IDs, sentence-group IDs or source audio hashes.
Source audio is mono16kHz, finite and at most30 seconds per supervised segment.
Targets fit the decoder without truncation. FLEURS test was not read for
preparation. Native-speaker reference review remains pending.

[Aggregate preparation audit and manifest hashes](preparation.json).
Source/prepared audio and transcripts remain ignored; original source cards
are retained beside the private prepared files. See [FLEURS attribution](https://huggingface.co/datasets/google/fleurs)
and [BeTraC attribution](../../datasets/betrac_2000/SOURCE_DATASET_CARD.md).

Next, diagnose greedy/beam and training-subset versus validation performance
using the frozen epoch12 adapter. Compare balanced sampling on the original
cohort before combining sampler and data changes. Use a separate run directory
and `configs/asr-lora-balanced.yaml`; warm-start weights and reset the optimizer
under recorded new-phase lineage, preserving the starting candidate.
See [the runbook](../../docs/ASR_IMPROVEMENT_WORK.md).

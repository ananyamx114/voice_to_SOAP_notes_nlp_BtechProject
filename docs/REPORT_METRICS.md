# Report metric handoff — 26 November 2026

Use [the dated interim report](../results/interim-2026-11-26/README.md) for tables,
ready-to-adapt prose, complete validation curves and exact JSON evidence. This is
an interim draft update while jobs continue, not a claim of a finished pipeline.

| Report topic | Source | Scope |
|---|---|---|
| Latest ASR continuation | [ASR validation CSV](../results/interim-2026-11-26/asr-validation.csv), [JSON](../results/interim-2026-11-26/asr-snapshot.json) | 12.03-hour cohort; validation only; epoch 10 selected |
| Completed older ASR test | [run2 paired test](../results/asr-lora-run2/paired-test-results.csv) | 71 selected recordings; different adapter/experiment |
| Qwen7B training | [training CSV](../results/interim-2026-11-26/soap-training.csv), [validation CSV](../results/interim-2026-11-26/soap-validation.csv) | DIRECT complete; HYBRID partial |
| Qwen7B note generation | [evaluation CSV](../results/interim-2026-11-26/soap-evaluation.csv), [JSON](../results/interim-2026-11-26/soap-snapshot.json) | Baseline 200/200; DIRECT 186/200; HYBRID not started |
| Stage entry points | [ASR](../results/asr/README.md), [DIRECT/baseline](../results/baseline/README.md), [HYBRID](../results/hybrid/README.md) | Links and experiment identity |

SOAP snapshot: **26 November, 14:42 IST**. The ASR observation is also 14:42 IST.
Report these cutoff times rather than implying files describe live jobs.

Use WER/CER percentages only for ASR. Token validation loss and ROUGE-L are
different quantities; neither is clinical accuracy. The ASR below-5% target is
not achieved for Hindi/Marathi. Keep epoch-11 regressions visible even though
epoch 10 remains selected. The first SOAP validation measurement already follows
100 updates; it is not an untrained loss baseline.

Invalid generated SOAP counts as zero in the new all-example ROUGE-L metric.
Do not mix it with older valid-only averages, hide invalid outputs or compare
DIRECT's partial subset against the full baseline as a completed paired gain.
The full baseline has 4/200 invalid outputs; partial DIRECT has 13/186. HYBRID
has no generation result yet. No factuality, clinician-reviewed quality, clinical
safety or end-to-end performance improvement has been established.

BeTraC-2000 is synthetic and selected from official training data; its internal
test split is not an official independent BeTraC test. Two test encounters occur
in ASR training, which must be handled separately for audio-to-SOAP claims.
Preserve [BeTraC attribution](../datasets/betrac_2000/SOURCE_DATASET_CARD.md),
[preparation methodology](BETRAC_2000_MIGRATION.md), and older ACI-Bench attribution.
Every generated note still requires clinician approval.

## Refresh without interrupting jobs

On the model host, with its own environment active:

```bash
python scripts/export_soap_interim.py --output .local/interim-soap-results.json
```

Transfer that aggregate JSON privately to the report maintainer. The exporter
reads existing saved checkpoints and per-encounter records only; it does not
load models, send signals, change training settings or export patient text,
prediction text, encounter identifiers, model weights or credentials. Incomplete
checkpoints/metrics are skipped or labelled partial, not represented as complete.

ASR evidence for this snapshot was extracted from checksum-verified private
Session005 and update-3330 continuation archives, with each selected epoch-metrics
file verified against its inventory. Only aggregates, configuration and archive
hashes are published. Live progress (3384) was recorded separately from the last
fully verified local archive (3330). Controller records and connection URLs must
never be copied into the shared repository.

To regenerate the tables from the already published aggregate inputs:

```bash
python scripts/build_interim_report.py \
  --asr results/interim-2026-11-26/asr-snapshot.json \
  --soap results/interim-2026-11-26/soap-snapshot.json \
  --outdir results/interim-2026-11-26
```

Preserve dated snapshots when later results arrive. Teammates should pull `main`
normally, preserving local changes; no model/dataset download is needed to read
or rebuild these report tables. Weights, raw data, private predictions and
execution artifacts remain outside Git.

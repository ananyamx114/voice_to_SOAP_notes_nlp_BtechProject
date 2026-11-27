# DIRECT / baseline SOAP results

Latest full 7B experiment: [completed DIRECT/base 200-example tests](../qwen7b-2026-11-26/README.md), [stage metrics](qwen7b-2026-11-26/). Older experiments below remain separate.


Latest: [26 November Qwen7B interim report](../interim-2026-11-26/README.md),
[training CSV](../interim-2026-11-26/soap-training.csv),
[evaluation CSV](../interim-2026-11-26/soap-evaluation.csv),
[exact SOAP snapshot](../interim-2026-11-26/soap-snapshot.json).

In the new experiment, **base** means pretrained Qwen2.5-7B-Instruct without an
adapter; **DIRECT** means its transcript-only rank-8 LoRA adapter. This team
directory historically also holds DIRECT artifacts; do not confuse them with
the unadapted control or mix older archive scores into the new experiment.

DIRECT training completed three epochs / 4,800 updates on BeTraC-2000 (1,600
train / 200 validation / 200 internal test). Best validation loss: **0.736586**.
The unadapted control completed 200 test encounters: **196 valid / 4 invalid**,
mean section ROUGE-L **0.229839**, counting invalid outputs as zero.

At the export cutoff, DIRECT test generation was **186/200**, with **173 valid /
13 invalid**, and subset ROUGE-L **0.426978**. This is partial, not a completed
paired improvement. The output-validity rates are 98% for the full baseline and
93.01% for the partial DIRECT subset; different coverage precludes a final
paired validity comparison. Report failures alongside overlap scores.

The timestamped snapshot is authoritative; these values do not update live.
Historical archives are preserved. References are synthetic/heuristically mapped;
ROUGE-L is not clinical accuracy and gold-transcript SOAP evaluation is not
audio-to-SOAP evaluation. All notes require clinician approval.

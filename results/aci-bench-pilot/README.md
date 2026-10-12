# ACI-Bench SOAP pilot results

These are results and prepared data from the exploratory Mac pilot using
`Qwen/Qwen2.5-0.5B-Instruct`, evaluated on the ACI-Bench test split.

## Dataset source and attribution

The included `data/processed/aci-bench-soap/{train,validation,test}.jsonl`
files and the dialogues/reference notes embedded in prediction outputs are
derived from ACI-BENCH, released under the Creative Commons Attribution 4.0
International (CC BY 4.0) license. The SOAP fields in the prepared files are
our heuristic transformation of the source notes, not original ACI-BENCH SOAP
annotations.

Please attribute the source dataset and cite:

> Yim, W.-w., Fu, Y., Ben Abacha, A., Snider, N., Lin, T., & Yetisgen, M.
> (2023). ACI-BENCH: A Novel Ambient Clinical Intelligence Dataset for
> Benchmarking Automatic Visit Note Generation. *Scientific Data*, 10, 586.
> https://doi.org/10.1038/s41597-023-02487-3

- Dataset and source repository: https://github.com/wyim/aci-bench
- Dataset download: https://figshare.com/articles/dataset/aci-bench-corpus_zip/22494601
- License: https://creativecommons.org/licenses/by/4.0/

The prediction JSONL files repeat full dialogue and reference-note text for
each test encounter; they are included to make this experiment inspectable.
They are research artifacts, not clinical advice or approved patient-care
notes. Do not infer that the predictions are safe or factually supported.

| Model | Valid JSON | Invalid JSON | Mean section ROUGE-L F1 |
|---|---:|---:|---:|
| Base Qwen2.5-0.5B-Instruct | 112/120 | 8/120 | 0.10452 |
| Qwen2.5-0.5B-Instruct + LoRA | 110/120 | 10/120 | 0.09888 |

ROUGE-L measures wording overlap with the prepared reference sections. It does
not measure clinical correctness, completeness, factual grounding, or safety.
The reference SOAP sections were mapped heuristically from ACI-Bench notes, not
reviewed as SOAP labels by clinicians. The LoRA pilot used only 26 of 67
training encounters and 8 of 20 validation encounters because longer sequences
were skipped on the Mac rather than truncated. This experiment does not
demonstrate an improvement from LoRA and is not suitable for clinical use.

Aggregate metrics, data-preparation counts, and per-encounter predictions are
in this directory. The prepared source examples are in
`../../data/processed/aci-bench-soap/`. Reproduce predictions using the
commands in the project README and the ACI-BENCH source dataset.

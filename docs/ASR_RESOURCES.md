# ASR resources, provenance and reproduction

The shared results entry point is [results/asr](../results/asr/README.md).
This document inventories resources actually used for ASR preparation, training,
evaluation and reporting. SOAP experiments remain separately documented in
`KAGGLE_TEXT_TO_SOAP.md`, `KAGGLE_LEGACY_MIXED.md` and the team's result folders.
No dataset exhaustion, universal accuracy gain or complete end-to-end clinical
validation is claimed.

## Models

| Resource | Recorded version/pin | Actual role |
|---|---|---|
| [Whisper Large-v3](https://huggingface.co/openai/whisper-large-v3) | `06f233fe06e710322aca913c1bc4249a0d71fce1` | Complete ASR base; FP16 on MPS/CUDA, FP32 on CPU, no quantization |
| Whisper Large-v3 LoRA | rank 16, alpha 32, dropout 0.05; 7,864,320 trainable parameters | Query/value projections in encoder and decoder attention, including cross-attention |
| [Wav2Vec2 Base 960h](https://huggingface.co/facebook/wav2vec2-base-960h) | `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` | English CTC forced alignment of existing reference transcripts; not pseudo-label generation |
| [Silero VAD](https://github.com/snakers4/silero-vad) | `silero-vad==6.2.3`; JIT SHA-256 `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720` | Earlier CPU speech-activity stage and BeTraC evaluation regions; gold boundary accuracy was not measured |
| Whisper Small | Historical CPU diagnostic only; see recorded baseline artifacts | Earlier reference, not the final model and not a replacement for Large-v3 |

The selected expanded-run epoch-5 adapter SHA-256 is
`2557815bc1214fd48fbc4364efe6b1fb074c4598a66309d57a50f661da97012f`.
It is task-specific and must be used with the pinned base. Obtain validated
weights through an approved private channel or reproduce training; no weights
are published in this ASR package. LoRA is described by
[Hu et al.](https://arxiv.org/abs/2106.09685). Model upstream terms still apply.

pyannote Community-1 and Qwen access were checked during broader project setup,
but they were not part of these ASR accuracy measurements. This run does not
assign doctor/patient roles, validate diarization, generate SOAP notes or measure
the team's Qwen adapters. Do not require a gated diarization login merely to
run the public Whisper/CTC models.

## Audio datasets and actual coverage

| Source | Immutable revision | Use in the completed expanded experiment |
|---|---|---|
| [BeTraC/betrac-2026](https://huggingface.co/datasets/BeTraC/betrac-2026) | `96f7b94e662dac0d1394cfdf55e8b8ae81422663` | Synthetic English; 1,045 accepted training segments from 36 consultations, 175 validation segments from four other consultations, two full diagnostic test consultations |
| [google/fleurs](https://huggingface.co/datasets/google/fleurs) | converted parquet `168de341b3db6859a9bac1c50a2ef5e3b47647e0`; card `70bb2e84b976b7e960aa89f1c648e09c59f894dd` | General read speech; Hindi 321 and Marathi 278 full training clips; 32 validation and 32 test clips per language |
| [Eka medical ASR evaluation](https://huggingface.co/datasets/ekacare/eka-medical-asr-evaluation-dataset) | `433e66037c8aebb8f396b938624a5ab6afc4ca36` | Test-only; two English and two Hindi diagnostic clips; never training data |
| [Sully open clinical conversations](https://huggingface.co/datasets/SullyHQ/open-clinical-conversations) | `06053cd9be2ea5fbfaefebb1db5be907b53a8c44` | One 457.92-second PriMock57 role-play consultation, `pm57-d1c01` |

BeTraC and FLEURS retain CC BY 4.0 attribution. Preserve the downloaded Sully
card, `ATTRIBUTION.md` and `CITATIONS.bib`, and review upstream terms for Eka and
every source before reuse/sharing. No source audio or text is redistributed by
the ASR results publication. Sully also includes ACI-Bench material: overlapping
encounters are not independent end-to-end tests.

Training audio totals 7,222.37 s English, 3,641.28 s Hindi and 3,631.14 s Marathi
(about 4.03 hours). The cached BeTraC training shard had 946 consultations/about
125.88 raw hours; cached FLEURS training had about 6.66 Hindi and 11.95 Marathi
raw hours. These are recorded availability checks, not training totals. They are
not exhausted. New experiments must preserve independent validation/test groups.

## Software, execution and artifacts

The actual CUDA run recorded PyTorch `2.11.0+cu130`, CUDA `13.0`, Transformers
`4.57.6`, PEFT `0.17.1`, NumPy `2.1.3`, and one visible Tesla T4. Local MPS
training used PyTorch `2.15.0` and NumPy `2.5.4`. These are experiment provenance,
not hardware/settings imposed on teammates. T4 x2 allocation was available,
but the trainer used one device; no distributed-training speedup is claimed.

- Python 3.12 is the team local setup target; the Kaggle image used Python 3.13.
- `requirements-local-asr.txt` and `requirements-asr-training.txt`: Torch,
  Transformers, Accelerate, PEFT, NumPy, SoundFile/SciPy, JiWER, PyArrow,
  PyYAML, num2words and psutil through the declared dependency profiles.
- `requirements-vad.txt`: optional preceding Silero VAD environment.
- FFmpeg is available for audio tooling; SoundFile/SciPy perform the shared
  decoding, mono conversion and resampling used by the component.
- JiWER computes WER/CER after case-folding, removing Unicode punctuation and
  collapsing whitespace. The completed cloud evaluator pinned JiWER `4.0.0`.
- Hugging Face Hub provides model/dataset downloads and optional personal
  authentication. Pin/size/checksum and free-space guards protect preparation.
- Kaggle CLI `2.2.5` with `six` handled personal OAuth, private uploads, GPU jobs,
  live logs and verified output download. Credentials stay in the user's home
  configuration, not source, notebook code or Git.
- Matplotlib `3.11.2` generated report PNG/SVG/PDF figures in an isolated report
  environment. Word export uses standard-library XML/ZIP, with no additional
  document library or external service. Optional plots use `requirements-reports.txt`.

The full run trained five epochs / 515 updates with AdamW, learning rate 5e-5,
weight decay 0.01, accumulation 16, warmup 0.1, clipping 1.0 and seed 42.
The configuration is `configs/asr-lora-expanded.yaml`. Every completed epoch
was retained; epoch 5 minimized macro-language validation WER. Selection did
not use test results. All 71 selected recordings completed matched base/adapter
CUDA evaluation; the detailed report includes the three English regressions.

MPS-to-CUDA migration preserved adapter, optimizer, scaler, CPU RNG and training
position, with an explicitly recorded fresh CUDA RNG seed. It does not provide
bitwise-identical cross-backend continuation. Safe checkpoint saves are atomic;
pause/resume waits for saved boundaries. A wrong-interpreter metric failure and
a stale live-log stream were corrected and documented; trained progress was
recovered. Final archives and per-file inventories were checksum-verified.

## Reproduce on a teammate's machine

Start with [TEAM_SETUP.md](TEAM_SETUP.md). Use `python3` on macOS/Linux or `py`
on Windows until activation. After activation, all commands below use `python`:

```text
python scripts/project.py system
python scripts/project.py setup --profile asr-training
python scripts/project.py test
python scripts/project.py download-data --dataset all
python scripts/project.py download-data --dataset betrac-train
python scripts/project.py download-data --dataset fleurs-training
python scripts/project.py download-data --dataset fleurs-test
python scripts/project.py prepare-asr-data
```

The downloader retrieves pinned source files into ignored `data/raw/`; it does
not retrieve previously trained private adapters. Public ASR sources do not
require a paid API or shared account. If HF authentication is needed, run
`python scripts/project.py hf-login` and enter your own token in the hidden
prompt. Only the account owner can accept gated terms for other resources.

Next follow [ASR_LORA_EXPANDED.md](ASR_LORA_EXPANDED.md) for separate validation,
test preparation and VAD prerequisites, then the full sequential training and
evaluation command. For inference with the validated adapter, follow
[ASR_TEAM_HANDOFF.md](ASR_TEAM_HANDOFF.md). For private CUDA migration, follow
[KAGGLE_ASR.md](KAGGLE_ASR.md). Do not start model stages concurrently or assume
another person's available RAM/GPU. Do not weaken the model or truncate labels
to pass a failed resource check.

Git contains code, portable config, setup guides, aggregate JSON/CSV and report
figures/documents. Keep audio, prepared source datasets, raw reference/prediction
text, weights, checkpoints, CLI credentials and machine-specific job/runtime
settings under ignored directories. The benchmark manifests and per-example
artifacts remain private; their coverage and checksums are documented in the
aggregate report.

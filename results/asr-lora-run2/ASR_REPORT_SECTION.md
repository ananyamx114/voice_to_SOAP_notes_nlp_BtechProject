# ASR: Whisper Large-v3 with LoRA

Evidence status: **complete**. Training finished all five epochs / 515 optimizer updates.

## Method

The audio-to-text component uses the complete pretrained Whisper Large-v3 encoder-decoder. The frozen base remains FP16 without quantization. LoRA adapts the query and value projections in encoder self-attention, decoder self-attention and decoder cross-attention. This is local model adaptation, rather than a hosted language-model transcription service.

The run trains 7,864,320 parameters out of 1,551,354,880 total parameters. LoRA rank is 16, alpha 32 and dropout 0.05. AdamW uses learning rate 5e-5, weight decay 0.01, gradient accumulation 16, warmup ratio 0.1 and gradient-norm clipping at 1.0. The seed is 42; microbatch size is one. Adapter weights remain FP32, with gradient scaling and non-reentrant gradient checkpointing.

Base checkpoint: `openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1`. Selected adapter SHA-256: `2557815bc1214fd48fbc4364efe6b1fb074c4598a66309d57a50f661da97012f`.

## Data and protocol

Training uses 1,644 aligned segments: 1,045 English BeTraC segments from 36 consultations (7,222.37 seconds), 321 Hindi FLEURS clips (3,641.28 seconds), and 278 Marathi FLEURS clips (3,631.14 seconds). This totals 4.03 hours. Validation contains 239 segments: 175 English, 32 Hindi and 32 Marathi. BeTraC English speech is synthetic; FLEURS is general-domain speech. These data do not establish clinical performance in Hindi or Marathi.

Inputs are 16 kHz mono. English segment boundaries use CTC forced alignment with rejection thresholds; alignment is not human-reviewed ground truth. FLEURS supervision uses the dataset's complete audio/reference pairs. Targets are checked without truncation. Training and validation IDs, encounter groups, source-audio hashes and full-encounter text are checked for overlap. Eka test recordings never enter training.

Checkpoint selection uses the equally weighted mean of English, Hindi and Marathi validation WER; epoch 5 was selected. Validation uses greedy short-segment decoding. Production evaluation uses native sequential decoding with temperature fallback and word timestamps. Within each paired test, base and adapter must match audio hashes, references, language hints, seeds, decoded regions, model revision, precision, device and decoding settings.

WER = (substitutions + deletions + insertions) / reference words; CER applies the same edit calculation to characters. Scoring case-folds text, removes Unicode punctuation and collapses whitespace. Aggregate WER is calculated over reference/hypothesis sequences, not by averaging per-recording percentages. Lower values are better.

## Validation results

The baseline validation used MPS; adapted validation used CUDA after checkpoint migration. This table is diagnostic validation evidence, not a same-backend paired test comparison.

| Language | Segments | Historical base WER | Selected WER | Historical base CER | Selected CER |
|---|---:|---:|---:|---:|---:|
| English | 175 | 8.48% | 1.32% | 6.32% | 0.54% |
| Hindi | 32 | 30.10% | 24.13% | 13.44% | 9.99% |
| Marathi | 32 | 81.55% | 59.59% | 23.55% | 15.87% |

## Paired CUDA evaluation

Measured coverage currently available: 71/71 selected recordings. The planned coverage is Eka 4, BeTraC 2, Sully/PriMock57 1 and FLEURS 64 (32 Hindi, 32 Marathi). This is selected-recording evaluation, not full-corpus benchmarking.

| Dataset | Language | Recordings | Base WER | LoRA WER | Change (pp) | Base CER | LoRA CER |
|---|---|---:|---:|---:|---:|---:|---:|
| eka | English | 2 | 13.33% | 16.67% | +3.33 | 10.83% | 14.01% |
| eka | Hindi | 2 | 50.00% | 39.13% | -10.87 | 49.32% | 42.53% |
| betrac | English | 2 | 2.98% | 4.56% | +1.58 | 2.62% | 2.63% |
| sully | English | 1 | 22.21% | 22.63% | +0.42 | 17.04% | 17.07% |
| fleurs | Hindi | 32 | 32.80% | 26.02% | -6.78 | 15.42% | 10.28% |
| fleurs | Marathi | 32 | 79.87% | 61.55% | -18.32 | 27.27% | 21.04% |

Across the available dataset/language groups, WER improved in 3, regressed in 3 and tied in 0. CER regressed in 3 groups. Negative percentage-point change means lower WER.

The selected shared adapter improved Hindi on Eka and FLEURS and Marathi on FLEURS, while all three English diagnostic groups regressed. The experiment therefore supports a multilingual adaptation tradeoff, rather than universal ASR improvement. The selected adapter was not changed after observing the test results.

## Discussion and limitations

The validation gains show adaptation to the selected development distribution; they do not guarantee generalization to other microphones, accents or clinical conversations. The known Eka recordings are diagnostic because they have been inspected in earlier experiments. A two-recording language subset is too small for broad accuracy claims. Marathi validation WER remains high. High validation improvement alongside a test regression requires error analysis; it does not by itself prove overfitting or underfitting.

The local MPS run resumed on one NVIDIA T4 using preserved adapter, AdamW, scaler, CPU RNG and training position. CUDA RNG was explicitly reseeded because MPS RNG cannot be translated. PyTorch/NumPy platform differences were recorded; bitwise-equivalent continuation is not claimed. A post-training coordinator interpreter error was corrected without retraining or deleting checkpoints.

WER/CER measure transcription overlap, not clinical correctness. Medication names, quantities, negation, omissions and timestamp boundaries need separate review. Numeric/negation inventory checks, when used, are lexical flags rather than clinical validation. Every downstream SOAP note still requires clinician approval. This ASR experiment does not measure Qwen SOAP quality or end-to-end clinical performance. Sully contains overlapping ACI-Bench material elsewhere in the collection; do not call overlapping encounters independent end-to-end test evidence.

The remaining existing training data can support a later experiment. Keep this run frozen, exclude validation/test overlap, use reference transcripts and vary data coverage in a separately recorded experiment. Repeated test-guided tuning requires an additional untouched final test set.

## Reproducibility and attribution

Configuration: `configs/asr-lora-expanded.yaml`. Source checkpoints and predictions remain ignored local/private artifacts. The summary records input archive hashes, model pin, adapter hash and software versions. Tables contain measured aggregate values only; no transcripts, audio, weights, credentials or machine-specific runtime settings are included.

Dataset pins: BeTraC `96f7b94e662dac0d1394cfdf55e8b8ae81422663`; FLEURS converted parquet `168de341b3db6859a9bac1c50a2ef5e3b47647e0`, with dataset card `70bb2e84b976b7e960aa89f1c648e09c59f894dd`.

- Whisper Large-v3 model card: https://huggingface.co/openai/whisper-large-v3
- Hu et al., LoRA: https://arxiv.org/abs/2106.09685
- BeTraC dataset and CC BY 4.0 attribution: https://huggingface.co/datasets/BeTraC/betrac-2026
- FLEURS dataset and CC BY 4.0 attribution: https://huggingface.co/datasets/google/fleurs
- Eka and Sully retain their upstream terms; the selected source recordings are not redistributed.

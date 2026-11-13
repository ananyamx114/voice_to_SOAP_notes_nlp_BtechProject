# Voice-to-SOAP Project Guide

This guide explains the project's goal, what the code currently does, how its
parts fit together, how to run the existing experiments, and what remains
before the system can be considered a dependable research prototype.

## 1. Project aim

Voice-to-SOAP is a research prototype for helping a clinician turn a
doctor-patient consultation into a structured clinical note:

- **Subjective (S):** symptoms and history reported by the patient.
- **Objective (O):** examination findings, measurements, and test results.
- **Assessment (A):** the clinician's stated impression or diagnosis.
- **Plan (P):** tests, treatment, advice, and follow-up stated in the visit.

The longer-term input is audio, including Hindi, Marathi, English, and
code-mixed conversations. The intended output is an English SOAP note that
keeps source evidence so a reviewer can check where each statement came from.
The system is intended as a documentation assistant, not a diagnostic,
prescribing, or autonomous clinical decision system. Clinician review is
required.

## 2. Current status at a glance

There are **two different conversation-to-SOAP implementations** in this
repository. Do not treat their outputs or training data as interchangeable.

| Track | What it does | Current state |
|---|---|---|
| Evidence-cited pipeline | Runs a provided transcript or audio through normalization, fact extraction, SOAP drafting, citation validation, and verification. | Implemented as a research scaffold. The default generation backend is a local OpenAI-compatible LLM. No fine-tuned adapter for this track is included. |
| ACI-Bench pilot | Converts full English dialogue/reference-note pairs to approximate four-section SOAP targets and trains/evaluates Qwen2.5-0.5B with LoRA. | Implemented and run on a Mac. The prepared ACI-Bench splits and aggregate/per-example results are committed under `data/processed/aci-bench-soap/` and `results/aci-bench-pilot/`. |

The pilot used weak, heuristic SOAP targets, not clinician-authored SOAP
labels. It is an experiment in data preparation and model training, not a
validated SOAP generator. The pilot's LoRA result did not improve the overall
ROUGE-L score over the base model. The adapter weights are not committed.

Some other components remain partial or require external access:

- Audio ingestion, VAD, Whisper ASR, diarization hooks, and speaker-role
  assignment are implemented, but real-world accuracy has not been validated
  for the project's target languages and recordings.
- Diarization is disabled by default and requires accepting the pyannote model
  terms and setting `HF_TOKEN`.
- The local terminology table is only a header until it is populated with a
  licensed, curated terminology release.
- Hindi/Marathi conversation-to-English-SOAP training and evaluation have not
  been completed.
- BioClinicalBERT is not part of the current production flow. Clinical NER is
  a possible experiment, not a trained or validated component of this
  pipeline.
- Passing unit tests verifies software behavior on test cases; it does not
  demonstrate clinical correctness or safety.

## 3. System architecture

The intended end-to-end path is:

```text
audio recording or supplied transcript
  -> input manifest validation
  -> mono, 16 kHz audio preparation (audio input only)
  -> voice activity detection (audio input only)
  -> Whisper ASR (audio input only)
  -> optional diarization and doctor/patient role assignment
  -> preserve raw transcript; apply reviewed normalization lookups
  -> extract structured facts with source utterance IDs
  -> link entities to a local terminology table (when populated)
  -> draft SOAP claims, each citing source utterance IDs
  -> validate citation IDs and check claims against their cited transcript
  -> detect selected omissions and dose-range warnings
  -> save draft, verification details, and review-required status
  -> clinician review
```

### Input modes

An encounter can provide:

1. A **transcript**, optionally with one utterance per line and speaker labels
   such as `[doctor]` and `[patient]`; or
2. An **audio path**, in which case the audio stages run first.

The CSV manifest has one row per encounter, not one row per utterance. Required
columns are `encounter_id`, `audio_path`, `transcript`, `reference_note`,
`language`, and `split`. Provide either an audio path or a transcript. For
transcripts, speaker labels are only retained when explicitly present; the
pipeline does not infer a doctor/patient mapping from unlabeled text.

### Audio path

- Audio is loaded/resampled to the configured target sample rate (16 kHz by
  default).
- Silero VAD detects likely speech segments. VAD identifies speech regions; it
  does not recognize words.
- Whisper (`openai/whisper-small` by default) transcribes the speech. The
  current configuration supports automatic language detection or an optional
  language setting.
- Optional pyannote diarization creates anonymous speaker turns. ASR timestamps
  are aligned to those turns, and a local LLM may assign `doctor`, `patient`,
  or `unknown` based on the transcript context.
- Diarization and role assignment are separate: a diarization ID is not itself
  a doctor/patient identity.

### Evidence-cited note path

- The normalizer preserves the raw text and creates a separate normalized
  transcript. It uses exact, reviewed lookup tables for fillers, drug aliases,
  and abbreviations. It does not silently fuzzy-correct medical words.
- A local LLM extracts atomic facts, including present/absent/uncertain/
  historical assertion, speaker, entity, and source utterance ID(s).
- The terminology linker uses exact local CSV matches. It reports `not_found`
  or `ambiguous` rather than inventing an identifier or silently picking among
  matches.
- SOAP generation returns claims grouped by S/O/A/P. Each claim must cite one
  or more known utterance IDs; the schema validator rejects missing or unknown
  citations.
- A verifier checks claims against their cited utterances. The pipeline
  retries a bounded number of times when it finds unsupported claims,
  omissions, or configured dose warnings.
- The final `soap_verified.json` contains supported claims, citations,
  omissions, warnings, attempt history, and a `review_required` flag. This is
  not a guarantee of factual correctness: the verifier is itself an LLM, and
  omission checks are limited heuristics.

The default SOAP generator uses the configured local OpenAI-compatible model
(defaults in `configs/base.yaml` point to an Ollama-compatible endpoint and
`qwen2.5:7b`). Alternatively, the evidence-cited generator can load a
Transformers/PEFT LoRA adapter when configured with `backend: lora`. That
adapter must be trained for the exact same base checkpoint. The ACI-Bench
pilot adapter is a different task and cannot be dropped into this evidence-
cited path.

### ACI-Bench pilot path

ACI-Bench contains English doctor-patient dialogues paired with full encounter
notes. The pilot maps note headings into four plain-text SOAP fields using
rules. Combined Assessment and Plan text is divided by action-language cues.
This is a heuristic transformation, and note content can include information
not stated in the dialogue.

The pilot fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` using LoRA and evaluates both
the base model and the adapter on the 120-example held-out test split. Its
output is four SOAP strings, without utterance-level evidence citations. These
results are included for reproducibility and inspection:

- Prepared examples: `data/processed/aci-bench-soap/`
- Metrics, predictions, and run notes: `results/aci-bench-pilot/`

The processed examples and predictions include ACI-Bench dialogue and
reference-note text. The dataset is distributed under CC BY 4.0; retain the
attribution and paper citation in `results/aci-bench-pilot/README.md` when
redistributing. Treat generated outputs as research artifacts, not clinical
notes.

## 4. Repository layout

```text
configs/
  base.yaml                 Runtime defaults for audio, LLM, normalization,
                            linking, generation, and verification.
data/
  lookup_tables/            Reviewed normalization and dose lookup CSVs.
  manifests/
    manifest.example.csv    Safe template for local encounter manifests.
  processed/
    aci-bench-soap/         Committed heuristic ACI-Bench train/valid/test
                            JSONL and a preparation report.
  terminology/
    concepts.csv            Local concept-linking table template.
docs/
  PROJECT_GUIDE.md          This project and implementation guide.
results/
  aci-bench-pilot/          Aggregate metrics and base/LoRA predictions.
src/voice_to_soap/
  aci_soap.py               ACI-Bench SOAP mapping, LoRA training/evaluation.
  asr_datasets.py           Eka ASR dataset import and WER/CER evaluation.
  audio.py                  Audio loading, VAD, Whisper, diarization helpers.
  cli.py                    Command-line entry point.
  config.py                 YAML configuration loading.
  datasets.py               Manifest validation and transcript parsing.
  entity_linking.py         Exact local terminology linking.
  llm.py                    Local OpenAI-compatible LLM client.
  normalize.py              Exact lookup normalization and dose checks.
  pipeline.py               Per-encounter end-to-end orchestration.
  roles.py                  LLM-assisted speaker-role assignment.
  schemas.py                Fact and evidence-cited SOAP validation.
  soap_lora.py              Evidence-cited SOAP LoRA data/training backend.
  steps.py                  Fact extraction, SOAP generation, verification.
  ner/                      Experimental NER training utilities.
tests/                      Unit tests for pipeline components and data rules.
```

Generated audio, raw datasets, environment directories, model weights, and
ordinary run outputs should stay out of Git unless there is a specific
redistribution decision. `.gitignore` excludes these by default. The prepared
ACI-Bench data and pilot predictions are deliberately committed with
attribution; do not assume that other datasets, especially Eka or local
recordings, have the same license or consent terms.

## 5. Technology stack

| Area | Current project choices |
|---|---|
| Language/runtime | Python; project metadata declares Python 3.10+; Python 3.11 is the documented initial environment. |
| Audio | PyTorch/TorchAudio, NumPy, SciPy, SoundFile, Librosa |
| Speech segmentation | Silero VAD |
| ASR | Hugging Face Transformers Whisper (`openai/whisper-small` default; large-v3 can be selected for GPU evaluation) |
| Speaker diarization | pyannote.audio (optional; gated model access) |
| Local LLM inference | OpenAI-compatible local API; defaults are compatible with Ollama |
| Fine-tuning | Transformers, Datasets, Accelerate, PEFT/LoRA; bitsandbytes is listed for supported quantized GPU workflows |
| Data/configuration | CSV, JSON/JSONL, PyYAML |
| Entity linking | Local exact-match terminology CSV; no hosted terminology calls |
| Evaluation | WER/CER tooling for ASR; ROUGE-L for ACI pilot text overlap |
| Tests | `unittest`-based test files; pytest is also a development dependency |

BioClinicalBERT is not currently configured as the fact extractor or SOAP
generator. The main structured extraction and verification stages use the
configured local generative LLM.

## 6. Setup

Use an isolated Python environment. On macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On the V100 machine, first install the PyTorch/TorchAudio builds compatible
with that host's NVIDIA driver and CUDA runtime, then install the remaining
requirements. Check the actual environment:

```bash
python --version
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
```

Two V100 cards do not automatically combine into one device with the sum of
their memory. Start with one GPU, measure model memory and sequence length,
then configure and test multi-GPU training explicitly if beneficial. Do not
assume a model fits until a real smoke run on the target server confirms it.

For the evidence-cited pipeline, configure an OpenAI-compatible local LLM
server, then set the environment variables expected by `configs/base.yaml`:

```bash
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=qwen2.5:7b
# Set LLM_API_KEY only if the chosen local server requires one.
```

Never put credentials or Hugging Face access tokens in YAML, source code, or
commits. The pyannote path reads `HF_TOKEN` from the process environment after
you have accepted the model terms.

## 7. Running the current code

Run commands from the repository root with `PYTHONPATH=src`.

### Run the evidence-cited pipeline on a transcript

1. Copy `data/manifests/manifest.example.csv` to a local manifest.
2. Add one row per encounter. Use a transcript and leave `audio_path` blank, or
   provide a local audio path and leave `transcript` blank.
3. Use explicit transcript lines such as:

   ```text
   [doctor] What brings you in today?
   [patient] I have had a cough since yesterday.
   ```

4. Run:

   ```bash
   PYTHONPATH=src python -m voice_to_soap.cli run \
     --manifest data/manifests/my_local_manifest.csv \
     --config configs/base.yaml \
     --outdir data/outputs/my-run
   ```

Each encounter gets its own directory with files such as:

| File | Contents |
|---|---|
| `audio.json` | Audio processing metadata, speech segments, and optional diarization/role results; transcript-only records mark their source. |
| `transcript_raw.json` | Utterances before lookup normalization. |
| `transcript.json` | Raw/normalized utterance data and recorded lookup changes. |
| `facts.json` | Extracted facts, assertion, evidence, and optional terminology-link result. |
| `soap_draft.json` | Draft SOAP claims with evidence and verifier verdicts. |
| `soap_verified.json` | Supported claims plus citations, omissions, warnings, attempts, and review status. |

`run.jsonl` at the output root records completion/failure per encounter. By
default the pipeline logs a failure and continues to the next encounter; set
`pipeline.stop_on_error: true` to stop on the first failure.

### Run audio preparation, VAD, and ASR separately

To prepare manifest audio without overwriting the originals:

```bash
PYTHONPATH=src python -m voice_to_soap.cli prepare-audio \
  --manifest data/manifests/my_local_manifest.csv \
  --outdir data/processed/audio \
  --sample-rate 16000
```

To inspect VAD regions:

```bash
PYTHONPATH=src python -m voice_to_soap.cli detect-speech \
  --manifest data/manifests/my_local_manifest.csv \
  --audio-dir data/processed/audio \
  --outdir data/processed/speech
```

To transcribe a single recording:

```bash
PYTHONPATH=src python -m voice_to_soap.cli transcribe-audio \
  --audio path/to/recording.wav \
  --out data/outputs/asr/visit-001.json \
  --language hindi
```

Omit `--language` for automatic detection. Test Hindi, Marathi, and
code-mixed speech on representative local recordings before selecting an ASR
model or language setting for a larger run.

### Run the ACI-Bench SOAP experiment

The committed processed split files let teammates evaluate without manually
reconstructing the preparation step. To prepare them again from a separately
downloaded ACI-Bench corpus:

```bash
PYTHONPATH=src python -m voice_to_soap.aci_soap prepare \
  --dataset-dir data/raw/aci-bench/aci-bench-corpus/challenge_data \
  --output-dir data/processed/aci-bench-soap
```

Evaluate the base model:

```bash
PYTHONPATH=src python -m voice_to_soap.aci_soap evaluate \
  --test data/processed/aci-bench-soap/test.jsonl \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir data/outputs/aci-base
```

Train a LoRA adapter:

```bash
PYTHONPATH=src python -m voice_to_soap.aci_soap train \
  --train data/processed/aci-bench-soap/train.jsonl \
  --validation data/processed/aci-bench-soap/validation.jsonl \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir data/models/aci-qwen-soap-lora \
  --epochs 1 --batch-size 1 --max-seq-length 2048 \
  --skip-long-examples
```

Evaluate the adapter on the unchanged test split:

```bash
PYTHONPATH=src python -m voice_to_soap.aci_soap evaluate \
  --test data/processed/aci-bench-soap/test.jsonl \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-dir data/models/aci-qwen-soap-lora \
  --output-dir data/outputs/aci-lora
```

The `--skip-long-examples` option is necessary only when some examples exceed
the selected sequence length; it skips whole encounters instead of silently
truncating them. Always inspect the saved `adapter_metadata.json` and report
the number of examples actually used. Do not use the test set for model
selection or prompt tuning.

### Run ASR evaluation on Eka

Eka is an ASR test resource, not a training split for this project. The local
manifest/audio export and output artifacts are ignored by Git. Check the
dataset access/redistribution terms before downloading or sharing it.

```bash
PYTHONPATH=src python -m voice_to_soap.cli import-eka-asr \
  --dataset-dir data/raw/eka-medical-asr-evaluation \
  --limit-per-language 2

PYTHONPATH=src python -m voice_to_soap.cli evaluate-asr \
  --manifest data/manifests/eka-medical-asr-evaluation.csv \
  --languages en,hi \
  --limit-per-language 2 \
  --outdir data/outputs/eka-whisper-smoke
```

This writes transcript predictions and WER/CER metrics locally. Do not turn
the evaluation test examples into training examples.

## 8. Data and training requirements for a meaningful model

For supervised dialogue-to-SOAP training, each training encounter needs:

- A complete, quality-checked conversation with stable speaker labels where
  possible.
- A clinician-reviewed English SOAP target, even when the input contains
  Hindi, Marathi, or code-mixed language.
- A separation between train, validation, and test **by encounter**, not by
  utterance.
- For the evidence-cited path, source utterance IDs for every SOAP claim.
- Appropriate de-identification, consent/ethics approval, access control, and
  retention rules for any locally recorded consultations.

ACI-Bench provides paired dialogues and full notes, but not gold SOAP sections
with utterance citations. The pilot's heading conversion is only a starting
point. Before using these targets for substantive training, have a qualified
reviewer correct section assignments and unsupported details, or obtain
clinician-authored SOAP labels. Do not claim Hindi-to-English capability from
English-only training and evaluation.

Recommended implementation sequence:

1. Freeze a data card: source, license/consent, fields, language, labels,
   intended use, and known issues for every dataset.
2. Validate and split encounters; keep a held-out test set untouched.
3. Audit transcripts and speaker roles on a manually reviewed sample.
4. Establish ASR baselines by language using WER/CER; diagnose errors before
   changing downstream note generation.
5. Improve SOAP targets and add clinician review, especially for medication,
   negation, uncertainty, temporality, and section assignment.
6. Train a small one-GPU LoRA smoke run on the V100. Record exact base model,
   license, adapter configuration, random seed, examples used/skipped, and
   validation metrics.
7. Compare with the unmodified base model using the same fixed test set.
   Supplement ROUGE with factuality, unsupported-claim, omission, citation,
   JSON-validity, and clinician-review measures.
8. Review complete encounters manually; only expand training or consider
   explicitly configured multi-GPU training after the small run is stable.
9. Keep an auditable clinician approval step and never present generated notes
   as independently verified medical advice.

Two V100 GPUs do not automatically provide the memory of one larger GPU.
Identify the exact V100 memory, CUDA/PyTorch versions, and model size on the
server before selecting full fine-tuning, LoRA, quantization, or distributed
training. The current ACI Mac pilot was Qwen2.5-0.5B with one LoRA epoch; it
does not establish what model will fit or perform best on the V100 server.

## 9. Interpreting the committed pilot results

The ACI-Bench pilot compared the base Qwen2.5-0.5B-Instruct model with the
same model plus a LoRA adapter on 120 held-out encounters:

| Model | Valid JSON | Invalid JSON | Overall mean section ROUGE-L F1 |
|---|---:|---:|---:|
| Base Qwen2.5-0.5B-Instruct | 112/120 | 8/120 | 0.10452 |
| Qwen2.5-0.5B-Instruct + LoRA | 110/120 | 10/120 | 0.09888 |

The LoRA run trained on 26 of 67 training encounters and validated on 8 of 20;
long examples were skipped because the Mac run used a 2,048-token limit.
ROUGE-L is text-overlap with heuristic targets. It is not a clinical
correctness or safety score. The pilot shows no overall improvement from LoRA.
See `results/aci-bench-pilot/README.md` and the JSON files there for details.

## 10. Tests and validation

Run the full unit-test suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

These tests exercise local data validation, transformations, mocked audio/LLM
paths, evidence-ID validation, and result writing. They do not download or
benchmark every pretrained model and do not validate clinical utility.

## 11. Safety, privacy, and reproducibility

- Never commit `.venv`, tokens, credentials, private recordings, or locally
  downloaded data without checking license, consent, and intended visibility.
- The public repository includes derived ACI-Bench dialogue/reference data
  and predictions under CC BY 4.0 attribution. Preserve the citation in
  `results/aci-bench-pilot/README.md`.
- The separate Eka evaluation manifest and raw audio exports should remain
  local unless redistribution permission has been confirmed.
- Keep original transcripts and audio for audit; normalized text is a separate
  view, not a replacement.
- A valid JSON schema or an LLM verifier does not prove a claim is medically
  correct. All generated notes require clinician review.
- Record model and dataset versions, split IDs, hardware/software versions,
  parameters, and all skipped/failed examples for each future experiment.

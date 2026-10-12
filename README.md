# Voice-to-SOAP

Voice-to-SOAP is a research project that aims to help a clinician turn a consultation into a structured SOAP note:

- **S — Subjective:** what the patient reports
- **O — Objective:** observed findings, measurements, or test results
- **A — Assessment:** the clinician's stated assessment
- **P — Plan:** what the clinician recommends or prescribes

The intended consultations may mix Hindi, Marathi, and English. The system is a documentation assistant, not a diagnostic or prescribing system. A clinician must review any generated note.

## Project status

This repository is being built in small, testable stages. The current code is an early scaffold, not a completed or clinically validated system. No model is trained or included in the repository. Model weights and datasets must be obtained separately, subject to their licences and access requirements.

## First step: prepare the development environment

Use **Python 3.11** for the initial setup. It is the project's target version while dependencies and GPU compatibility are tested. Do not use the system Python on a shared GPU server; create a project environment.

On macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Check which Python and PyTorch are active:

```bash
python --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

On the computer that has the V100 cards, install a PyTorch build compatible with that machine's NVIDIA driver before installing the rest of the requirements. Use the official PyTorch install selector for the server's operating system and CUDA option. The two GPUs' memory is not automatically combined into one larger GPU.

If CUDA is unavailable on your personal computer, that is okay for this first setup step. We will verify GPU access on the machine where the V100s are installed.

## Planned system at a glance

```text
consultation audio
  -> prepare audio and detect speech
  -> convert speech to text
  -> identify speakers
  -> normalize transcript and flag uncertain terms
  -> extract facts with evidence references
  -> draft an evidence-cited SOAP note
  -> check claims and omissions
  -> clinician reviews the note
```

This describes the intended project, not a claim that every stage is implemented. BioClinicalBERT is planned as an English clinical entity-recognition experiment and needs annotated training examples. Hindi, Marathi, and code-mixed text require a separate multilingual-model experiment. Neither model choice has been validated on the project's datasets yet.

## Stage 2: dataset manifest and input checks

Each dataset will eventually have its own adapter. Before writing those adapters, the project uses one small CSV manifest format to describe encounters consistently. Copy `data/manifests/manifest.example.csv` and add one row per encounter.

The manifest columns are:

| Column | Required value |
|---|---|
| `encounter_id` | Unique ID for this encounter |
| `audio_path` | Existing local audio file path, relative to the manifest or absolute; blank if using a transcript |
| `transcript` | Existing transcript; blank if using audio |
| `reference_note` | Reference note when the dataset provides one; otherwise blank |
| `language` | Language label such as `en`, `mr`, `hi-en`, or `mr-en` |
| `split` | Dataset split such as `train`, `validation`, or `test` |

Every row must have a unique ID, a language, a split, and at least one data source (audio or transcript). Relative audio paths are resolved from the manifest's folder and checked for existence. The loader rejects malformed rows early so they do not fail later inside model code.

Run the manifest-stage tests without installing model or GPU libraries:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_dataset_manifest.py" -v
```

## Stage 3: audio ingestion and standardization

Before speech recognition, recordings need a consistent format. This stage reads each audio file listed in the manifest, converts stereo to mono, resamples it to 16 kHz, and saves a new WAV file plus metadata. It never overwrites the original recording. Transcript-only encounters are skipped.

```bash
PYTHONPATH=src python -m voice_to_soap.cli prepare-audio \
  --manifest data/manifests/manifest.csv \
  --outdir data/processed/audio \
  --sample-rate 16000
```

For an encounter named `visit-001`, the result is saved under `data/processed/audio/visit-001/`. The metadata records original and prepared sample rates, channel counts, duration, sample count, peak level, and clipping fraction. This step does **not** detect speech, transcribe audio, or identify speakers; those are later stages.

Run its tests with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_audio_preparation.py" -v
```

## Stage 4: voice activity detection

Prepared recordings can contain pauses and silence. Voice Activity Detection (VAD) marks time ranges that likely contain speech. This project uses Silero VAD for that stage; it does not decide what words were spoken. ASR will process the saved speech segments in a later stage.

Run VAD on the prepared WAVs:

```bash
PYTHONPATH=src python -m voice_to_soap.cli detect-speech \
  --manifest data/manifests/manifest.csv \
  --audio-dir data/processed/audio \
  --outdir data/processed/speech
```

The command saves each detected region as a separate WAV under the encounter's `segments/` folder and writes `speech_segments.json` with each segment's start/end time in the prepared recording. If VAD finds no speech, the JSON records an empty segment list; if a file or dependency is missing, the command reports the failure.

VAD is not perfect: background noise can look like speech, and quiet speech can be missed. Keep the original and standardized audio and inspect sample segment boundaries before large runs. The standalone VAD command is useful for inspection; the `run` pipeline also detects speech in memory and passes those ranges to ASR.

Run the segment handling and CLI tests without downloading VAD model weights:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_voice_activity.py" -v
```

## Stage 5: ASR with Whisper

ASR (Automatic Speech Recognition) converts speech into text. The default checkpoint is the multilingual open-weight `openai/whisper-small`, loaded locally through Transformers. It is a practical choice for development on a laptop; it is not a claim that it is best for every Hindi, Marathi, or code-mixed recording. You can select `openai/whisper-large-v3` for a later GPU evaluation.

Try one audio file without running the later LLM stages:

```bash
PYTHONPATH=src python -m voice_to_soap.cli transcribe-audio \
  --audio path/to/recording.wav \
  --out data/outputs/asr/visit-001.json
```

The command loads the ASR model named in `configs/base.yaml` (or pass `--model openai/whisper-large-v3` to select large-v3), converts the input to the configured sample rate, then writes the recognized text and any word timestamps to the requested JSON file. The first run downloads model weights from Hugging Face; later runs use the local cache. The same model checkpoint can be used on a laptop or the V100 server; `asr_device: auto` selects CUDA, Apple MPS when available, or CPU. The hardware changes speed and memory use, not the model's identity. Large-v3 may be slow or memory-heavy on a laptop, so use it on the V100 server for full evaluation.

If the language is known, pass `--language marathi` or `--language hindi`; otherwise leave it unset for Whisper's language detection. For code-mixed speech, automatic detection is only a starting point and must be evaluated on your data.

When running the full pipeline, VAD runs first and Whisper transcribes each detected range. Word timestamps are shifted back to the original recording's timeline. ASR unit tests mock the model call, so they verify our input/output handling but do not validate Whisper accuracy. Model accuracy requires running it on audio with trusted reference transcripts.

### Eka Medical ASR evaluation data

The Eka dataset is published with English (`en`) and Hindi (`hi`) test subsets. It has no ASR training split, so keep its references out of training and use it only for held-out ASR evaluation. The import command exports embedded audio to per-example files, writes the correct text only to `reference_transcript`, and creates an ASR-specific manifest. The original Parquet shards remain unchanged under `data/raw/eka-medical-asr-evaluation/`.

Import all available English and Hindi examples:

```bash
PYTHONPATH=src python -m voice_to_soap.cli import-eka-asr
```

For a small smoke test, pass `--limit-per-language 2`. The command writes audio files and `data/processed/eka-medical-asr-evaluation/asr_manifest.csv`. The ordinary `transcript` input column is blank, preventing the pipeline from treating the answer key as ASR input. Rows are explicitly marked `task=asr_evaluation` and `split=test`.

Evaluate the configured ASR model on the exported samples. With the current `whisper-small` default, this command evaluates the full selected manifest; add `--limit-per-language 2` to process only two English and two Hindi samples:

```bash
PYTHONPATH=src python -m voice_to_soap.cli evaluate-asr \
  --manifest data/manifests/eka-medical-asr-evaluation.csv \
  --limit-per-language 2 \
  --outdir data/outputs/eka-whisper-small-smoke
```

Change `--limit-per-language` to control how many samples to process per language, use `--languages en` or `--languages hi` to select a subset, and pass `--model openai/whisper-large-v3` to switch to large-v3. The evaluator runs the shared audio loader and ASR function, then writes every recognized transcript alongside its reference to `data/outputs/eka-whisper-small-smoke/predictions.jsonl`; aggregate WER and CER are in `data/outputs/eka-whisper-small-smoke/metrics.json`. Scores use case-folded, punctuation-stripped text. These metrics are benchmark measurements, not proof of clinical safety.

The evaluator writes WER/CER overall, by language, and by recording context (conversation, sentence narration, or entity narration), so results do not hide differences between recording styles. Text scoring case-folds and strips punctuation. The generated manifest lives at `data/manifests/eka-medical-asr-evaluation.csv`; its audio paths are relative to that manifest and point into `data/processed/eka-medical-asr-evaluation/`. Override this path with `--manifest-out` if needed. The published Eka data contains only a test split. Do not create a training split from it; an independent ASR training corpus is needed to fine-tune ASR. This dataset is separate from ACI-Bench, MTS-Dialog, and MediTOD, which serve text-based note generation or dialogue-understanding tasks.

### Run diarization on manifest audio

The manifest-level diarization command runs pyannote on the audio itself and saves anonymous, timestamped speaker turns. It does not require ASR transcripts. To test two English and two Hindi examples:

```bash
PYTHONPATH=src python -m voice_to_soap.cli diarize-manifest \
  --manifest data/manifests/eka-medical-asr-evaluation.csv \
  --languages en,hi \
  --limit-per-language 2 \
  --out data/outputs/eka-diarization-smoke/diarization.jsonl
```

This requires the `pyannote.audio` dependency, permission to access `pyannote/speaker-diarization-community-1` on Hugging Face, and an `HF_TOKEN` environment variable. Each JSONL row contains the encounter ID, language, audio path, model ID, and turns with `start`, `end`, and `speaker_id`. The model estimates the speakers by default; use `--num-speakers` only when the recording's speaker count is known. This command reports diarization only; mapping those turns to transcript words happens in the full pipeline using timestamps.

## Stage 6: speaker diarization

Diarization answers “who spoke when?” It produces anonymous IDs such as `speaker_0` and `speaker_1`; it does not decide which person is the doctor or patient. The default candidate is the local `pyannote/speaker-diarization-community-1` model. Its model page requires accepting the model terms and using a Hugging Face access token. Set the token in the environment as `HF_TOKEN`; do not paste it into YAML or commit it.

After accepting the model terms and setting the token, enable `audio.diarization.enabled` in `configs/base.yaml`. The main pipeline runs diarization on the original recording, then assigns each timestamped ASR word the speaker turn with the greatest overlap. Words without timestamps or without an overlapping turn remain `unknown`. The pipeline records diarization turns in `audio.json` and speaker IDs on each utterance in `transcript_raw.json`.

To test diarization on one recording separately:

```bash
PYTHONPATH=src python -m voice_to_soap.cli diarize-audio \
  --audio path/to/recording.wav \
  --out data/outputs/diarization/visit-001.json \
  --num-speakers 2
```

The model runs locally and can use CUDA when available. `--num-speakers 2` is appropriate only when the recording is known to have exactly two speakers; otherwise omit it and let the model estimate. You may need FFmpeg and a compatible pyannote/PyTorch setup on the machine. The tests mock the pretrained pipeline: they validate timestamps, speaker assignment, options, and output files, but do not measure diarization quality on real speech.

## Stage 7: doctor/patient role assignment

Diarization only says which anonymous voice spoke when. When diarization is enabled in the full pipeline, the local LLM reads the ASR text grouped by speaker and assigns `doctor`, `patient`, or `unknown`. It must cite utterance IDs from that speaker for a doctor/patient assignment; the code checks those IDs. If evidence is unclear, the prompt directs it to return `unknown`. Role decisions and their cited utterances are recorded in `audio.json`, and each transcript utterance receives its role.

The role guess is still a model prediction, not ground truth. Check it against datasets with known speaker roles and manually review a sample before using roles in SOAP generation. Explicit doctor/patient labels already present in an input transcript are retained instead of being guessed again. If diarization is disabled, audio speakers remain `unknown`.

## Stage 8: transcript normalization

The pipeline keeps the ASR or supplied transcript unchanged in `text_raw` and creates a separate `text_normalized` for downstream extraction. The normalizer applies three auditable, case-insensitive, whole-word/whole-phrase lookup transformations:

1. Remove configured verbal fillers such as `um`, `uh`, `erm`, and `er` from `disfluencies.csv`.
2. Map listed heard drug names or brand phrases to their chosen generic ingredient using `drug_names.csv` (`heard,generic`). Add only mappings reviewed for the exact product/formulation in your data; do not map a brand family if it can refer to combination products.
3. Expand listed abbreviations from `abbreviations.csv` (`heard,canonical`).

The matching is exact apart from capitalization: it does not guess spellings, use fuzzy matching, or remove repeated words. This is intentional for clinical safety: repeated negations or medication names can be meaningful, and a mistaken automatic correction could alter the record. Add Hindi/Marathi fillers or local brand spellings to the relevant CSV only after reviewing examples from the dataset. Each changed utterance is recorded in the `changes` array in `transcript.json`, including the before/after text and each transformation. The raw text is retained for comparison and review.

Lookup tables are used because the project can explicitly review and control each clinical term replacement. They are predictable, easy to audit, and do not let a language model silently guess a drug identity. They are not complete medical dictionaries; coverage depends on the aliases added and validated for the target dataset.

## Stage 9: structured fact extraction and terminology linking

After normalization, the local LLM extracts atomic facts rather than drafting a SOAP note directly. Each fact records its category, concise meaning, assertion (`present`, `absent`, `uncertain`, or `historical`), supporting speaker, entity type/text, and one or more source utterance IDs. For example, “No fever” should become a symptom entity `fever` with assertion `absent`, not a positive fever. The validator rejects missing fields, unsupported assertion/entity types, unknown evidence IDs, and a named speaker that does not match the cited utterances. A legacy `past` assertion is normalized to `historical`.

Facts are linked locally using `data/terminology/concepts.csv`. Its columns are `term,vocabulary,concept_id,canonical_name,entity_type`; add one row per exact alias and concept using an official RxNorm RxCUI for drugs or UMLS CUI for symptoms/conditions. Matching ignores capitalization and repeated whitespace but is otherwise exact. The linker prefers RxNorm for drugs and UMLS for symptoms/conditions. Multiple matching concepts are returned as `ambiguous`; unmatched mentions are explicitly marked `not_found`. It never invents IDs or silently selects one ambiguous concept.

The terminology table in this starter project contains only its header, so facts will currently be marked `not_found` until it is populated with an appropriate terminology release. The code reads a local file and sends no transcript or extracted clinical text to a terminology service. Obtain and use UMLS data only under its applicable account and licence terms; RxNorm releases are available from NLM. Keep any downloaded terminology source/version and local alias curation documented for reproducibility. This exact-match linker does not perform fuzzy spell correction or resolve context-dependent abbreviations automatically.

The structured output, including `entity_link.status` and any canonical concept identifiers, is saved in each encounter's `facts.json`. Review extracted assertions, evidence, and concept matches before downstream use; validation of JSON structure does not prove that the LLM understood the clinical statement correctly.

## Stage 10: LoRA fine-tuning for cited SOAP generation

The existing SOAP step already requires each note claim to include source utterance IDs, and `validate_soap` rejects missing or unknown IDs. By default, that generation still uses the configured local OpenAI-compatible LLM. An optional Transformers/PEFT backend now supports training and loading a LoRA adapter for SOAP generation; fact extraction and claim verification continue to use the configured local LLM.

The LoRA base checkpoint is intentionally **not selected** in `configs/base.yaml`. Choose a model whose licence permits your research, and use the exact same base checkpoint when training and running its adapter. For example, Qwen 2.5 7B (the current local default) is an open-weight model and could be the base if you use its compatible Transformers checkpoint. Llama 3 8B or BioMistral 7B would instead be different base models. An adapter trained for one base checkpoint is not interchangeable with another. Ollama is a local serving runtime; this training path loads the checkpoint through Transformers and PEFT and does not train an adapter inside Ollama.

Training JSONL uses one JSON object per encounter with `utterances`, optional `facts`, and a clinician-reviewed `soap` object. Each SOAP item must contain `text` and evidence utterance IDs that exist in that example. Keep encounters—not individual utterances—separated between training and validation to avoid leakage. Only use de-identified or appropriately consented/approved data.

```bash
PYTHONPATH=src python -m voice_to_soap.soap_lora \
  --train data/training/soap_train.jsonl \
  --validation data/training/soap_validation.jsonl \
  --base-model <chosen-open-weight-checkpoint> \
  --output-dir data/models/soap-lora
```

The script trains LoRA adapters on attention projection layers, masks transcript/prompt tokens from the training loss (the loss is on the reviewed SOAP JSON target), evaluates and saves per epoch, and refuses to truncate overlong examples because that could remove cited source lines. It saves adapter metadata with the base checkpoint identifier. Start with a small pilot and inspect GPU memory on the target V100 before scaling up; the code does not assume that any named model fits or is suitable without testing.

After training, set `soap_generation.backend: lora`, `base_model` to the exact checkpoint used for training, and `adapter_path` to the adapter output directory. The production pipeline loads that adapter for SOAP drafting and validates every generated claim's citations against the encounter's utterance IDs. No adapter for this evidence-cited production path is included: the ACI-Bench pilot described below uses heuristically mapped targets without utterance-level evidence citations. Unit tests validate data handling; they do not demonstrate clinical performance.

### ACI-Bench conversation-to-SOAP pilot

ACI-Bench contains full English doctor-patient dialogue paired with full encounter notes. The following experimental adapter maps recognizable note headings into the four SOAP fields and fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` with LoRA. This is a separate research path from the evidence-cited production generator above: ACI-Bench does not supply sentence-level transcript evidence links, so its output schema is four SOAP strings rather than evidence-cited claims.

Prepare the official train/validation/test split files without mixing encounters:

```bash
PYTHONPATH=src python -m voice_to_soap.aci_soap prepare
```

The default output is `data/processed/aci-bench-soap/`. The mapping is heuristic: note headings such as Chief Complaint and Review of Systems go to Subjective; exam, vitals, and results go to Objective; assessment/diagnosis headings go to Assessment; and instructions/orders go to Plan. Combined Assessment and Plan text is split using action-language cues. Review `preparation_report.json` and the individual JSONL examples before interpreting the experiment. These are weak labels, not clinician-reviewed SOAP annotations, and a source note may contain details not stated in its paired dialogue.

The 0.5B checkpoint is chosen for a constrained 16 GB Mac pilot; it is not the intended final model recommendation. The model weights are downloaded from Hugging Face on first use. First measure the base model on the held-out set, then train one LoRA epoch on the 67 training encounters with the 20 validation encounters:

```bash
PYTHONPATH=src python -m voice_to_soap.aci_soap evaluate \
  --test data/processed/aci-bench-soap/test.jsonl \
  --output-dir data/outputs/aci-qwen25-05b-baseline

PYTHONPATH=src python -m voice_to_soap.aci_soap train \
  --train data/processed/aci-bench-soap/train.jsonl \
  --validation data/processed/aci-bench-soap/validation.jsonl \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir data/models/aci-qwen25-05b-soap-lora \
  --epochs 1 --batch-size 1 --max-seq-length 2048 \
  --skip-long-examples

PYTHONPATH=src python -m voice_to_soap.aci_soap evaluate \
  --test data/processed/aci-bench-soap/test.jsonl \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-dir data/models/aci-qwen25-05b-soap-lora \
  --output-dir data/outputs/aci-qwen25-05b-lora
```

Evaluation uses the 120 held-out ACI-Bench encounters and reports section-level ROUGE-L overlap with the heuristic SOAP targets; predictions are saved in each output directory's `predictions.jsonl`, and scores in `metrics.json`. This lexical metric does not establish clinical correctness, factual grounding, or safety. Review outputs manually and compare the base-model and LoRA scores. ACI-Bench is English-only, so this pilot does not train or evaluate Hindi-to-English note generation. Its repository also warns that some dialogue speaker tags are swapped; this experiment preserves the published labels rather than correcting them silently.

Mac smoke-run artifacts (not production results) are under `data/outputs/aci-qwen25-05b-*`. In the completed run, the 16 GB Apple Silicon laptop trained one epoch using only the 26 training encounters fitting the 2,048-token limit and evaluated on 8 fitting validation encounters; the remaining long examples were reported and skipped, never truncated. Full held-out evaluation was run on all 120 test encounters for both base and adapter. See `data/outputs/aci-qwen25-05b-comparison/comparison.json`. The adapter's overall mean section ROUGE-L was lower than the base model in this small pilot, so this run provides no evidence that LoRA improved results. For substantive training, use the V100 machine, a larger sequence-memory budget, and preferably clinician-reviewed SOAP/evidence targets.

There is more than one research design for this task. MEDIQA-Sum 2023 separately evaluated dialogue-to-topic/section classification, section-conditioned note generation, and full-encounter note generation; this pilot focuses on the last task and converts full notes into approximate SOAP targets. Another research direction is explicit factuality evaluation: a 2025 release provides expert ACI-Bench annotations for hallucinations, inferences, misunderstandings, and severity. A future evaluation should use human/factuality review in addition to ROUGE-L. Embeddings or retrieval can supply reference material, but do not teach SOAP structure or replace paired conversation-to-note targets. See the [ACI-Bench paper and repository](https://github.com/wyim/aci-bench), the [MEDIQA-Sum task description](https://www.imageclef.org/2023/medical/mediqa), and the [ACI-Bench hallucination annotations](https://github.com/amazon-science/acibench-hallucination-annotations).

## Stage 11: verification and bounded feedback

Each SOAP draft goes through a configurable number of attempts (`verification.max_attempts`, default 2). Every claim must cite existing utterance IDs; malformed citations or structure cause a corrective re-prompt. For structurally valid claims, the configured local LLM checks whether each claim is supported by the cited transcript. The code also checks whether fact concepts and their present/absent/uncertain/historical status are represented in supported claims, and checks medication quantities against any configured dose ranges.

If checks find unsupported claims, omissions, or dose-range/unit warnings, the next SOAP prompt includes those issues as feedback. The system does not silently change a dose: a flagged dose should be omitted or explicitly referred for clinician review. Retries are bounded to avoid loops. After the final attempt, only claims rated `supported` are included in the SOAP shown as verified; unresolved omissions, warnings, per-attempt feedback, and `review_required` remain in `soap_verified.json` for review. Invalid SOAP output that still cites unknown IDs after all attempts fails that encounter rather than being presented as verified.

The checks are safeguards, not a proof of clinical correctness. In particular, semantic support checking still uses an LLM and can be wrong; omission detection uses a conservative term/assertion cue check and may miss paraphrases; configured ranges do not account for every patient factor or complete prescribing context. Clinician review remains required.

The starter `data/lookup_tables/dose_ranges.csv` contains only the header, so no dose is currently judged in/out of range. Populate it only from an authoritative, clinically reviewed source and document the source/version. Current columns:

```csv
drug,min_value,max_value,unit,basis,route
example-drug,250,500,mg,per_dose,oral
```

Use the normalized generic drug name; enter a clinically meaningful minimum and maximum for the stated basis (`per_dose` or another explicitly reviewed basis), unit, and route. Do not copy a general daily maximum into a per-dose row. Different formulations, strengths, routes, indications, and patient groups can require different limits; this starter schema cannot represent all those dimensions, so add separate well-defined rules only with clinical guidance or extend the schema before using those cases. Supported unit conversion currently covers mass units (`mg`, `g`, `mcg`/`ug`/`μg`); volume and count units are compared only to the same unit. A unit mismatch is flagged rather than converted.

## Data and privacy

Before using a dataset, record what it contains (audio, transcript, speaker labels, notes, language, and official splits) and check its licence and access conditions. Keep test data separate from training and prompt development.

Keep raw audio, identifiable transcripts, model weights, access tokens, and other private data out of Git. If the project later uses locally recorded consultations, obtain the required ethics approval and consent, and agree on secure storage and retention before recording.

## Next development steps

1. Confirm this README and the Python/dependency setup on the development machine.
2. Inventory the available datasets and their licences.
3. Choose one dataset and implement a small, tested data loader for it.
4. Add model components one at a time and evaluate each on held-out data.

Do not treat generated notes as suitable for patient care without appropriate clinical validation and clinician approval.

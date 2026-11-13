# Repository audit and implementation plan

Subsequent status: main commit `751da5a` fixed clinician-approval flag separation.
Portable setup and independent local hardware inspection were then added; see
TEAM_SETUP.md and PIPELINE_EXECUTION.md. The inventory below records the initial
audit findings rather than claiming every issue is still unresolved.

Audit date: 13 November 2026. Scope: repository source, configuration, tests,
lookup tables, both project-guide copies, all committed JSON/JSONL artifacts,
and the two pasted ChatGPT conversations. No report PDF or PPT was present;
claims in the pasted conversation about their contents remain unverified.
Git internals and generated environments are not application source.

## Finding

Continue this repository. It contains useful modular Python code and 72 unit
tests, but is not a complete application. Preserve two distinct tracks:

1. Evidence-cited documentation: manifest -> transcript/audio -> normalization
   -> facts -> optional terminology linking -> cited SOAP -> bounded verification.
2. ACI-Bench pilot: English dialogue -> heuristic four-string SOAP -> LoRA and
   ROUGE-L evaluation. Its outputs and adapter are incompatible with track 1.

The first deliverable should be a reproducible synthetic transcript run with
traceable claims and explicit pending clinician approval. Audio, UI, translation,
training and research evaluation follow in separate phases.

## Environment and executed checks

- Apple Silicon macOS; available Python is 3.12.4, within the declared >=3.10
  requirement. Python 3.11 was not found on PATH. No Python downgrade performed.
- FFmpeg exists. Ollama and `code` were absent from PATH. VS Code itself exists
  under /Applications, and its CLI was used directly to inspect extensions.
- Codex (`openai.chatgpt`), Microsoft Python, Pylance, Python Debugger and Jupyter
  are already installed. Ruff was absent; it is an optional recommendation.
- Created an isolated `.venv` and installed the lightweight development packages.
- Before installation: unittest discovery reported 38 entries with 10 errors,
  mostly test-module import failures; this was not the full suite.
- After installation: `PYTHONPATH=src .venv/bin/python -m unittest discover -s
  tests -v` executed **72 tests, all passing**. Model calls are mocked.
- `pip check`: no broken requirements in that environment.
- Both PROJECT_GUIDE.md copies are byte-identical. Root copy was already an
  untracked user file and was preserved.
- Parsed all 207 prepared ACI examples: train 67, validation 20, test 120;
  encounter IDs are unique within and disjoint across splits. This does not
  establish absence of every form of semantic or patient-level leakage.
- Parsed both 120-row prediction artifacts: 8 base and 10 LoRA invalid outputs,
  matching committed aggregates. Did not rerun generation or training.

No live ASR, diarization, LoRA training, clinical review or multilingual benchmark
was completed during this audit. See DEVELOPMENT_SETUP.md for final setup checks.

## Source-file inventory

| File under src/voice_to_soap | Responsibility and assessment |
| --- | --- |
| `__init__.py` | Package version 0.1.0. |
| `config.py` | YAML mapping and environment lookup; lacks nested typed validation. Paths depend on working directory. |
| `datasets.py` | Manifest checks and labeled-line transcript parser; missing safe encounter-ID validation and explicit policy when audio and transcript are both supplied. |
| `audio.py` | Decode, mono/resample, save, Silero VAD, timestamp restoration, Whisper and pyannote hooks, overlap speaker alignment. Real model compatibility/accuracy unverified. VAD loads per call; ASR/diarization pipelines cached. Word-level ASR yields word-level utterance IDs. |
| `roles.py` | LLM role attribution with same-speaker evidence checks; unknown is supported. No measured role accuracy. |
| `normalize.py` | Exact auditable lookups; raw text retained. Dose rules support mass conversion but do not enforce all route/patient/formulation context. Context-free abbreviation expansion can be ambiguous. |
| `entity_linking.py` | Exact local CSV match, ambiguity/not-found statuses. Empty production table. `Any` annotation is missing its import (postponed annotations hide it at runtime). |
| `schemas.py` | Dataclass utterance and manual validators. Citations checked for existence, not entailment. SOAP sections may be missing and `{}` becomes an empty note. Malformed unhashable fact classification fields can raise TypeError before string validation. |
| `llm.py` | OpenAI SDK transport to configured server and JSON parsing. Catches every exception then repeats the request without JSON mode; needs bounded transport behavior and narrower fallback. No project-specific timeout configuration. |
| `steps.py` | Fact prompting, cited SOAP, LLM claim verification, lexical assertion-aware omissions, retries. `review_required` can be false without human approval. Attempts omit full successful draft snapshots. Verifier sees cited text without speaker roles. |
| `pipeline.py` | Writes per-encounter intermediates and run log. Reuses output directories; no immutable run identity, full model provenance or resume stage tracking. Errors are logged/continued by default. Translation is an unprotected local-LLM prompt. |
| `io_utils.py` | Temporary-file JSON replacement and append JSONL; no concurrent encounter writer coordination. |
| `cli.py` | Commands for run, audio preparation, VAD, ASR, diarization, Eka import/evaluation. No doctor command, service or UI. Default run can exit successfully with failed encounter entries. |
| `evaluate.py` | Text ROUGE-L on intersecting IDs; missing IDs/duplicates and coverage not reported. Does not consume nested pipeline artifacts directly. |
| `asr_datasets.py` | Eka Parquet/audio import and test-only WER/CER evaluation. Explicit en/hi restriction; Marathi benchmarking needs a separate loader/extension. |
| `soap_lora.py` | Evidence-cited training loader, prompt-loss masking, LoRA training/inference. No cited adapter provided. Loader checks base checkpoint but not metadata task; train/validation encounter leakage checks and richer reproducibility metadata needed. Inference does not explicitly place the model on GPU. |
| `aci_soap.py` | Heuristic note mapping, split preparation, separate LoRA training/generation/ROUGE. Preparation checks IDs across splits, but training can take arbitrary split files without rechecking. CUDA training-device metadata can incorrectly report cpu; model loading uses float32. ROUGE excludes invalid generations and models have different valid subsets. |
| `ner/__init__.py` | Experimental package marker. |
| `ner/train.py` | Experimental span-to-BIO transformer training. Not in the pipeline. Hard-coded fp16, truncation and batched row-offset alignment require repair before relying on training results. No trained clinical NER included. |

## Remaining files and artifacts

| Files | Assessment |
| --- | --- |
| `configs/base.yaml` | Local qwen2.5:7b endpoint; Whisper small; diarization/translation disabled; maximum two SOAP attempts; continue on error. Candidate settings, not validated deployment. |
| `pyproject.toml` | Initially empty dependency metadata; lightweight runtime dependencies and dev extras added during setup. |
| `requirements.txt` | Broad unpinned audio/training/API/evaluation stack. SDK was incorrectly commented out and is now required. Full stack still needs a platform-specific verified environment. |
| `.gitignore` | Excludes .venv, outputs, raw data, models, manifests and .env. Already tracked ACI files remain tracked despite processed-data ignore. |
| `README.md` | Extensive scaffold instructions; some historical output references point to ignored directories. New setup/audit links added. |
| `PROJECT_GUIDE.md`, `docs/PROJECT_GUIDE.md` | Accurate broad status; duplicate-copy drift risk. Use docs copy as the implementation reference, preserve supplied root copy. |
| `data/lookup_tables/abbreviations.csv` | Six reviewed-intent aliases; no context model for ambiguous abbreviations such as OD. |
| `data/lookup_tables/disfluencies.csv` | Four English fillers; no established Hindi/Marathi coverage. |
| `data/lookup_tables/drug_names.csv` | Two aliases; limited coverage. |
| `data/lookup_tables/dose_ranges.csv` | Header only: no active dosage-range checks. |
| `data/terminology/concepts.csv` | Header only: no active concept vocabulary. Do not populate with invented IDs. |
| `data/manifests/manifest.example.csv` | Header-only template; not a runnable example. |
| `data/processed/aci-bench-soap/{train,validation,test}.jsonl` | All rows parsed and schema checked; heuristic labels, no utterance citations. |
| `data/processed/aci-bench-soap/preparation_report.json` | Split counts and mapping limitations; matches saved pilot preparation report. |
| `results/aci-bench-pilot/README.md` | Dataset attribution and limitations; retain it. |
| `results/aci-bench-pilot/{baseline,lora}-metrics.json` | Saved test aggregates, not newly measured results. |
| `results/aci-bench-pilot/{baseline,lora}-predictions.jsonl` | Full test outputs; all rows parsed; invalid counts checked. No new clinical labels assigned. |
| `results/aci-bench-pilot/comparison.json` | Base 0.104521 vs LoRA 0.098875 mean section ROUGE-L; regression. Valid-example subsets differ. |
| `results/aci-bench-pilot/training-summary.json` | One epoch, rank 8, 2048 tokens; 26/67 train and 8/20 validation used. |
| `results/aci-bench-pilot/preparation-report.json` | 39/67 train, 10/20 validation, 57/120 test examples have mapping warnings. |

Test inventory: `test_aci_soap.py` (mapping/training masks/evaluation failures),
`test_asr.py` (mocked ASR/options/timestamps/CLI), `test_asr_datasets.py` (synthetic
Parquet import/scoring), `test_audio_preparation.py` (real synthetic WAV conversion),
`test_core.py` (parsing/citations/rendering), `test_dataset_manifest.py` (validation),
`test_diarization.py` (mocked speaker turns/alignment/CLI), `test_entity_linking.py`
(exact matching/ambiguity/fact attribution), `test_normalization.py` (audit/lookups),
`test_pipeline_audio_stage.py` (mocked audio integration), `test_roles.py` (role
evidence), `test_soap_lora.py` (data/masks/citations), `test_verification.py`
(retry/omission/dose behavior), `test_voice_activity.py` (segments/CLI).
There is no complete transcript-to-files integration test or live-model test.

## Priority findings and exact first changes

1. **Clinician approval ambiguity:** separate automated verification issues from
   human approval. Modify steps.py, pipeline.py and verification tests. Every
   generated note starts pending human review; keep all claims and issue history.
2. **Output path containment:** validate encounter IDs centrally in datasets.py
   and check output containment in writers/orchestration before exposing uploads.
   Add manifest tests for absolute paths, traversal, separators and reserved IDs.
3. **Baseline contract:** tighten schemas.py; add synthetic encounter fixture
   under examples/ and transcript-to-files integration tests. Record empty-input
   and malformed model-output behavior explicitly. Keep valid empty SOAP sections.
4. **Transport/config:** narrow llm.py fallback, configure timeouts/retries, add
   validated configuration, and make CLI failed-run status observable. Do not
   silently switch from local inference to a paid/cloud provider.
5. **Provenance:** persist full draft per attempt, source/config/model identifiers,
   run IDs and final status; define overwrite/retry behavior before persistence/UI.

These application changes are proposed, not applied during setup. Relevant setup
files were added and the SDK dependency omission corrected. No clinical output
behavior, experiment results, datasets or pretrained weights were changed.

## Build phases and completion evidence

| Order | Work and likely files | Completion evidence |
| --- | --- | --- |
| 1 | Above baseline changes; steps, schemas, datasets, llm, pipeline, cli; new integration fixture/tests | Passing regressions, mock integration artifacts, then one separately identified real local-model transcript run with cited claims and pending human review |
| 2 | Audio/roles; language fixtures, audio configs and benchmark scripts | Actual English/Hindi/Marathi/code-mixed recordings evaluated separately; WER/CER, dose/negation errors, timestamp/role review and latency; model versions recorded |
| 3 | Structured facts and deterministic claim checks; schemas/steps/normalization | Reviewed fact and claim labels; numeric/unit/negation/speaker cases; citation metrics and omission errors |
| 4 | Backend encounter/review/export API, storage and frontend | Upload/paste -> progress -> transcript/citation review -> versioned clinician edits/approval -> JSON/PDF; route and integration tests |
| 5 | Separate protected Plan translation module | hi/mr placeholder integrity and bilingual review, explicit source approval; backend choice remains provisional |
| 6 | Dataset cards and reproducible LoRA experiments; training scripts/configs | Frozen encounter splits, reviewed evidence targets, seed/context/hardware metadata; matched base/adapter runs and all failure counts |
| 7 | Evaluation/ablation framework and paper materials | Same-encounter direct/fact/cited/verified comparisons, factuality/omission labels, confidence intervals where justified, actual tables and error analysis |

Preserve existing intermediate JSON contracts where possible. FastAPI/SQLite and
a React/TypeScript review app are proposed; no service or frontend exists yet.
Use the Sites workflow when actually implementing the website, per session skill
instructions. No database service/account connection is necessary for this audit.

## External dependencies and missing context

Local LLM serving/model weights, actual hardware budget, Hugging Face diarization
model access/token, licensed terminology, authorized multilingual samples, and
clinician-reviewed evaluation labels remain external inputs. The report and PPT
would help reconcile promised deliverables and diagrams; they are not required
to start the baseline. The pasted report descriptions are not evidence that those
features exist. A previous mention of two V100 GPUs is unverified in this session.

Clinical generation, translation and research claims remain unvalidated. Unit
tests establish only the exercised software behavior.

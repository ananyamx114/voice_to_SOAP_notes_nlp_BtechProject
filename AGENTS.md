# Voice-to-SOAP working context

Start with `docs/TEAM_SETUP.md`. Read `docs/REPOSITORY_AUDIT.md`, `docs/DEVELOPMENT_SETUP.md`, and
`docs/PROJECT_GUIDE.md` before implementation. The root PROJECT_GUIDE.md is a
user-provided copy; preserve it. The report/PPT have not yet been supplied.

Preserve the existing evidence-cited pipeline. Keep the ACI-Bench heuristic
four-string SOAP experiment separate from the evidence-cited claim schema;
adapters are task-specific as well as base-checkpoint-specific.

Inspect each computer independently with `python scripts/project.py system`.
Use `python3` on macOS/Linux or `py` on Windows until an environment is activated.
The portable launcher selects this computer's .venv interpreter and sets the
working directory and cache automatically. Set up with `scripts/project.py setup`
and test with `scripts/project.py test` through the appropriate Python command.
Full audio/training dependencies are in requirements.txt and need a separately
verified target environment. Do not download large models as part of unit tests.

Implement one phase at a time, preserving raw transcripts, timestamps, evidence
IDs and experiment artifacts. Use synthetic data for development. Never invent
clinical facts, terminology IDs, metrics, dataset permissions or validation.
Every generated note requires clinician approval, regardless of verifier output.
Clinician approval separation was fixed in 751da5a: all generated notes remain
pending approval, independently of automated verification issues.
Keep credentials, private audio, raw datasets, outputs and
model weights out of Git. Retain ACI-Bench attribution.

The audit is complete. The user subsequently authorized local dataset/model
downloads and local compute. See `docs/LOCAL_BENCHMARKS.md` for current data and
historical benchmark context. Never assume teammates share the original machine's
specifications. Store hardware reports, local configs and dependency snapshots
under ignored .local/ or *.local.yaml, never in shared runtime settings.
BeTraC dev, Eka test and one Sully/PriMock57
sample are under ignored data/raw; never commit the corpora or model weights.
Keep Sully's overlapping ACI-Bench encounters out of independent test claims.
New BeTraC import code preserves dev/validation labels and separates gold
transcripts from audio input. Run model stages sequentially. Gated access terms
must be accepted by the user; no account sharing or credentials in the repository.

Team preference: everyone works on `main`, using separate accounts and local
clones. Do not create task branches unless requested. Fetch/review remote changes
before integrating; preserve local edits, never force-push shared main, and
coordinate changes to the same files. The intended deliverable is the complete
pipeline; implement and validate every stage without claiming completion early.
Before large downloads, full-dataset benchmarks, training or long model runs,
commit and push the reviewed code, dependencies, configs, and run instructions
to main, preserving any newer remote changes. Teammates should be able to pull
the same implementation before each person executes it. Use real model stages;
mock tests and small validation subsets do not replace full pipeline features.

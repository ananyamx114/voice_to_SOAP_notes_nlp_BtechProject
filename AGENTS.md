# Voice-to-SOAP working context

Start with `docs/TEAM_SETUP.md`. Read `docs/REPOSITORY_AUDIT.md`, `docs/DEVELOPMENT_SETUP.md`, and
`docs/PROJECT_GUIDE.md` before implementation. The root PROJECT_GUIDE.md is a
user-provided copy; preserve it. The report/PPT have not yet been supplied.

Preserve the existing evidence-cited pipeline. Keep the ACI-Bench heuristic
four-string SOAP experiment separate from the evidence-cited claim schema;
adapters are task-specific as well as base-checkpoint-specific.

Use `.venv/bin/python`. Install lightweight development dependencies with
`.venv/bin/python -m pip install -r requirements-dev.txt`. Run from the root:
`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`.
Full audio/training dependencies are in requirements.txt and need a separately
verified target environment. Do not download large models as part of unit tests.

Implement one phase at a time, preserving raw transcripts, timestamps, evidence
IDs and experiment artifacts. Use synthetic data for development. Never invent
clinical facts, terminology IDs, metrics, dataset permissions or validation.
Every generated note requires clinician approval, regardless of verifier output.
Never treat `review_required: false` as human approval; fixing that ambiguity is
a first-phase task. Keep credentials, private audio, raw datasets, outputs and
model weights out of Git. Retain ACI-Bench attribution.

The audit is complete. The user subsequently authorized local dataset/model
downloads and local compute. See `docs/LOCAL_BENCHMARKS.md` for current data and
hardware constraints: M1, 8 GB RAM. BeTraC dev, Eka test and one Sully/PriMock57
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

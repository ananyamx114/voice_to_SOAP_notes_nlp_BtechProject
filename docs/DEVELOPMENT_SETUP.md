# Development setup

For current teammate onboarding, use [TEAM_SETUP.md](TEAM_SETUP.md). Diarization
now supports the Hugging Face CLI's saved login as well as an explicit token.

Update: local ASR dependencies, audio datasets and an actual CPU smoke experiment
were subsequently added. See [local benchmarks](LOCAL_BENCHMARKS.md) for current
status. The sections below record the initial lightweight setup; its dependency
snapshot remains useful for tests alone.

Prepared 13 November 2026 for this Apple Silicon workspace. Start with
[the audit and build plan](REPOSITORY_AUDIT.md) and [project guide](PROJECT_GUIDE.md).

## What is ready

The local `.venv` uses Python 3.12.4. Lightweight runtime and test dependencies
are installed. All 72 existing tests pass without pretrained model downloads.
VS Code already has Codex, Python, Pylance, Python Debugger and Jupyter installed.
The repository now provides interpreter/test settings and two VS Code tasks.
Ruff is recommended but optional and was not installed automatically.

Final verification: editable package installation succeeded; all 72 tests passed
again; all three CLI help commands succeeded; VS Code JSON files parsed; pip
dependency checks and `git diff --check` passed. The local model-list probe failed
both inside and outside the network sandbox. The configured inference connection
is not ready. PyTorch, Transformers, Silero, pyannote and PEFT are not installed
in this lightweight environment, and HF_TOKEN is not set.

Codex can already read/edit/run commands in this workspace. Signing in with a
ChatGPT account is a supported Codex authentication route; an application calling
an LLM has its own inference configuration. See [OpenAI authentication guidance](https://learn.chatgpt.com/docs/auth)
and [Codex IDE extension](https://learn.chatgpt.com/docs/codex/ide).
No browser-session or subscription credential is used as the project's LLM key.
Only supplied conversations and available workspace files were reviewed here.

## Recreate the lightweight environment

From the repository root on macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python scripts/check_setup.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

Python >=3.10 is declared; 3.12.4 is the version exercised here. The full ML stack
is not installed. `requirements-dev.txt` is deliberately sufficient for mocked
model tests and real synthetic audio I/O. `requirements.txt` remains the broader
model/audio/training stack and needs platform-specific verification before use.
Do not assume bitsandbytes/CUDA training works on every computer. Use the
portable launcher in TEAM_SETUP.md. Machine-specific dependency snapshots were
removed from shared setup; record local versions under ignored .local/ instead.

Open this folder in VS Code. If an interpreter was previously selected, use
**Python: Select Interpreter** and select `.venv/bin/python`; the configured
default may not replace a previous selection. Run **Tasks: Run Task** and choose
the unit-tests or check-setup task. Existing tests use unittest.

## Model connection remains a separate step

The configuration expects an OpenAI-compatible local server at
`http://localhost:11434/v1`, model `qwen2.5:7b`. The Ollama command was not found
on PATH during the audit. No Ollama installation or model download was performed.
Select the model after checking memory/latency and run a real synthetic encounter
before claiming inference readiness. No paid provider is enabled automatically.

Once your local server and selected model are available:

```bash
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=qwen2.5:7b
.venv/bin/python scripts/check_setup.py --check-llm
```

The probe queries `/models` on local endpoints only and never sends consultation
text. A model-list success is connectivity evidence, not a generation test.
The setup-only check returns zero when development modules are importable; missing
optional models do not make it fail. `--check-llm` also requires a listed model.

`.env.example` is a template. The CLI does **not** automatically load `.env`.
Export variables in the terminal, or copy the example to an ignored `.env` and
explicitly source it in bash. VS Code's python.envFile is for Python tooling and
does not guarantee shell tasks load the file. Never commit tokens.

## Current run commands

```bash
PYTHONPATH=src .venv/bin/python -m voice_to_soap.cli --help
PYTHONPATH=src .venv/bin/python -m voice_to_soap.aci_soap --help
PYTHONPATH=src .venv/bin/python -m voice_to_soap.soap_lora --help
```

The example manifest is header-only. To run inference, create a local CSV with
all six required columns and a synthetic transcript, then use:

```bash
PYTHONPATH=src .venv/bin/python -m voice_to_soap.cli run \
  --manifest data/manifests/my_local_manifest.csv \
  --config configs/base.yaml --outdir data/outputs/my-run
```

Inspect `run.jsonl`: the existing CLI can finish with encounter failures. Inspect
the cited draft, facts and verifier report. Every generated note is pending
clinician approval: `review_required` remains true, while
`automated_review_required` separately reports unresolved verifier issues.
`clinician_approval_status` is `pending` until a future review workflow records
approval. Neither the API nor clinician review UI exists yet.

Whisper/VAD need additional dependencies and weights. Diarization additionally
requires the chosen model's terms and `HF_TOKEN`; it stays disabled. Terminology
and dose tables remain empty templates. Translation remains disabled.

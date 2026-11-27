# Reproduce the completed full Qwen7B SOAP experiments

These scripts publish the DGX experiment implementation, with GPU selection made explicit for each new machine. The numerical training/evaluation protocol is unchanged. Original execution used two V100 32GB cards; the current copies have been syntax-checked and existing software tests pass, but have not been rerun on GPUs after portability edits. They are Linux/CUDA research runners (async mode additionally requires tmux and /proc), not CPU or Windows launchers. All runtime pins, caches and recovery files remain under ignored `.local/` and `data/outputs/`.

Start with [team setup](../../docs/TEAM_SETUP.md), inspect the machine independently, and install the full training dependencies in its own environment. The original environment used Python3.11, PyTorch2.11.0+cu126 and CPU TorchCodec0.11.0+cpu. FP16 and SDPA are used; no quantized model or FlashAttention requirement is introduced. Verify driver/framework compatibility and decoding locally. Do not share environments or credentials. Obtain your own Hugging Face access using the project's `hf-login` command.

Prepare BeTraC-2000 using [its preparation instructions](../../docs/BETRAC_2000_MIGRATION.md). The preparation helper expects the exact private `datasets/betrac_2000/prepared.zip` checksum documented in its source. It verifies existing splits rather than overwriting differences. Its ASR overlap audit additionally expects private restored train/validation manifests under `.local/dgx-migration/restored/data/outputs/asr-lora-experiment-run3/data/`. Those are not distributed on Git; independently prepare the same manifests or arrange an authorized private transfer before this audit. No script downloads patient data implicitly.

From the repository root, after preparing those inputs:

```bash
export HF_HOME="$PWD/.local/huggingface"
python scripts/qwen7b/prepare_soap_dgx.py
python scripts/qwen7b/audit_qwen7b_lengths.py
```

Preparation downloads full Qwen2.5-7B weights at revision `a09a35458c702b33eeacc393d103063234e8bc28` and creates a local pin with that machine's snapshot path. Existing pins are preserved; verify the revision before reproducing these results. The length audit checks all three splits without truncation, but test content is never used in training/checkpoint selection. All 2,000 direct/hybrid encodings fit 4,096 tokens in the recorded experiment.

Choose permitted physical GPU IDs on the current host. Values below are examples, not a claim of exclusive access. The probe performs one disposable real 4,096-token backward/optimizer step and saves no adapter. Start these long jobs in tmux:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export QWEN_TRAIN_GPU=0
export QWEN_VALIDATION_GPU=1
CUDA_VISIBLE_DEVICES="$QWEN_TRAIN_GPU" python scripts/qwen7b/probe_qwen7b_training.py
CUDA_VISIBLE_DEVICES="$QWEN_TRAIN_GPU" python -u scripts/qwen7b/train_qwen7b_soap.py --approach direct
CUDA_VISIBLE_DEVICES="$QWEN_VALIDATION_GPU" python -u scripts/qwen7b/evaluate_qwen7b_resumable.py --stage paired
CUDA_VISIBLE_DEVICES="$QWEN_TRAIN_GPU" python -u scripts/qwen7b/train_qwen7b_soap.py --approach hybrid
```

Training saves recovery checkpoints every100 updates and handles SIGINT/SIGTERM by saving at the next completed optimizer step. Rerun the same command to resume; a completed run is preserved. Generation saves atomic per-encounter records and checks protocol/data hashes before reusing them. Different machine GPU metadata can cause old run-config guards to reject a migrated directory: preserve that directory and review the migration explicitly rather than deleting its guards or overwriting artifacts.

Optional two-card asynchronous HYBRID validation, matching the recorded day2 optimization: while the synchronous HYBRID trainer is running, invoke `python scripts/qwen7b/switch_hybrid_async.py`. It checks the owned trainer PID/cwd, requests a checkpointed pause, verifies a backup of latest/selected checkpoints, then starts one training worker and one validation worker using the two configured GPU IDs. Never run the switch twice or launch duplicate workers. Require `VALIDATION_PARITY_PASSED` before trusting async selection. Training reaching4800 is insufficient: wait for `TRAINING AND VALIDATION COMPLETE` and `completed.json`. The switch is for Linux with tmux and an activated `.venv`.

After HYBRID completion, `CUDA_VISIBLE_DEVICES="$QWEN_VALIDATION_GPU" python scripts/qwen7b/evaluate_qwen7b_resumable.py --stage hybrid` runs its test evaluation. **This has not yet been done in the published experiment.** Use `python scripts/export_soap_interim.py --output .local/latest-soap.json` to export aggregate progress without interrupting jobs. Keep raw predictions, weights, optimizer state, credentials and caches private; report provenance, invalid generations and all measured regressions.

Results: [completed training and available tests](../../results/qwen7b-2026-11-26/README.md). The evidence-cited claim pipeline remains a separate task; these heuristic four-string SOAP adapters must not be substituted into it. Every generated note requires clinician approval.

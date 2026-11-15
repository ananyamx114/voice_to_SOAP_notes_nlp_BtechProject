# Kaggle CLI and ASR checkpoint migration

Use your own Kaggle account. The CLI can submit Python scripts, monitor runs,
and download outputs from a local terminal; it does not turn the local shell
into a remote GPU shell. No notebook rewrite is required.

## Connect the CLI

From the repository root on macOS/Linux:

```bash
python3 -m venv .local/kaggle-cli-venv
.local/kaggle-cli-venv/bin/python -m pip install kaggle six
.local/kaggle-cli-venv/bin/kaggle auth login
.local/kaggle-cli-venv/bin/kaggle datasets list --mine
```

On Windows, use `py -3.12` to create the environment and replace `bin/kaggle`
with `Scripts/kaggle.exe`, and `bin/python` with `Scripts/python.exe`.
Complete the official browser authorization yourself. Credentials are managed
by Kaggle under your home directory, never in the repository or transfer files.
The extra `six` dependency is needed by the tested Kaggle CLI 2.2.5 package.
Do not paste tokens into chat. Kaggle may require account/phone verification
and available accelerator quota before allowing GPU jobs.

Official references: [authentication](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md),
[kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md),
[dataset upload commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).

## Safe transfer order

1. Verify authenticated API access and a short private CUDA/FP16 GPU check.
2. Prepare immutable audio/code assets with the original relative directory
   layout. No credentials, virtual environments, model caches or Git internals
   enter the allowlisted archive. Preserve original references, timestamps,
   source audio hashes, dataset split IDs and attribution. Current training
   uses BeTraC and FLEURS under CC BY 4.0; retain their source references in
   `LOCAL_BENCHMARKS.md` and the preparation metadata. Selected evaluation
   recordings remain private and retain their original terms.
3. Prepare the assets for a **private** Kaggle dataset; never use `--public`.
   Asset upload may proceed while the local checkpoint-boundary pause is pending.
4. Request local pause with `python3 scripts/project.py asr-job pause`.
   Wait until `asr-job status` reports paused: requesting a pause does not mean
   the process has stopped. The controller waits for the next atomic optimizer
   checkpoint (currently every ten updates).
5. Export the stopped training directory separately, including checkpoint,
   validation metrics, adapter selections, epoch archives and session history.
   The snapshot builder refuses to export a running job. Retain local files.
6. Upload the checkpoint snapshot as another private dataset. For training
   continuation, place a `kaggle-asr-run.json` beside its tar/inventory with
   `{"mode": "resume_training", "expected_updates": YOUR_SAVED_UPDATE}`. Attach
   the private assets/checkpoint datasets to a Python script kernel using
   `scripts/kaggle_asr_entry.py`. Enable internet for the pinned public base.
   The update value must be the actual safely saved optimizer count, not the
   last partly processed batch. The entrypoint checks the checkpoint itself.
7. Verify restored hashes, exact optimizer update and epoch/sample offset,
   GPU identity, successful checkpoint load, finite loss and a newly saved CUDA
   optimizer checkpoint. Only then call GPU continuation verified.

Archive preparation, with an activated training environment:

```bash
python scripts/prepare_kaggle_asr.py assets
# Only after local pause is confirmed:
python scripts/prepare_kaggle_asr.py checkpoint
```

Outputs go under ignored `.local/kaggle-asr/transfer/`. Each tar has an inventory
and SHA-256; the destination verifies every restored member. Existing snapshots
are never overwritten. Choose a fresh `--outdir` for a new export.
Kaggle automatically unpacks uploaded tar files into named input folders. The
entry script supports both intact archives and these unpacked inputs, checking
every file against the inventory before training.
The working source tree is packaged directly, including unpublished ASR changes;
this does not commit or push to GitHub.

## What resume preserves

`asr_checkpoint_migration.py` runs on the real CUDA destination and checks the
base/config, unchanged manifest bytes, ordered segment IDs, audio provenance,
split disjointness, trainer source hash, epoch/sample offset and update count.
It pins Transformers/PEFT to the source versions. It records CUDA PyTorch/NumPy
version changes explicitly, preserves the original checkpoint and metadata, and
retains adapter tensors, AdamW moments/steps, gradient scaler, scheduler position,
CPU RNG and training position. MPS and CUDA RNG states are incompatible: CUDA
gets a recorded fresh seed. This is continuation of learned training state,
not bitwise identical continuation across different GPU backends.

Only the adapter is trained. The complete pinned Whisper Large-v3 base remains
FP16, without quantization or a smaller replacement. The current trainer uses
**one CUDA GPU**. T4 x2 does not combine both devices into one memory pool;
distributed training would be a separate implementation and validation task.

## Cloud session persistence

The cloud wrapper calls the unchanged trainer and ends after an atomic checkpoint
when its eight-hour budget is reached, leaving headroom before Kaggle's session
limit. Time spent in a long validation pass can delay that safe boundary.
Saved `resume-state.tar` and its inventory contain training and completed
evaluation artifacts. Download them after each session; attach the previous
kernel output to the next private kernel in addition to the original two input
datasets. Use a new kernel slug for each session so the preceding saved output
can be attached through `kernel_sources`. An actively running notebook's memory
is not a persistent checkpoint.

```bash
kaggle kernels status YOUR_USER/YOUR_SESSION
kaggle kernels output YOUR_USER/YOUR_SESSION -p .local/kaggle-asr/downloaded-session-001
```

After training completes, cloud evaluation runs both base and adapter on CUDA
in a separate output directory. Historical MPS controls remain historical;
they cannot serve as matched CUDA controls. Existing evaluation coverage is
71 selected recordings, not the full corpora. Report regressions as well as
improvements; no WER reduction is guaranteed by more compute.

Do not start local and cloud training simultaneously from the same experiment
checkpoint. The original local stopped run remains a recovery path, but it does
not incorporate subsequent CUDA updates unless explicitly migrated back.
When continuing an existing migration, first inspect the ignored local handoff
record `.local/kaggle-asr/migration.local.json` and query its remote job. A local
`asr-job status` describes the stopped source job, not the cloud continuation.

## Periodic monitoring

The independent local monitor checks every five minutes, follows the **live**
Kaggle log stream, records update/checkpoint/validation progress, and flags WER
and CER regressions separately. A lower loss is never treated as a WER result.
Logged validation scores describe the best checkpoint so far; every epoch's
metrics and paired CUDA comparisons are collected from the final output.
The historical MPS validation baseline is explicitly labeled diagnostic.

```bash
python scripts/monitor_kaggle_asr.py --kernel YOUR_USER/YOUR_SESSION
```

It writes ignored `.local/kaggle-asr/monitor/STATUS.md`, `status.json`,
`history.jsonl`, and `live.log`. After the remote session terminates, it downloads
only `resume-state.tar` and its inventory, verifies archive and individual-file
hashes, and extracts metrics into its JSON report without extracting or loading
model weights. Transient status failures are retried on the next check.
It does not change hyperparameters, relaunch GPU jobs, publish data, or send
messages. Flags require a human/Codex review; it cannot guarantee improvements.

The monitor needs the laptop awake with internet access. Kaggle batch training
continues independently if the local monitor stops or the laptop sleeps.

After session 001 completed all 515 training updates, the cloud entrypoint was
made evaluation recovery the safe default: it requires a `resume-state` input whose
training metadata is complete at update 515. Missing inputs fail before any
training/model loading. Use a private dataset containing the verified resume
archive if Kaggle rejects a failed kernel as a `kernel_sources` dependency.
The evaluation coordinator must use the same `.venv` Python as training so its
metric dependencies are available. The monitor reconnects stale live streams
and reads persisted logs on terminal status before reporting final progress.

To reproduce the earlier training continuation, the explicit `resume_training`
mode above accepts a safely paused checkpoint with its exact declared update
count. It never silently falls back to an older snapshot. Subsequent training
sessions also attach the previous verified `resume-state` archive; if a failed
kernel cannot be used as `kernel_sources`, upload that archive and inventory as
a private dataset instead. The default `evaluate_completed` mode retains the
finished-run update-515 guard. A settings example is
`configs/kaggle-asr-run.example.json`; copy it only into an ignored private
upload folder and name the copy `kaggle-asr-run.json`.

Create each person's `kernel-metadata.json` under ignored `.local/`, using their
own `USER/SLUG`, `kernel_type: script`, the entry script as `code_file`,
`is_private: true`, `enable_gpu: true`, `enable_internet: true`, and their private
dataset IDs in `dataset_sources`. Submit with
`kaggle kernels push -p YOUR_PRIVATE_KERNEL_FOLDER --accelerator NvidiaTeslaT4`.
Check the returned input warnings: missing datasets must be fixed, not ignored.
The repository never contains a teammate's account IDs, login files or tokens.

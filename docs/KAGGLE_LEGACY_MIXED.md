# Kaggle guide: Qwen text-to-SOAP LoRA run

This guide explains how to run the prepared **direct transcript-to-SOAP**
experiment on a Kaggle GPU. It does not run Whisper or the hybrid fact-extraction
approach.

## What this run does

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Training method: LoRA; the base weights are frozen
- Input: English, speaker-labelled consultation transcript
- Target: four SOAP text sections
- Prepared matched split: 338 train, 13 validation, 87 held-out test encounters
- Training configuration: one epoch, batch size 1, learning rate `1e-4`, max
  input length 6,144 tokens, LoRA rank 8, seed 42

The prepared training data combines 67 ACI-Bench training examples and 400
synthetic BeTraC examples. Validation and test are ACI-Bench only. The SOAP
targets were heuristically mapped, not clinician-reviewed ground truth. These
results measure text agreement with the prepared references; they do not
establish clinical correctness or safety. Read
[`LOCAL_BENCHMARKS.md`](LOCAL_BENCHMARKS.md) and the bundled
`preparation_report.json` for source and license details.

The 338/13/87 examples were selected to fit both the direct and hybrid inputs
at the configured sequence length. This guide trains only the direct model
(approach A). Do not use the test set for training or setting selection.

## 1. Create a Kaggle account and notebook

1. Sign in or create your own account at [Kaggle](https://www.kaggle.com/).
2. Complete Kaggle's account/phone verification if it requires it for GPU use.
   Use your own number and never share verification codes.
3. Open [Kaggle Notebooks](https://www.kaggle.com/code) and create a notebook.
4. In notebook settings, select **GPU** (the project was tested on a Tesla T4).
   Turn notebook **Internet** on so Hugging Face can download Qwen.
5. Run this cell and make sure `CUDA available` is `True`:

   ```python
   import torch

   print("CUDA available:", torch.cuda.is_available())
   print("GPU count:", torch.cuda.device_count())
   for i in range(torch.cuda.device_count()):
       p = torch.cuda.get_device_properties(i)
       print(f"GPU {i}: {p.name}, {p.total_memory / 1024**3:.1f} GiB")
   ```

If Kaggle shows two T4s, this does not combine them into one GPU's memory.
Resource availability, GPU type, and session limits can change by account and
over time. Do not start training unless CUDA is available.

## 2. Add the code and data to Kaggle

Use the provided `soap-kaggle-upload.tar.gz` archive. The person preparing the
run can share it with teammates through an approved private channel. It contains
the required experiment code and prepared JSONL splits; it does not contain
Qwen weights.

1. In Kaggle, create a **private Dataset** from the archive and wait until
   creation/processing finishes. Alternatively, add a Kaggle Dataset that
   already contains the archive's expanded files.
2. In the notebook, choose **Add Input** and attach that dataset.
3. Check that the input appears:

   ```python
   from pathlib import Path
   for p in Path("/kaggle/input").rglob("*"):
       if p.is_file() and not p.name.startswith("._"):
           print(p)
   ```

Do not upload identifiable patient information. Keep the Kaggle Dataset private
unless the data's license and intended redistribution explicitly allow public
sharing. Each teammate needs their own Kaggle account/GPU access; sharing a
dataset does not share GPU quota.

## 3. Copy the files into writable notebook storage

Kaggle input datasets are read-only. This cell handles either the uploaded
archive or the expanded directory layout and copies files into
`/kaggle/working/project`:

```python
from pathlib import Path
import json
import shutil
import tarfile

input_root = Path("/kaggle/input")
project = Path("/kaggle/working/project")
project.mkdir(parents=True, exist_ok=True)

archives = list(input_root.rglob("soap-kaggle-upload.tar.gz"))
if archives:
    with tarfile.open(archives[0], "r:gz") as archive:
        archive.extractall(project, filter="data")
else:
    code_files = [
        p for p in input_root.rglob("aci_soap.py")
        if p.parent.name == "voice_to_soap" and p.parent.parent.name == "src"
    ]
    if not code_files:
        raise FileNotFoundError(
            "Could not find soap-kaggle-upload.tar.gz or src/voice_to_soap/aci_soap.py under /kaggle/input"
        )
    source_root = code_files[0].parents[2]
    shutil.copytree(
        source_root / "src",
        project / "src",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("._*"),
    )
    shutil.copytree(
        source_root / "data",
        project / "data",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("._*"),
    )

data_dir = project / "data/processed/aci-betrac-soap-v1/kaggle-matched"
expected = {"train": 338, "validation": 13, "test": 87}
ids = {}
for split, count in expected.items():
    path = data_dir / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split: {path}")
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids[split] = {json.loads(line)["encounter_id"] for line in rows}
    print(f"{split}: {len(rows)} examples (expected {count})")
    assert len(rows) == count, f"Unexpected {split} example count"
    assert len(ids[split]) == count, f"Duplicate encounter IDs in {split}"

assert not (ids["train"] & ids["validation"])
assert not (ids["train"] & ids["test"])
assert not (ids["validation"] & ids["test"])
print("Code/data copied; split counts and leakage checks passed.")
```

## 4. Install and verify training libraries

Do not reinstall PyTorch: use the CUDA-enabled PyTorch supplied by Kaggle.

Install only what is missing, and do not replace Kaggle's CUDA PyTorch:

```python
%pip install -q transformers peft accelerate rouge-score
```

If Kaggle says a kernel restart is needed, restart it, then rerun the GPU check,
the copy/verification cell above, and this import check:

```python
import sys
sys.path.insert(0, "/kaggle/working/project/src")

import torch
import transformers
import peft
import accelerate
from voice_to_soap.aci_soap import train_aci_soap_lora

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("accelerate:", accelerate.__version__)
print("SOAP training code imported.")
assert torch.cuda.is_available()
```

If PEFT fails with an error saying the installed `torchao` version is
incompatible, this project does not need `torchao` for ordinary LoRA. In a
separate cell run:

```python
%pip uninstall -y torchao
```

Restart the kernel, then rerun the GPU check, copy/verification cell, and import
check. Do not blindly upgrade PyTorch or `torchao`; those must match Kaggle's
CUDA environment. Record the package versions printed by the successful import
check.

## 5. Run a tiny smoke test

The smoke test checks that the model loads, one short LoRA training run finishes,
and adapter files can be saved. It is not a model-quality result.

```python
from pathlib import Path
import json
import os
import subprocess
import sys

project = Path("/kaggle/working/project")
data = project / "data/processed/aci-betrac-soap-v1/kaggle-matched"
smoke = Path("/kaggle/working/soap-smoke")
smoke.mkdir(parents=True, exist_ok=True)

for split, count in (("train", 2), ("validation", 1)):
    rows = [
        json.loads(line)
        for line in (data / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    (smoke / f"{split}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows[:count]),
        encoding="utf-8",
    )

env = os.environ.copy()
env["PYTHONPATH"] = str(project / "src")
command = [
    sys.executable, "-m", "voice_to_soap.aci_soap", "train",
    "--train", str(smoke / "train.jsonl"),
    "--validation", str(smoke / "validation.jsonl"),
    "--base-model", "Qwen/Qwen2.5-0.5B-Instruct",
    "--model-revision", "7ae557604adf67be50417f59c2c2f167def9a775",
    "--output-dir", str(smoke / "adapter"),
    "--epochs", "1", "--batch-size", "1", "--learning-rate", "1e-4",
    "--max-seq-length", "6144", "--lora-rank", "8", "--seed", "42",
    "--skip-long-examples",
]
subprocess.run(command, cwd=project, env=env, check=True)
assert (smoke / "adapter/adapter_model.safetensors").is_file()
print("Smoke test passed; adapter saved.")
```

If this fails, do not start the full run. Resolve the specific setup/runtime
error first. A CUDA warning from TensorFlow/JAX does not by itself mean PyTorch
cannot see the GPU; check `torch.cuda.is_available()` and the training metadata.

## 6. Train the direct model

Run this cell **once** after the smoke test passes. It uses 338 train and 13
validation examples; the test split is not read during training.

```python
%cd /kaggle/working/project

!PYTHONPATH=src python -m voice_to_soap.aci_soap train \
  --train data/processed/aci-betrac-soap-v1/kaggle-matched/train.jsonl \
  --validation data/processed/aci-betrac-soap-v1/kaggle-matched/validation.jsonl \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --model-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --output-dir /kaggle/working/qwen05b-direct-run-01/adapter \
  --epochs 1 --batch-size 1 --learning-rate 1e-4 \
  --max-seq-length 6144 --lora-rank 8 --seed 42 \
  --skip-long-examples
```

In the recorded Kaggle T4 run this took about nine minutes, but runtime varies.
The successful run used CUDA and reported no additional examples skipped. Wait
for the final JSON result and confirm it reports `training_device: "cuda"` and
`trained_examples: 338`. If the notebook page looks idle, check whether the cell
is still running; do not click Run again or stop the session while it is active.

## 7. Save the adapter before ending the session

Files in `/kaggle/working` are temporary and can disappear when the session is
stopped. Save the adapter as soon as training finishes:

```python
from pathlib import Path
import shutil
from IPython.display import FileLink, display

adapter = Path("/kaggle/working/qwen05b-direct-run-01/adapter")
assert (adapter / "adapter_model.safetensors").is_file()
archive = shutil.make_archive(
    "/kaggle/working/qwen05b-direct-adapter",
    "gztar",
    root_dir=adapter,
)
display(FileLink(archive))
```

Download the linked archive to your computer and confirm it exists before
stopping the Kaggle session. It contains the trained LoRA adapter and tokenizer
files, not the base Qwen model. Keep a note of the base model revision and
package versions printed earlier.

## 8. Evaluate once on the held-out test set

Only evaluate after training has completed and the adapter is saved. This
evaluates all 87 held-out test examples and writes metrics and per-example
predictions. Generation can take several minutes or longer, and this function
does not print per-example progress. Do not rerun or interrupt the cell merely
because it is quiet.

```python
from pathlib import Path
import sys
sys.path.insert(0, "/kaggle/working/project/src")

from voice_to_soap.aci_soap import evaluate_aci_soap_lora

adapter = Path("/kaggle/working/qwen05b-direct-run-01/adapter")
test_file = Path(
    "/kaggle/working/project/data/processed/aci-betrac-soap-v1/kaggle-matched/test.jsonl"
)
output = Path("/kaggle/working/qwen05b-direct-run-01/evaluation")
assert (adapter / "adapter_model.safetensors").is_file()
assert test_file.is_file()

metrics = evaluate_aci_soap_lora(
    test_path=test_file,
    base_model="Qwen/Qwen2.5-0.5B-Instruct",
    adapter_dir=adapter,
    output_dir=output,
    device="cuda",
    max_new_tokens=1024,
    model_revision="7ae557604adf67be50417f59c2c2f167def9a775",
)
print("Test examples:", metrics["examples"])
print("Valid predictions:", metrics["valid_predictions"])
print("Invalid predictions:", metrics["invalid_predictions"])
print("Mean section ROUGE-L:", metrics["overall_mean_section_rougeL_fmeasure"])
print("Saved:", output / "metrics.json")
print("Saved:", output / "predictions.jsonl")
```

ROUGE-L is text overlap with the prepared heuristic references, not clinical
correctness, factuality, or safety. Metrics and predictions are written only
after the full evaluation loop completes; stopping the cell early will not
produce a complete result. Inspect invalid outputs and a sample of predictions;
do not claim the model is clinically validated.

## 9. Save the final run artifacts

After evaluation completes, create a single download archive containing the
adapter, its metadata, metrics, and predictions:

```python
from pathlib import Path
import shutil
from IPython.display import FileLink, display

run_dir = Path("/kaggle/working/qwen05b-direct-run-01")
archive = shutil.make_archive(
    "/kaggle/working/qwen05b-direct-run-01-results",
    "gztar",
    root_dir=run_dir,
)
display(FileLink(archive))
```

Download and verify the archive before stopping the session. The notebook
account holder should keep it with the dataset preparation report and record
the actual GPU, Python/PyTorch/Transformers/PEFT versions, model revision,
settings, and any errors or skipped examples. Do not publish the data or model
artifacts without checking source licenses and attribution requirements.

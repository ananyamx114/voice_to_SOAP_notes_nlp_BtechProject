"""Export a completed epoch selected using language validation, without retraining."""
import argparse
import json
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from voice_to_soap.asr_component import file_sha256
from voice_to_soap.io_utils import write_json
from voice_to_soap.whisper_lora import validate_adapter


def export_epoch(run_dir, language, output):
    root, target = Path(run_dir), Path(output)
    training = json.loads((root / "training/run.json").read_text(encoding="utf-8"))
    if training["status"] != "complete":
        raise ValueError("Finish the declared training experiment before selecting its epoch")
    candidates = [(int(path.parent.name.split("-")[-1]), json.loads(path.read_text(encoding="utf-8")))
                  for path in (root / "training").glob("epoch-*/metrics.json")]
    epoch, metrics = min(candidates, key=lambda item: (item[1]["by_language"][language]["wer"], item[0]))
    checkpoint_path = root / f"training/epoch-{epoch}/checkpoint.pt"
    if not checkpoint_path.is_file():
        # Compatibility with the first measured run's retained local archives.
        checkpoint_path = root / f"epoch-archives/epoch-{epoch}/checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    updates_per_epoch = (training["data_audit"]["train_segments"] +
                         training["config"]["training"]["gradient_accumulation"] - 1) // training["config"]["training"]["gradient_accumulation"]
    if checkpoint["state"]["updates"] != epoch * updates_per_epoch:
        raise ValueError("Archived weights do not correspond to the selected completed epoch")
    original = root / "training/best-adapter"
    with safe_open(original / "adapter_model.safetensors", framework="pt", device="cpu") as stream:
        if set(stream.keys()) != set(checkpoint["adapter"]):
            raise ValueError("Archived adapter parameter names differ from the trained PEFT adapter")
    target.mkdir(parents=True, exist_ok=False)
    for path in original.iterdir():
        if path.is_file() and path.name not in {"adapter_model.safetensors", "asr_adapter.json"}:
            shutil.copy2(path, target / path.name)
    save_file({key: value.contiguous() for key, value in checkpoint["adapter"].items()},
              target / "adapter_model.safetensors", metadata={"format": "pt"})
    metadata = json.loads((original / "asr_adapter.json").read_text(encoding="utf-8"))
    baseline = metadata["baseline_validation"]
    metadata.update(epoch=epoch, validation=metrics, selection_metric="language_validation_wer",
        selection_language=language, selection_score=metrics["by_language"][language]["wer"],
        selection_rule="Minimum language validation WER across completed epochs; ties choose earlier epoch",
        adapter_sha256=file_sha256(target / "adapter_model.safetensors"),
        source_checkpoint_sha256=file_sha256(checkpoint_path),
        validation_wer_change=metrics["overall"]["wer"] - baseline["overall"]["wer"],
        improved_validation_wer=metrics["overall"]["wer"] < baseline["overall"]["wer"],
        independent_test_evaluated=False)
    write_json(target / "asr_adapter.json", metadata)
    validate_adapter(target, metadata["model_id"], metadata["revision"])
    return {"epoch": epoch, "language": language, "validation_wer": metadata["selection_score"],
            "adapter_sha256": metadata["adapter_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--language", required=True, choices=("en", "hi", "mr"))
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    print(json.dumps(export_epoch(args.run_dir, args.language, args.outdir), indent=2))

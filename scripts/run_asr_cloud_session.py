"""Resume the real Whisper trainer on CUDA; end cleanly at saved checkpoints."""
import argparse
import json
from pathlib import Path
from time import monotonic

from voice_to_soap import whisper_lora
from voice_to_soap.asr_checkpoint_migration import migrate
from voice_to_soap.config import load_config
from voice_to_soap.io_utils import write_json


class SavedSessionBoundary(BaseException):
    """Unwind only after a successful atomic save; trainer finally still runs."""


def run(args):
    config = load_config(args.config)
    root = Path(args.training)
    migration = migrate(root, args.train, args.validation, config)
    print("Backend migration verified:", json.dumps(migration), flush=True)
    initial = json.loads((root / "run.json").read_text())
    if initial["status"] == "complete":
        print("Training is already complete", flush=True)
        return
    started = monotonic()
    original_save = whisper_lora.save_checkpoint

    def saved_boundary(directory, model, optimizer, scaler, state, device):
        original_save(directory, model, optimizer, scaler, state, device)
        print("CLOUD_CHECKPOINT_SAVED", json.dumps(state), flush=True)
        if monotonic() - started >= args.seconds:
            raise SavedSessionBoundary()

    whisper_lora.save_checkpoint = saved_boundary
    try:
        whisper_lora.train(config, args.train, args.validation, root, device="cuda", resume=True)
    except SavedSessionBoundary:
        metadata = json.loads((root / "run.json").read_text())
        metadata.update(status="paused", pause_reason="Cloud session time budget reached after atomic optimizer checkpoint")
        write_json(root / "run.json", metadata)
        print("CLOUD_SESSION_PAUSED_SAFELY", flush=True)
    finally:
        whisper_lora.save_checkpoint = original_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/asr-lora-expanded.yaml")
    parser.add_argument("--training", default="data/outputs/asr-lora-experiment-run2/training")
    parser.add_argument("--train", default="data/outputs/asr-lora-experiment-run2/data/train.jsonl")
    parser.add_argument("--validation", default="data/outputs/asr-lora-experiment-run2/data/validation.jsonl")
    parser.add_argument("--seconds", type=float, default=8 * 3600)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("Session budget must be positive")
    run(args)


if __name__ == "__main__":
    main()

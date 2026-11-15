"""Full-checkpoint Whisper LoRA training and matched held-out validation."""
import argparse
import gc
from contextlib import nullcontext
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
from time import perf_counter, time_ns

import numpy as np
import soundfile as sf

from .asr_component import file_sha256, peak_rss_bytes
from .asr_datasets import _metric_summary
from .asr_training_data import check_disjoint, encode_labels, read_segments
from .config import load_config
from .io_utils import append_jsonl, write_json

TASK = "whisper_asr_lora_v1"


def validate_config(config):
    if config.get("model_id") != "openai/whisper-large-v3":
        raise ValueError("This experiment requires the complete Whisper Large-v3 checkpoint")
    if not re.fullmatch(r"[a-f0-9]{40}", config.get("revision", "")):
        raise ValueError("Pin an immutable base checkpoint revision")
    lora, training = config["lora"], config["training"]
    if training.get("selection_metric", "overall_wer") not in {"overall_wer", "macro_language_wer"}:
        raise ValueError("Unknown validation checkpoint-selection metric")
    if lora["scope"] not in {"encoder_decoder", "decoder"}:
        raise ValueError("LoRA scope must be explicit: encoder_decoder or decoder")
    for value in (lora["rank"], lora["alpha"], training["epochs"],
                  training["gradient_accumulation"], training["early_stopping_patience"]):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("Ranks, epochs, accumulation and patience must be positive integers")
    if not isinstance(training["checkpoint_every_updates"], int) or training["checkpoint_every_updates"] <= 0:
        raise ValueError("Checkpoint interval must be a positive integer")
    for key in ("learning_rate", "max_grad_norm"):
        if not math.isfinite(training[key]) or training[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not 0 <= training["weight_decay"] or not math.isfinite(training["weight_decay"]):
        raise ValueError("Invalid weight decay")
    if not 0 <= lora["dropout"] < 1 or not 0 <= training["warmup_ratio"] < 1:
        raise ValueError("Invalid dropout/warmup ratio")


def attach_lora(model, settings):
    from peft import LoraConfig, get_peft_model

    targets = [name for name, module in model.named_modules()
               if name.endswith(("q_proj", "v_proj"))
               and (settings["scope"] == "encoder_decoder" or ".decoder." in name)]
    if not targets:
        raise ValueError("No Whisper attention projections matched")
    # Generic PEFT wrapper preserves input_features; text-only task wrappers do not.
    model = get_peft_model(model, LoraConfig(r=settings["rank"],
        lora_alpha=settings["alpha"], lora_dropout=settings["dropout"],
        target_modules=targets, bias="none"))
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Only LoRA parameters may be trainable")
    model.config.use_cache = False
    # Non-reentrant checkpointing works with frozen feature-extractor inputs.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, {"target_modules": targets,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters())}


def validate_adapter(path, model_id, revision):
    root = Path(path).resolve()
    metadata = json.loads((root / "asr_adapter.json").read_text(encoding="utf-8"))
    if (metadata.get("task") != TASK or metadata.get("model_id") != model_id
            or metadata.get("revision") != revision):
        raise ValueError("Adapter task/base checkpoint/revision mismatch")
    if metadata.get("status") != "complete":
        raise ValueError("Adapter did not complete training/validation; resource probes are not deployable")
    if file_sha256(root / "adapter_model.safetensors") != metadata.get("adapter_sha256"):
        raise ValueError("Adapter weights changed after validation")
    adapter_config = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
    if adapter_config.get("base_model_name_or_path") != model_id:
        raise ValueError("PEFT base model differs from adapter provenance")
    return metadata


def batch_for(row, processor, maximum, device):
    import torch

    info = sf.info(row["audio_path"])
    if info.samplerate != 16000 or info.channels != 1:
        raise ValueError("Training input must already be 16kHz mono; preserve raw source separately")
    first, last = round(row["start"] * 16000), round(row["end"] * 16000)
    with sf.SoundFile(row["audio_path"]) as audio:
        audio.seek(first)
        waveform = audio.read(last - first, dtype="float32")
    if not len(waveform) or len(waveform) > 480000 or not np.isfinite(waveform).all():
        raise ValueError("Invalid or oversized aligned waveform; refusing truncation")
    features = processor.feature_extractor(waveform, sampling_rate=16000,
        return_tensors="pt", return_attention_mask=True, truncation=False)
    labels = encode_labels(processor.tokenizer, row["text"], row["language"], maximum)
    # Bound decoder graph shapes on MPS without changing supervised tokens.
    # Causal attention prevents ignored future padding from affecting the prefix.
    padded_length = min(maximum, math.ceil(len(labels) / 32) * 32)
    labels = labels + [-100] * (padded_length - len(labels))
    return {"input_features": features.input_features.to(device),
            "attention_mask": features.attention_mask.to(device),
            "labels": torch.tensor([labels], dtype=torch.long, device=device)}


def amp_context(device):
    import torch
    return torch.autocast(device_type=device.split(":")[0], dtype=torch.float16) \
        if device.split(":")[0] in {"cuda", "mps"} else nullcontext()


def evaluate(model, processor, rows, device, output):
    import torch

    model.eval()
    records = []
    for index, row in enumerate(rows, 1):
        # Labels are deliberately removed before calling generation.
        batch = batch_for(row, processor, model.config.max_target_positions, device)
        del batch["labels"]
        language = {"en": "english", "hi": "hindi", "mr": "marathi"}[row["language"]]
        with torch.inference_mode(), amp_context(device):
            tokens = model.generate(**batch, language=language, task="transcribe",
                do_sample=False, num_beams=1, max_new_tokens=440,
                return_timestamps=False, use_cache=True)
        prediction = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        if device.startswith("mps"):
            torch.mps.synchronize()
        del batch, tokens
        if device.startswith("mps"):
            gc.collect()
            torch.mps.empty_cache()
        records.append({"id": row["id"], "encounter_id": row["encounter_id"],
                        "language": row["language"], "reference_transcript": row["text"],
                        "prediction": prediction})
        print(f"  validation {index}/{len(rows)}: {row['id']}", flush=True)
    summary = {"overall": _metric_summary(records),
               "by_language": {lang: _metric_summary([r for r in records if r["language"] == lang])
                               for lang in sorted({r["language"] for r in records})},
               "segments": len(rows), "encounters": len({r["encounter_id"] for r in rows})}
    write_json(Path(output) / "predictions.json", records)
    write_json(Path(output) / "metrics.json", summary)
    return summary


def learning_rate_scale(update, total, warmup_ratio):
    warmup = int(total * warmup_ratio)
    if warmup and update < warmup:
        return (update + 1) / warmup
    return max(0.0, (total - update) / max(1, total - warmup))


def selection_score(metrics, settings):
    if settings.get("selection_metric", "overall_wer") == "macro_language_wer":
        return sum(m["wer"] for m in metrics["by_language"].values()) / len(metrics["by_language"])
    return metrics["overall"]["wer"]


def save_checkpoint(root, model, optimizer, scaler, state, device):
    """One atomically replaced local checkpoint; never serialize the frozen base."""
    import torch
    from peft import get_peft_model_state_dict

    checkpoint = {"adapter": {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "state": state,
        "torch_rng": torch.get_rng_state()}
    if device.startswith("cuda"):
        checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()
    if device.startswith("mps"):
        checkpoint["mps_rng"] = torch.mps.get_rng_state()
    temporary = root / "checkpoint.pt.partial"
    torch.save(checkpoint, temporary)
    temporary.replace(root / "checkpoint.pt")


def archive_epoch_checkpoint(root, epoch):
    """Retain each completed epoch for validation-based selection after training."""
    destination = root / f"epoch-{epoch}" / "checkpoint.pt"
    source = root / "checkpoint.pt"
    if destination.exists():
        if file_sha256(destination) != file_sha256(source):
            raise ValueError("Refusing to overwrite a different completed epoch checkpoint")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.partial")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def train(config, train_path, validation_path, output, device="auto", probe=False, resume=False,
          resume_source_patch=None):
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    from .audio import _resolve_asr_device

    validate_config(config)
    train_rows, validation_rows = read_segments(train_path), read_segments(validation_path)
    audit = check_disjoint(train_rows, validation_rows)
    selected = _resolve_asr_device(device)
    root = Path(output).resolve()
    if probe and resume:
        raise ValueError("Resource probes cannot resume substantive training")
    root.mkdir(parents=True, exist_ok=resume)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dtype = torch.float16 if selected.split(":")[0] in {"mps", "cuda"} else torch.float32
    metadata = {"task": TASK, "status": "running", "model_id": config["model_id"],
        "revision": config["revision"], "config": config, "device": selected,
        "base_precision": str(dtype), "quantization": "none", "probe_only": probe,
        "implementation_sha256": file_sha256(Path(__file__)),
        "runtime_limits": {k: os.environ.get(k) for k in
                           ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO")},
        "decoder_padding_policy": "multiples_of_32; -100 loss mask; no truncation",
        "mps_microbatch_cleanup": True,
        "data_audit": audit, "train_manifest_sha256": file_sha256(Path(train_path)),
        "validation_manifest_sha256": file_sha256(Path(validation_path)),
        "train_ids": [r["id"] for r in train_rows],
        "validation_ids": [r["id"] for r in validation_rows],
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "numpy")},
        "limitations": ["Selected on validation; this is not an independent test result.",
          "WER/CER do not measure clinical correctness; dose/negation review remains required.",
          "English synthetic training does not establish Hindi/Marathi improvement."]}
    if resume:
        previous = json.loads((root / "run.json").read_text(encoding="utf-8"))
        for key in ("task", "config", "device", "probe_only", "train_manifest_sha256",
                    "validation_manifest_sha256", "versions"):
            if previous.get(key) != metadata[key]:
                raise ValueError(f"Resume mismatch in {key}; preserve the old run and choose a new directory")
        if previous.get("implementation_sha256") != metadata["implementation_sha256"]:
            if not resume_source_patch:
                raise ValueError("Resume implementation changed; require an explicitly documented compatible source patch")
            metadata["resume_source_patch"] = {
                "previous_implementation_sha256": previous.get("implementation_sha256"),
                "current_implementation_sha256": metadata["implementation_sha256"],
                "reason": resume_source_patch}
        history = root / "session-history"
        history.mkdir(exist_ok=True)
        write_json(history / f"before-resume-{time_ns()}.json", previous)
        if not (root / "checkpoint.pt").is_file():
            raise ValueError("No completed optimizer checkpoint to resume")
        metadata["resumed_checkpoint_sha256"] = file_sha256(root / "checkpoint.pt")
    write_json(root / "run.json", metadata)
    started = perf_counter()
    try:
        print(f"Loading processor and checking {len(train_rows)} training / {len(validation_rows)} validation segments", flush=True)
        processor = AutoProcessor.from_pretrained(config["model_id"], revision=config["revision"])
        # Validate every target before loading the full model. No skipped/shortened targets.
        label_lengths = []
        for row in train_rows + validation_rows:
            label_lengths.append(len(encode_labels(processor.tokenizer, row["text"], row["language"], 448)))
        metadata["maximum_training_label_tokens"] = max(label_lengths[:len(train_rows)])
        print(f"Loading full {config['model_id']} on {selected}, base dtype={dtype}", flush=True)
        model = WhisperForConditionalGeneration.from_pretrained(config["model_id"],
            revision=config["revision"], torch_dtype=dtype, low_cpu_mem_usage=True,
            use_safetensors=True, attn_implementation="sdpa").to(selected)
        model.generation_config.forced_decoder_ids = None
        model.config.forced_decoder_ids = None
        # Baseline uses identical segments/settings and the unmodified checkpoint.
        if resume and (root / "base-validation/metrics.json").is_file():
            baseline = json.loads((root / "base-validation/metrics.json").read_text(encoding="utf-8"))
        else:
            baseline = None if probe else evaluate(model, processor, validation_rows,
                                                    selected, root / "base-validation")
        model, parameters = attach_lora(model, config["lora"])
        metadata["parameters"] = parameters
        write_json(root / "run.json", metadata)
        print(f"LoRA attached: {parameters['trainable_parameters']:,} trainable / {parameters['total_parameters']:,} total parameters", flush=True)
        if any(p.dtype != torch.float32 for p in model.parameters() if p.requires_grad):
            raise RuntimeError("Trainable adapters must remain float32 for stable scaled updates")
        settings = config["training"]
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
            lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        scaler = torch.amp.GradScaler(selected.split(":")[0],
            enabled=selected.split(":")[0] in {"mps", "cuda"})
        accumulation = settings["gradient_accumulation"]
        total_updates = math.ceil(len(train_rows) / accumulation) * settings["epochs"]
        updates, best_wer, stale, next_epoch, next_offset = 0, float("inf"), 0, 0, 0
        best_metrics = None
        if resume:
            from peft import set_peft_model_state_dict
            checkpoint = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=True)
            set_peft_model_state_dict(model, checkpoint["adapter"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            torch.set_rng_state(checkpoint["torch_rng"])
            if selected.startswith("cuda"):
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
            if selected.startswith("mps"):
                torch.mps.set_rng_state(checkpoint["mps_rng"])
            state = checkpoint["state"]
            updates, best_wer, stale = state["updates"], state["best_wer"], state["stale"]
            best_metrics = state["best_metrics"]
            next_epoch, next_offset = state["next_epoch"], state["next_offset"]
            del checkpoint
            if next_epoch > 0 and next_offset == 0:
                # Recover a crash after checkpoint replacement but before archive.
                archive_epoch_checkpoint(root, next_epoch)
        first_adapter = next(p for n, p in model.named_parameters() if "lora_B" in n)
        original_adapter = first_adapter.detach().cpu().clone()
        for epoch in range(next_epoch, 1 if probe else settings["epochs"]):
            if stale >= settings["early_stopping_patience"]:
                break
            model.train()
            indices = list(range(len(train_rows)))
            if probe:
                # Test the largest decoder targets before committing to a long run.
                indices.sort(key=lambda i: label_lengths[i], reverse=True)
            else:
                random.Random(seed + epoch).shuffle(indices)
            for start in range(next_offset if epoch == next_epoch else 0, len(indices), accumulation):
                group = indices[start:start + accumulation]
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for microstep, index in enumerate(group, 1):
                    micro_started = perf_counter()
                    batch = batch_for(train_rows[index], processor, model.config.max_target_positions, selected)
                    with amp_context(selected):
                        loss = model(**batch, use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite loss; stopping without publishing an adapter")
                    loss_value = loss.detach().float().item()
                    loss_sum += loss_value
                    scaler.scale(loss / len(group)).backward()
                    if selected.startswith("mps"):
                        torch.mps.synchronize()
                    del loss, batch
                    if selected.startswith("mps"):
                        gc.collect()
                        torch.mps.empty_cache()
                    print(f"  update {updates+1}, microbatch {microstep}/{len(group)}: "
                          f"{train_rows[index]['id']}, loss={loss_value:.5f}, "
                          f"{perf_counter()-micro_started:.1f}s", flush=True)
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                    settings["max_grad_norm"], error_if_nonfinite=True)
                lr = settings["learning_rate"] * learning_rate_scale(updates, total_updates,
                                                                       settings["warmup_ratio"])
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = lr
                scaler.step(optimizer)
                scaler.update()
                updates += 1
                if selected.startswith("mps"):
                    torch.mps.synchronize()
                if selected.startswith("cuda"):
                    torch.cuda.synchronize()
                event = {"epoch": epoch + 1, "update": updates, "loss": loss_sum / len(group),
                         "gradient_norm": float(norm), "learning_rate": lr,
                         "elapsed_seconds": perf_counter() - started,
                         "peak_process_rss_bytes": peak_rss_bytes()}
                if selected.startswith("mps"):
                    event["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
                if selected.startswith("cuda"):
                    event["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                append_jsonl(root / "training.jsonl", event)
                print(f"epoch {epoch+1} update {updates}: loss={event['loss']:.5f}", flush=True)
                if probe:
                    if torch.equal(original_adapter, first_adapter.detach().cpu()):
                        raise RuntimeError("Resource probe did not change adapter weights")
                    metadata.update(status="resource_probe_passed", optimizer_updates=updates,
                                    adapter_changed=True)
                    return metadata
                if updates % settings["checkpoint_every_updates"] == 0 or start + len(group) == len(indices):
                    save_checkpoint(root, model, optimizer, scaler,
                        {"updates": updates, "best_wer": best_wer, "stale": stale,
                         "best_metrics": best_metrics,
                         "next_epoch": epoch, "next_offset": start + len(group)}, selected)
            metrics = evaluate(model, processor, validation_rows, selected, root / f"epoch-{epoch+1}")
            wer = selection_score(metrics, settings)
            if wer < best_wer:
                best_wer, stale = wer, 0
                best_metrics = metrics
                adapter = root / "best-adapter"
                model.save_pretrained(adapter, safe_serialization=True)
                processor.save_pretrained(adapter)
                adapter_metadata = {"task": TASK, "status": "complete",
                    "model_id": config["model_id"], "revision": config["revision"],
                    "adapter_sha256": file_sha256(adapter / "adapter_model.safetensors"),
                    "epoch": epoch + 1, "validation": metrics, "baseline_validation": baseline,
                    "selection_metric": settings.get("selection_metric", "overall_wer"),
                    "selection_score": best_wer,
                    "validation_wer_change": metrics["overall"]["wer"] - baseline["overall"]["wer"],
                    "improved_validation_wer": metrics["overall"]["wer"] < baseline["overall"]["wer"],
                    "training_manifest_sha256": metadata["train_manifest_sha256"],
                    "validation_manifest_sha256": metadata["validation_manifest_sha256"],
                    "config": config, "independent_test_evaluated": False}
                write_json(adapter / "asr_adapter.json", adapter_metadata)
            else:
                stale += 1
            write_json(root / f"epoch-{epoch+1}" / "training-state.json",
                       {"optimizer_updates": updates, "best_validation_selection_score": best_wer})
            save_checkpoint(root, model, optimizer, scaler,
                {"updates": updates, "best_wer": best_wer, "stale": stale,
                 "best_metrics": best_metrics,
                 "next_epoch": epoch + 1, "next_offset": 0}, selected)
            archive_epoch_checkpoint(root, epoch + 1)
            if stale >= settings["early_stopping_patience"]:
                break
        metadata.update(status="complete", optimizer_updates=updates,
            best_validation_wer=best_metrics["overall"]["wer"], best_validation_selection_score=best_wer,
            best_validation_metrics=best_metrics,
            baseline_validation_wer=baseline["overall"]["wer"],
            validation_wer_change=best_metrics["overall"]["wer"] - baseline["overall"]["wer"],
            improved_validation_wer=best_metrics["overall"]["wer"] < baseline["overall"]["wer"])
        return metadata
    except Exception as exc:
        metadata.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        metadata.update(elapsed_seconds=perf_counter() - started, peak_process_rss_bytes=peak_rss_bytes())
        write_json(root / "run.json", metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/asr-lora.yaml")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--probe", action="store_true", help="One full-model optimizer update; not a trained deliverable")
    parser.add_argument("--resume", action="store_true", help="Resume matching local adapter/optimizer/RNG checkpoint")
    parser.add_argument("--resume-source-patch", help="Explicit reason for a compatible code correction; all other resume guards remain")
    parser.add_argument("--audit-only", action="store_true", help="Validate sources/splits without loading models")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    if args.audit_only:
        print(json.dumps(check_disjoint(read_segments(args.train), read_segments(args.validation)), indent=2))
        return
    train(config, args.train, args.validation, args.outdir, args.device, args.probe, args.resume,
          args.resume_source_patch)


if __name__ == "__main__":
    main()

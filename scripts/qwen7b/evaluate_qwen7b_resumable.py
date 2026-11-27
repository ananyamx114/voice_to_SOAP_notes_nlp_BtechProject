"""Paired full-test SOAP generation with atomic per-encounter recovery and failure-aware scoring."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".local/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def write(path, value):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def summarize(records):
    sections = ("subjective", "objective", "assessment", "plan")
    valid = [r for r in records if r["prediction_soap"] is not None]
    return {"examples": len(records), "valid_predictions": len(valid),
            "invalid_predictions": len(records) - len(valid),
            "by_section_rougeL_all_examples": {
                s: sum(r["section_rougeL"][s] for r in records) / len(records) for s in sections},
            "overall_mean_section_rougeL_all_examples": sum(r["mean_section_rougeL"] for r in records) / len(records),
            "overall_mean_section_rougeL_valid_only": (
                sum(r["mean_section_rougeL"] for r in valid) / len(valid) if valid else None),
            "generation_seconds": sum(r["seconds"] for r in records),
            "invalid_outputs_scored_as_zero": True,
            "clinical_factuality_measured": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("paired", "hybrid"), default="paired")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Explicitly select one permitted GPU"
    import torch
    from transformers import set_seed
    from rouge_score import rouge_scorer
    from voice_to_soap.aci_soap import (AciSoapLoRAGenerator, InvalidGeneratedSoapError,
                                       SOAP_SECTIONS, load_aci_soap_examples)
    assert torch.cuda.device_count() == 1
    destination = ROOT / "data/outputs/dgx-qwen7b-soap-evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / "evaluation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pin = json.loads((ROOT / ".local/dgx-migration/qwen7b-model-pin.json").read_text())
    dataset = ROOT / "data/processed/betrac_2000/test.jsonl"
    manifest = json.loads((dataset.parent / "split_manifest.json").read_text())
    data_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert data_sha == manifest["sha256"]["test"]
    examples = load_aci_soap_examples(dataset)
    assert len(examples) == 200
    ids = [e["encounter_id"] for e in examples]
    assert ids == manifest["encounter_ids"]["test"] and len(set(ids)) == 200
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    summaries = {}
    for stage in (("base", "direct") if args.stage == "paired" else ("hybrid",)):
        adapter = None
        adapter_sha = None
        if stage != "base":
            run = ROOT / "data/outputs/dgx-qwen7b-soap" / stage
            complete = json.loads((run / "completed.json").read_text())
            assert complete["global_step"] == 4800 and complete["approach"] == stage
            assert complete["base_model"] == pin["model_id"] and complete["model_revision"] == pin["revision"]
            adapter = run / "selected-adapter"
            adapter_sha = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
        folder = destination / stage
        records_dir = folder / "encounters"
        records_dir.mkdir(parents=True, exist_ok=True)
        config = {"model_id": pin["model_id"], "revision": pin["revision"], "stage": stage,
                  "adapter_sha256": adapter_sha, "test_sha256": data_sha,
                  "max_new_tokens": 1024, "do_sample": False, "seed": 42,
                  "precision": "float16_base", "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "examples": 200,
                  "scope": "text-only internal BeTraC test; not clinical validation",
                  "known_test_encounters_in_asr_training": 2}
        if (folder / "run_config.json").exists():
            assert json.loads((folder / "run_config.json").read_text()) == config, "Evaluation protocol changed"
        else:
            write(folder / "run_config.json", config)
        generator = None
        records = []
        for index, example in enumerate(examples, 1):
            encounter = example["encounter_id"]
            assert Path(encounter).name == encounter
            output = records_dir / (encounter + ".json")
            if output.exists():
                record = json.loads(output.read_text())
                assert record["encounter_id"] == encounter and record["stage"] == stage
            else:
                if generator is None:
                    set_seed(42)
                    generator = AciSoapLoRAGenerator(pin["model_id"], adapter,
                        device="cuda:0", model_revision=pin["revision"])
                facts = None
                if stage == "hybrid":
                    from voice_to_soap.clinical_facts import extract_clinical_facts
                    facts = extract_clinical_facts(example["dialogue"])
                torch.cuda.synchronize()
                started = time.monotonic()
                prediction = None
                error = None
                raw = None
                try:
                    prediction = generator.generate(example["dialogue"], max_new_tokens=1024, facts=facts)
                except InvalidGeneratedSoapError as exc:
                    error, raw = str(exc), exc.raw_output
                torch.cuda.synchronize()
                scores = {s: scorer.score(example["soap"][s], prediction[s])["rougeL"].fmeasure
                          if prediction is not None else 0.0 for s in SOAP_SECTIONS}
                record = {"encounter_id": encounter, "stage": stage, "prediction_soap": prediction,
                          "reference_soap": example["soap"], "dialogue": example["dialogue"],
                          "generation_error": error, "invalid_raw_output": raw,
                          "section_rougeL": scores, "mean_section_rougeL": sum(scores.values()) / 4,
                          "seconds": time.monotonic() - started, "clinician_approval": "pending"}
                write(output, record)
            records.append(record)
            print(f"{stage}: {index}/200; valid={record['prediction_soap'] is not None}", flush=True)
        summary = {**config, **summarize(records)}
        write(folder / "metrics.json", summary)
        summaries[stage] = summary
        if generator is not None:
            del generator
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps(summary, indent=2), flush=True)
    if args.stage == "paired":
        write(destination / "paired-comparison.json", {
            "base": summaries["base"], "direct": summaries["direct"],
            "direct_minus_base_rougeL": summaries["direct"]["overall_mean_section_rougeL_all_examples"]
                - summaries["base"]["overall_mean_section_rougeL_all_examples"],
            "test_used_for_training_or_checkpoint_selection": False})
    print("EVALUATION COMPLETE: all requested encounters saved.", flush=True)


if __name__ == "__main__":
    main()

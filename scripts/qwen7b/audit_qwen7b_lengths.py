"""Measure unchanged DIRECT/HYBRID SOAP encodings using the downloaded 7B tokenizer."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from transformers import AutoTokenizer
from voice_to_soap.aci_soap import _encode_example, load_aci_soap_examples

local = ROOT / ".local/dgx-migration"
pin = json.loads((local / "qwen7b-model-pin.json").read_text())
tokenizer = AutoTokenizer.from_pretrained(pin["snapshot_path"], local_files_only=True)
data = ROOT / "data/processed/betrac_2000"
manifest = json.loads((data / "split_manifest.json").read_text())
report = {"model_id": pin["model_id"], "revision": pin["revision"],
          "sequence_limit": 4096, "truncation_used": False, "training_started": False, "splits": {}}
for split, count in (("train", 1600), ("validation", 200), ("test", 200)):
    path = data / (split + ".jsonl")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"][split]
    rows = load_aci_soap_examples(path)
    assert len(rows) == count
    report["splits"][split] = {}
    for approach in ("direct", "hybrid"):
        # A high ceiling here measures length without truncation; the run limit remains 4096.
        lengths = [len(_encode_example(tokenizer, row, sys.maxsize, approach)["input_ids"]) for row in rows]
        report["splits"][split][approach] = {
            "examples": len(lengths), "min_tokens": min(lengths), "max_tokens": max(lengths),
            "over_4096": sum(length > 4096 for length in lengths)}
        print(split, approach, report["splits"][split][approach], flush=True)
report["all_fit"] = all(m["over_4096"] == 0 for split in report["splits"].values() for m in split.values())
(local / "qwen7b-length-audit.json").write_text(json.dumps(report, indent=2) + "\n")
print("All examples fit without truncation:", report["all_fit"], flush=True)
if not report["all_fit"]:
    raise SystemExit("Training remains stopped: review sequence lengths before configuring the run.")

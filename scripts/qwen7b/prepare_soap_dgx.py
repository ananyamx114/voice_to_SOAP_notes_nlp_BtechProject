"""Verify/extract existing SOAP splits, audit ASR overlap, download pinned Qwen weights."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / ".local/dgx-migration"
LOCAL.mkdir(parents=True, exist_ok=True)
PREFIX = "data/processed/betrac_2000/"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    archive = ROOT / "datasets/betrac_2000/prepared.zip"
    expected = "4692124f4fc7c45258d5697fcc24c142165f0cc79c4d6ae560407dc721eb6665"
    assert digest(archive) == expected, "SOAP archive hash mismatch"
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names))
        for info in z.infolist():
            path = PurePosixPath(info.filename)
            assert info.filename.startswith(PREFIX) and not path.is_absolute() and ".." not in path.parts
            assert ((info.external_attr >> 16) & 0o170000) != 0o120000
            if not info.is_dir() and (ROOT / info.filename).exists():
                assert (ROOT / info.filename).read_bytes() == z.read(info), "Existing SOAP file differs; preserved"
        manifest = json.loads(z.read(PREFIX + "split_manifest.json"))
        selection = json.loads(z.read(PREFIX + "selection_report.json"))
        assert selection["status"] == "complete"
        ids = {}
        for split, count in (("train", 1600), ("validation", 200), ("test", 200)):
            content = z.read(PREFIX + split + ".jsonl")
            assert hashlib.sha256(content).hexdigest() == manifest["sha256"][split]
            rows = [json.loads(line) for line in content.splitlines() if line.strip()]
            ordered = [r["encounter_id"] for r in rows]
            assert len(rows) == len(set(ordered)) == count
            assert ordered == manifest["encounter_ids"][split]
            assert all(r["assigned_split"] == split for r in rows)
            ids[split] = set(ordered)
        assert not (ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["test"] & ids["validation"])
        for info in z.infolist():
            target = ROOT / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(z.read(info))

    asr_ids = {}
    for split in ("train", "validation"):
        path = LOCAL / "restored/data/outputs/asr-lora-experiment-run3/data" / (split + ".jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        asr_ids[split] = {r["encounter_id"] for r in rows if r.get("dataset") == "BeTraC/betrac-2026"}
    overlaps = {f"soap_{soap}_asr_{asr}": len(ids[soap] & asr_ids[asr])
                for soap in ids for asr in asr_ids}
    report = {"soap_dataset": "BeTraC-2000", "verified_counts": {k: len(v) for k, v in ids.items()},
              "asr_encounter_overlap_counts": overlaps,
              "token_lengths_need_recheck_for_7b": True,
              "soap_encounter_audio_downloaded_by_this_script": False,
              "training_started": False}
    (LOCAL / "soap-data-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)

    os.environ.setdefault("HF_HOME", str(ROOT / ".local/huggingface"))
    from huggingface_hub import HfApi, snapshot_download
    pin_file = LOCAL / "qwen7b-model-pin.json"
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    if pin_file.exists():
        pin = json.loads(pin_file.read_text())
        assert pin["model_id"] == model_id
        assert pin["revision"] == "a09a35458c702b33eeacc393d103063234e8bc28", "Existing pin differs from published experiment"
    else:
        pin = {"model_id": model_id, "revision": "a09a35458c702b33eeacc393d103063234e8bc28"}
        pin_file.write_text(json.dumps(pin, indent=2) + "\n")
    print("Downloading full Qwen weights at pinned revision: " + pin["revision"], flush=True)
    location = snapshot_download(repo_id=model_id, revision=pin["revision"],
                                 allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja", "LICENSE*", "README*"])
    pin["snapshot_path"] = location
    pin_file.write_text(json.dumps(pin, indent=2) + "\n")
    print("Qwen2.5-7B-Instruct download complete. No model loaded or training started.", flush=True)


if __name__ == "__main__":
    main()

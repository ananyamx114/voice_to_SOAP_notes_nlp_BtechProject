import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts.prepare_kaggle_asr import archive, digest, manifest_assets


class MigrationArchiveTests(unittest.TestCase):
    def test_cloud_settings_default_to_completed_evaluation_and_validate_explicit_mode(self):
        with patch.dict(os.environ):
            from scripts import kaggle_asr_entry as entry
        with tempfile.TemporaryDirectory() as tmp, patch.object(entry, "INPUT", Path(tmp)):
            self.assertEqual(entry.run_settings(), {"mode": "evaluate_completed", "expected_updates": 515})
            path = Path(tmp) / "kaggle-asr-run.json"
            path.write_text(json.dumps({"mode": "resume_training", "expected_updates": 90}))
            self.assertEqual(entry.run_settings()["expected_updates"], 90)
            path.write_text(json.dumps({"mode": "resume_training", "expected_updates": True}))
            with self.assertRaisesRegex(ValueError, "positive expected_updates"):
                entry.run_settings()

    def test_cloud_training_resume_requires_explicit_paused_checkpoint_position(self):
        with patch.dict(os.environ):
            from scripts.kaggle_asr_entry import validate_source
        validate_source({"status": "paused", "optimizer_updates": 90},
                        {"mode": "resume_training", "expected_updates": 90})
        with self.assertRaisesRegex(ValueError, "Source update count"):
            validate_source({"status": "paused", "optimizer_updates": 90},
                            {"mode": "resume_training", "expected_updates": 515})
        with self.assertRaisesRegex(ValueError, "safely paused"):
            validate_source({"status": "running", "optimizer_updates": 90},
                            {"mode": "resume_training", "expected_updates": 90})

    def test_evaluation_resume_refuses_old_or_partial_training(self):
        with patch.dict(os.environ):
            from scripts.kaggle_asr_entry import require_completed_training
        for state in ({"status": "paused", "optimizer_updates": 90},
                      {"status": "complete", "optimizer_updates": 90},
                      {"status": "running", "optimizer_updates": 515}):
            with self.assertRaisesRegex(ValueError, "completed update-515"):
                require_completed_training(state)
        require_completed_training({"status": "complete", "optimizer_updates": 515})

    def test_preserves_relative_paths_and_checks_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "data").mkdir()
            audio = root / "audio.flac"
            audio.write_bytes(b"test fixture")
            manifest = root / "data/train.jsonl"
            row = {"audio_path": "../audio.flac", "source_sha256": digest(audio)}
            manifest.write_text(json.dumps(row) + "\n")
            files = manifest_assets(manifest, root)
            self.assertEqual(files, {audio, manifest})
            result = archive(files, root / "transfer.tar", root)
            self.assertEqual(result["sha256"], digest(root / "transfer.tar"))
            with tarfile.open(root / "transfer.tar") as saved:
                self.assertEqual(saved.extractfile("data/train.jsonl").read(), manifest.read_bytes())
            with self.assertRaises(FileExistsError):
                archive(files, root / "transfer.tar", root)
            audio.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "provenance"):
                manifest_assets(manifest, root)

    def test_rejects_outside_repo_and_absolute_audio_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "train.jsonl"
            source.write_text(json.dumps({"audio_path": "/private/credentials"}))
            with self.assertRaisesRegex(ValueError, "absolute"):
                manifest_assets(source, root)

    def test_cloud_loader_accepts_tar_and_kaggle_unpacked_inputs(self):
        with patch.dict(os.environ):
            from scripts import kaggle_asr_entry as entry
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "source"
            source.mkdir()
            (source / "sample").write_bytes(b"preserve these exact bytes")
            inputs = root / "input"
            inputs.mkdir()
            archive({source / "sample"}, inputs / "assets.tar", source)
            destination = root / "working"
            destination.mkdir()
            with patch.object(entry, "WORK", destination), patch.object(entry, "INPUT", inputs):
                self.assertTrue(entry.restore("assets"))
                self.assertEqual((destination / "sample").read_bytes(), (source / "sample").read_bytes())
                with tarfile.open(inputs / "assets.tar") as saved:
                    saved.extractall(inputs / "assets", filter="data")
                (inputs / "assets.tar").unlink()
                (destination / "sample").unlink()
                self.assertTrue(entry.restore("assets"))
                self.assertEqual((destination / "sample").read_bytes(), (source / "sample").read_bytes())
                (inputs / "assets/sample").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    entry.restore("assets")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional training dependencies")
    def test_conversion_keeps_all_learned_and_position_state(self):
        import torch
        from voice_to_soap.asr_checkpoint_migration import convert_checkpoint
        checkpoint = {"adapter": {"lora": torch.randn(2, 2)},
                      "optimizer": {"state": {0: {"step": torch.tensor(80), "exp_avg": torch.randn(2, 2)}}},
                      "scaler": {"scale": 65536},
                      "state": {"updates": 80, "next_epoch": 0, "next_offset": 1280},
                      "torch_rng": torch.get_rng_state(), "mps_rng": torch.tensor([1, 2], dtype=torch.uint8)}
        converted = convert_checkpoint(checkpoint, [torch.get_rng_state()])
        for key in checkpoint:
            self.assertIs(converted[key], checkpoint[key])
        self.assertNotIn("cuda_rng", checkpoint)
        self.assertIn("cuda_rng", converted)
        with self.assertRaisesRegex(ValueError, "original MPS"):
            convert_checkpoint(converted, [torch.get_rng_state()])

    @unittest.skipUnless(importlib.util.find_spec("peft"), "Optional training dependencies")
    def test_session_budget_pauses_only_after_successful_save(self):
        from scripts import run_asr_cloud_session as session
        from voice_to_soap.io_utils import write_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "run.json", {"status": "paused"})
            saved = []
            def original_save(*args):
                saved.append(True)
            def training(*args, **kwargs):
                # Mirror trainer finally: the wrapper must finalize paused after it.
                try:
                    session.whisper_lora.save_checkpoint(root, None, None, None, {"updates": 90}, "cuda")
                    self.fail("Training continued after its safe session boundary")
                finally:
                    write_json(root / "run.json", {"status": "running"})
            args = SimpleNamespace(config="unused", training=root, train="train", validation="val", seconds=1)
            with patch.object(session, "migrate", return_value={}), \
                 patch.object(session, "load_config", return_value={}), \
                 patch.object(session, "monotonic", side_effect=[0, 2]), \
                 patch.object(session.whisper_lora, "save_checkpoint", side_effect=original_save) as save, \
                 patch.object(session.whisper_lora, "train", side_effect=training):
                session.run(args)
                self.assertIs(session.whisper_lora.save_checkpoint, save)
            self.assertEqual(saved, [True])
            self.assertEqual(json.loads((root / "run.json").read_text())["status"], "paused")


if __name__ == "__main__":
    unittest.main()

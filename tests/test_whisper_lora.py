import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from voice_to_soap.asr_alignment import ctc_path, transcript_units
from voice_to_soap.asr_component import file_sha256
from voice_to_soap.asr_comparison import compare
from voice_to_soap.asr_training_data import check_disjoint, encode_labels, read_segments
from voice_to_soap.whisper_lora import TASK, attach_lora, save_checkpoint, validate_adapter


class AsrSupervisionTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Optional dataset preparation dependencies")
    def test_duration_selection_requires_full_target_and_is_reproducible(self):
        from scripts.prepare_fleurs_asr import select_candidates
        rows = [{"num_samples": 16000 * seconds, "raw_transcription": text}
                for seconds, text in [(4, "one"), (5, "two"), (6, "three"), (40, "long"), (3, "")]]
        selected, rejected = select_candidates(rows, 42, seconds=10)
        self.assertEqual(selected, select_candidates(rows, 42, seconds=10)[0])
        self.assertGreaterEqual(sum(rows[index]["num_samples"] / 16000 for index in selected), 10)
        self.assertTrue(selected <= {0, 1, 2})
        with self.assertRaisesRegex(ValueError, "exhausted"):
            select_candidates(rows, 42, seconds=16)
        with self.assertRaisesRegex(ValueError, "exhausted"):
            select_candidates(rows, 42, seconds=10, accepts=lambda row: row["raw_transcription"] == "one")

    def row(self, root):
        sf.write(root / "audio.wav", np.zeros(32000), 16000)
        return {"id": "encounter-1-001", "encounter_id": "encounter-1", "group_id": "family-1",
            "dataset": "synthetic-test", "dataset_revision": "a" * 40, "split": "train",
            "audio_path": "audio.wav", "source_sha256": file_sha256(root / "audio.wav"),
            "start": 0, "end": 2, "text": "नमस्ते", "language": "hi",
            "alignment": {"method": "human_reviewed", "status": "accepted"}}

    def test_alignment_source_hash_and_boundary_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.row(root)
            path = root / "segments.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(read_segments(path)[0]["text"], "नमस्ते")
            row["end"] = 31
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "never truncate"):
                read_segments(path)
            row["end"], row["source_sha256"] = 2, "0" * 64
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source hash"):
                read_segments(path)

    def test_rejects_eka_training_and_unaccepted_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.row(root)
            path = root / "segments.jsonl"
            row["dataset"] = "ekacare/eka-medical-asr-evaluation-dataset"
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evaluation-only"):
                read_segments(path)
            row["dataset"] = "synthetic-test"
            row["alignment"]["status"] = "rejected"
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "accepted"):
                read_segments(path)

    def test_whole_clip_labels_cannot_be_assigned_to_an_arbitrary_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.row(root)
            row["alignment"] = {"method": "dataset_clip_pair", "status": "accepted"}
            path = root / "segments.jsonl"
            path.write_text(json.dumps(row), encoding="utf-8")
            self.assertEqual(len(read_segments(path)), 1)
            row["start"] = 0.5
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete source clip"):
                read_segments(path)

    def test_detects_group_leakage_across_different_encounter_ids(self):
        train = {"id": "train1", "encounter_id": "a", "group_id": "same-family",
                 "source_sha256": "hash1", "split": "train", "start": 0, "end": 1, "text": "hello"}
        val = {**train, "id": "val1", "encounter_id": "b", "source_sha256": "hash2", "split": "validation"}
        with self.assertRaisesRegex(ValueError, "group_id"):
            check_disjoint([train], [val])
        val["group_id"] = "other-family"
        self.assertEqual(check_disjoint([train], [val])["shared_segment_texts"], 1)
        val["split"] = "test"
        with self.assertRaisesRegex(ValueError, "non-validation"):
            check_disjoint([train], [val])

    def test_labels_remove_decoder_start_but_keep_language_and_task(self):
        class Tokenizer:
            def set_prefix_tokens(self, **kwargs):
                self.prefix = kwargs
            def convert_tokens_to_ids(self, token):
                return 10
            def __call__(self, text, truncation):
                from types import SimpleNamespace
                assert truncation is False
                return SimpleNamespace(input_ids=[10, 11, 12, 13, 20, 2])
        tokenizer = Tokenizer()
        self.assertEqual(encode_labels(tokenizer, "hello", "en", 6), [11, 12, 13, 20, 2])
        self.assertEqual(tokenizer.prefix["task"], "transcribe")
        with self.assertRaisesRegex(ValueError, "never truncate"):
            encode_labels(tokenizer, "hello", "en", 3)

    def test_text_units_do_not_create_audio_times_or_keep_stage_directions(self):
        text = "DOCTOR: [sigh] Hello.\\n\\n How are you?\nPATIENT: I'm *fine*."
        units = transcript_units(text, 4)
        self.assertEqual(" ".join(units), "Hello. How are you? I'm fine.")
        self.assertEqual(transcript_units("one two .", 2), ["one two ."])

    def test_ctc_repeated_letters_require_blank_and_complete_reference(self):
        logits = np.full((5, 3), -20.0, dtype=np.float32)
        logits[np.arange(5), [0, 1, 0, 1, 0]] = 0
        spans = ctc_path(logits, [1, 1])
        self.assertEqual([(s[0], s[1]) for s in spans], [(1, 1), (3, 3)])
        with self.assertRaisesRegex(ValueError, "too few"):
            ctc_path(logits[:2], [1, 1])
        with self.assertRaisesRegex(ValueError, "memory budget"):
            ctc_path(logits, [1, 1], max_cells=2)

    def test_ctc_path_matches_exhaustive_best_alignment(self):
        scores = np.random.default_rng(42).normal(size=(5, 3)).astype(np.float32)
        spans = ctc_path(scores, [1, 2])
        predicted_path = np.zeros(5, dtype=int)
        for token, (first, last, _) in zip((1, 2), spans):
            predicted_path[first:last + 1] = token
        cost = scores[np.arange(5), predicted_path].sum()
        best = -np.inf
        for path in itertools.product(range(3), repeat=5):
            collapsed = [token for i, token in enumerate(path)
                         if token and (i == 0 or token != path[i - 1])]
            if collapsed == [1, 2]:
                best = max(best, scores[np.arange(5), path].sum())
        self.assertAlmostEqual(float(cost), float(best), places=5)

    def test_adapter_loader_rejects_soap_or_other_revision_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = {"task": "cited_soap", "model_id": "openai/whisper-large-v3", "revision": "a" * 40}
            (root / "asr_adapter.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                validate_adapter(root, metadata["model_id"], metadata["revision"])

    def test_paired_comparison_reports_regression_and_rejects_changed_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {"encounter_id": "visit-1", "status": "complete", "audio_sha256": "same-source",
                "reference_transcript": "take two tablets", "prediction": "take two tablets",
                "language": "en", "decoded_regions": [{"start_seconds": 0, "end_seconds": 1}],
                "seed": 42, "model": {"model_id": "openai/whisper-large-v3", "revision": "a" * 40,
                    "precision": "float32", "device": "cpu", "generation": {"num_beams": 1},
                    "algorithm": "sequential", "timestamps": "word", "adapter": None}}
            for name in ("base", "adapted"):
                directory = root / name
                (directory / "visit-1").mkdir(parents=True)
                (directory / "summary.json").write_text(json.dumps({"requested": 1, "completed": 1,
                    "failed": 0, "not_processed": 0, "encounters": [{"encounter_id": "visit-1"}]}), encoding="utf-8")
                if name == "adapted":
                    row = {**row, "prediction": "take ten tablets",
                           "model": {**row["model"], "adapter": {"sha256": "adapter"}}}
                (directory / "visit-1/asr.json").write_text(json.dumps(row), encoding="utf-8")
            report = compare(root / "base", root / "adapted", root / "comparison.json")
            self.assertGreater(report["by_language"]["en"]["wer_change"], 0)
            self.assertFalse(report["by_language"]["en"]["improved_wer"])
            row["audio_sha256"] = "different-source"
            (root / "adapted/visit-1/asr.json").write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "audio_sha256"):
                compare(root / "base", root / "adapted", root / "comparison.json")


@unittest.skipUnless(importlib.util.find_spec("peft") and importlib.util.find_spec("torch"),
                     "Requires the optional ASR training profile; no model downloads")
class RealPeftContractTests(unittest.TestCase):
    def test_selected_timestamp_heads_match_original_hf_dtw_exactly(self):
        import torch
        from transformers import WhisperConfig, WhisperForConditionalGeneration
        from transformers.generation.utils import GenerateEncoderDecoderOutput
        from voice_to_soap.whisper_asr import select_timestamp_heads_before_concatenation

        config = WhisperConfig(vocab_size=32, num_mel_bins=4, d_model=8,
            encoder_layers=1, decoder_layers=3, encoder_attention_heads=2,
            decoder_attention_heads=2, encoder_ffn_dim=16, decoder_ffn_dim=16,
            max_source_positions=8, max_target_positions=16, pad_token_id=0,
            eos_token_id=2, decoder_start_token_id=1, median_filter_width=3)
        model = WhisperForConditionalGeneration(config)
        torch.manual_seed(42)
        attention = tuple(tuple(torch.rand(2, 2, length, 12) for _ in range(3))
                          for length in (2, 1, 1, 1))
        outputs = GenerateEncoderDecoderOutput(sequences=torch.ones(2, 6, dtype=torch.long),
                                                cross_attentions=attention)
        heads = [(2, 1), (0, 0), (2, 0)]
        cases = (None, 24, [24, 20], torch.tensor([24, 20]))
        expected = [model._extract_token_timestamps(outputs, heads, num_frames=frames,
                                                    num_input_ids=2) for frames in cases]
        select_timestamp_heads_before_concatenation(model)
        for frames, reference in zip(cases, expected):
            actual = model._extract_token_timestamps(outputs, heads, num_frames=frames, num_input_ids=2)
            self.assertTrue(torch.equal(actual, reference))
        self.assertEqual(model.config.decoder_layers, 3)
        self.assertIs(outputs.cross_attentions, attention)

    def test_validation_generation_never_receives_reference_labels(self):
        import torch
        from types import SimpleNamespace
        from voice_to_soap.whisper_lora import evaluate

        class Tokenizer:
            def set_prefix_tokens(self, **kwargs):
                pass
            def convert_tokens_to_ids(self, token):
                return 1
            def __call__(self, text, truncation):
                return SimpleNamespace(input_ids=[1, 5, 6, 2])
            def batch_decode(self, tokens, skip_special_tokens):
                return ["नमस्ते"]
        class Processor:
            tokenizer = Tokenizer()
            def feature_extractor(self, waveform, **kwargs):
                return SimpleNamespace(input_features=torch.ones(1, 4, 16),
                                       attention_mask=torch.ones(1, 16, dtype=torch.long))
        class Model:
            config = SimpleNamespace(max_target_positions=16)
            def eval(self):
                return self
            def generate(self, **kwargs):
                if "labels" in kwargs or "decoder_input_ids" in kwargs or "decoder_attention_mask" in kwargs:
                    raise AssertionError("Reference supervision leaked into generation")
                return torch.tensor([[5, 6, 2]])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = AsrSupervisionTests().row(root)
            row["audio_path"] = str(root / row["audio_path"])
            metrics = evaluate(Model(), Processor(), [row], "cpu", root / "out")
            self.assertEqual(metrics["overall"]["wer"], 0)

    def test_model_loading_cannot_advance_the_decoding_rng(self):
        import torch
        from voice_to_soap.whisper_asr import preserve_loading_rng

        torch.manual_seed(1234)
        original = torch.get_rng_state().clone()
        with preserve_loading_rng("cpu"):
            torch.rand(200)
        self.assertTrue(torch.equal(original, torch.get_rng_state()))
        with self.assertRaises(RuntimeError):
            with preserve_loading_rng("cpu"):
                torch.rand(200)
                raise RuntimeError("loading failed")
        self.assertTrue(torch.equal(original, torch.get_rng_state()))

    def test_interrupted_training_restores_adapter_optimizer_and_rng(self):
        import torch
        from types import SimpleNamespace
        from transformers import WhisperConfig, WhisperForConditionalGeneration
        from voice_to_soap.config import load_config
        from voice_to_soap.io_utils import write_json
        from voice_to_soap.whisper_lora import train

        model_config = WhisperConfig(vocab_size=32, num_mel_bins=4, d_model=8,
            encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
            decoder_attention_heads=2, encoder_ffn_dim=16, decoder_ffn_dim=16,
            max_source_positions=8, max_target_positions=16, pad_token_id=0,
            eos_token_id=2, decoder_start_token_id=1)
        base = WhisperForConditionalGeneration(model_config)
        initial_state = {n: p.detach().clone() for n, p in base.state_dict().items()}
        def load_model(*args, **kwargs):
            model = WhisperForConditionalGeneration(model_config)
            model.load_state_dict(initial_state)
            return model
        class Tokenizer:
            def set_prefix_tokens(self, **kwargs):
                pass
            def convert_tokens_to_ids(self, token):
                return 1
            def __call__(self, text, truncation):
                return SimpleNamespace(input_ids=[1, 5, 6, 2])
        class Processor:
            tokenizer = Tokenizer()
            def feature_extractor(self, waveform, **kwargs):
                return SimpleNamespace(input_features=torch.ones(1, 4, 16),
                                       attention_mask=torch.ones(1, 16, dtype=torch.long))
            def save_pretrained(self, directory):
                pass
        def synthetic_metrics(model, processor, rows, device, output):
            # Isolate recovery from generation: these are test fixtures, not scores.
            metrics = {"overall": {"wer": 0.4 if hasattr(model, "peft_config") else 0.5}}
            metrics["by_language"] = {"hi": dict(metrics["overall"])}
            write_json(Path(output) / "metrics.json", metrics)
            return metrics
        def interrupt_after_save(*args, **kwargs):
            save_checkpoint(*args, **kwargs)
            raise RuntimeError("simulated interruption after atomic checkpoint")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = AsrSupervisionTests().row(root)
            row["end"] = 1
            another = {**row, "id": "train-second", "start": 1, "end": 2}
            validation = {**row, "id": "val", "encounter_id": "val", "group_id": "val",
                          "split": "validation", "audio_path": "val.wav"}
            sf.write(root / "val.wav", np.ones(32000) * 0.1, 16000)
            validation["source_sha256"] = file_sha256(root / "val.wav")
            (root / "train.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(another), encoding="utf-8")
            (root / "validation.jsonl").write_text(json.dumps(validation), encoding="utf-8")
            settings = load_config("configs/asr-lora.yaml")
            settings["lora"].update(rank=2, alpha=4, dropout=0.2)
            settings["training"].update(epochs=2, gradient_accumulation=1, checkpoint_every_updates=1)
            with patch("transformers.AutoProcessor.from_pretrained", return_value=Processor()), \
                 patch("transformers.WhisperForConditionalGeneration.from_pretrained", side_effect=load_model), \
                 patch("voice_to_soap.whisper_lora.evaluate", side_effect=synthetic_metrics):
                train(settings, root / "train.jsonl", root / "validation.jsonl", root / "uninterrupted", "cpu")
                with patch("voice_to_soap.whisper_lora.save_checkpoint", side_effect=interrupt_after_save):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        train(settings, root / "train.jsonl", root / "validation.jsonl", root / "recovered", "cpu")
                result = train(settings, root / "train.jsonl", root / "validation.jsonl",
                               root / "recovered", "cpu", resume=True)
                with patch("voice_to_soap.whisper_lora.archive_epoch_checkpoint",
                           side_effect=RuntimeError("interrupted before epoch archive")):
                    with self.assertRaisesRegex(RuntimeError, "before epoch archive"):
                        train(settings, root / "train.jsonl", root / "validation.jsonl",
                              root / "boundary-recovered", "cpu")
                train(settings, root / "train.jsonl", root / "validation.jsonl",
                      root / "boundary-recovered", "cpu", resume=True)
                # A new phase starts from selected weights, not the old optimizer/step.
                source = root / "uninterrupted/best-adapter"
                adapter_config = json.loads((source / "adapter_config.json").read_text())
                adapter_config["base_model_name_or_path"] = settings["model_id"]
                write_json(source / "adapter_config.json", adapter_config)
                warm = train(settings, root / "train.jsonl", root / "validation.jsonl",
                             root / "warm-start", "cpu", initial_adapter=source)
                self.assertTrue(warm["initial_adapter"]["optimizer_reset"])
                selected = json.loads((root / "warm-start/best-adapter/asr_adapter.json").read_text())
                self.assertEqual(selected["epoch"], 0)
                self.assertEqual(selected["adapter_sha256"], file_sha256(source / "adapter_model.safetensors"))
                self.assertEqual(warm["optimizer_updates"], 4)
            self.assertEqual(result["status"], "complete")
            continuous = torch.load(root / "uninterrupted/checkpoint.pt", weights_only=True)
            recovered = torch.load(root / "recovered/checkpoint.pt", weights_only=True)
            self.assertEqual(recovered["state"], continuous["state"])
            for name in continuous["adapter"]:
                self.assertTrue(torch.equal(continuous["adapter"][name], recovered["adapter"][name]), name)
            boundary = torch.load(root / "boundary-recovered/checkpoint.pt", weights_only=True)
            for name in continuous["adapter"]:
                self.assertTrue(torch.equal(continuous["adapter"][name], boundary["adapter"][name]), name)
            self.assertTrue((root / "boundary-recovered/epoch-1/checkpoint.pt").is_file())
            first = torch.load(root / "uninterrupted/epoch-1/checkpoint.pt", weights_only=True)
            last = torch.load(root / "uninterrupted/epoch-2/checkpoint.pt", weights_only=True)
            self.assertEqual(first["state"]["updates"], 2)
            self.assertEqual(last["state"]["updates"], 4)
            self.assertTrue(any(not torch.equal(first["adapter"][name], last["adapter"][name])
                                for name in first["adapter"]))
            from voice_to_soap.whisper_lora import archive_epoch_checkpoint
            with self.assertRaisesRegex(ValueError, "overwrite"):
                archive_epoch_checkpoint(root / "uninterrupted", 1)

    def test_real_whisper_forward_backward_freezes_base_and_adapter_roundtrips(self):
        import torch
        from peft import PeftModel
        from transformers import WhisperConfig, WhisperForConditionalGeneration

        # Random tiny model verifies architecture/API contracts, not ASR quality.
        config = WhisperConfig(vocab_size=32, num_mel_bins=4, d_model=8,
            encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
            decoder_attention_heads=2, encoder_ffn_dim=16, decoder_ffn_dim=16,
            max_source_positions=8, max_target_positions=16, pad_token_id=0,
            bos_token_id=1, eos_token_id=2, decoder_start_token_id=1)
        torch.manual_seed(42)
        base = WhisperForConditionalGeneration(config)
        initial = {n: p.detach().clone() for n, p in base.named_parameters()}
        base_state = {n: p.detach().clone() for n, p in base.state_dict().items()}
        model, counts = attach_lora(base, {"rank": 2, "alpha": 4, "dropout": 0,
                                           "scope": "encoder_decoder"})
        self.assertGreater(counts["trainable_parameters"], 0)
        self.assertTrue(any("encoder" in n for n in counts["target_modules"]))
        self.assertTrue(any("decoder" in n for n in counts["target_modules"]))
        features = torch.randn(1, 4, 16)
        labels = torch.tensor([[5, 6, 2]])
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        loss = model(input_features=features, labels=labels).loss
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n, p in model.named_parameters() if "lora_B" in n))
        optimizer.step()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                original_name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                self.assertTrue(torch.equal(parameter, initial[original_name]), name)
        model.eval()
        with torch.no_grad():
            expected = model(input_features=features, labels=labels).logits
            # Bucketing target shapes must neither train on padding nor alter
            # the actual prefix targets through causal attention.
            padded = torch.tensor([[5, 6, 2, -100, -100, -100, -100, -100]])
            original_loss = model(input_features=features, labels=labels).loss
            padded_result = model(input_features=features, labels=padded)
            self.assertTrue(torch.allclose(original_loss, padded_result.loss, atol=1e-6))
            self.assertTrue(torch.allclose(expected, padded_result.logits[:, :3], atol=1e-6))
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp, safe_serialization=True)
            fresh = WhisperForConditionalGeneration(config)
            fresh.load_state_dict(base_state)
            loaded = PeftModel.from_pretrained(fresh, tmp).eval()
            with torch.no_grad():
                actual = loaded(input_features=features, labels=labels).logits
            self.assertTrue(torch.allclose(expected, actual, atol=1e-6))
            scaler = torch.amp.GradScaler("cpu", enabled=False)
            save_checkpoint(Path(tmp), model, optimizer, scaler,
                {"updates": 1, "next_epoch": 0, "next_offset": 1, "best_wer": float("inf"), "stale": 0}, "cpu")
            checkpoint = torch.load(Path(tmp) / "checkpoint.pt", weights_only=True)
            self.assertEqual(checkpoint["state"]["updates"], 1)
            self.assertTrue(checkpoint["optimizer"]["state"])
            self.assertTrue(all("lora_" in n for n in checkpoint["adapter"]))
            self.assertFalse((Path(tmp) / "checkpoint.pt.partial").exists())

    def test_probe_exercises_training_loop_and_is_not_a_deployable_adapter(self):
        import torch
        from types import SimpleNamespace
        from transformers import WhisperConfig, WhisperForConditionalGeneration
        from voice_to_soap.config import load_config
        from voice_to_soap.whisper_lora import train

        config = WhisperConfig(vocab_size=32, num_mel_bins=4, d_model=8,
            encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
            decoder_attention_heads=2, encoder_ffn_dim=16, decoder_ffn_dim=16,
            max_source_positions=8, max_target_positions=16, pad_token_id=0,
            eos_token_id=2, decoder_start_token_id=1)
        class Tokenizer:
            def set_prefix_tokens(self, **kwargs):
                pass
            def convert_tokens_to_ids(self, token):
                return 1
            def __call__(self, text, truncation):
                return SimpleNamespace(input_ids=[1, 5, 6, 2])
        class Processor:
            tokenizer = Tokenizer()
            def feature_extractor(self, waveform, **kwargs):
                # Deterministic synthetic fixture; never used in runtime code.
                return SimpleNamespace(input_features=torch.ones(1, 4, 16),
                                       attention_mask=torch.ones(1, 16, dtype=torch.long))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_row = AsrSupervisionTests().row(root)
            val_row = {**train_row, "id": "validation-1", "encounter_id": "val-1",
                       "group_id": "val-family", "split": "validation", "audio_path": "val.wav"}
            sf.write(root / "val.wav", np.ones(32000) * 0.1, 16000)
            val_row["source_sha256"] = file_sha256(root / "val.wav")
            for name, row in (("train", train_row), ("validation", val_row)):
                (root / (name + ".jsonl")).write_text(json.dumps(row), encoding="utf-8")
            settings = load_config("configs/asr-lora.yaml")
            settings["training"]["gradient_accumulation"] = 1
            with patch("transformers.AutoProcessor.from_pretrained", return_value=Processor()), \
                 patch("transformers.WhisperForConditionalGeneration.from_pretrained",
                       return_value=WhisperForConditionalGeneration(config)):
                report = train(settings, root / "train.jsonl", root / "validation.jsonl",
                               root / "out", "cpu", probe=True)
            self.assertEqual(report["status"], "resource_probe_passed")
            self.assertTrue(report["adapter_changed"])
            self.assertEqual(report["optimizer_updates"], 1)
            self.assertFalse((root / "out/best-adapter").exists())


if __name__ == "__main__":
    unittest.main()

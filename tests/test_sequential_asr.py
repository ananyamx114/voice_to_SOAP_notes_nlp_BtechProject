import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from voice_to_soap.whisper_asr import sequential_transcribe, speech_context_regions, omit_unused_encoder_attention_outputs
from voice_to_soap.asr_component import run_component, ResumeMismatchError


OPTIONS = {"mode": "sequential", "revision": "a" * 40,
           "generation": {"condition_on_prev_tokens": False}}


class SequentialAsrTests(unittest.TestCase):
    def test_encoder_memory_change_preserves_inputs_and_decoder(self):
        encoder = SimpleNamespace(forward=Mock(return_value="unchanged encoder result"))
        original = encoder.forward
        decoder = SimpleNamespace(forward=Mock())
        model = SimpleNamespace(get_encoder=lambda: encoder, decoder=decoder)
        omit_unused_encoder_attention_outputs(model)
        self.assertEqual(encoder.forward("features", output_attentions=True), "unchanged encoder result")
        original.assert_called_once_with("features", output_attentions=False)
        decoder.forward.assert_not_called()

    @patch("voice_to_soap.whisper_asr.version", return_value="test-version")
    @patch("voice_to_soap.audio._resolve_asr_device", return_value="mps")
    @patch("voice_to_soap.whisper_asr.load_sequential_pipeline")
    def test_native_longform_and_pinned_full_precision_selection(self, load, resolve, version):
        load.return_value = Mock(return_value={"text": "hello", "chunks": [
            {"text": "hello", "timestamp": (0.1, 0.5)}]})
        result = sequential_transcribe(np.ones(16000), 16000, "full-checkpoint", "auto", "hindi", OPTIONS)
        load.assert_called_once_with("full-checkpoint", "a" * 40, "mps", "float16")
        kwargs = load.return_value.call_args.kwargs
        self.assertNotIn("chunk_length_s", kwargs)
        self.assertEqual(kwargs["generate_kwargs"]["task"], "transcribe")
        self.assertEqual(kwargs["generate_kwargs"]["language"], "hindi")
        self.assertFalse(kwargs["generate_kwargs"]["condition_on_prev_tokens"])
        self.assertEqual(result["metadata"]["quantization"], "none")
        self.assertIn("monitor_progress", kwargs["generate_kwargs"])
        json.dumps(result) # Runtime progress callback must not leak into saved metadata.

    def test_unpinned_and_translation_overrides_fail_before_loading(self):
        with self.assertRaisesRegex(ValueError, "immutable"):
            sequential_transcribe(np.ones(16000), 16000, "model", "cpu", None, {"revision": "main"})
        with patch("voice_to_soap.audio._resolve_asr_device", return_value="cpu"):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                sequential_transcribe(np.ones(16000), 16000, "model", "cpu", None,
                    {**OPTIONS, "generation": {"task": "translate"}})

    def test_vad_context_preserves_short_pauses_and_original_offsets(self):
        regions = speech_context_regions([
            {"start_seconds": 1, "end_seconds": 2},
            {"start_seconds": 3, "end_seconds": 4},
            {"start_seconds": 10, "end_seconds": 12}], 15, 2)
        self.assertEqual(regions, [{"start_seconds": 1, "end_seconds": 4},
                                  {"start_seconds": 10, "end_seconds": 12}])
        with self.assertRaisesRegex(ValueError, "No VAD"):
            speech_context_regions([], 15)
        with self.assertRaisesRegex(ValueError, "outside"):
            speech_context_regions([{"start_seconds": 0, "end_seconds": 20}], 15)

    @patch("voice_to_soap.asr_component.sequential_transcribe")
    def test_component_writes_full_outputs_scores_and_safe_resume(self, transcribe):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sf.write(root / "input.wav", np.ones(16000, dtype=np.float32) * 0.1, 16000)
            manifest = root / "manifest.csv"
            manifest.write_text('encounter_id,audio_path,transcript,reference_note,language,split,reference_transcript\nvisit-1,input.wav,,,en,test,hello\n')
            config = {"audio": {"asr_model_id": "model", "asr_options": OPTIONS}}
            transcribe.return_value = {"text": "hello", "chunks": [{"text": "hello",
                "start": 0.1, "end": 0.5, "confidence": None}], "warnings": [],
                "metadata": {"model_id": "model", "precision": "float32"}}
            with patch.dict("sys.modules", {"torch": SimpleNamespace(manual_seed=Mock())}):
                summary = run_component(config, [str(manifest)], str(root / "out"), "cpu")
                self.assertEqual(summary["completed"], 1)
                self.assertEqual(summary["overall"]["wer"], 0)
                args = transcribe.call_args.args
                self.assertTrue(isinstance(args[0], np.ndarray))
                self.assertNotIn("reference_transcript", str(args[2:]))
                run_component(config, [str(manifest)], str(root / "out"), "cpu", resume=True)
                self.assertEqual(transcribe.call_count, 1)
                before = (root / "out/visit-1/asr.json").read_text()
                manifest.write_text(manifest.read_text().replace(',hello\n', ',changed\n'))
                with self.assertRaises(ResumeMismatchError):
                    run_component(config, [str(manifest)], str(root / "out"), "cpu", resume=True)
                self.assertEqual((root / "out/visit-1/asr.json").read_text(), before)

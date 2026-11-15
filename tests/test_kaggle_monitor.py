import json
from pathlib import Path
import tempfile
import unittest

from scripts.monitor_kaggle_asr import paired_flags, progress, verify_download
from scripts.prepare_kaggle_asr import archive


class KaggleMonitorTests(unittest.TestCase):
    def test_evaluation_recovery_reports_restored_training_and_new_case_progress(self):
        saved = {"updates": 515, "next_epoch": 5, "next_offset": 0,
                 "saved_validation_epochs": [1, 2, 3, 4, 5]}
        state = progress("Starting stage: betrac_base\n[1/2] ASR fixture-id (en)\n", {}, saved)
        self.assertEqual(state["latest_update"], 515)
        self.assertEqual(state["completed_validation_epochs"], [1, 2, 3, 4, 5])
        self.assertEqual(state["latest_evaluation_stage"], "betrac_base")
        self.assertEqual(state["latest_asr_record"], ["1", "2", "fixture-id"])

    def test_stream_replay_does_not_regress_known_update(self):
        text = ('epoch 5 update 515: loss=0.1\n' +
                'CLOUD_CHECKPOINT_SAVED ' + json.dumps({"updates": 515, "next_epoch": 5, "next_offset": 0}) + '\n' +
                'epoch 2 update 187: loss=0.3\n' +
                'CLOUD_CHECKPOINT_SAVED ' + json.dumps({"updates": 180, "next_epoch": 1, "next_offset": 1232}) + '\n')
        state = progress(text, {})
        self.assertEqual(state["latest_update"], 515)
        self.assertEqual(state["last_saved_checkpoint"]["updates"], 515)

    def test_final_cuda_regressions_are_flagged_separately(self):
        result = {"by_language": {"mr": {"base": {"wer": .8, "cer": .2},
                                          "adapter": {"wer": .7, "cer": .25}}}}
        self.assertEqual(paired_flags({"path/fleurs-comparison.json": result}),
                         ["Paired CUDA test regression: fleurs/mr CER"])

    def test_flags_cer_regression_even_when_wer_improves(self):
        baseline = {"by_language": {"en": {"wer": .1, "cer": .05}}}
        best = {"overall": {"wer": .08, "cer": .06},
                "by_language": {"en": {"wer": .08, "cer": .06}}}
        state = {"updates": 103, "next_epoch": 1, "next_offset": 0, "best_metrics": best}
        later = {**state, "updates": 110, "next_epoch": 1, "next_offset": 112}
        text = ('epoch 2 update 110: loss=0.321\n  validation 4/239: fixture\n' +
                'CLOUD_CHECKPOINT_SAVED ' + json.dumps(state) + '\n' +
                'CLOUD_CHECKPOINT_SAVED ' + json.dumps(later) + '\n')
        result = progress(text, baseline)
        self.assertEqual(result["latest_update"], 110)
        self.assertEqual(result["completed_validation_epochs"], [1])
        self.assertEqual(result["best_validation"], best)
        self.assertEqual(result["flags"], ["en best-validation CER exceeds historical MPS base"])
        self.assertIn("not every epoch", result["comparison_limit"])

    def test_incomplete_checkpoint_line_is_not_a_saved_checkpoint(self):
        result = progress('CLOUD_CHECKPOINT_SAVED {"updates": 100', {})
        self.assertEqual(result["last_saved_checkpoint"], {})
        self.assertIsNone(result["best_validation"])

    def test_downloaded_archive_is_verified_before_claiming_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"synthetic test checkpoint")
            metric = root / "data/outputs/asr-lora-experiment-run2/training/epoch-1/metrics.json"
            metric.parent.mkdir(parents=True)
            metric.write_text(json.dumps({"fixture": True}))
            archive({checkpoint, metric}, root / "resume-state.tar", root)
            result = verify_download(root)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["files_verified"], 2)
            self.assertEqual(len(result["epoch_validation_metrics"]), 1)
            with (root / "resume-state.tar").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_download(root)


if __name__ == "__main__":
    unittest.main()

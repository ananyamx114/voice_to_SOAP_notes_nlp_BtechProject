import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "export_soap_interim", Path(__file__).resolve().parents[1] / "scripts/export_soap_interim.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExportSoapInterimTests(unittest.TestCase):
    def fixture(self, root):
        pin = root / ".local/dgx-migration/qwen7b-model-pin.json"
        pin.parent.mkdir(parents=True)
        pin.write_text(json.dumps({"model_id": "Qwen/Qwen2.5-7B-Instruct", "revision": "example",
                                  "snapshot_path": "/private/machine/path"}))

    def test_completed_metrics_export_only_whitelisted_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            folder = root / "data/outputs/dgx-qwen7b-soap-evaluation/base"
            folder.mkdir(parents=True)
            (folder / "metrics.json").write_text(json.dumps({"examples": 200,
                "valid_predictions": 196, "invalid_predictions": 4,
                "overall_mean_section_rougeL_all_examples": .23,
                "dialogue": "PRIVATE_TRANSCRIPT", "adapter_dir": "/private/adapter",
                "encounter_ids": ["PRIVATE_ENCOUNTER"]}))
            report = module.export(root)
            self.assertEqual(report["evaluation"]["base"]["status"], "complete")
            self.assertEqual(report["evaluation"]["base"]["invalid_predictions"], 4)
            rendered = json.dumps(report)
            self.assertNotIn("PRIVATE_", rendered)
            self.assertNotIn("/private/", rendered)
            self.assertEqual(report["evaluation"]["direct"]["status"], "not_started")

    def test_partial_evaluation_keeps_failures_in_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            folder = root / "data/outputs/dgx-qwen7b-soap-evaluation/direct/encounters"
            folder.mkdir(parents=True)
            for i, (prediction, score) in enumerate((({}, 1.0), (None, 0.0))):
                (folder / f"example{i}.json").write_text(json.dumps({"stage": "direct",
                    "prediction_soap": prediction, "mean_section_rougeL": score}))
            report = module.export(root)["evaluation"]["direct"]
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["examples"], 2)
            self.assertEqual(report["invalid_predictions"], 1)
            self.assertEqual(report["overall_mean_section_rougeL_all_examples"], .5)

    def test_async_validation_overrides_stale_trainer_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            folder = root / "data/outputs/dgx-qwen7b-soap/hybrid"
            checkpoint = folder / "checkpoint-1400"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text(json.dumps({
                "global_step": 1400, "epoch": .875, "best_metric": 1.0,
                "best_model_checkpoint": "checkpoint-1200", "log_history": []}))
            (folder / "async-validation-summary.json").write_text(json.dumps({
                "best_validation_loss": .5, "selected_checkpoint_step": 1300,
                "last_evaluated_update": 1300, "validation_history": [
                    {"step": 1300, "epoch": .8125, "eval_loss": .5, "eval_runtime": 1}]}))
            result = module.export(root)["training"]["hybrid"]
            self.assertEqual(result["saved_updates"], 1400)
            self.assertEqual(result["last_validated_update"], 1300)
            self.assertEqual(result["best_validation_loss"], .5)
            self.assertEqual(result["selected_checkpoint_step"], 1300)


if __name__ == "__main__":
    unittest.main()

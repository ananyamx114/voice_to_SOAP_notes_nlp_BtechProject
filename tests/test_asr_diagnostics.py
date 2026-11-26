import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("diagnose_asr", Path(__file__).resolve().parents[1] / "scripts/diagnose_asr.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DiagnosticSelectionTests(unittest.TestCase):
    def test_subset_is_seeded_language_balanced_and_group_distinct(self):
        rows = [{"language": l, "group_id": f"{l}-{g}", "id": f"{l}-{g}-{speaker}"}
                for l in ("en", "hi", "mr") for g in range(8) for speaker in range(2)]
        selected = module.subset(rows, 4, 42)
        self.assertEqual(selected, module.subset(rows, 4, 42))
        self.assertEqual(len(selected), 12)
        for language in ("en", "hi", "mr"):
            self.assertEqual(len({r["group_id"] for r in selected if r["language"] == language}), 4)

    def test_insufficient_distinct_groups_cannot_be_silently_repeated(self):
        with self.assertRaises(ValueError):
            module.subset([{"language": "mr", "group_id": "one"}] * 10, 4, 42)


if __name__ == "__main__":
    unittest.main()

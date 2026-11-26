from collections import Counter
import random
import unittest

from voice_to_soap.whisper_lora import epoch_indices


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"language": lang} for lang, n in (("en", 21), ("hi", 12), ("mr", 10)) for _ in range(n)]

    def test_uniform_policy_preserves_historical_shuffle_exactly(self):
        expected = list(range(len(self.rows)))
        random.Random(45).shuffle(expected)
        self.assertEqual(epoch_indices(self.rows, 42, 3), expected)

    def test_language_balance_keeps_the_same_total_update_budget(self):
        order = epoch_indices(self.rows, 42, 0, "language_balanced")
        counts = Counter(self.rows[index]["language"] for index in order)
        self.assertEqual(len(order), len(self.rows))
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(order, epoch_indices(self.rows, 42, 0, "language_balanced"))
        self.assertNotEqual(order, epoch_indices(self.rows, 42, 1, "language_balanced"))
        for language in ("hi", "mr"):
            self.assertEqual({i for i in order if self.rows[i]["language"] == language},
                             {i for i, row in enumerate(self.rows) if row["language"] == language})

    def test_unknown_policy_is_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            epoch_indices(self.rows, 42, 0, "unknown")


if __name__ == "__main__":
    unittest.main()

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from casa.data import ContentCSVCTRDataset, IndexedCSVCTRDataset


class IndexedDatasetTests(unittest.TestCase):
    def test_indexed_matches_in_memory_content_dataset(self):
        fields = [
            "history", "candidate", "label", "history_tags", "candidate_tag",
            "history_authors", "candidate_author", "history_durations",
            "candidate_duration", "long_view_label",
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "examples.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "history": "1 2 3", "candidate": 4, "label": 1,
                    "history_tags": "2 3 4", "candidate_tag": 5,
                    "history_authors": "3 4 5", "candidate_author": 6,
                    "history_durations": "1 2 3", "candidate_duration": 4,
                    "long_view_label": 0,
                })
            memory = ContentCSVCTRDataset(path, max_sequence_length=5)
            indexed = IndexedCSVCTRDataset(path, max_sequence_length=5, use_content_features=True)
            self.assertEqual(len(memory), len(indexed))
            for left, right in zip(memory[0], indexed[0]):
                self.assertTrue(torch.equal(left, right))
            indexed.close()


if __name__ == "__main__":
    unittest.main()

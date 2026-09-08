import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HEADER = ["user_id", "video_id", "time_ms", "is_click", "long_view"]


class PreprocessTests(unittest.TestCase):
    def test_candidate_does_not_leak_into_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root, data, output = Path(temp), Path(temp) / "data", Path(temp) / "out"
            data.mkdir()
            train = data / "log_standard_4_08_to_4_21_pure.csv"
            future = data / "log_standard_4_22_to_5_08_pure.csv"
            with train.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=HEADER); writer.writeheader()
                for index in range(1, 5):
                    writer.writerow({"user_id": 1, "video_id": index, "time_ms": index, "is_click": 1, "long_view": 0})
            with future.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=HEADER); writer.writeheader()
                writer.writerow({"user_id": 1, "video_id": 8, "time_ms": 10, "is_click": 0, "long_view": 0})
                writer.writerow({"user_id": 1, "video_id": 9, "time_ms": 20, "is_click": 1, "long_view": 0})
            subprocess.run([sys.executable, str(Path(__file__).parents[1] / "preprocess_kuairand.py"), "--data-dir", str(data), "--output-dir", str(output), "--min-history", "2", "--include-behavior-features"], check=True, capture_output=True, text=True)
            with (output / "train.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["history"], "2 3")
            self.assertNotIn(rows[0]["candidate"], rows[0]["history"].split())
            self.assertEqual(len(rows[0]["history_actions"].split()), len(rows[0]["history"].split()))
            self.assertEqual(len(rows[0]["history_time_gap_buckets"].split()), len(rows[0]["history"].split()))


if __name__ == "__main__":
    unittest.main()

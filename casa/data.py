from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SyntheticCTRDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Reproducible toy data for verifying the complete training pipeline.

    Items are assigned latent categories. A click is more likely when the
    candidate category occurs repeatedly in the user's recent history.
    """

    def __init__(
        self,
        num_samples: int = 5000,
        num_items: int = 1000,
        sequence_length: int = 100,
        num_categories: int = 20,
        seed: int = 42,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.histories = torch.randint(
            1, num_items + 1, (num_samples, sequence_length), generator=generator
        )
        self.candidates = torch.randint(1, num_items + 1, (num_samples,), generator=generator)
        item_categories = torch.arange(num_items + 1) % num_categories
        history_categories = item_categories[self.histories]
        candidate_categories = item_categories[self.candidates].unsqueeze(1)
        match_rate = history_categories.eq(candidate_categories).float().mean(dim=1)
        # Nonlinear but learnable signal with balanced stochastic labels.
        click_probability = torch.sigmoid((match_rate - 1.0 / num_categories) * 28.0)
        self.labels = torch.bernoulli(click_probability, generator=generator)

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.histories[index], self.candidates[index], self.labels[index]


@dataclass
class CSVSchema:
    history_column: str = "history"
    candidate_column: str = "candidate"
    label_column: str = "label"
    history_separator: str = " "


class CSVCTRDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Loads preprocessed CTR examples from a simple CSV file.

    Required default columns:
      history: space-separated integer item IDs
      candidate: integer item ID
      label: 0 or 1
    """

    def __init__(self, path: str | Path, max_sequence_length: int, schema: CSVSchema | None = None):
        self.schema = schema or CSVSchema()
        self.max_sequence_length = max_sequence_length
        self.rows: list[tuple[Tensor, Tensor, Tensor]] = []
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                self.schema.history_column,
                self.schema.candidate_column,
                self.schema.label_column,
            }
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"CSV must contain columns: {sorted(required)}")
            for row in reader:
                values = [
                    int(value)
                    for value in row[self.schema.history_column].split(self.schema.history_separator)
                    if value
                ][-max_sequence_length:]
                if not values:
                    continue
                padded = [0] * (max_sequence_length - len(values)) + values
                self.rows.append(
                    (
                        torch.tensor(padded, dtype=torch.long),
                        torch.tensor(int(row[self.schema.candidate_column]), dtype=torch.long),
                        torch.tensor(float(row[self.schema.label_column]), dtype=torch.float32),
                    )
                )
        if not self.rows:
            raise ValueError("CSV contains no usable examples")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.rows[index]


class ContentCSVCTRDataset(Dataset):
    """CTR dataset with aligned video tag, author-bucket and duration features."""

    COLUMNS = {
        "history", "candidate", "label", "history_tags", "candidate_tag",
        "history_authors", "candidate_author", "history_durations", "candidate_duration",
        "long_view_label",
    }
    BEHAVIOR_COLUMNS = {
        "history_actions", "history_watch_buckets", "history_time_gap_buckets",
    }

    def __init__(
        self, path: str | Path, max_sequence_length: int,
        use_behavior_features: bool = False,
    ):
        self.rows = []
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = self.COLUMNS | (self.BEHAVIOR_COLUMNS if use_behavior_features else set())
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"content CSV is missing columns: {sorted(missing)}")
            for row in reader:
                sequences = []
                for column in ("history", "history_tags", "history_authors", "history_durations"):
                    values = [int(value) for value in row[column].split() if value]
                    sequences.append(values[-max_sequence_length:])
                length = len(sequences[0])
                if length == 0 or any(len(values) != length for values in sequences):
                    continue
                padded = [
                    torch.tensor([0] * (max_sequence_length - length) + values, dtype=torch.long)
                    for values in sequences
                ]
                result = (
                    padded[0],
                    torch.tensor(int(row["candidate"]), dtype=torch.long),
                    torch.tensor(float(row["label"]), dtype=torch.float32),
                    padded[1],
                    torch.tensor(int(row["candidate_tag"]), dtype=torch.long),
                    padded[2],
                    torch.tensor(int(row["candidate_author"]), dtype=torch.long),
                    padded[3],
                    torch.tensor(int(row["candidate_duration"]), dtype=torch.long),
                    torch.tensor(float(row["long_view_label"]), dtype=torch.float32),
                )
                if use_behavior_features:
                    behavior = tuple(
                        _pad_sequence(row[column], max_sequence_length)
                        for column in (
                            "history_actions", "history_watch_buckets",
                            "history_time_gap_buckets",
                        )
                    )
                    result = result + behavior
                self.rows.append(result)
        if not self.rows:
            raise ValueError("content CSV contains no usable examples")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def _pad_sequence(text: str, max_sequence_length: int) -> Tensor:
    values = [int(value) for value in text.split() if value][-max_sequence_length:]
    return torch.tensor(
        [0] * (max_sequence_length - len(values)) + values, dtype=torch.long
    )


class IndexedCSVCTRDataset(Dataset):
    """Memory-efficient random-access CSV dataset.

    Only byte offsets are kept in memory. Rows are decoded on demand, so this
    class can train on hundreds of thousands of long-sequence examples without
    materializing every tensor at startup.
    """

    def __init__(
        self, path: str | Path, max_sequence_length: int,
        use_content_features: bool, use_behavior_features: bool = False,
    ):
        self.path = Path(path)
        self.max_sequence_length = max_sequence_length
        self.use_content_features = use_content_features
        self.use_behavior_features = use_behavior_features
        if use_behavior_features and not use_content_features:
            raise ValueError("behavior features currently require content features")
        self._handle = None
        self.offsets: list[int] = []
        with self.path.open("rb") as handle:
            header = handle.readline().decode("utf-8-sig").rstrip("\r\n")
            self.fieldnames = next(csv.reader([header]))
            required = ContentCSVCTRDataset.COLUMNS if use_content_features else {
                "history", "candidate", "label"
            }
            if use_behavior_features:
                required = required | ContentCSVCTRDataset.BEHAVIOR_COLUMNS
            missing = required.difference(self.fieldnames)
            if missing:
                raise ValueError(f"indexed CSV is missing columns: {sorted(missing)}")
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError("indexed CSV contains no examples")

    def __len__(self) -> int:
        return len(self.offsets)

    def _open(self):
        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("rb")
        return self._handle

    def __getitem__(self, index: int):
        handle = self._open()
        handle.seek(self.offsets[index])
        line = handle.readline().decode("utf-8").rstrip("\r\n")
        row = next(csv.DictReader([line], fieldnames=self.fieldnames))
        base = (
            _pad_sequence(row["history"], self.max_sequence_length),
            torch.tensor(int(row["candidate"]), dtype=torch.long),
            torch.tensor(float(row["label"]), dtype=torch.float32),
        )
        if not self.use_content_features:
            return base
        result = base + (
            _pad_sequence(row["history_tags"], self.max_sequence_length),
            torch.tensor(int(row["candidate_tag"]), dtype=torch.long),
            _pad_sequence(row["history_authors"], self.max_sequence_length),
            torch.tensor(int(row["candidate_author"]), dtype=torch.long),
            _pad_sequence(row["history_durations"], self.max_sequence_length),
            torch.tensor(int(row["candidate_duration"]), dtype=torch.long),
            torch.tensor(float(row["long_view_label"]), dtype=torch.float32),
        )
        if self.use_behavior_features:
            result = result + tuple(
                _pad_sequence(row[column], self.max_sequence_length)
                for column in (
                    "history_actions", "history_watch_buckets",
                    "history_time_gap_buckets",
                )
            )
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def close(self):
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
        self._handle = None

    def __del__(self):
        self.close()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict, deque
from pathlib import Path


REQUIRED_COLUMNS = {"user_id", "video_id", "time_ms", "is_click", "long_view"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert KuaiRand-Pure logs into causal CTR sequences")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/kuairand"))
    parser.add_argument("--max-history", type=int, default=1000)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--history-event", choices=["click", "long_view", "click_or_long_view"], default="click_or_long_view")
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--max-output-rows", type=int, default=None, help="Debug-only limit per split")
    parser.add_argument(
        "--test-output-offset", type=int, default=0,
        help="Skip this many eligible test examples before writing a disjoint holdout",
    )
    parser.add_argument("--include-content-features", action="store_true")
    parser.add_argument("--include-behavior-features", action="store_true")
    parser.add_argument("--author-buckets", type=int, default=2048)
    return parser.parse_args()


def is_positive_history_event(row: dict[str, str], rule: str) -> bool:
    click = int(row["is_click"]) == 1
    long_view = int(row["long_view"]) == 1
    return click if rule == "click" else long_view if rule == "long_view" else click or long_view


def inspect_file(path: Path) -> tuple[int, int, int]:
    rows, min_time, max_time = 0, 2**63 - 1, 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        for row in reader:
            timestamp = int(row["time_ms"])
            rows += 1
            min_time, max_time = min(min_time, timestamp), max(max_time, timestamp)
    return rows, min_time, max_time


def duration_bucket(milliseconds: float) -> int:
    seconds = max(milliseconds, 0.0) / 1000.0
    for index, upper in enumerate((15, 30, 60, 120, 300), start=1):
        if seconds <= upper:
            return index
    return 6


def action_strength_bucket(row: dict[str, str]) -> int:
    """Encode increasingly strong observable feedback without target leakage."""
    social = sum(int(row.get(name) or 0) for name in (
        "is_like", "is_follow", "is_comment", "is_forward"
    ))
    if social >= 2:
        return 4
    if social == 1:
        return 3
    if int(row.get("long_view") or 0) == 1:
        return 2
    return 1


def watch_completion_bucket(row: dict[str, str]) -> int:
    play = float(row.get("play_time_ms") or 0)
    duration = float(row.get("duration_ms") or 0)
    if duration <= 0:
        return 0
    ratio = max(0.0, min(play / duration, 1.5))
    for index, upper in enumerate((0.1, 0.3, 0.6, 0.9), start=1):
        if ratio <= upper:
            return index
    return 5


def time_gap_bucket(gap_ms: int) -> int:
    hours = max(gap_ms, 0) / 3_600_000
    for index, upper in enumerate((1, 6, 24, 72, 168), start=1):
        if hours <= upper:
            return index
    return 6


def load_video_features(path: Path, author_buckets: int):
    records = []
    raw_tags = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            match = re.search(r"-?\d+", row.get("tag", ""))
            raw_tag = int(match.group()) if match else None
            if raw_tag is not None:
                raw_tags.add(raw_tag)
            records.append((row, raw_tag))
    tag_mapping = {raw: index + 1 for index, raw in enumerate(sorted(raw_tags))}
    features = {}
    for row, raw_tag in records:
        video_id = int(row["video_id"])
        author_raw = int(float(row.get("author_id") or 0))
        duration_raw = float(row.get("video_duration") or 0)
        features[video_id] = (
            tag_mapping.get(raw_tag, 0),
            author_raw % author_buckets + 1 if author_raw else 0,
            duration_bucket(duration_raw),
        )
    return features, tag_mapping


def write_examples(
    source, writers, histories, args, split_timestamp, counters,
    eligible_counters, video_features
) -> None:
    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            user_id = int(row["user_id"])
            candidate_id = int(row["video_id"]) + 1  # zero is reserved for padding
            candidate_features = video_features.get(int(row["video_id"]), (0, 0, 0))
            timestamp = int(row["time_ms"])
            history = histories[user_id]
            split = "train" if split_timestamp is None else (
                "validation" if timestamp <= split_timestamp else "test"
            )
            if len(history) >= args.min_history:
                eligible_index = eligible_counters[split]
                eligible_counters[split] += 1
                offset = args.test_output_offset if split == "test" else 0
                within_limit = args.max_output_rows is None or counters[split] < args.max_output_rows
                if eligible_index >= offset and within_limit:
                    example = {
                        "user_id": user_id,
                        "timestamp": timestamp,
                        "history": " ".join(str(item[0]) for item in history),
                        "candidate": candidate_id,
                        "label": int(row["is_click"]),
                        "long_view_label": int(row["long_view"]),
                    }
                    if args.include_content_features:
                        example.update({
                            "history_tags": " ".join(str(item[1]) for item in history),
                            "candidate_tag": candidate_features[0],
                            "history_authors": " ".join(str(item[2]) for item in history),
                            "candidate_author": candidate_features[1],
                            "history_durations": " ".join(str(item[3]) for item in history),
                            "candidate_duration": candidate_features[2],
                        })
                    if args.include_behavior_features:
                        example.update({
                            "history_actions": " ".join(str(item[5]) for item in history),
                            "history_watch_buckets": " ".join(str(item[6]) for item in history),
                            "history_time_gap_buckets": " ".join(
                                str(time_gap_bucket(timestamp - item[4])) for item in history
                            ),
                        })
                    writers[split].writerow(example)
                    counters[split] += 1
            # Update after writing to prevent candidate leakage into its own history.
            if is_positive_history_event(row, args.history_event):
                history.append((
                    candidate_id, *candidate_features, timestamp,
                    action_strength_bucket(row), watch_completion_bucket(row),
                ))


def main() -> None:
    args = parse_args()
    if args.max_history < 1 or args.min_history < 1:
        raise ValueError("history lengths must be positive")
    if args.test_output_offset < 0:
        raise ValueError("test-output-offset must be non-negative")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    train_source = args.data_dir / "log_standard_4_08_to_4_21_pure.csv"
    future_source = args.data_dir / "log_standard_4_22_to_5_08_pure.csv"
    feature_source = args.data_dir / "video_features_basic_pure.csv"
    for source in (train_source, future_source):
        if not source.exists():
            raise FileNotFoundError(source)

    train_rows, train_min, train_max = inspect_file(train_source)
    future_rows, future_min, future_max = inspect_file(future_source)
    split_timestamp = int(future_min + (future_max - future_min) * args.validation_fraction)
    if args.include_content_features:
        video_features, tag_mapping = load_video_features(feature_source, args.author_buckets)
    else:
        video_features, tag_mapping = {}, {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: args.output_dir / f"{name}.csv" for name in ("train", "validation", "test")}
    handles = {name: path.open("w", encoding="utf-8", newline="") for name, path in paths.items()}
    fields = ["user_id", "timestamp", "history", "candidate", "label", "long_view_label"]
    if args.include_content_features:
        fields += [
            "history_tags", "candidate_tag", "history_authors", "candidate_author",
            "history_durations", "candidate_duration",
        ]
    if args.include_behavior_features:
        fields += [
            "history_actions", "history_watch_buckets", "history_time_gap_buckets",
        ]
    writers = {name: csv.DictWriter(handle, fieldnames=fields) for name, handle in handles.items()}
    for writer in writers.values():
        writer.writeheader()
    histories = defaultdict(lambda: deque(maxlen=args.max_history))
    counters = {"train": 0, "validation": 0, "test": 0}
    eligible_counters = {"train": 0, "validation": 0, "test": 0}
    try:
        write_examples(
            train_source, writers, histories, args, None, counters,
            eligible_counters, video_features
        )
        write_examples(
            future_source, writers, histories, args, split_timestamp, counters,
            eligible_counters, video_features
        )
    finally:
        for handle in handles.values():
            handle.close()
    metadata = {
        "source_train_rows": train_rows,
        "source_future_rows": future_rows,
        "source_train_time_range": [train_min, train_max],
        "source_future_time_range": [future_min, future_max],
        "validation_test_split_timestamp": split_timestamp,
        "output_rows": counters,
        "eligible_rows_seen": eligible_counters,
        "test_output_offset": args.test_output_offset,
        "max_history": args.max_history,
        "min_history": args.min_history,
        "history_event": args.history_event,
        "video_id_offset": 1,
        "include_content_features": args.include_content_features,
        "include_behavior_features": args.include_behavior_features,
        "num_tags": len(tag_mapping),
        "num_author_buckets": args.author_buckets if args.include_content_features else 0,
        "num_duration_buckets": 6 if args.include_content_features else 0,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from casa.data import CSVCTRDataset
from casa.model import ContentAdaptiveSparseAttentionCTR
from train import evaluate


MODES = ("full", "recent", "random", "similarity", "learned")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fair CASA baseline comparisons")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--num-items", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs/baselines"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    test_dataset = CSVCTRDataset(args.test_csv, max_sequence_length=args.sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    results = []

    for mode in MODES:
        mode_output = args.output / mode
        command = [
            sys.executable,
            str(Path(__file__).with_name("train.py")),
            "--train-csv", str(args.train_csv),
            "--validation-csv", str(args.validation_csv),
            "--num-items", str(args.num_items),
            "--sequence-length", str(args.sequence_length),
            "--top-k", str(args.top_k),
            "--embedding-dim", str(args.embedding_dim),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--seed", str(args.seed),
            "--selection-mode", mode,
            "--output", str(mode_output),
        ]
        print(f"\n===== Training {mode} =====", flush=True)
        subprocess.run(command, check=True)
        checkpoint = torch.load(mode_output / "best_model.pt", map_location="cpu", weights_only=True)
        model = ContentAdaptiveSparseAttentionCTR(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        test_metrics = evaluate(model, test_loader, torch.device("cpu"))
        row = {
            "mode": mode,
            "best_validation_auc": checkpoint["metrics"]["auc"],
            "test_auc": test_metrics["auc"],
            "test_logloss": test_metrics["logloss"],
            "test_examples_per_second": test_metrics["examples_per_second"],
            "test_milliseconds_per_example": test_metrics["milliseconds_per_example"],
            "seed": args.seed,
            "sequence_length": args.sequence_length,
            "top_k": args.top_k if mode != "full" else args.sequence_length,
        }
        results.append(row)
        print(json.dumps(row, indent=2), flush=True)

    (args.output / "comparison.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nComparison saved to {args.output / 'comparison.csv'}")


if __name__ == "__main__":
    main()

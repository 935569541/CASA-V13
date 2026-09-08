from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


SEEDS = (7, 42, 2026)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed three-seed V13 behavior-aware validation protocol."
    )
    parser.add_argument("--project", type=Path, default=Path(r"C:\casa"))
    parser.add_argument(
        "--data", type=Path,
        default=Path(r"C:\casa\data\processed\kuairand_v13_behavior_50k"),
    )
    parser.add_argument(
        "--teacher", type=Path,
        default=Path(r"C:\casa\runs\full_content_len500\best_model.pt"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(r"C:\casa\runs\v13_validation_suite"),
    )
    return parser.parse_args()


def best_auc(history_path: Path) -> float:
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError(f"Invalid or empty history: {history_path}")
    return max(float(row["auc"]) for row in history)


def train_one(args: argparse.Namespace, seed: int) -> dict[str, float | int | str]:
    output = args.output_root / f"seed{seed}"
    command = [
        sys.executable, str(args.project / "train.py"),
        "--train-csv", str(args.data / "train.csv"),
        "--validation-csv", str(args.data / "validation.csv"),
        "--num-items", "7583",
        "--sequence-length", "500",
        "--top-k", "100",
        "--embedding-dim", "32",
        "--epochs", "8",
        "--batch-size", "128",
        "--learning-rate", "0.0005",
        "--selection-mode", "learned",
        "--use-content-features",
        "--use-behavior-features",
        "--num-tags", "43",
        "--num-author-buckets", "2048",
        "--num-duration-buckets", "6",
        "--content-embedding-dim", "8",
        "--teacher-checkpoint", str(args.teacher),
        "--initialize-from-teacher",
        "--distillation-weight", "0.5",
        "--representation-weight", "0.1",
        "--distillation-temperature", "2.0",
        "--streaming-data",
        "--early-stopping-patience", "2",
        "--seed", str(seed),
        "--output", str(output),
    ]
    print(f"\n===== V13 seed {seed} =====", flush=True)
    subprocess.run(command, cwd=args.project, check=True)
    return {
        "model": "v13_behavior_aware",
        "seed": seed,
        "best_validation_auc": best_auc(output / "history.json"),
        "history": str(output / "history.json"),
    }


def add_existing_baselines(args: argparse.Namespace, rows: list[dict]) -> None:
    paths = {
        "v12_temporal_aware": {
            7: args.project / r"runs\v12_validation_suite\seed7\history.json",
            42: args.project / r"runs\v12_validation_suite\seed42\history.json",
            2026: args.project / r"runs\v12_validation_suite\seed2026\history.json",
        },
        "recent": {
            7: args.project / r"runs\repeat\recent_seed7\history.json",
            42: args.project / r"runs\recent_content_len500_top100\history.json",
            2026: args.project / r"runs\repeat\recent_seed2026\history.json",
        },
        "random": {
            7: args.project / r"runs\repeat\random_seed7\history.json",
            42: args.project / r"runs\random_content_len500_top100\history.json",
            2026: args.project / r"runs\repeat\random_seed2026\history.json",
        },
    }
    for model, seed_paths in paths.items():
        for seed, path in seed_paths.items():
            if path.exists():
                rows.append({
                    "model": model,
                    "seed": seed,
                    "best_validation_auc": best_auc(path),
                    "history": str(path),
                })


def write_reports(output_root: Path, rows: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "validation_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(
            float(row["best_validation_auc"])
        )
    summary = []
    for model, values in grouped.items():
        summary.append({
            "model": model,
            "runs": len(values),
            "mean_validation_auc": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min_validation_auc": min(values),
            "max_validation_auc": max(values),
        })
    summary.sort(key=lambda row: row["mean_validation_auc"], reverse=True)
    json_path = output_root / "validation_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n===== FINAL VALIDATION SUMMARY =====")
    for row in summary:
        print(
            f"{row['model']}: {row['mean_validation_auc']:.6f} "
            f"+/- {row['sample_std']:.6f} (n={row['runs']})"
        )
    print(f"Runs: {csv_path}")
    print(f"Summary: {json_path}")


def main() -> None:
    args = parse_args()
    required = [
        args.project / "train.py", args.data / "train.csv",
        args.data / "validation.csv", args.teacher,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    rows: list[dict] = []
    for seed in SEEDS:
        rows.append(train_one(args, seed))
    add_existing_baselines(args, rows)
    write_reports(args.output_root, rows)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


PROJECT = Path(r"C:\casa")
DATA = PROJECT / r"data\processed\kuairand_v13_behavior_50k"
RAW = PROJECT / r"data\raw\KuaiRand-Pure\KuaiRand-Pure\data"
TEACHER = PROJECT / r"runs\full_content_len500\best_model.pt"
BASELINES = PROJECT / r"runs\v13_fair_baselines"
HOLDOUT = PROJECT / r"data\processed\kuairand_v13_final_offset100k"
FINAL = PROJECT / r"runs\v13_locked_final_evaluation"
SEEDS = (7, 42, 2026)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT, check=True)


def best_auc(history: Path) -> float:
    rows = json.loads(history.read_text(encoding="utf-8"))
    return max(float(row["auc"]) for row in rows)


def train_baseline(mode: str, seed: int) -> Path:
    output = BASELINES / mode / f"seed{seed}"
    print(f"===== TRAINING FAIR {mode.upper()} SEED {seed} =====", flush=True)
    run([
        sys.executable, str(PROJECT / "train.py"),
        "--train-csv", str(DATA / "train.csv"),
        "--validation-csv", str(DATA / "validation.csv"),
        "--num-items", "7583", "--sequence-length", "500", "--top-k", "100",
        "--embedding-dim", "32", "--epochs", "8", "--batch-size", "128",
        "--learning-rate", "0.0005", "--selection-mode", mode,
        "--use-content-features", "--use-behavior-features",
        "--num-tags", "43", "--num-author-buckets", "2048",
        "--num-duration-buckets", "6", "--content-embedding-dim", "8",
        "--teacher-checkpoint", str(TEACHER), "--initialize-from-teacher",
        "--distillation-weight", "0.5", "--representation-weight", "0.1",
        "--distillation-temperature", "2.0", "--streaming-data",
        "--early-stopping-patience", "2", "--seed", str(seed),
        "--output", str(output),
    ])
    return output / "best_model.pt"


def v13_checkpoint(seed: int) -> Path:
    return PROJECT / f"runs/v13_validation_suite/seed{seed}/best_model.pt"


def validation_summary(checkpoints: dict[str, list[tuple[int, Path]]]) -> None:
    summary = []
    for model, runs in checkpoints.items():
        values = [best_auc(path.parent / "history.json") for _, path in runs]
        summary.append({
            "model": model,
            "mean_validation_auc": statistics.mean(values),
            "sample_std": statistics.stdev(values),
            "runs": len(values),
        })
    summary.sort(key=lambda row: row["mean_validation_auc"], reverse=True)
    BASELINES.mkdir(parents=True, exist_ok=True)
    (BASELINES / "fair_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n===== FAIR BEHAVIOR-AWARE VALIDATION SUMMARY =====")
    for row in summary:
        print(
            f"{row['model']}: {row['mean_validation_auc']:.6f} +/- "
            f"{row['sample_std']:.6f} (n={row['runs']})"
        )


def build_fresh_holdout() -> Path:
    print("\n===== BUILDING NEVER-SEEN OFFSET-100K HOLDOUT =====", flush=True)
    run([
        sys.executable, str(PROJECT / "preprocess_kuairand.py"),
        "--data-dir", str(RAW), "--output-dir", str(HOLDOUT),
        "--max-history", "500", "--min-history", "5",
        "--history-event", "click_or_long_view",
        "--max-output-rows", "37000", "--test-output-offset", "100000",
        "--include-content-features", "--include-behavior-features",
        "--author-buckets", "2048",
    ])
    metadata = json.loads((HOLDOUT / "metadata.json").read_text(encoding="utf-8"))
    rows = int(metadata["output_rows"]["test"])
    if metadata["test_output_offset"] != 100000 or rows < 30000:
        raise RuntimeError(f"Fresh final holdout is invalid or too small: {rows} rows")
    print(f"Fresh final examples: {rows}")
    return HOLDOUT / "test.csv"


def evaluate_one(model: str, seed: int, checkpoint: Path, test_csv: Path) -> dict:
    output = FINAL / f"{model}_seed{seed}.json"
    print(f"===== FINAL EVALUATION {model.upper()} SEED {seed} =====", flush=True)
    run([
        sys.executable, str(PROJECT / "evaluate_checkpoint.py"),
        "--checkpoint", str(checkpoint), "--csv", str(test_csv),
        "--sequence-length", "500", "--batch-size", "128",
        "--streaming-data", "--seed", str(seed), "--output", str(output),
    ])
    metrics = json.loads(output.read_text(encoding="utf-8"))
    return {
        "model": model, "seed": seed, "auc": float(metrics["auc"]),
        "logloss": float(metrics["logloss"]),
        "milliseconds_per_example": float(metrics["milliseconds_per_example"]),
        "checkpoint": str(checkpoint),
    }


def final_summary(rows: list[dict]) -> None:
    runs_path = FINAL / "locked_final_runs.csv"
    with runs_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)
    summary = []
    for model, group in grouped.items():
        aucs = [row["auc"] for row in group]
        summary.append({
            "model": model, "runs": len(group),
            "mean_auc": statistics.mean(aucs),
            "auc_sample_std": statistics.stdev(aucs),
            "mean_logloss": statistics.mean(row["logloss"] for row in group),
            "mean_milliseconds_per_example": statistics.mean(
                row["milliseconds_per_example"] for row in group
            ),
        })
    summary.sort(key=lambda row: row["mean_auc"], reverse=True)
    summary_path = FINAL / "locked_final_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n===== V13 FAIR LOCKED FINAL SUMMARY =====")
    for row in summary:
        print(
            f"{row['model']}: AUC {row['mean_auc']:.6f} +/- "
            f"{row['auc_sample_std']:.6f}; LogLoss {row['mean_logloss']:.6f}; "
            f"latency {row['mean_milliseconds_per_example']:.5f} ms (n={row['runs']})"
        )
    print(f"Runs: {runs_path}")
    print(f"Summary: {summary_path}")


def main() -> None:
    required = [DATA / "train.csv", DATA / "validation.csv", TEACHER]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    checkpoints: dict[str, list[tuple[int, Path]]] = {
        "v13_behavior_aware": [(seed, v13_checkpoint(seed)) for seed in SEEDS]
    }
    for mode in ("recent", "random", "full"):
        checkpoints[mode] = [(seed, train_baseline(mode, seed)) for seed in SEEDS]
    validation_summary(checkpoints)
    test_csv = build_fresh_holdout()
    FINAL.mkdir(parents=True, exist_ok=True)
    rows = [
        evaluate_one(model, seed, checkpoint, test_csv)
        for model, runs in checkpoints.items() for seed, checkpoint in runs
    ]
    final_summary(rows)


if __name__ == "__main__":
    main()

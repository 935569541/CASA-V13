from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


PROJECT = Path(r"C:\casa")
RAW_DATA = Path(r"C:\casa\data\raw\KuaiRand-Pure\KuaiRand-Pure\data")
HOLDOUT = Path(r"C:\casa\data\processed\kuairand_final_holdout_offset50k")
RESULTS = Path(r"C:\casa\runs\final_holdout_evaluation")


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT, check=True)


def create_disjoint_holdout() -> Path:
    print("===== BUILDING DISJOINT FINAL HOLDOUT =====", flush=True)
    run([
        sys.executable, str(PROJECT / "preprocess_kuairand.py"),
        "--data-dir", str(RAW_DATA),
        "--output-dir", str(HOLDOUT),
        "--max-history", "500",
        "--min-history", "5",
        "--history-event", "click_or_long_view",
        "--max-output-rows", "50000",
        "--test-output-offset", "50000",
        "--include-content-features",
        "--author-buckets", "2048",
    ])
    metadata = json.loads((HOLDOUT / "metadata.json").read_text(encoding="utf-8"))
    if metadata["test_output_offset"] != 50000:
        raise RuntimeError("Final holdout offset was not applied")
    if metadata["output_rows"]["test"] != 50000:
        raise RuntimeError("Final holdout does not contain exactly 50,000 test rows")
    return HOLDOUT / "test.csv"


def checkpoints() -> list[tuple[str, int, Path]]:
    return [
        ("v12_temporal_aware", 7, PROJECT / r"runs\v12_validation_suite\seed7\best_model.pt"),
        ("v12_temporal_aware", 42, PROJECT / r"runs\v12_validation_suite\seed42\best_model.pt"),
        ("v12_temporal_aware", 2026, PROJECT / r"runs\v12_validation_suite\seed2026\best_model.pt"),
        ("recent", 7, PROJECT / r"runs\repeat\recent_seed7\best_model.pt"),
        ("recent", 42, PROJECT / r"runs\recent_content_len500_top100\best_model.pt"),
        ("recent", 2026, PROJECT / r"runs\repeat\recent_seed2026\best_model.pt"),
        ("random", 7, PROJECT / r"runs\repeat\random_seed7\best_model.pt"),
        ("random", 42, PROJECT / r"runs\random_content_len500_top100\best_model.pt"),
        ("random", 2026, PROJECT / r"runs\repeat\random_seed2026\best_model.pt"),
        ("full", 42, PROJECT / r"runs\full_content_len500\best_model.pt"),
    ]


def evaluate_all(test_csv: Path) -> list[dict]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, seed, checkpoint in checkpoints():
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        output = RESULTS / f"{model}_seed{seed}.json"
        print(f"===== EVALUATING {model} SEED {seed} =====", flush=True)
        run([
            sys.executable, str(PROJECT / "evaluate_checkpoint.py"),
            "--checkpoint", str(checkpoint),
            "--csv", str(test_csv),
            "--sequence-length", "500",
            "--batch-size", "128",
            "--streaming-data",
            "--seed", str(seed),
            "--output", str(output),
        ])
        metrics = json.loads(output.read_text(encoding="utf-8"))
        rows.append({
            "model": model,
            "seed": seed,
            "auc": float(metrics["auc"]),
            "logloss": float(metrics["logloss"]),
            "milliseconds_per_example": float(metrics["milliseconds_per_example"]),
            "checkpoint": str(checkpoint),
        })
    return rows


def summarize(rows: list[dict]) -> None:
    runs_path = RESULTS / "final_test_runs.csv"
    with runs_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)
    summary = []
    for model, group in grouped.items():
        aucs = [row["auc"] for row in group]
        losses = [row["logloss"] for row in group]
        times = [row["milliseconds_per_example"] for row in group]
        summary.append({
            "model": model,
            "runs": len(group),
            "mean_auc": statistics.mean(aucs),
            "auc_sample_std": statistics.stdev(aucs) if len(aucs) > 1 else 0.0,
            "mean_logloss": statistics.mean(losses),
            "mean_milliseconds_per_example": statistics.mean(times),
        })
    summary.sort(key=lambda row: row["mean_auc"], reverse=True)
    summary_path = RESULTS / "final_test_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n===== LOCKED FINAL HOLDOUT SUMMARY =====")
    for row in summary:
        print(
            f"{row['model']}: AUC {row['mean_auc']:.6f} +/- "
            f"{row['auc_sample_std']:.6f}; LogLoss {row['mean_logloss']:.6f}; "
            f"latency {row['mean_milliseconds_per_example']:.5f} ms (n={row['runs']})"
        )
    print(f"Runs: {runs_path}")
    print(f"Summary: {summary_path}")


def main() -> None:
    test_csv = create_disjoint_holdout()
    rows = evaluate_all(test_csv)
    summarize(rows)


if __name__ == "__main__":
    main()

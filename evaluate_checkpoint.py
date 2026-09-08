from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from casa.data import CSVCTRDataset, ContentCSVCTRDataset, IndexedCSVCTRDataset
from casa.data import seed_everything
from casa.model import ContentAdaptiveSparseAttentionCTR
from train import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved CASA checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--streaming-data", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = checkpoint["model_config"]
    model = ContentAdaptiveSparseAttentionCTR(**config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    use_content = config.get("use_content_features", False)
    use_behavior = config.get("use_behavior_features", False)
    if args.streaming_data:
        dataset = IndexedCSVCTRDataset(
            args.csv, max_sequence_length=args.sequence_length,
            use_content_features=use_content,
            use_behavior_features=use_behavior,
        )
    else:
        dataset_class = ContentCSVCTRDataset if use_content else CSVCTRDataset
        if use_content:
            dataset = dataset_class(
                args.csv, max_sequence_length=args.sequence_length,
                use_behavior_features=use_behavior,
            )
        else:
            dataset = dataset_class(args.csv, max_sequence_length=args.sequence_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    metrics = evaluate(model, loader, device)
    result = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.csv),
        "examples": len(dataset),
        "device": str(device),
        **metrics,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

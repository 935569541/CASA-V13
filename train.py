from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from casa.data import (
    CSVCTRDataset, ContentCSVCTRDataset, IndexedCSVCTRDataset,
    SyntheticCTRDataset, seed_everything,
)
from casa.model import ContentAdaptiveSparseAttentionCTR


def binary_auc(labels: Tensor, probabilities: Tensor) -> float:
    labels = labels.detach().cpu().float()
    probabilities = probabilities.detach().cpu().float()
    positives = int(labels.sum().item())
    negatives = labels.numel() - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(probabilities)
    sorted_scores = probabilities[order]
    ranks = torch.arange(1, labels.numel() + 1, dtype=torch.float32)
    # Average tied ranks.
    start = 0
    while start < sorted_scores.numel():
        end = start + 1
        while end < sorted_scores.numel() and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = ranks[start:end].mean()
        start = end
    ranked_labels = labels[order]
    positive_rank_sum = ranks[ranked_labels.bool()].sum().item()
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def unpack_batch(batch, device):
    histories, candidates, labels = (tensor.to(device) for tensor in batch[:3])
    if len(batch) == 3:
        return histories, candidates, labels, None, {}
    kwargs = {
        "history_tags": batch[3].to(device),
        "candidate_tags": batch[4].to(device),
        "history_authors": batch[5].to(device),
        "candidate_authors": batch[6].to(device),
        "history_durations": batch[7].to(device),
        "candidate_durations": batch[8].to(device),
    }
    long_view_labels = batch[9].to(device) if len(batch) > 9 else None
    if len(batch) >= 13:
        kwargs.update({
            "history_actions": batch[10].to(device),
            "history_watch_buckets": batch[11].to(device),
            "history_time_gap_buckets": batch[12].to(device),
        })
    return histories, candidates, labels, long_view_labels, kwargs


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    all_logits, all_labels = [], []
    start = time.perf_counter()
    for batch in loader:
        histories, candidates, labels, _, kwargs = unpack_batch(batch, device)
        output = model(histories, candidates, **kwargs)
        all_logits.append(output.logits.cpu())
        all_labels.append(labels.cpu())
    elapsed = time.perf_counter() - start
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).float()
    probabilities = torch.sigmoid(logits)
    logloss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    return {
        "auc": binary_auc(labels, probabilities),
        "logloss": logloss,
        "examples_per_second": labels.numel() / max(elapsed, 1e-9),
        "milliseconds_per_example": elapsed * 1000.0 / labels.numel(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Content-Adaptive Sparse Attention CTR model")
    parser.add_argument("--csv", type=Path, default=None, help="Optional preprocessed CSV dataset")
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--validation-csv", type=Path, default=None)
    parser.add_argument("--num-items", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sparsity-weight", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--selection-mode",
        choices=["learned", "hybrid", "full", "recent", "random", "similarity"],
        default="learned",
    )
    parser.add_argument("--recent-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--distillation-weight", type=float, default=0.0)
    parser.add_argument("--representation-weight", type=float, default=0.0)
    parser.add_argument("--attention-distillation-weight", type=float, default=0.0)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--initialize-from-teacher", action="store_true")
    parser.add_argument("--use-content-features", action="store_true")
    parser.add_argument("--use-behavior-features", action="store_true")
    parser.add_argument("--num-tags", type=int, default=0)
    parser.add_argument("--num-author-buckets", type=int, default=0)
    parser.add_argument("--num-duration-buckets", type=int, default=0)
    parser.add_argument("--content-embedding-dim", type=int, default=8)
    parser.add_argument("--multitask-long-view", action="store_true")
    parser.add_argument("--long-view-weight", type=float, default=0.2)
    parser.add_argument("--use-global-summary", action="store_true")
    parser.add_argument("--streaming-data", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")

    if bool(args.train_csv) != bool(args.validation_csv):
        raise ValueError("--train-csv and --validation-csv must be provided together")
    dataset_class = ContentCSVCTRDataset if args.use_content_features else CSVCTRDataset
    def load_csv(path):
        if args.streaming_data:
            return IndexedCSVCTRDataset(
                path, max_sequence_length=args.sequence_length,
                use_content_features=args.use_content_features,
                use_behavior_features=args.use_behavior_features,
            )
        if args.use_content_features:
            return dataset_class(
                path, max_sequence_length=args.sequence_length,
                use_behavior_features=args.use_behavior_features,
            )
        return dataset_class(path, max_sequence_length=args.sequence_length)
    if args.train_csv and args.validation_csv:
        train_dataset = load_csv(args.train_csv)
        validation_dataset = load_csv(args.validation_csv)
    elif args.csv:
        dataset = CSVCTRDataset(args.csv, max_sequence_length=args.sequence_length)
    else:
        dataset = SyntheticCTRDataset(
            num_samples=args.num_samples,
            num_items=args.num_items,
            sequence_length=args.sequence_length,
            seed=args.seed,
        )

    if not (args.train_csv and args.validation_csv):
        train_size = int(len(dataset) * 0.8)
        validation_size = len(dataset) - train_size
        train_dataset, validation_dataset = random_split(
            dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(args.seed)
        )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    model = ContentAdaptiveSparseAttentionCTR(
        num_items=args.num_items,
        embedding_dim=args.embedding_dim,
        top_k=args.top_k,
        temperature=args.temperature,
        selection_mode=args.selection_mode,
        recent_k=args.recent_k,
        use_content_features=args.use_content_features,
        use_behavior_features=args.use_behavior_features,
        num_tags=args.num_tags,
        num_author_buckets=args.num_author_buckets,
        num_duration_buckets=args.num_duration_buckets,
        content_embedding_dim=args.content_embedding_dim,
        use_multitask=args.multitask_long_view,
        use_global_summary=args.use_global_summary,
    ).to(device)
    teacher = None
    teacher_checkpoint = None
    if args.teacher_checkpoint:
        teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location=device, weights_only=True)
        teacher = ContentAdaptiveSparseAttentionCTR(**teacher_checkpoint["model_config"]).to(device)
        # strict=False keeps older full-attention teachers compatible with
        # selector-only parameters introduced in later model versions.
        teacher.load_state_dict(teacher_checkpoint["model_state"], strict=False)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        if args.initialize_from_teacher:
            model.load_state_dict(teacher_checkpoint["model_state"], strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    target_rate = min(args.top_k / args.sequence_length, 1.0)
    best_auc = -math.inf
    history = []
    epochs_without_improvement = 0
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        examples = 0
        running_distillation = 0.0
        running_representation = 0.0
        running_attention_distillation = 0.0
        running_long_view = 0.0
        for batch in train_loader:
            histories, candidates, labels, long_view_labels, kwargs = unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(histories, candidates, **kwargs)
            loss, _ = model.loss(
                output,
                labels,
                sparsity_weight=args.sparsity_weight,
                target_selection_rate=target_rate,
            )
            distillation_loss = torch.zeros((), device=device)
            representation_loss = torch.zeros((), device=device)
            attention_distillation_loss = torch.zeros((), device=device)
            long_view_loss = torch.zeros((), device=device)
            if args.multitask_long_view:
                if long_view_labels is None or output.long_view_logits is None:
                    raise ValueError("multitask training requires long_view_label in the dataset")
                long_view_loss = F.binary_cross_entropy_with_logits(
                    output.long_view_logits, long_view_labels.float()
                )
                loss = loss + args.long_view_weight * long_view_loss
            if teacher is not None:
                with torch.no_grad():
                    # The teacher and student consume the same batch features.
                    # This is required when a content-aware full-attention model
                    # is used as the distillation teacher.
                    teacher_output = teacher(histories, candidates, **kwargs)
                temperature = args.distillation_temperature
                soft_targets = torch.sigmoid(teacher_output.logits / temperature)
                distillation_loss = F.binary_cross_entropy_with_logits(
                    output.logits / temperature, soft_targets
                ) * temperature**2
                representation_loss = 1.0 - F.cosine_similarity(
                    output.user_interest, teacher_output.user_interest, dim=-1
                ).mean()
                if output.attention_weights.shape == teacher_output.attention_weights.shape:
                    attention_distillation_loss = F.kl_div(
                        torch.log(output.attention_weights.clamp_min(1e-8)),
                        teacher_output.attention_weights.clamp_min(1e-8),
                        reduction="batchmean",
                    )
                loss = (
                    loss
                    + args.distillation_weight * distillation_loss
                    + args.representation_weight * representation_loss
                    + args.attention_distillation_weight * attention_distillation_loss
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * labels.numel()
            running_distillation += distillation_loss.item() * labels.numel()
            running_representation += representation_loss.item() * labels.numel()
            running_attention_distillation += attention_distillation_loss.item() * labels.numel()
            running_long_view += long_view_loss.item() * labels.numel()
            examples += labels.numel()

        metrics = evaluate(model, validation_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running_loss / examples
        if teacher is not None:
            metrics["distillation_loss"] = running_distillation / examples
            metrics["representation_loss"] = running_representation / examples
            metrics["attention_distillation_loss"] = (
                running_attention_distillation / examples
            )
        if args.multitask_long_view:
            metrics["long_view_loss"] = running_long_view / examples
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))
        if not math.isnan(metrics["auc"]) and metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": {
                        "num_items": args.num_items,
                        "embedding_dim": args.embedding_dim,
                        "top_k": args.top_k,
                        "temperature": args.temperature,
                        "selection_mode": args.selection_mode,
                        "recent_k": args.recent_k,
                        "use_content_features": args.use_content_features,
                        "use_behavior_features": args.use_behavior_features,
                        "num_tags": args.num_tags,
                        "num_author_buckets": args.num_author_buckets,
                        "num_duration_buckets": args.num_duration_buckets,
                        "content_embedding_dim": args.content_embedding_dim,
                        "use_multitask": args.multitask_long_view,
                        "use_global_summary": args.use_global_summary,
                    },
                    "metrics": metrics,
                },
                args.output / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping after {epoch} epochs; "
                f"validation AUC did not improve for {epochs_without_improvement} epochs."
            )
            break

    (args.output / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Finished on {device}. Best validation AUC: {best_auc:.4f}")
    print(f"Checkpoint: {args.output / 'best_model.pt'}")


if __name__ == "__main__":
    main()

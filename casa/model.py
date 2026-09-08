from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class CASAOutput:
    logits: Tensor
    user_interest: Tensor
    long_view_logits: Tensor | None
    gate_probabilities: Tensor
    selected_mask: Tensor
    selected_indices: Tensor
    attention_weights: Tensor


class ContentAdaptiveSparseAttentionCTR(nn.Module):
    """Candidate-aware sparse attention model for binary CTR prediction.

    Training uses a dense Gumbel relaxation so every valid behaviour receives
    a selection gradient. Evaluation uses hard Top-k and gathers only the
    selected behaviours before fine-grained attention.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 32,
        gate_hidden_dim: int = 64,
        predictor_hidden_dim: int = 64,
        top_k: int = 20,
        temperature: float = 1.0,
        dropout: float = 0.1,
        selection_mode: str = "learned",
        recent_k: int = 5,
        use_content_features: bool = False,
        use_behavior_features: bool = False,
        num_tags: int = 0,
        num_author_buckets: int = 0,
        num_duration_buckets: int = 0,
        content_embedding_dim: int = 8,
        use_multitask: bool = False,
        use_global_summary: bool = False,
    ) -> None:
        super().__init__()
        if num_items < 2:
            raise ValueError("num_items must be at least 2")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if selection_mode not in {"learned", "hybrid", "full", "recent", "random", "similarity"}:
            raise ValueError(f"unsupported selection_mode: {selection_mode}")
        if recent_k < 0 or (selection_mode == "hybrid" and recent_k > top_k):
            raise ValueError("recent_k must be between 0 and top_k")

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.temperature = temperature
        self.selection_mode = selection_mode
        self.recent_k = recent_k
        self.use_content_features = use_content_features
        self.use_behavior_features = use_behavior_features
        self.use_multitask = use_multitask
        self.use_global_summary = use_global_summary
        if use_global_summary:
            self.global_summary_scale = nn.Parameter(torch.zeros(()))

        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        if use_content_features:
            if min(num_tags, num_author_buckets, num_duration_buckets) < 1:
                raise ValueError("content feature cardinalities must be positive")
            self.tag_embedding = nn.Embedding(num_tags + 1, content_embedding_dim, padding_idx=0)
            self.author_embedding = nn.Embedding(
                num_author_buckets + 1, content_embedding_dim, padding_idx=0
            )
            self.duration_embedding = nn.Embedding(
                num_duration_buckets + 1, content_embedding_dim, padding_idx=0
            )
            self.content_projection = nn.Linear(content_embedding_dim * 3, embedding_dim)
        if use_behavior_features:
            self.behavior_projection = nn.Linear(3, embedding_dim)
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 4, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
        )
        # Candidate-aware recency prior. The scalar base captures the strong
        # short-term signal, while the candidate projection learns when older
        # semantically related events deserve more weight.
        self.recency_base = nn.Parameter(torch.tensor(1.0))
        self.recency_candidate = nn.Linear(embedding_dim, 1, bias=False)
        self.query_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 4, predictor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden_dim, 1),
        )
        if use_multitask:
            self.long_view_predictor = nn.Sequential(
                nn.Linear(embedding_dim * 4, predictor_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(predictor_hidden_dim, 1),
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Content starts as a no-op residual adapter, preserving a pretrained
        # ID-only teacher at initialization and learning content incrementally.
        if self.use_content_features:
            nn.init.zeros_(self.content_projection.weight)
            nn.init.zeros_(self.content_projection.bias)
        if self.use_behavior_features:
            nn.init.zeros_(self.behavior_projection.weight)
            nn.init.zeros_(self.behavior_projection.bias)
        nn.init.zeros_(self.recency_candidate.weight)

    @staticmethod
    def _sample_gumbel_like(tensor: Tensor) -> Tensor:
        uniform = torch.rand_like(tensor).clamp_(1e-6, 1.0 - 1e-6)
        return -torch.log(-torch.log(uniform))

    def _gate_logits(self, history: Tensor, candidate: Tensor) -> Tensor:
        repeated_candidate = candidate.unsqueeze(1).expand_as(history)
        features = torch.cat(
            [
                history,
                repeated_candidate,
                history * repeated_candidate,
                torch.abs(history - repeated_candidate),
            ],
            dim=-1,
        )
        return self.gate(features).squeeze(-1)

    def _selection_scores(self, history: Tensor, candidate: Tensor, valid_mask: Tensor) -> Tensor:
        if self.selection_mode in {"learned", "hybrid"}:
            scores = self._gate_logits(history, candidate)
            positions = torch.arange(
                history.size(1), device=history.device, dtype=history.dtype
            ).unsqueeze(0)
            last_valid = (valid_mask.sum(dim=1, keepdim=True) - 1).clamp_min(1)
            normalized_recency = positions / last_valid.to(history.dtype)
            candidate_adjustment = self.recency_candidate(candidate)
            recency_strength = self.recency_base + candidate_adjustment
            scores = scores + recency_strength * normalized_recency
        elif self.selection_mode == "recent":
            scores = torch.arange(history.size(1), device=history.device, dtype=history.dtype)
            scores = scores.unsqueeze(0).expand(history.size(0), -1)
        elif self.selection_mode == "random":
            scores = torch.rand(history.shape[:2], device=history.device, dtype=history.dtype)
        elif self.selection_mode == "similarity":
            scores = F.cosine_similarity(history, candidate.unsqueeze(1), dim=-1)
        else:
            scores = torch.zeros(history.shape[:2], device=history.device, dtype=history.dtype)
        return scores.masked_fill(~valid_mask, -20.0)

    def _embed_with_content(
        self, item_ids: Tensor, tags: Tensor | None, authors: Tensor | None, durations: Tensor | None
    ) -> Tensor:
        embedding = self.item_embedding(item_ids)
        if not self.use_content_features:
            return embedding
        if tags is None or authors is None or durations is None:
            raise ValueError("content feature tensors are required")
        content = torch.cat([
            self.tag_embedding(tags), self.author_embedding(authors), self.duration_embedding(durations)
        ], dim=-1)
        return embedding + self.content_projection(content)

    def _selection_mask_and_indices(
        self, scores: Tensor, valid_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.selection_mode != "hybrid" or self.recent_k == 0:
            return self._hard_topk_mask(scores, valid_mask, self.top_k)
        recent_count = min(self.recent_k, self.top_k, scores.size(1))
        positions = torch.arange(scores.size(1), device=scores.device, dtype=scores.dtype)
        position_scores = positions.unsqueeze(0).expand_as(scores)
        recent_mask, recent_indices = self._hard_topk_mask(
            position_scores, valid_mask, recent_count
        )
        remaining = self.top_k - recent_count
        if remaining == 0:
            return recent_mask, recent_indices
        eligible = valid_mask & ~recent_mask.bool()
        learned_mask, learned_indices = self._hard_topk_mask(scores, eligible, remaining)
        return torch.maximum(recent_mask, learned_mask), torch.cat(
            [recent_indices, learned_indices], dim=1
        )

    @staticmethod
    def _hard_topk_mask(scores: Tensor, valid_mask: Tensor, top_k: int) -> tuple[Tensor, Tensor]:
        sequence_length = scores.size(1)
        k = min(top_k, sequence_length)
        masked_scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        indices = masked_scores.topk(k=k, dim=1).indices
        chosen_valid = valid_mask.gather(1, indices)
        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(1, indices, chosen_valid.to(scores.dtype))
        return hard_mask, indices

    def _training_forward(
        self,
        history: Tensor,
        candidate: Tensor,
        gate_logits: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        noisy_logits = gate_logits + self._sample_gumbel_like(gate_logits)
        if self.selection_mode == "hybrid" and self.recent_k > 0:
            positions = torch.arange(history.size(1), device=history.device, dtype=history.dtype)
            position_scores = positions.unsqueeze(0).expand_as(noisy_logits)
            recent_count = min(self.recent_k, self.top_k, history.size(1))
            recent_mask, _ = self._hard_topk_mask(position_scores, valid_mask, recent_count)
            eligible = valid_mask & ~recent_mask.bool()
            masked_noisy_logits = noisy_logits.masked_fill(~eligible, -1e9)
            learned_relaxed = F.softmax(
                masked_noisy_logits / max(self.temperature, 1e-4), dim=1
            ) * eligible.to(history.dtype)
            learned_relaxed = learned_relaxed / learned_relaxed.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            relaxed = recent_mask + learned_relaxed * max(self.top_k - recent_count, 0)
            relaxed = relaxed.clamp(max=1.0)
        else:
            masked_noisy_logits = noisy_logits.masked_fill(~valid_mask, -1e9)
            relaxed = F.softmax(
                masked_noisy_logits / max(self.temperature, 1e-4), dim=1
            )
            relaxed = (relaxed * min(self.top_k, history.size(1))).clamp(max=1.0)
        hard_mask, indices = self._selection_mask_and_indices(noisy_logits, valid_mask)
        query = self.query_projection(candidate).unsqueeze(1)
        keys = self.key_projection(history)
        values = self.value_projection(history)
        attention_logits = (query * keys).sum(dim=-1) / math.sqrt(self.embedding_dim)
        attention_logits = attention_logits + torch.log(relaxed.clamp_min(1e-8))
        attention_logits = attention_logits.masked_fill(~valid_mask, -1e9)
        attention = F.softmax(attention_logits, dim=1)
        user_interest = torch.bmm(attention.unsqueeze(1), values).squeeze(1)
        return user_interest, torch.sigmoid(gate_logits), hard_mask, attention

    def _inference_forward(
        self,
        history: Tensor,
        candidate: Tensor,
        gate_logits: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        hard_mask, indices = self._selection_mask_and_indices(gate_logits, valid_mask)
        gather_index = indices.unsqueeze(-1).expand(-1, -1, history.size(-1))
        selected_history = history.gather(1, gather_index)
        selected_valid = valid_mask.gather(1, indices)

        query = self.query_projection(candidate).unsqueeze(1)
        keys = self.key_projection(selected_history)
        values = self.value_projection(selected_history)
        attention_logits = (query * keys).sum(dim=-1) / math.sqrt(self.embedding_dim)
        attention_logits = attention_logits.masked_fill(~selected_valid, -1e9)
        attention = F.softmax(attention_logits, dim=1)
        attention = attention * selected_valid.to(attention.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        user_interest = torch.bmm(attention.unsqueeze(1), values).squeeze(1)
        return user_interest, torch.sigmoid(gate_logits), hard_mask, attention, indices

    def _full_attention_forward(
        self, history: Tensor, candidate: Tensor, valid_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        query = self.query_projection(candidate).unsqueeze(1)
        keys = self.key_projection(history)
        values = self.value_projection(history)
        attention_logits = (query * keys).sum(dim=-1) / math.sqrt(self.embedding_dim)
        attention_logits = attention_logits.masked_fill(~valid_mask, -1e9)
        attention = F.softmax(attention_logits, dim=1) * valid_mask.to(history.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        user_interest = torch.bmm(attention.unsqueeze(1), values).squeeze(1)
        indices = torch.arange(history.size(1), device=history.device).unsqueeze(0)
        indices = indices.expand(history.size(0), -1)
        selected_mask = valid_mask.to(history.dtype)
        return user_interest, selected_mask, selected_mask, attention, indices

    def forward(
        self,
        history_ids: Tensor,
        candidate_ids: Tensor,
        history_tags: Tensor | None = None,
        candidate_tags: Tensor | None = None,
        history_authors: Tensor | None = None,
        candidate_authors: Tensor | None = None,
        history_durations: Tensor | None = None,
        candidate_durations: Tensor | None = None,
        history_actions: Tensor | None = None,
        history_watch_buckets: Tensor | None = None,
        history_time_gap_buckets: Tensor | None = None,
    ) -> CASAOutput:
        if history_ids.ndim != 2:
            raise ValueError("history_ids must have shape [batch, sequence_length]")
        if candidate_ids.ndim != 1 or candidate_ids.size(0) != history_ids.size(0):
            raise ValueError("candidate_ids must have shape [batch]")

        valid_mask = history_ids.ne(0)
        if not valid_mask.any(dim=1).all():
            raise ValueError("each example must contain at least one non-padding history item")

        history = self._embed_with_content(
            history_ids, history_tags, history_authors, history_durations
        )
        if self.use_behavior_features:
            if any(value is None for value in (
                history_actions, history_watch_buckets, history_time_gap_buckets
            )):
                raise ValueError("behavior feature tensors are required")
            action_strength = history_actions.to(history.dtype) / 4.0
            watch_completion = history_watch_buckets.to(history.dtype) / 5.0
            freshness = torch.where(
                history_time_gap_buckets.gt(0),
                (7.0 - history_time_gap_buckets.to(history.dtype)) / 6.0,
                torch.zeros_like(history_time_gap_buckets, dtype=history.dtype),
            )
            behavior = torch.stack(
                [action_strength, watch_completion, freshness], dim=-1
            )
            history = history + self.behavior_projection(behavior)
        candidate = self._embed_with_content(
            candidate_ids, candidate_tags, candidate_authors, candidate_durations
        )
        selection_scores = self._selection_scores(history, candidate, valid_mask)

        if self.selection_mode == "full":
            user_interest, gate_prob, selected_mask, attention, indices = self._full_attention_forward(
                history, candidate, valid_mask
            )
        elif self.selection_mode in {"learned", "hybrid"} and self.training:
            user_interest, gate_prob, selected_mask, attention = self._training_forward(
                history, candidate, selection_scores, valid_mask
            )
            _, indices = self._selection_mask_and_indices(selection_scores, valid_mask)
        else:
            user_interest, gate_prob, selected_mask, attention, indices = self._inference_forward(
                history, candidate, selection_scores, valid_mask
            )

        if self.use_global_summary:
            global_summary = (history * valid_mask.unsqueeze(-1)).sum(dim=1)
            global_summary = global_summary / valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
            user_interest = user_interest + torch.tanh(self.global_summary_scale) * global_summary

        predictor_input = torch.cat(
            [candidate, user_interest, candidate * user_interest, torch.abs(candidate - user_interest)],
            dim=-1,
        )
        logits = self.predictor(predictor_input).squeeze(-1)
        long_view_logits = (
            self.long_view_predictor(predictor_input).squeeze(-1)
            if self.use_multitask else None
        )
        return CASAOutput(
            logits=logits,
            user_interest=user_interest,
            long_view_logits=long_view_logits,
            gate_probabilities=gate_prob,
            selected_mask=selected_mask,
            selected_indices=indices,
            attention_weights=attention,
        )

    @staticmethod
    def loss(
        output: CASAOutput,
        labels: Tensor,
        sparsity_weight: float = 1e-3,
        target_selection_rate: float | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        ctr_loss = F.binary_cross_entropy_with_logits(output.logits, labels.float())
        selection_rate = output.gate_probabilities.mean()
        if target_selection_rate is None:
            sparsity_loss = selection_rate
        else:
            target = torch.as_tensor(target_selection_rate, device=labels.device)
            sparsity_loss = (selection_rate - target).square()
        total = ctr_loss + sparsity_weight * sparsity_loss
        return total, {
            "total_loss": total.detach(),
            "ctr_loss": ctr_loss.detach(),
            "sparsity_loss": sparsity_loss.detach(),
            "selection_rate": selection_rate.detach(),
        }

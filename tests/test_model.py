import unittest

import torch

from casa.model import ContentAdaptiveSparseAttentionCTR


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = ContentAdaptiveSparseAttentionCTR(
            num_items=100, embedding_dim=16, top_k=3, dropout=0.0
        )
        self.histories = torch.tensor(
            [[1, 2, 3, 4, 5, 0], [6, 7, 8, 9, 10, 11]], dtype=torch.long
        )
        self.candidates = torch.tensor([12, 13], dtype=torch.long)
        self.labels = torch.tensor([1.0, 0.0])

    def test_training_forward_and_backward(self):
        self.model.train()
        output = self.model(self.histories, self.candidates)
        self.assertEqual(tuple(output.logits.shape), (2,))
        self.assertEqual(tuple(output.user_interest.shape), (2, 16))
        self.assertTrue(torch.equal(output.selected_mask.sum(dim=1), torch.tensor([3.0, 3.0])))
        loss, _ = self.model.loss(output, self.labels)
        loss.backward()
        self.assertIsNotNone(self.model.gate[0].weight.grad)
        self.assertGreater(self.model.gate[0].weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(self.model.recency_base.grad)
        self.assertGreater(self.model.recency_base.grad.abs().item(), 0.0)
        self.assertTrue(torch.allclose(output.attention_weights.sum(dim=1), torch.ones(2)))

    def test_sparse_inference(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.histories, self.candidates)
        self.assertEqual(tuple(output.selected_indices.shape), (2, 3))
        self.assertEqual(tuple(output.attention_weights.shape), (2, 3))
        self.assertTrue(torch.allclose(output.attention_weights.sum(dim=1), torch.ones(2)))

    def test_padding_is_never_selected(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.histories, self.candidates)
        self.assertEqual(output.selected_mask[0, -1].item(), 0.0)

    def test_all_baseline_modes(self):
        for mode in ("full", "recent", "random", "similarity", "hybrid"):
            model = ContentAdaptiveSparseAttentionCTR(
                num_items=100, embedding_dim=16, top_k=3, recent_k=2,
                dropout=0.0, selection_mode=mode
            )
            output = model(self.histories, self.candidates)
            self.assertEqual(tuple(output.logits.shape), (2,))
            expected = torch.tensor([5.0, 6.0]) if mode == "full" else torch.tensor([3.0, 3.0])
            self.assertTrue(torch.equal(output.selected_mask.sum(dim=1), expected))

    def test_recent_selects_last_valid_items(self):
        model = ContentAdaptiveSparseAttentionCTR(
            num_items=100, embedding_dim=16, top_k=2, dropout=0.0, selection_mode="recent"
        )
        model.eval()
        output = model(self.histories, self.candidates)
        self.assertEqual(set(output.selected_indices[0].tolist()), {3, 4})

    def test_hybrid_always_keeps_recent_quota(self):
        model = ContentAdaptiveSparseAttentionCTR(
            num_items=100, embedding_dim=16, top_k=4, recent_k=2,
            dropout=0.0, selection_mode="hybrid"
        )
        model.eval()
        output = model(self.histories, self.candidates)
        selected = set(output.selected_indices[0].tolist())
        self.assertTrue({3, 4}.issubset(selected))

    def test_content_feature_forward(self):
        model = ContentAdaptiveSparseAttentionCTR(
            num_items=100, embedding_dim=16, top_k=3, selection_mode="learned",
            use_content_features=True, num_tags=10, num_author_buckets=16,
            num_duration_buckets=6, content_embedding_dim=4,
        )
        history_tags = torch.tensor([[1, 2, 3, 4, 5, 0], [1, 1, 2, 2, 3, 3]])
        history_authors = torch.tensor([[1, 2, 3, 4, 5, 0], [2, 3, 4, 5, 6, 7]])
        history_durations = torch.tensor([[1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 6]])
        output = model(
            self.histories, self.candidates,
            history_tags=history_tags, candidate_tags=torch.tensor([2, 3]),
            history_authors=history_authors, candidate_authors=torch.tensor([3, 4]),
            history_durations=history_durations, candidate_durations=torch.tensor([2, 4]),
        )
        self.assertEqual(tuple(output.logits.shape), (2,))

    def test_multitask_output(self):
        model = ContentAdaptiveSparseAttentionCTR(
            num_items=100, embedding_dim=16, top_k=3, use_multitask=True
        )
        output = model(self.histories, self.candidates)
        self.assertEqual(tuple(output.long_view_logits.shape), (2,))

    def test_behavior_feature_forward_and_gradient(self):
        model = ContentAdaptiveSparseAttentionCTR(
            num_items=100, embedding_dim=16, top_k=3,
            use_behavior_features=True,
        )
        output = model(
            self.histories, self.candidates,
            history_actions=torch.tensor([[1, 2, 3, 4, 1, 0], [1, 1, 2, 3, 4, 2]]),
            history_watch_buckets=torch.tensor([[1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 2]]),
            history_time_gap_buckets=torch.tensor([[6, 5, 4, 3, 1, 0], [6, 5, 4, 3, 2, 1]]),
        )
        output.logits.sum().backward()
        self.assertIsNotNone(model.behavior_projection.weight.grad)
        self.assertGreater(model.behavior_projection.weight.grad.abs().sum().item(), 0.0)

    def test_global_summary_receives_gradient(self):
        model = ContentAdaptiveSparseAttentionCTR(
            num_items=100, embedding_dim=16, top_k=3, use_global_summary=True
        )
        output = model(self.histories, self.candidates)
        output.logits.sum().backward()
        self.assertIsNotNone(model.global_summary_scale.grad)


if __name__ == "__main__":
    unittest.main()

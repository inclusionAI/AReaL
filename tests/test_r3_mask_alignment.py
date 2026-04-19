import pytest
import torch

from areal.engine.megatron_engine_r3_patch import _align_routed_experts_to_mask


class TestAlignRoutedExpertsToMask:
    """Tests for _align_routed_experts_to_mask: seq and batch alignment."""

    def _make_left_padded_re(self, bs, seqlen, num_layers=2, topk=3, pad_val=0):
        re = torch.randint(1, 64, (bs, seqlen, num_layers, topk))
        re[:, 0, :, :] = pad_val
        re[:, 1, :, :] = pad_val
        return re

    def test_same_shape_no_change(self):
        routed_experts = torch.randint(1, 64, (4, 10, 2, 3))
        attention_mask = torch.ones(4, 10, dtype=torch.long)
        result = _align_routed_experts_to_mask(routed_experts, attention_mask)
        torch.testing.assert_close(result, routed_experts)

    def test_seq_dim_shorter_right_pad(self):
        routed_experts = torch.randint(1, 64, (2, 5, 2, 3))
        attention_mask = torch.ones(2, 8, dtype=torch.long)
        result = _align_routed_experts_to_mask(routed_experts, attention_mask)
        assert result.shape == (2, 8, 2, 3)
        torch.testing.assert_close(result[:, :5, :, :], routed_experts)
        assert (result[:, 5:, :, :] == 0).all()

    def test_seq_dim_longer_left_padded_to_left_aligned(self):
        bs, re_seqlen, mask_seqlen, num_layers, topk = 3, 10, 6, 2, 3
        routed_experts = torch.zeros(bs, re_seqlen, num_layers, topk, dtype=torch.long)
        routed_experts[:, 4:, :, :] = torch.randint(1, 64, (bs, 6, num_layers, topk))
        attention_mask = torch.ones(bs, mask_seqlen, dtype=torch.long)
        result = _align_routed_experts_to_mask(routed_experts, attention_mask)
        assert result.shape == (bs, mask_seqlen, num_layers, topk)
        torch.testing.assert_close(result, routed_experts[:, 4:, :, :])

    def test_seq_dim_longer_with_varying_lengths(self):
        bs, re_seqlen, mask_seqlen, num_layers, topk = 2, 10, 8, 2, 3
        routed_experts = torch.zeros(bs, re_seqlen, num_layers, topk, dtype=torch.long)
        routed_experts[0, 3:, :, :] = torch.randint(1, 64, (7, num_layers, topk))
        routed_experts[1, 5:, :, :] = torch.randint(1, 64, (5, num_layers, topk))
        attention_mask = torch.zeros(bs, mask_seqlen, dtype=torch.long)
        attention_mask[0, :7] = 1
        attention_mask[1, :5] = 1
        result = _align_routed_experts_to_mask(routed_experts, attention_mask)
        assert result.shape == (bs, mask_seqlen, num_layers, topk)
        torch.testing.assert_close(result[0, :7, :, :], routed_experts[0, 3:, :, :])
        torch.testing.assert_close(result[1, :5, :, :], routed_experts[1, 5:, :, :])
        assert (result[0, 7:, :, :] == 0).all()
        assert (result[1, 5:, :, :] == 0).all()

    def test_batch_dim_smaller_padded(self):
        routed_experts = torch.randint(1, 64, (3, 8, 2, 3))
        attention_mask = torch.ones(5, 8, dtype=torch.long)
        result = _align_routed_experts_to_mask(routed_experts, attention_mask)
        assert result.shape == (5, 8, 2, 3)
        torch.testing.assert_close(result[:3, :, :, :], routed_experts)
        assert (result[3:, :, :, :] == 0).all()

    def test_batch_dim_larger_truncated(self):
        routed_experts = torch.randint(1, 64, (5, 8, 2, 3))
        attention_mask = torch.ones(3, 8, dtype=torch.long)
        result = _align_routed_experts_to_mask(routed_experts, attention_mask)
        assert result.shape == (3, 8, 2, 3)
        torch.testing.assert_close(result, routed_experts[:3])

    def test_both_batch_and_seq_mismatch(self):
        routed_experts = torch.zeros(2, 10, 2, 3, dtype=torch.long)
        routed_experts[:, 4:, :, :] = torch.randint(1, 64, (2, 6, 2, 3))
        attention_mask = torch.ones(4, 6, dtype=torch.long)
        result = _align_routed_experts_to_mask(routed_experts, attention_mask)
        assert result.shape == (4, 6, 2, 3)
        torch.testing.assert_close(result[:2, :, :, :], routed_experts[:, 4:, :, :])
        assert (result[2:, :, :, :] == 0).all()

    def test_empty_attention_mask_same_seqlen(self):
        routed_experts = torch.randint(1, 64, (2, 8, 2, 3))
        attention_mask = torch.zeros(2, 8, dtype=torch.long)
        result = _align_routed_experts_to_mask(routed_experts, attention_mask)
        assert result.shape == (2, 8, 2, 3)
        torch.testing.assert_close(result, routed_experts)

    def test_empty_attention_mask_longer_re_seqlen(self):
        routed_experts = torch.zeros(2, 10, 2, 3, dtype=torch.long)
        routed_experts[:, 4:, :, :] = torch.randint(1, 64, (2, 6, 2, 3))
        attention_mask = torch.zeros(2, 6, dtype=torch.long)
        result = _align_routed_experts_to_mask(routed_experts, attention_mask)
        assert result.shape == (2, 6, 2, 3)
        assert (result == 0).all()


class TestLogMoeRoutingMetricsMaskFallback:
    """Tests for log_moe_routing_metrics: attn_mask seq_len mismatch handling."""

    def test_matching_mask_uses_real_mask(self):
        from areal.trainer.ppo.actor_r3_patch import log_moe_routing_metrics

        bs, seq_len, num_layers, topk = 2, 10, 2, 3
        re = torch.randint(1, 64, (bs, seq_len, num_layers, topk))
        attn_mask = torch.ones(bs, seq_len, dtype=torch.long)
        attn_mask[0, 7:] = 0
        data = {"routed_experts": re, "attention_mask": attn_mask}
        log_moe_routing_metrics(data, scope="test_moe")

    def test_shorter_mask_falls_back_to_all_ones(self):
        from areal.trainer.ppo.actor_r3_patch import log_moe_routing_metrics

        bs, re_seqlen, num_layers, topk = 2, 20, 2, 3
        mask_seqlen = 12
        re = torch.randint(1, 64, (bs, re_seqlen, num_layers, topk))
        attn_mask = torch.ones(bs, mask_seqlen, dtype=torch.long)
        data = {"routed_experts": re, "attention_mask": attn_mask}
        log_moe_routing_metrics(data, scope="test_moe")

    def test_longer_mask_falls_back_to_all_ones(self):
        from areal.trainer.ppo.actor_r3_patch import log_moe_routing_metrics

        bs, re_seqlen, num_layers, topk = 2, 10, 2, 3
        mask_seqlen = 20
        re = torch.randint(1, 64, (bs, re_seqlen, num_layers, topk))
        attn_mask = torch.ones(bs, mask_seqlen, dtype=torch.long)
        data = {"routed_experts": re, "attention_mask": attn_mask}
        log_moe_routing_metrics(data, scope="test_moe")

    def test_no_mask_falls_back_to_all_ones(self):
        from areal.trainer.ppo.actor_r3_patch import log_moe_routing_metrics

        bs, seq_len, num_layers, topk = 2, 10, 2, 3
        re = torch.randint(1, 64, (bs, seq_len, num_layers, topk))
        data = {"routed_experts": re}
        log_moe_routing_metrics(data, scope="test_moe")

    def test_none_routed_experts_returns_early(self):
        from areal.trainer.ppo.actor_r3_patch import log_moe_routing_metrics

        data = {"attention_mask": torch.ones(2, 10)}
        log_moe_routing_metrics(data, scope="test_moe")

    def test_low_dim_routed_experts_returns_early(self):
        from areal.trainer.ppo.actor_r3_patch import log_moe_routing_metrics

        data = {"routed_experts": torch.randint(1, 64, (2, 10))}
        log_moe_routing_metrics(data, scope="test_moe")

"""Tests for ebsd_ai.features.fusion module."""

from __future__ import annotations

import torch
import pytest

from ebsd_ai.features.fusion import AttentionGate, MultimodalFusion


class TestAttentionGate:
    def test_output_shape(self) -> None:
        gate = AttentionGate(pattern_dim=256, eds_dim=64)
        p = torch.randn(4, 256)
        e = torch.randn(4, 64)
        alpha = gate(p, e)
        assert alpha.shape == (4, 1)

    def test_output_range(self) -> None:
        """Alpha should be in [0, 1] due to sigmoid."""
        gate = AttentionGate(pattern_dim=256, eds_dim=64)
        gate.eval()
        p = torch.randn(8, 256)
        e = torch.randn(8, 64)
        with torch.no_grad():
            alpha = gate(p, e)
        assert torch.all(alpha >= 0.0)
        assert torch.all(alpha <= 1.0)

    def test_batch_1(self) -> None:
        gate = AttentionGate(pattern_dim=128, eds_dim=32)
        p = torch.randn(1, 128)
        e = torch.randn(1, 32)
        alpha = gate(p, e)
        assert alpha.shape == (1, 1)

    def test_gradient_flows(self) -> None:
        gate = AttentionGate(pattern_dim=64, eds_dim=32)
        p = torch.randn(4, 64, requires_grad=True)
        e = torch.randn(4, 32, requires_grad=True)
        alpha = gate(p, e)
        alpha.sum().backward()
        assert p.grad is not None
        assert e.grad is not None


class TestMultimodalFusion:
    def test_output_shape_default(self) -> None:
        fusion = MultimodalFusion(pattern_dim=256, eds_dim=64, fused_dim=128)
        p = torch.randn(4, 256)
        e = torch.randn(4, 64)
        fused, alpha = fusion(p, e)
        assert fused.shape == (4, 128)
        assert alpha.shape == (4, 1)

    def test_output_shape_batch_1(self) -> None:
        fusion = MultimodalFusion(pattern_dim=256, eds_dim=64, fused_dim=128)
        p = torch.randn(1, 256)
        e = torch.randn(1, 64)
        fused, alpha = fusion(p, e)
        assert fused.shape == (1, 128)
        assert alpha.shape == (1, 1)

    def test_output_shape_batch_16(self) -> None:
        fusion = MultimodalFusion(pattern_dim=256, eds_dim=64, fused_dim=128)
        p = torch.randn(16, 256)
        e = torch.randn(16, 64)
        fused, alpha = fusion(p, e)
        assert fused.shape == (16, 128)

    def test_custom_dims(self) -> None:
        fusion = MultimodalFusion(
            pattern_dim=512, eds_dim=128, fused_dim=256, dropout=0.1
        )
        p = torch.randn(4, 512)
        e = torch.randn(4, 128)
        fused, alpha = fusion(p, e)
        assert fused.shape == (4, 256)
        assert alpha.shape == (4, 1)

    def test_no_nan(self) -> None:
        fusion = MultimodalFusion()
        fusion.eval()
        p = torch.randn(8, 256)
        e = torch.randn(8, 64)
        with torch.no_grad():
            fused, alpha = fusion(p, e)
        assert not torch.any(torch.isnan(fused))
        assert not torch.any(torch.isinf(fused))
        assert not torch.any(torch.isnan(alpha))

    def test_alpha_range(self) -> None:
        """Alpha must be in [0, 1]."""
        fusion = MultimodalFusion()
        fusion.eval()
        p = torch.randn(16, 256)
        e = torch.randn(16, 64)
        with torch.no_grad():
            _, alpha = fusion(p, e)
        assert torch.all(alpha >= 0.0)
        assert torch.all(alpha <= 1.0)

    def test_zero_eds_does_not_crash(self) -> None:
        """When EDS is all zeros the fusion should still produce valid output."""
        fusion = MultimodalFusion()
        fusion.eval()
        p = torch.randn(4, 256)
        e = torch.zeros(4, 64)
        with torch.no_grad():
            fused, alpha = fusion(p, e)
        assert not torch.any(torch.isnan(fused))
        assert fused.shape == (4, 128)

    def test_gradient_flows(self) -> None:
        fusion = MultimodalFusion()
        p = torch.randn(4, 256, requires_grad=True)
        e = torch.randn(4, 64, requires_grad=True)
        fused, alpha = fusion(p, e)
        fused.sum().backward()
        assert p.grad is not None
        assert e.grad is not None

    def test_eds_contribution_interpretation(self) -> None:
        """eds_contribution = 1 - alpha should also be in [0, 1]."""
        fusion = MultimodalFusion()
        fusion.eval()
        p = torch.randn(8, 256)
        e = torch.randn(8, 64)
        with torch.no_grad():
            _, alpha = fusion(p, e)
        eds_contribution = 1.0 - alpha
        assert torch.all(eds_contribution >= 0.0)
        assert torch.all(eds_contribution <= 1.0)

    def test_parameter_count(self) -> None:
        """Fusion module should be lightweight."""
        fusion = MultimodalFusion()
        total_params = sum(p.numel() for p in fusion.parameters())
        # Should be well under 500K params for the fusion module alone
        assert total_params < 500_000

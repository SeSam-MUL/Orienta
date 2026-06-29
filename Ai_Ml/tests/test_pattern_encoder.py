"""Tests for ebsd_ai.features.pattern_encoder module."""

from __future__ import annotations

import torch
import pytest

from ebsd_ai.features.pattern_encoder import PatternEncoder, ResidualBlock


class TestResidualBlock:
    def test_same_channels(self) -> None:
        block = ResidualBlock(32, 32, stride=1)
        x = torch.randn(2, 32, 16, 16)
        out = block(x)
        assert out.shape == (2, 32, 16, 16)

    def test_downsample(self) -> None:
        block = ResidualBlock(32, 64, stride=2)
        x = torch.randn(2, 32, 16, 16)
        out = block(x)
        assert out.shape == (2, 64, 8, 8)

    def test_no_nan(self) -> None:
        block = ResidualBlock(16, 32, stride=2)
        x = torch.randn(4, 16, 32, 32)
        out = block(x)
        assert not torch.any(torch.isnan(out))


class TestPatternEncoder:
    def test_output_shape_default(self) -> None:
        encoder = PatternEncoder(feature_dim=256, pattern_size=128)
        x = torch.randn(4, 1, 128, 128)
        out = encoder(x)
        assert out.shape == (4, 256)

    def test_output_shape_batch_1(self) -> None:
        encoder = PatternEncoder()
        x = torch.randn(1, 1, 128, 128)
        out = encoder(x)
        assert out.shape == (1, 256)

    def test_output_shape_batch_16(self) -> None:
        encoder = PatternEncoder()
        x = torch.randn(16, 1, 128, 128)
        out = encoder(x)
        assert out.shape == (16, 256)

    def test_custom_feature_dim(self) -> None:
        encoder = PatternEncoder(feature_dim=512)
        x = torch.randn(2, 1, 128, 128)
        out = encoder(x)
        assert out.shape == (2, 512)

    def test_no_nan_random_input(self) -> None:
        encoder = PatternEncoder()
        encoder.eval()
        x = torch.randn(4, 1, 128, 128)
        with torch.no_grad():
            out = encoder(x)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))

    def test_parameter_count(self) -> None:
        encoder = PatternEncoder()
        total_params = sum(p.numel() for p in encoder.parameters())
        assert total_params < 5_000_000  # < 5M params

    def test_freeze_early_layers(self) -> None:
        encoder = PatternEncoder()
        encoder.freeze_early_layers()

        # Stem and layers 1-2 should be frozen
        for name, param in encoder.named_parameters():
            if any(n in name for n in ["stem", "layer1", "layer2"]):
                assert not param.requires_grad, f"{name} should be frozen"

        # layer3, layer4, fc should still be trainable
        for name, param in encoder.named_parameters():
            if any(n in name for n in ["layer3", "layer4", "fc"]):
                assert param.requires_grad, f"{name} should be trainable"

    def test_unfreeze_all(self) -> None:
        encoder = PatternEncoder()
        encoder.freeze_early_layers()
        encoder.unfreeze_all()
        for param in encoder.parameters():
            assert param.requires_grad

    def test_gradient_flows(self) -> None:
        encoder = PatternEncoder()
        x = torch.randn(2, 1, 128, 128)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        # Check that gradients exist on stem conv
        assert encoder.stem[0].weight.grad is not None

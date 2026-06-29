"""Tests for the PatternEnhancer U-Net model.

Covers:
- Model construction and configuration
- Forward pass shapes for various batch sizes
- FiLM conditioning (with and without detector metadata)
- Skip connection handling
- Single-pattern enhancement API
- Batch scan enhancement API
- Selection mask handling
- Serialization (save/load roundtrip)
- No NaN/Inf in outputs
- Gradient flow
- Parameter count sanity check
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ebsd_ai.models.pattern_enhancer import (
    ConvBlock,
    EnhancerConfig,
    FiLMLayer,
    PatternEnhancer,
)
from tests.conftest import make_synthetic_detector_info, make_synthetic_pattern

# ---------------------------------------------------------------------------
# EnhancerConfig tests
# ---------------------------------------------------------------------------


class TestEnhancerConfig:
    """Tests for EnhancerConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = EnhancerConfig()
        assert cfg.pattern_size == 128
        assert cfg.base_channels == 32
        assert cfg.depth == 4
        assert cfg.use_film is True
        assert cfg.detector_encoding_dim == 15
        assert cfg.dropout == 0.1

    def test_roundtrip(self) -> None:
        cfg = EnhancerConfig(base_channels=16, depth=3, use_film=False)
        d = cfg.to_dict()
        cfg2 = EnhancerConfig.from_dict(d)
        assert cfg2.base_channels == 16
        assert cfg2.depth == 3
        assert cfg2.use_film is False

    def test_from_dict_ignores_extra_keys(self) -> None:
        d = EnhancerConfig().to_dict()
        d["unknown_key"] = 42
        cfg = EnhancerConfig.from_dict(d)
        assert cfg.base_channels == 32


# ---------------------------------------------------------------------------
# Building block tests
# ---------------------------------------------------------------------------


class TestConvBlock:
    """Tests for the ConvBlock building block."""

    def test_output_shape(self) -> None:
        block = ConvBlock(1, 32)
        x = torch.randn(2, 1, 64, 64)
        out = block(x)
        assert out.shape == (2, 32, 64, 64)

    def test_channel_change(self) -> None:
        block = ConvBlock(32, 64)
        x = torch.randn(2, 32, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)


class TestFiLMLayer:
    """Tests for the FiLM conditioning layer."""

    def test_output_shape(self) -> None:
        film = FiLMLayer(conditioning_dim=15, n_channels=64)
        x = torch.randn(4, 64, 8, 8)
        cond = torch.randn(4, 15)
        out = film(x, cond)
        assert out.shape == (4, 64, 8, 8)

    def test_modulation_changes_values(self) -> None:
        film = FiLMLayer(conditioning_dim=15, n_channels=32)
        x = torch.randn(2, 32, 16, 16)
        cond = torch.randn(2, 15)
        out = film(x, cond)
        # The output should differ from input (with untrained random weights)
        assert not torch.allclose(x, out)

    def test_gradient_flow(self) -> None:
        film = FiLMLayer(conditioning_dim=15, n_channels=32)
        x = torch.randn(2, 32, 8, 8, requires_grad=True)
        cond = torch.randn(2, 15, requires_grad=True)
        out = film(x, cond)
        out.sum().backward()
        assert x.grad is not None
        assert cond.grad is not None


# ---------------------------------------------------------------------------
# PatternEnhancer model tests
# ---------------------------------------------------------------------------


class TestPatternEnhancerInit:
    """Tests for model construction."""

    def test_default_init(self) -> None:
        model = PatternEnhancer()
        assert model.config.depth == 4
        assert model.config.base_channels == 32
        assert model.film is not None

    def test_custom_config(self) -> None:
        cfg = EnhancerConfig(base_channels=16, depth=3, use_film=False)
        model = PatternEnhancer(config=cfg)
        assert model.config.depth == 3
        assert model.film is None
        assert len(model.encoders) == 3
        assert len(model.decoders) == 3

    def test_no_film(self) -> None:
        cfg = EnhancerConfig(use_film=False)
        model = PatternEnhancer(config=cfg)
        assert model.film is None

    def test_parameter_count_reasonable(self) -> None:
        model = PatternEnhancer()
        n_params = sum(p.numel() for p in model.parameters())
        # Should be reasonable — under 15M
        assert n_params < 15_000_000
        # Should have a meaningful number of parameters
        assert n_params > 10_000


class TestPatternEnhancerForward:
    """Tests for forward pass shapes and correctness."""

    @pytest.fixture
    def model(self) -> PatternEnhancer:
        return PatternEnhancer()

    @pytest.fixture
    def small_model(self) -> PatternEnhancer:
        cfg = EnhancerConfig(base_channels=8, depth=3)
        return PatternEnhancer(config=cfg)

    def test_basic_forward(self, model: PatternEnhancer) -> None:
        x = torch.randn(2, 1, 128, 128)
        out = model(x)
        assert out.shape == (2, 1, 128, 128)

    def test_batch_size_1(self, model: PatternEnhancer) -> None:
        x = torch.randn(1, 1, 128, 128)
        out = model(x)
        assert out.shape == (1, 1, 128, 128)

    def test_batch_size_8(self, small_model: PatternEnhancer) -> None:
        x = torch.randn(8, 1, 128, 128)
        out = small_model(x)
        assert out.shape == (8, 1, 128, 128)

    def test_no_nan_inf(self, model: PatternEnhancer) -> None:
        x = torch.randn(4, 1, 128, 128)
        out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_with_detector_encoding(self, model: PatternEnhancer) -> None:
        x = torch.randn(2, 1, 128, 128)
        det = torch.randn(2, 15)
        out = model(x, detector_encoding=det)
        assert out.shape == (2, 1, 128, 128)

    def test_without_detector_encoding(self, model: PatternEnhancer) -> None:
        x = torch.randn(2, 1, 128, 128)
        # FiLM is enabled but no detector encoding provided — should work
        out = model(x, detector_encoding=None)
        assert out.shape == (2, 1, 128, 128)

    def test_film_disabled(self) -> None:
        cfg = EnhancerConfig(use_film=False)
        model = PatternEnhancer(config=cfg)
        x = torch.randn(2, 1, 128, 128)
        # Should ignore detector encoding when FiLM is disabled
        det = torch.randn(2, 15)
        out = model(x, detector_encoding=det)
        assert out.shape == (2, 1, 128, 128)

    def test_gradient_flow(self, small_model: PatternEnhancer) -> None:
        x = torch.randn(2, 1, 128, 128, requires_grad=True)
        out = small_model(x)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_flow_with_film(self, small_model: PatternEnhancer) -> None:
        x = torch.randn(2, 1, 128, 128)
        det = torch.randn(2, 15, requires_grad=True)
        out = small_model(x, detector_encoding=det)
        loss = out.mean()
        loss.backward()
        assert det.grad is not None

    def test_different_depths(self) -> None:
        for depth in [2, 3, 4]:
            cfg = EnhancerConfig(base_channels=8, depth=depth)
            model = PatternEnhancer(config=cfg)
            x = torch.randn(1, 1, 128, 128)
            out = model(x)
            assert out.shape == (1, 1, 128, 128), f"Failed for depth={depth}"

    def test_eval_vs_train_mode(self, model: PatternEnhancer) -> None:
        x = torch.randn(2, 1, 128, 128)
        model.train()
        out_train = model(x)
        model.eval()
        out_eval = model(x)
        # Shapes should be the same in both modes
        assert out_train.shape == out_eval.shape


# ---------------------------------------------------------------------------
# High-level API tests
# ---------------------------------------------------------------------------


class TestEnhance:
    """Tests for the single-pattern enhance() method."""

    @pytest.fixture
    def model(self) -> PatternEnhancer:
        cfg = EnhancerConfig(base_channels=8, depth=3)
        return PatternEnhancer(config=cfg)

    def test_basic_enhance(
        self, model: PatternEnhancer, rng: np.random.Generator
    ) -> None:
        pattern = make_synthetic_pattern(120, 160, rng)
        result = model.enhance(pattern)
        assert result.shape == (128, 128)
        assert result.dtype == np.float32

    def test_enhance_various_sizes(
        self, model: PatternEnhancer, rng: np.random.Generator
    ) -> None:
        for h, w in [(60, 60), (80, 60), (120, 120), (320, 240)]:
            pattern = make_synthetic_pattern(h, w, rng)
            result = model.enhance(pattern)
            assert result.shape == (128, 128)

    def test_enhance_with_detector_info(
        self, model: PatternEnhancer, rng: np.random.Generator
    ) -> None:
        pattern = make_synthetic_pattern(120, 160, rng)
        det = make_synthetic_detector_info(rng=rng)
        result = model.enhance(pattern, detector_info=det)
        assert result.shape == (128, 128)

    def test_enhance_no_nan(
        self, model: PatternEnhancer, rng: np.random.Generator
    ) -> None:
        pattern = make_synthetic_pattern(128, 128, rng)
        result = model.enhance(pattern)
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

    def test_enhance_restores_training_mode(
        self, model: PatternEnhancer, rng: np.random.Generator
    ) -> None:
        model.train()
        pattern = make_synthetic_pattern(128, 128, rng)
        model.enhance(pattern)
        assert model.training


class TestEnhanceScan:
    """Tests for the batch scan enhance_scan() method."""

    @pytest.fixture
    def model(self) -> PatternEnhancer:
        cfg = EnhancerConfig(base_channels=8, depth=3)
        return PatternEnhancer(config=cfg)

    def test_basic_scan(
        self, model: PatternEnhancer, rng: np.random.Generator
    ) -> None:
        patterns = rng.integers(
            0, 256, size=(3, 4, 60, 80), dtype=np.uint8
        )
        result = model.enhance_scan(patterns, batch_size=4)
        assert result.shape == (3, 4, 128, 128)
        assert result.dtype == np.float32

    def test_scan_with_mask(
        self, model: PatternEnhancer, rng: np.random.Generator
    ) -> None:
        patterns = rng.integers(
            0, 256, size=(3, 4, 60, 80), dtype=np.uint8
        )
        mask = np.zeros((3, 4), dtype=bool)
        mask[0, 0] = True
        mask[1, 2] = True
        result = model.enhance_scan(patterns, selection_mask=mask)
        # Selected pixels should have non-zero values
        assert not np.allclose(result[0, 0], 0.0)
        assert not np.allclose(result[1, 2], 0.0)
        # Unselected pixels should be zero
        assert np.allclose(result[2, 3], 0.0)

    def test_scan_empty_mask(
        self, model: PatternEnhancer, rng: np.random.Generator
    ) -> None:
        patterns = rng.integers(
            0, 256, size=(2, 3, 60, 80), dtype=np.uint8
        )
        mask = np.zeros((2, 3), dtype=bool)
        result = model.enhance_scan(patterns, selection_mask=mask)
        assert np.allclose(result, 0.0)

    def test_scan_with_detector_info(
        self, model: PatternEnhancer, rng: np.random.Generator
    ) -> None:
        patterns = rng.integers(
            0, 256, size=(2, 2, 60, 80), dtype=np.uint8
        )
        det = make_synthetic_detector_info(rng=rng)
        result = model.enhance_scan(patterns, detector_info=det)
        assert result.shape == (2, 2, 128, 128)

    def test_scan_wrong_dimensions(
        self, model: PatternEnhancer, rng: np.random.Generator
    ) -> None:
        patterns_3d = rng.integers(0, 256, size=(4, 60, 80), dtype=np.uint8)
        with pytest.raises(ValueError, match="4-D"):
            model.enhance_scan(patterns_3d)

    def test_scan_mask_shape_mismatch(
        self, model: PatternEnhancer, rng: np.random.Generator
    ) -> None:
        patterns = rng.integers(
            0, 256, size=(3, 4, 60, 80), dtype=np.uint8
        )
        bad_mask = np.ones((5, 5), dtype=bool)
        with pytest.raises(ValueError, match="selection_mask shape"):
            model.enhance_scan(patterns, selection_mask=bad_mask)

    def test_scan_restores_training_mode(
        self, model: PatternEnhancer, rng: np.random.Generator
    ) -> None:
        model.train()
        patterns = rng.integers(
            0, 256, size=(2, 2, 60, 80), dtype=np.uint8
        )
        model.enhance_scan(patterns)
        assert model.training


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    """Tests for save/load roundtrip."""

    def test_save_dict_roundtrip(self) -> None:
        cfg = EnhancerConfig(base_channels=16, depth=3, use_film=True)
        model = PatternEnhancer(config=cfg)
        save_dict = model.get_save_dict()

        loaded = PatternEnhancer.from_save_dict(save_dict)
        assert loaded.config.base_channels == 16
        assert loaded.config.depth == 3
        assert loaded.config.use_film is True

        # Verify weights match
        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), loaded.state_dict().items()
        ):
            assert k1 == k2
            assert torch.equal(v1, v2)

    def test_save_load_via_torch(self, tmp_path: Path) -> None:
        cfg = EnhancerConfig(base_channels=8, depth=2)
        model = PatternEnhancer(config=cfg)
        save_path = tmp_path / "enhancer.pt"
        torch.save(model.get_save_dict(), save_path)

        save_dict = torch.load(save_path, weights_only=False)
        loaded = PatternEnhancer.from_save_dict(save_dict)

        # Outputs should match
        x = torch.randn(1, 1, 128, 128)
        model.eval()
        loaded.eval()
        out1 = model(x)
        out2 = loaded(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_save_dict_contains_keys(self) -> None:
        model = PatternEnhancer()
        save_dict = model.get_save_dict()
        assert "state_dict" in save_dict
        assert "config" in save_dict

    def test_from_save_dict_with_device(self) -> None:
        model = PatternEnhancer(EnhancerConfig(base_channels=8, depth=2))
        save_dict = model.get_save_dict()
        loaded = PatternEnhancer.from_save_dict(
            save_dict, device=torch.device("cpu")
        )
        device = next(loaded.parameters()).device
        assert device == torch.device("cpu")

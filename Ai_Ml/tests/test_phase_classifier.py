"""Tests for ebsd_ai.models.phase_classifier module."""

from __future__ import annotations

import pytest
import torch

from ebsd_ai.config import ModelConfig
from ebsd_ai.models.phase_classifier import PhaseClassifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(
    batch: int = 4,
    pattern_size: int = 128,
    eds_dim: int = 108,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random pattern and EDS tensors."""
    pattern = torch.randn(batch, 1, pattern_size, pattern_size)
    eds_input = torch.randn(batch, eds_dim)
    return pattern, eds_input


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPhaseClassifierInit:
    def test_default_config(self) -> None:
        model = PhaseClassifier()
        assert model.n_phases == 2
        assert model.phase_names == ["phase_0", "phase_1"]

    def test_custom_phase_names(self) -> None:
        cfg = ModelConfig(n_phases=3)
        names = ["Ferrite", "Austenite", "Cementite"]
        model = PhaseClassifier(config=cfg, phase_names=names)
        assert model.n_phases == 3
        assert model.phase_names == names

    def test_mismatched_phase_names_raises(self) -> None:
        cfg = ModelConfig(n_phases=2)
        with pytest.raises(ValueError, match="len\\(phase_names\\)=3"):
            PhaseClassifier(config=cfg, phase_names=["A", "B", "C"])


class TestForwardPass:
    def test_output_shapes_default(self) -> None:
        model = PhaseClassifier()
        model.eval()
        pat, eds = _make_inputs(batch=4)
        with torch.no_grad():
            logits, alpha = model(pat, eds)
        assert logits.shape == (4, 2)
        assert alpha.shape == (4, 1)

    def test_output_shapes_batch_1(self) -> None:
        model = PhaseClassifier()
        model.eval()
        pat, eds = _make_inputs(batch=1)
        with torch.no_grad():
            logits, alpha = model(pat, eds)
        assert logits.shape == (1, 2)
        assert alpha.shape == (1, 1)

    def test_output_shapes_batch_16(self) -> None:
        cfg = ModelConfig(n_phases=5)
        model = PhaseClassifier(config=cfg)
        model.eval()
        pat, eds = _make_inputs(batch=16)
        with torch.no_grad():
            logits, alpha = model(pat, eds)
        assert logits.shape == (16, 5)

    def test_no_nan(self) -> None:
        model = PhaseClassifier()
        model.eval()
        pat, eds = _make_inputs(batch=8)
        with torch.no_grad():
            logits, alpha = model(pat, eds)
        assert not torch.any(torch.isnan(logits))
        assert not torch.any(torch.isinf(logits))
        assert not torch.any(torch.isnan(alpha))

    def test_zero_eds_no_nan(self) -> None:
        """All-zero EDS input should still produce valid output."""
        model = PhaseClassifier()
        model.eval()
        pat = torch.randn(4, 1, 128, 128)
        eds = torch.zeros(4, 108)
        with torch.no_grad():
            logits, alpha = model(pat, eds)
        assert not torch.any(torch.isnan(logits))
        assert logits.shape == (4, 2)

    def test_gradient_flows(self) -> None:
        model = PhaseClassifier()
        pat, eds = _make_inputs(batch=2)
        pat.requires_grad_(True)
        eds.requires_grad_(True)
        logits, alpha = model(pat, eds)
        logits.sum().backward()
        assert pat.grad is not None
        assert eds.grad is not None


class TestPredictTopK:
    def test_returns_correct_structure(self) -> None:
        cfg = ModelConfig(n_phases=4)
        names = ["Ferrite", "Austenite", "Cementite", "Martensite"]
        model = PhaseClassifier(config=cfg, phase_names=names)
        pat, eds = _make_inputs(batch=3)
        results = model.predict_top_k(pat, eds, k=2)
        assert len(results) == 3
        for sample in results:
            assert len(sample) == 2
            for name, prob in sample:
                assert isinstance(name, str)
                assert name in names
                assert 0.0 <= prob <= 1.0

    def test_probabilities_sum_to_one(self) -> None:
        """Top-k probs for k=n_phases should sum to ~1."""
        cfg = ModelConfig(n_phases=3)
        model = PhaseClassifier(config=cfg)
        pat, eds = _make_inputs(batch=4)
        results = model.predict_top_k(pat, eds, k=3)
        for sample in results:
            total = sum(p for _, p in sample)
            assert abs(total - 1.0) < 1e-5

    def test_sorted_descending(self) -> None:
        cfg = ModelConfig(n_phases=5)
        model = PhaseClassifier(config=cfg)
        pat, eds = _make_inputs(batch=4)
        results = model.predict_top_k(pat, eds, k=5)
        for sample in results:
            probs = [p for _, p in sample]
            assert probs == sorted(probs, reverse=True)

    def test_k_larger_than_n_phases(self) -> None:
        """k > n_phases should clamp to n_phases."""
        cfg = ModelConfig(n_phases=2)
        model = PhaseClassifier(config=cfg)
        pat, eds = _make_inputs(batch=2)
        results = model.predict_top_k(pat, eds, k=10)
        for sample in results:
            assert len(sample) == 2

    def test_restores_training_mode(self) -> None:
        model = PhaseClassifier()
        model.train()
        pat, eds = _make_inputs(batch=2)
        model.predict_top_k(pat, eds)
        assert model.training


class TestGetAttentionWeights:
    def test_shape_and_range(self) -> None:
        model = PhaseClassifier()
        pat, eds = _make_inputs(batch=8)
        alpha = model.get_attention_weights(pat, eds)
        assert alpha.shape == (8, 1)
        assert torch.all(alpha >= 0.0)
        assert torch.all(alpha <= 1.0)

    def test_restores_training_mode(self) -> None:
        model = PhaseClassifier()
        model.train()
        pat, eds = _make_inputs(batch=2)
        model.get_attention_weights(pat, eds)
        assert model.training


class TestAddPhase:
    def test_adds_new_phase(self) -> None:
        cfg = ModelConfig(n_phases=2)
        model = PhaseClassifier(config=cfg, phase_names=["A", "B"])
        idx = model.add_phase("C")
        assert idx == 2
        assert model.n_phases == 3
        assert model.phase_names == ["A", "B", "C"]
        assert model.config.n_phases == 3

    def test_forward_after_add(self) -> None:
        """Model should produce correct output shape after adding a phase."""
        cfg = ModelConfig(n_phases=2)
        model = PhaseClassifier(config=cfg, phase_names=["A", "B"])
        model.add_phase("C")
        model.eval()
        pat, eds = _make_inputs(batch=4)
        with torch.no_grad():
            logits, alpha = model(pat, eds)
        assert logits.shape == (4, 3)

    def test_existing_predictions_preserved(self) -> None:
        """Adding a phase should not change predictions for existing phases."""
        cfg = ModelConfig(n_phases=2)
        model = PhaseClassifier(config=cfg, phase_names=["A", "B"])
        model.eval()
        pat, eds = _make_inputs(batch=4)
        with torch.no_grad():
            logits_before, _ = model(pat, eds)
        model.add_phase("C")
        model.eval()
        with torch.no_grad():
            logits_after, _ = model(pat, eds)
        # First 2 columns should be identical (same weights)
        assert torch.allclose(logits_before, logits_after[:, :2])

    def test_duplicate_raises(self) -> None:
        model = PhaseClassifier(config=ModelConfig(n_phases=2), phase_names=["A", "B"])
        with pytest.raises(ValueError, match="already exists"):
            model.add_phase("A")

    def test_new_phase_near_zero_prob(self) -> None:
        """Newly added phase should have near-zero probability (zero-init)."""
        cfg = ModelConfig(n_phases=3)
        model = PhaseClassifier(config=cfg)
        model.eval()
        model.add_phase("new_phase")
        pat, eds = _make_inputs(batch=8)
        with torch.no_grad():
            logits, _ = model(pat, eds)
        probs = torch.softmax(logits, dim=1)
        # The new phase (index 3) should have small probability
        new_probs = probs[:, 3]
        assert torch.all(new_probs < 0.5)


class TestSerialization:
    def test_save_load_roundtrip(self, tmp_path: object) -> None:
        cfg = ModelConfig(
            n_phases=3,
            pattern_feature_dim=128,
            eds_feature_dim=32,
            fused_feature_dim=64,
        )
        names = ["Ferrite", "Austenite", "Cementite"]
        model = PhaseClassifier(config=cfg, phase_names=names)
        model.eval()

        # Get a reference output
        pat, eds = _make_inputs(batch=2, eds_dim=108)
        with torch.no_grad():
            logits_orig, alpha_orig = model(pat, eds)

        # Save and reload
        save_dict = model.get_save_dict()
        restored = PhaseClassifier.from_save_dict(save_dict)
        restored.eval()

        with torch.no_grad():
            logits_rest, alpha_rest = restored(pat, eds)

        assert torch.allclose(logits_orig, logits_rest, atol=1e-6)
        assert torch.allclose(alpha_orig, alpha_rest, atol=1e-6)
        assert restored.phase_names == names
        assert restored.config.pattern_feature_dim == 128

    def test_save_dict_keys(self) -> None:
        model = PhaseClassifier()
        sd = model.get_save_dict()
        assert "state_dict" in sd
        assert "config" in sd
        assert "phase_names" in sd

    def test_roundtrip_with_added_phase(self) -> None:
        model = PhaseClassifier(
            config=ModelConfig(n_phases=2), phase_names=["A", "B"]
        )
        model.add_phase("C")
        model.eval()

        sd = model.get_save_dict()
        restored = PhaseClassifier.from_save_dict(sd)
        restored.eval()

        assert restored.n_phases == 3
        assert restored.phase_names == ["A", "B", "C"]

        pat, eds = _make_inputs(batch=2)
        with torch.no_grad():
            logits_orig, _ = model(pat, eds)
            logits_rest, _ = restored(pat, eds)
        assert torch.allclose(logits_orig, logits_rest, atol=1e-6)


class TestParameterCount:
    def test_total_under_15m(self) -> None:
        """Total parameter count should be under 15M per spec."""
        model = PhaseClassifier()
        total = sum(p.numel() for p in model.parameters())
        assert total < 15_000_000

    def test_reasonable_size(self) -> None:
        """With default config, should be around 2-4M params."""
        model = PhaseClassifier()
        total = sum(p.numel() for p in model.parameters())
        assert 1_000_000 < total < 5_000_000

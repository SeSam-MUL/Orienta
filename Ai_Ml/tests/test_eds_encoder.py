"""Tests for ebsd_ai.features.eds_encoder module."""

from __future__ import annotations

import torch
import pytest

from ebsd_ai.config import NUM_ELEMENTS, DetectorInfo
from ebsd_ai.features.eds_encoder import EDSEncoder, EDS_INPUT_DIM


class TestEDSEncoder:
    def test_input_dim_constant(self) -> None:
        assert EDS_INPUT_DIM == NUM_ELEMENTS + DetectorInfo.encoded_length() + 1
        assert EDS_INPUT_DIM == 108

    def test_output_shape_default(self) -> None:
        encoder = EDSEncoder(feature_dim=64)
        x = torch.randn(4, EDS_INPUT_DIM)
        out = encoder(x)
        assert out.shape == (4, 64)

    def test_output_shape_batch_1(self) -> None:
        encoder = EDSEncoder()
        # BatchNorm1d requires batch > 1 in training mode
        encoder.eval()
        x = torch.randn(1, EDS_INPUT_DIM)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (1, 64)

    def test_output_shape_batch_16(self) -> None:
        encoder = EDSEncoder()
        x = torch.randn(16, EDS_INPUT_DIM)
        out = encoder(x)
        assert out.shape == (16, 64)

    def test_custom_feature_dim(self) -> None:
        encoder = EDSEncoder(feature_dim=128)
        x = torch.randn(4, EDS_INPUT_DIM)
        out = encoder(x)
        assert out.shape == (4, 128)

    def test_no_nan_random_input(self) -> None:
        encoder = EDSEncoder()
        encoder.eval()
        x = torch.randn(8, EDS_INPUT_DIM)
        with torch.no_grad():
            out = encoder(x)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))

    def test_all_zero_eds(self) -> None:
        """Model should handle the case where EDS is all zeros (no EDS)."""
        encoder = EDSEncoder()
        encoder.eval()
        x = torch.zeros(4, EDS_INPUT_DIM)
        with torch.no_grad():
            out = encoder(x)
        assert not torch.any(torch.isnan(out))
        assert out.shape == (4, 64)

    def test_gradient_flows(self) -> None:
        encoder = EDSEncoder()
        x = torch.randn(4, EDS_INPUT_DIM, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestBuildInput:
    def test_concatenation(self) -> None:
        eds = torch.randn(4, NUM_ELEMENTS)
        det = torch.randn(4, DetectorInfo.encoded_length())
        flag = torch.ones(4, 1)

        combined = EDSEncoder.build_input(eds, det, flag)
        assert combined.shape == (4, EDS_INPUT_DIM)

    def test_values_preserved(self) -> None:
        eds = torch.ones(2, NUM_ELEMENTS) * 5.0
        det = torch.ones(2, DetectorInfo.encoded_length()) * 3.0
        flag = torch.ones(2, 1) * 1.0

        combined = EDSEncoder.build_input(eds, det, flag)
        # EDS portion
        assert torch.all(combined[:, :NUM_ELEMENTS] == 5.0)
        # Detector portion
        det_start = NUM_ELEMENTS
        det_end = det_start + DetectorInfo.encoded_length()
        assert torch.all(combined[:, det_start:det_end] == 3.0)
        # Flag
        assert torch.all(combined[:, -1] == 1.0)

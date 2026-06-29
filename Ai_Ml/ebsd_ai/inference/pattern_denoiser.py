"""Inference API for EBSD pattern denoising.

Provides PatternDenoiser -- a high-level wrapper around PatternEnhancer
for single-pattern and batch enhancement.  Gracefully falls back to
passthrough (normalized input) when no model is loaded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from ebsd_ai.config import DetectorInfo
from ebsd_ai.data.pattern_io import normalize_pattern, resize_pattern
from ebsd_ai.models.pattern_enhancer import PatternEnhancer


class PatternDenoiser:
    """High-level denoising inference API.

    Parameters
    ----------
    model : PatternEnhancer or None
        Trained model.  If None, enhance() returns normalized input.
    device : str or torch.device
        Device for inference.
    """

    def __init__(
        self,
        model: PatternEnhancer | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> PatternDenoiser:
        """Load a PatternDenoiser from a saved checkpoint file.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to best.pt or last.pt.
        device : str or torch.device
            Device to place the model on.

        Returns
        -------
        PatternDenoiser
        """
        save_dict = torch.load(
            checkpoint_path, map_location=device, weights_only=False,
        )
        model = PatternEnhancer.from_save_dict(save_dict, torch.device(device))
        return cls(model=model, device=device)

    def enhance(
        self,
        pattern: np.ndarray,
        detector_info: Optional[DetectorInfo] = None,
        target_size: int = 128,
    ) -> np.ndarray:
        """Enhance a single pattern.

        Parameters
        ----------
        pattern : np.ndarray
            2-D raw or normalized pattern.
        detector_info : DetectorInfo or None
            For FiLM conditioning.
        target_size : int
            Output size.

        Returns
        -------
        np.ndarray
            Enhanced pattern of shape (target_size, target_size), float32.
        """
        resized = resize_pattern(pattern, target_size)
        normalized, _ = normalize_pattern(resized)

        if self.model is None:
            return normalized.astype(np.float32)

        with torch.no_grad():
            tensor = (
                torch.from_numpy(normalized)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )

            det_t = None
            if detector_info is not None and self.model.config.use_film:
                det_enc = detector_info.encode()
                det_t = (
                    torch.from_numpy(det_enc)
                    .float()
                    .unsqueeze(0)
                    .to(self.device)
                )

            out = self.model(tensor, detector_encoding=det_t)
            return out.squeeze(0).squeeze(0).cpu().numpy()

    def enhance_batch(
        self,
        patterns: np.ndarray,
        detector_info: Optional[DetectorInfo] = None,
        batch_size: int = 64,
        target_size: int = 128,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """Enhance multiple patterns in batches.

        Parameters
        ----------
        patterns : np.ndarray
            (N, H, W) array of patterns.
        detector_info : DetectorInfo or None
            Shared detector info.
        batch_size : int
            Patterns per forward pass.
        target_size : int
            Output size.
        progress_callback : callable, optional
            Called with (processed_count, total_count).

        Returns
        -------
        np.ndarray
            (N, target_size, target_size) enhanced patterns, float32.
        """
        n = patterns.shape[0]
        result = np.zeros((n, target_size, target_size), dtype=np.float32)

        if self.model is None:
            for i in range(n):
                resized = resize_pattern(patterns[i], target_size)
                normalized, _ = normalize_pattern(resized)
                result[i] = normalized.astype(np.float32)
            return result

        # Prepare detector encoding once
        det_batch_single = None
        if detector_info is not None and self.model.config.use_film:
            det_enc = detector_info.encode()
            det_batch_single = (
                torch.from_numpy(det_enc)
                .float()
                .unsqueeze(0)
                .to(self.device)
            )

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                bs = end - start

                # Prepare batch tensor
                batch_t = torch.zeros(
                    bs, 1, target_size, target_size, device=self.device,
                )
                for i, idx in enumerate(range(start, end)):
                    resized = resize_pattern(patterns[idx], target_size)
                    normalized, _ = normalize_pattern(resized)
                    batch_t[i, 0] = torch.from_numpy(normalized).float()

                det_t = None
                if det_batch_single is not None:
                    det_t = det_batch_single.expand(bs, -1)

                out = self.model(batch_t, detector_encoding=det_t)
                result[start:end] = out.squeeze(1).cpu().numpy()

                if progress_callback is not None:
                    progress_callback(end, n)

        return result

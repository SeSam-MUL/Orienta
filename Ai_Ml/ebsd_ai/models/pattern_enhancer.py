"""U-Net for EBSD pattern denoising and enhancement.

A convolutional encoder-decoder network with skip connections that maps
noisy/degraded EBSD patterns to clean, enhanced versions. The network can
be conditioned on detector metadata (kV, PC, detector type) via FiLM
(Feature-wise Linear Modulation) layers.

The enhanced patterns preserve crystallographic information (Kikuchi band
positions and widths) while removing noise and improving contrast. This
directly improves downstream indexing quality for Dictionary, Spherical,
and Hough Indexing — not just the ML classifier.

Architecture
------------
Input: ``(B, 1, 128, 128)`` noisy pattern
Output: ``(B, 1, 128, 128)`` enhanced pattern

Encoder:  Conv blocks with downsampling (128 → 64 → 32 → 16 → 8)
Bottleneck: Conv block + FiLM conditioning
Decoder:  Transpose-conv upsampling with skip connections (8 → 16 → 32 → 64 → 128)

Training losses:
- L1 pixel-wise reconstruction
- Optional perceptual loss via PatternEncoder features
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np
import torch
import torch.nn as nn

from ebsd_ai.config import DEFAULT_PATTERN_SIZE, DetectorInfo
from ebsd_ai.data.pattern_io import prepare_pattern

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EnhancerConfig:
    """Hyperparameters for the pattern enhancer U-Net.

    Parameters
    ----------
    pattern_size : int
        Square side length for input/output patterns.
    base_channels : int
        Number of channels in the first encoder block. Subsequent blocks
        double the channel count.
    depth : int
        Number of encoder/decoder stages (excluding bottleneck).
    use_film : bool
        Whether to condition the bottleneck on detector metadata via FiLM.
    detector_encoding_dim : int
        Dimension of the detector metadata encoding (from ``DetectorInfo.encode()``).
    dropout : float
        Dropout probability in bottleneck.
    """

    pattern_size: int = DEFAULT_PATTERN_SIZE
    base_channels: int = 32
    depth: int = 4
    use_film: bool = True
    detector_encoding_dim: int = DetectorInfo.encoded_length()
    dropout: float = 0.1

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain dict."""
        return {
            "pattern_size": self.pattern_size,
            "base_channels": self.base_channels,
            "depth": self.depth,
            "use_film": self.use_film,
            "detector_encoding_dim": self.detector_encoding_dim,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> EnhancerConfig:
        """Reconstruct from a plain dict."""
        filtered = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**cast(dict[str, Any], filtered))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU.

    Parameters
    ----------
    in_channels : int
        Input channel count.
    out_channels : int
        Output channel count.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, C_out, H, W)``.
        """
        result: torch.Tensor = self.block(x)
        return result


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for conditioning on metadata.

    Learns affine transforms ``gamma`` and ``beta`` from a conditioning
    vector and applies them channel-wise: ``y = gamma * x + beta``.

    Parameters
    ----------
    conditioning_dim : int
        Dimension of the conditioning vector (detector metadata).
    n_channels : int
        Number of feature map channels to modulate.
    """

    def __init__(self, conditioning_dim: int, n_channels: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(conditioning_dim, n_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels * 2, n_channels * 2),
        )

    def forward(
        self, x: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """Apply FiLM modulation.

        Parameters
        ----------
        x : torch.Tensor
            Feature maps of shape ``(B, C, H, W)``.
        conditioning : torch.Tensor
            Conditioning vector of shape ``(B, conditioning_dim)``.

        Returns
        -------
        torch.Tensor
            Modulated feature maps of shape ``(B, C, H, W)``.
        """
        params = self.fc(conditioning)  # (B, 2*C)
        gamma, beta = params.chunk(2, dim=1)  # each (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        result: torch.Tensor = gamma * x + beta
        return result


# ---------------------------------------------------------------------------
# U-Net model
# ---------------------------------------------------------------------------


class PatternEnhancer(nn.Module):
    """U-Net for EBSD pattern denoising and enhancement.

    Encoder-decoder network with skip connections that reconstructs
    clean patterns from noisy/degraded inputs. Optionally conditioned
    on detector metadata via FiLM layers in the bottleneck.

    Parameters
    ----------
    config : EnhancerConfig or None
        Model hyperparameters. Uses defaults if *None*.

    Examples
    --------
    >>> enhancer = PatternEnhancer()
    >>> noisy = torch.randn(4, 1, 128, 128)
    >>> clean = enhancer(noisy)
    >>> clean.shape
    torch.Size([4, 1, 128, 128])

    With detector conditioning:

    >>> det_vec = torch.randn(4, 15)
    >>> clean = enhancer(noisy, detector_encoding=det_vec)
    """

    def __init__(
        self,
        config: EnhancerConfig | None = None,
        residual_learning: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or EnhancerConfig()
        self.residual_learning = residual_learning

        base = self.config.base_channels
        depth = self.config.depth

        # --- Encoder ---
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = 1
        encoder_channels: list[int] = []
        for i in range(depth):
            out_ch = base * (2 ** i)
            self.encoders.append(ConvBlock(in_ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            encoder_channels.append(out_ch)
            in_ch = out_ch

        # --- Bottleneck ---
        bottleneck_ch = base * (2 ** depth)
        self.bottleneck = ConvBlock(in_ch, bottleneck_ch)
        self.bottleneck_dropout = nn.Dropout2d(self.config.dropout)

        # FiLM conditioning in bottleneck
        self.film: FiLMLayer | None = None
        if self.config.use_film:
            self.film = FiLMLayer(
                self.config.detector_encoding_dim, bottleneck_ch
            )

        # --- Decoder ---
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        in_ch = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            out_ch = encoder_channels[i]
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
            )
            # Skip connection doubles input channels
            self.decoders.append(ConvBlock(out_ch * 2, out_ch))
            in_ch = out_ch

        # --- Output ---
        self.output_conv = nn.Conv2d(base, 1, 1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        detector_encoding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Enhance a batch of EBSD patterns.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 1, H, W)`` noisy patterns.
        detector_encoding : torch.Tensor or None
            Shape ``(B, detector_encoding_dim)`` detector metadata.
            Only used when ``config.use_film`` is True.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1, H, W)`` enhanced patterns.
        """
        # Encoder path with skip connections
        skips: list[torch.Tensor] = []
        out = x
        for encoder, pool in zip(self.encoders, self.pools):
            out = encoder(out)
            skips.append(out)
            out = pool(out)

        # Bottleneck
        out = self.bottleneck(out)
        out = self.bottleneck_dropout(out)

        # FiLM conditioning
        if self.film is not None and detector_encoding is not None:
            out = self.film(out, detector_encoding)

        # Decoder path with skip connections
        for i, (upconv, decoder) in enumerate(
            zip(self.upconvs, self.decoders)
        ):
            out = upconv(out)
            skip = skips[-(i + 1)]

            # Handle size mismatches from non-power-of-2 inputs
            if out.shape != skip.shape:
                out = nn.functional.interpolate(
                    out, size=skip.shape[2:], mode="bilinear",
                    align_corners=False,
                )

            out = torch.cat([out, skip], dim=1)
            out = decoder(out)

        out = self.output_conv(out)
        if self.residual_learning:
            out = x + out
        result: torch.Tensor = out
        return result

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def enhance(
        self,
        pattern: np.ndarray,
        detector_info: DetectorInfo | None = None,
        device: torch.device | None = None,
    ) -> np.ndarray:
        """Enhance a single EBSD pattern.

        Handles all preprocessing (resize, normalize) and postprocessing
        internally.

        Parameters
        ----------
        pattern : np.ndarray
            Raw 2-D grayscale pattern of any size.
        detector_info : DetectorInfo or None
            Detector metadata for FiLM conditioning. Ignored if the model
            was built without FiLM.
        device : torch.device or None
            Device to run inference on. Defaults to the model's device.

        Returns
        -------
        np.ndarray
            Enhanced pattern of shape ``(pattern_size, pattern_size)``,
            float32, in the normalized domain.
        """
        if device is None:
            device = next(self.parameters()).device

        was_training = self.training
        self.eval()
        try:
            # Prepare input
            size = self.config.pattern_size
            normalized, _ = prepare_pattern(pattern, target_size=size)
            tensor = (
                torch.from_numpy(normalized)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            # Detector conditioning
            det_t: torch.Tensor | None = None
            if self.film is not None and detector_info is not None:
                det_encoded = detector_info.encode()
                det_t = (
                    torch.from_numpy(det_encoded)
                    .unsqueeze(0)
                    .to(device)
                )

            enhanced = self.forward(tensor, det_t)
            return enhanced.squeeze(0).squeeze(0).cpu().numpy()
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def enhance_scan(
        self,
        patterns_4d: np.ndarray,
        detector_info: DetectorInfo | None = None,
        selection_mask: np.ndarray | None = None,
        batch_size: int = 64,
        device: torch.device | None = None,
    ) -> np.ndarray:
        """Enhance all patterns in an EBSD scan.

        Parameters
        ----------
        patterns_4d : np.ndarray
            Shape ``(n_rows, n_cols, height, width)`` — all patterns.
        detector_info : DetectorInfo or None
            Shared detector metadata for the whole scan.
        selection_mask : np.ndarray or None
            Boolean array ``(n_rows, n_cols)``. Only enhance where True.
            If None, enhances all pixels.
        batch_size : int
            Number of patterns to process at once.
        device : torch.device or None
            Device for inference.

        Returns
        -------
        np.ndarray
            Shape ``(n_rows, n_cols, pattern_size, pattern_size)``
            enhanced patterns. Unselected pixels are zero-filled.

        Raises
        ------
        ValueError
            If *patterns_4d* is not 4-D.
        """
        if patterns_4d.ndim != 4:
            raise ValueError(
                f"patterns_4d must be 4-D (n_rows, n_cols, H, W), "
                f"got {patterns_4d.ndim}-D with shape {patterns_4d.shape}"
            )

        if device is None:
            device = next(self.parameters()).device

        was_training = self.training
        self.eval()
        try:
            n_rows, n_cols = patterns_4d.shape[:2]
            size = self.config.pattern_size

            # Output array
            result = np.zeros(
                (n_rows, n_cols, size, size), dtype=np.float32
            )

            # Determine which pixels to process
            if selection_mask is not None:
                if selection_mask.shape != (n_rows, n_cols):
                    raise ValueError(
                        f"selection_mask shape {selection_mask.shape} doesn't "
                        f"match scan shape ({n_rows}, {n_cols})"
                    )
                flat_mask = selection_mask.ravel()
            else:
                flat_mask = np.ones(n_rows * n_cols, dtype=bool)

            selected_indices = np.where(flat_mask)[0]
            n_selected = len(selected_indices)

            if n_selected == 0:
                return result

            flat_patterns = patterns_4d.reshape(
                n_rows * n_cols, *patterns_4d.shape[2:]
            )

            # Prepare detector encoding once
            if self.film is not None and detector_info is not None:
                det_encoded = detector_info.encode()
                det_t_single = (
                    torch.from_numpy(det_encoded).unsqueeze(0).to(device)
                )
            else:
                det_t_single = None

            # Process in batches
            for start in range(0, n_selected, batch_size):
                end = min(start + batch_size, n_selected)
                batch_indices = selected_indices[start:end]
                bs = end - start

                # Prepare pattern batch
                pattern_batch = torch.zeros(
                    bs, 1, size, size, device=device
                )
                for i, pixel_idx in enumerate(batch_indices):
                    raw = flat_patterns[pixel_idx]
                    normalized, _ = prepare_pattern(
                        raw, target_size=size
                    )
                    pattern_batch[i, 0] = torch.from_numpy(normalized)

                # Expand detector encoding to batch
                det_batch: torch.Tensor | None = None
                if det_t_single is not None:
                    det_batch = det_t_single.expand(bs, -1)

                # Forward pass
                enhanced = self.forward(pattern_batch, det_batch)
                enhanced_np = enhanced.squeeze(1).cpu().numpy()

                # Write results
                for i, pixel_idx in enumerate(batch_indices):
                    row = pixel_idx // n_cols
                    col = pixel_idx % n_cols
                    result[row, col] = enhanced_np[i]

            return result
        finally:
            if was_training:
                self.train()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_save_dict(self) -> dict[str, object]:
        """Build a checkpoint dictionary for :func:`torch.save`.

        Returns
        -------
        dict
            Contains ``state_dict`` and ``config``.
        """
        return {
            "state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            "residual_learning": self.residual_learning,
        }

    @classmethod
    def from_save_dict(
        cls,
        save_dict: dict[str, object],
        device: Optional[torch.device] = None,
    ) -> PatternEnhancer:
        """Reconstruct from a checkpoint dictionary.

        Parameters
        ----------
        save_dict : dict
            As produced by :meth:`get_save_dict`.
        device : torch.device or None
            Device to place the model on. Defaults to CPU.

        Returns
        -------
        PatternEnhancer
            Restored model with weights loaded.
        """
        config = EnhancerConfig.from_dict(
            save_dict["config"]  # type: ignore[arg-type]
        )
        residual = bool(save_dict.get("residual_learning", False))
        model = cls(config=config, residual_learning=residual)
        model.load_state_dict(
            save_dict["state_dict"]  # type: ignore[arg-type]
        )
        if device is not None:
            model = model.to(device)
        return model

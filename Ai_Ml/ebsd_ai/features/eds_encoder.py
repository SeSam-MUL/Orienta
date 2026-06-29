"""MLP encoder for EDS chemistry data and detector metadata.

The EDS encoder takes a concatenated vector of:
- 92-element EDS atomic percent values
- Encoded detector metadata (15 floats: PC + one-hot convention/mfr
  + normalized kV/WD/tilt)
- 1 has_eds flag

Total input: 92 + 15 + 1 = 108 floats

Produces a compact feature vector suitable for fusion with the
pattern encoder output.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ebsd_ai.config import NUM_ELEMENTS, DetectorInfo

# Total input = EDS (92) + detector encoding (15) + has_eds flag (1)
EDS_INPUT_DIM = NUM_ELEMENTS + DetectorInfo.encoded_length() + 1


class EDSEncoder(nn.Module):
    """MLP encoder for EDS chemistry and detector metadata.

    Input is a concatenation of the 92-element EDS vector,
    15-float encoded detector info, and a 1-float has_eds flag.

    Architecture:
        Linear(108, 256) -> BN -> ReLU -> Dropout
        Linear(256, 128) -> BN -> ReLU -> Dropout
        Linear(128, feature_dim)

    Parameters
    ----------
    feature_dim : int
        Output feature vector dimension. Default 64.
    dropout : float
        Dropout probability. Default 0.3.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.input_dim = EDS_INPUT_DIM

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode EDS + metadata vector.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, input_dim)`` where input_dim = 108.
            The vector is the concatenation of:
            [eds_vector (92), detector_encoded (15), has_eds (1)].

        Returns
        -------
        torch.Tensor
            Shape ``(B, feature_dim)`` feature vectors.
        """
        result: torch.Tensor = self.mlp(x)
        return result

    @staticmethod
    def build_input(
        eds_vector: torch.Tensor,
        detector_encoded: torch.Tensor,
        has_eds_flag: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate EDS, detector, and has_eds into the encoder input.

        Parameters
        ----------
        eds_vector : torch.Tensor
            Shape ``(B, 92)`` float32.
        detector_encoded : torch.Tensor
            Shape ``(B, 15)`` float32.
        has_eds_flag : torch.Tensor
            Shape ``(B, 1)`` float32 (0 or 1).

        Returns
        -------
        torch.Tensor
            Shape ``(B, 108)`` concatenated input.
        """
        return torch.cat([eds_vector, detector_encoded, has_eds_flag], dim=1)

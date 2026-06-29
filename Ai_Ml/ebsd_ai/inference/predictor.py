"""Main prediction interface for EBSD phase classification.

Provides :class:`PhasePredictor` — the primary public API that wraps a
trained :class:`PhaseClassifier` and handles all input preprocessing,
device management, and output formatting.

Key features:
- Single-pattern prediction via :meth:`predict_phase`
- Batch scan prediction via :meth:`predict_scan`
- Graceful degradation when EDS or pattern is missing
- Thread-safe (uses ``torch.no_grad`` and model eval mode per call)
- Automatic device selection (CPU/GPU)
"""

from __future__ import annotations

import numpy as np
import torch

from ebsd_ai.config import (
    NUM_ELEMENTS,
    DetectorInfo,
    PhasePrediction,
    eds_dict_to_vector,
    has_eds,
)
from ebsd_ai.data.pattern_io import prepare_pattern
from ebsd_ai.features.eds_encoder import EDSEncoder
from ebsd_ai.models.phase_classifier import PhaseClassifier

# ---------------------------------------------------------------------------
# Scan result container
# ---------------------------------------------------------------------------


class ScanPredictionResult:
    """Container for batch scan prediction results.

    Parameters
    ----------
    phase_names : np.ndarray
        Shape ``(n_pixels, k)`` — top-k phase name per pixel.
    probabilities : np.ndarray
        Shape ``(n_pixels, k)`` — probability for each top-k phase.
    confidence : np.ndarray
        Shape ``(n_pixels,)`` — top-1 probability per pixel.
    eds_contribution : np.ndarray
        Shape ``(n_pixels,)`` — how much EDS influenced each pixel
        (``1 - alpha``).
    scan_shape : tuple[int, int]
        Original ``(n_rows, n_cols)`` for reshaping.
    k : int
        Number of top candidates per pixel.
    """

    def __init__(
        self,
        phase_names: np.ndarray,
        probabilities: np.ndarray,
        confidence: np.ndarray,
        eds_contribution: np.ndarray,
        scan_shape: tuple[int, int],
        k: int,
    ) -> None:
        self.phase_names = phase_names
        self.probabilities = probabilities
        self.confidence = confidence
        self.eds_contribution = eds_contribution
        self.scan_shape = scan_shape
        self.k = k

    @property
    def n_pixels(self) -> int:
        """Total number of predicted pixels."""
        return len(self.confidence)


# ---------------------------------------------------------------------------
# PhasePredictor
# ---------------------------------------------------------------------------


_DEFAULT_BATCH_SIZE = 64


class PhasePredictor:
    """High-level prediction interface for EBSD phase classification.

    Wraps a trained :class:`PhaseClassifier` and handles preprocessing,
    batching, device transfers, and output formatting.

    Parameters
    ----------
    model_path : str or None
        Path to a saved model checkpoint (``best.pt`` or similar).
        If *None*, the predictor works in fallback mode (no ML predictions).
    device : str
        ``"auto"`` picks CUDA if available, else CPU.
        Can also be ``"cpu"`` or ``"cuda"``.
    batch_size : int
        Batch size for scan-level prediction.

    Examples
    --------
    >>> predictor = PhasePredictor("checkpoints/best.pt")
    >>> result = predictor.predict_phase(
    ...     pattern=my_pattern,
    ...     eds_data={"Fe": 65.2, "C": 8.1},
    ...     detector_info=my_detector,
    ... )
    >>> result.top_k
    [("Ferrite", 0.85), ("Austenite", 0.12), ("Cementite", 0.03)]
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.batch_size = batch_size
        self._model: PhaseClassifier | None = None

        if model_path is not None:
            self._load_model(model_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str) -> None:
        """Load model from checkpoint file.

        Parameters
        ----------
        model_path : str
            Path to a ``.pt`` checkpoint as produced by :class:`Trainer`.
        """
        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )
        # Support both Trainer checkpoints (with "model" key) and
        # direct PhaseClassifier.get_save_dict() outputs.
        if "model" in checkpoint:
            save_dict = checkpoint["model"]
        elif "state_dict" in checkpoint and "config" in checkpoint:
            save_dict = checkpoint
        else:
            raise ValueError(
                f"Unrecognised checkpoint format at '{model_path}'. "
                "Expected a Trainer checkpoint (with 'model' key) or "
                "a PhaseClassifier.get_save_dict() output."
            )
        self._model = PhaseClassifier.from_save_dict(
            save_dict, device=self.device
        )
        self._model.eval()

    @classmethod
    def from_model(
        cls,
        model: PhaseClassifier,
        device: str = "auto",
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> PhasePredictor:
        """Create a predictor from an already-loaded model instance.

        Parameters
        ----------
        model : PhaseClassifier
            A trained model (will be moved to device and set to eval).
        device : str
            Device string (``"auto"``, ``"cpu"``, or ``"cuda"``).
        batch_size : int
            Batch size for scan prediction.

        Returns
        -------
        PhasePredictor
        """
        predictor = cls(model_path=None, device=device, batch_size=batch_size)
        predictor._model = model.to(predictor.device)
        predictor._model.eval()
        return predictor

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_model(self) -> bool:
        """Whether a trained model is loaded."""
        return self._model is not None

    @property
    def phase_names(self) -> list[str]:
        """Phase names known to the model, or empty if no model."""
        if self._model is not None:
            return self._model.phase_names
        return []

    # ------------------------------------------------------------------
    # Input preparation helpers
    # ------------------------------------------------------------------

    def _prepare_pattern_tensor(
        self,
        pattern: np.ndarray | None,
    ) -> torch.Tensor:
        """Resize, normalize, and convert a pattern to a model-ready tensor.

        Parameters
        ----------
        pattern : np.ndarray or None
            Raw 2-D grayscale pattern of any size, or None.

        Returns
        -------
        torch.Tensor
            Shape ``(1, 1, H, W)`` float32 on ``self.device``.
            All zeros if *pattern* is None.
        """
        size = 128
        if self._model is not None:
            size = self._model.config.pattern_size

        if pattern is None:
            return torch.zeros(1, 1, size, size, device=self.device)

        normalized, _ = prepare_pattern(pattern, target_size=size)
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def _prepare_eds_tensor(
        self,
        eds_data: dict[str, float] | None,
        detector_info: DetectorInfo | None,
    ) -> torch.Tensor:
        """Build the EDS+metadata input tensor.

        Parameters
        ----------
        eds_data : dict or None
            Element symbol → atomic percent.
        detector_info : DetectorInfo or None
            Detector metadata.

        Returns
        -------
        torch.Tensor
            Shape ``(1, 108)`` float32 on ``self.device``.
        """
        eds_vec = eds_dict_to_vector(eds_data)
        det_encoded = (
            detector_info.encode()
            if detector_info is not None
            else DetectorInfo().encode()
        )
        has_eds_val = 1.0 if has_eds(eds_vec) else 0.0

        eds_t = torch.from_numpy(eds_vec).unsqueeze(0)
        det_t = torch.from_numpy(det_encoded).unsqueeze(0)
        flag_t = torch.tensor([[has_eds_val]], dtype=torch.float32)

        combined = EDSEncoder.build_input(eds_t, det_t, flag_t)
        return combined.to(self.device)

    # ------------------------------------------------------------------
    # Single-pattern prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_phase(
        self,
        pattern: np.ndarray | None = None,
        eds_data: dict[str, float] | None = None,
        detector_info: DetectorInfo | None = None,
        k: int = 3,
    ) -> PhasePrediction:
        """Predict the crystal phase for a single EBSD pattern.

        Parameters
        ----------
        pattern : np.ndarray or None
            Raw 2-D grayscale pattern (any size). If *None*, uses EDS only.
        eds_data : dict[str, float] or None
            Element symbol → At.%. If *None*, uses pattern only.
        detector_info : DetectorInfo or None
            Detector metadata. Uses defaults if *None*.
        k : int
            Number of top candidates to return.

        Returns
        -------
        PhasePrediction
            Contains ``top_k``, ``confidence``, and ``eds_contribution``.

        Raises
        ------
        RuntimeError
            If no model is loaded and no fallback is possible.
        ValueError
            If both *pattern* and *eds_data* are None.
        """
        if pattern is None and eds_data is None:
            raise ValueError(
                "At least one of 'pattern' or 'eds_data' must be provided."
            )

        # Validate pattern
        if pattern is not None:
            if not isinstance(pattern, np.ndarray):
                raise TypeError(
                    f"pattern must be a numpy array, "
                    f"got {type(pattern).__name__}"
                )
            if pattern.ndim != 2:
                raise ValueError(
                    f"pattern must be 2-D (H, W), got {pattern.ndim}-D "
                    f"with shape {pattern.shape}"
                )
            if pattern.size == 0:
                raise ValueError("pattern is empty (zero-size array)")

        # Validate EDS data
        if eds_data is not None and not isinstance(eds_data, dict):
            raise TypeError(
                f"eds_data must be a dict mapping element symbols to "
                f"atomic percentages, got {type(eds_data).__name__}"
            )

        # Validate k
        if not isinstance(k, int) or k < 1:
            raise ValueError(
                f"k must be a positive integer, got {k!r}"
            )

        if not self.has_model:
            raise RuntimeError(
                "No model loaded. Load a model via PhasePredictor('model.pt') "
                "or PhasePredictor.from_model(model) before calling "
                "predict_phase()."
            )

        assert self._model is not None  # for type checker

        pattern_t = self._prepare_pattern_tensor(pattern)
        eds_t = self._prepare_eds_tensor(eds_data, detector_info)

        logits, alpha = self._model(pattern_t, eds_t)
        probs = torch.softmax(logits, dim=1).cpu()
        alpha_val = float(alpha.cpu().item())

        k_clamped = min(k, self._model.n_phases)
        top_probs, top_idx = torch.topk(probs[0], k_clamped)

        top_k: list[tuple[str, float]] = []
        for i in range(k_clamped):
            idx = int(top_idx[i].item())
            prob = float(top_probs[i].item())
            top_k.append((self._model.phase_names[idx], prob))

        return PhasePrediction(
            top_k=top_k,
            confidence=top_k[0][1] if top_k else 0.0,
            eds_contribution=1.0 - alpha_val,
        )

    # ------------------------------------------------------------------
    # Batch scan prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_scan(
        self,
        patterns_4d: np.ndarray,
        eds_maps: dict[str, np.ndarray] | None = None,
        detector_info: DetectorInfo | None = None,
        selection_mask: np.ndarray | None = None,
        k: int = 3,
    ) -> ScanPredictionResult:
        """Predict phases for an entire EBSD scan.

        Parameters
        ----------
        patterns_4d : np.ndarray
            Shape ``(n_rows, n_cols, height, width)`` — all patterns.
        eds_maps : dict[str, np.ndarray] or None
            Element symbol → flattened array of shape ``(n_rows * n_cols,)``.
        detector_info : DetectorInfo or None
            Shared detector metadata for the whole scan.
        selection_mask : np.ndarray or None
            Boolean array of shape ``(n_rows, n_cols)``. Only predict
            where *True*. If *None*, predict all pixels.
        k : int
            Number of top candidates per pixel.

        Returns
        -------
        ScanPredictionResult
            Results for all selected pixels.

        Raises
        ------
        RuntimeError
            If no model is loaded.
        ValueError
            If *patterns_4d* has wrong number of dimensions.
        """
        if not self.has_model:
            raise RuntimeError(
                "No model loaded. Cannot predict scan without a trained model."
            )
        assert self._model is not None

        # Validate patterns_4d
        if not isinstance(patterns_4d, np.ndarray):
            raise TypeError(
                f"patterns_4d must be a numpy array, "
                f"got {type(patterns_4d).__name__}"
            )
        if patterns_4d.ndim != 4:
            raise ValueError(
                f"patterns_4d must be 4-D (n_rows, n_cols, H, W), "
                f"got {patterns_4d.ndim}-D with shape {patterns_4d.shape}"
            )

        # Validate k
        if not isinstance(k, int) or k < 1:
            raise ValueError(
                f"k must be a positive integer, got {k!r}"
            )

        # Validate EDS maps
        if eds_maps is not None:
            if not isinstance(eds_maps, dict):
                raise TypeError(
                    f"eds_maps must be a dict mapping element symbols to "
                    f"arrays, got {type(eds_maps).__name__}"
                )
            n_total = patterns_4d.shape[0] * patterns_4d.shape[1]
            for symbol, arr in eds_maps.items():
                if not isinstance(arr, np.ndarray):
                    raise TypeError(
                        f"eds_maps['{symbol}'] must be a numpy array, "
                        f"got {type(arr).__name__}"
                    )
                if arr.size < n_total:
                    raise ValueError(
                        f"eds_maps['{symbol}'] has {arr.size} elements, "
                        f"expected at least {n_total} "
                        f"(n_rows * n_cols = {patterns_4d.shape[0]} * "
                        f"{patterns_4d.shape[1]})"
                    )

        n_rows, n_cols = patterns_4d.shape[:2]
        scan_shape = (n_rows, n_cols)

        # Determine which pixels to predict
        if selection_mask is not None:
            if selection_mask.shape != (n_rows, n_cols):
                raise ValueError(
                    f"selection_mask shape {selection_mask.shape} doesn't match "
                    f"scan shape ({n_rows}, {n_cols})"
                )
            flat_mask = selection_mask.ravel()
        else:
            flat_mask = np.ones(n_rows * n_cols, dtype=bool)

        selected_indices = np.where(flat_mask)[0]
        n_selected = len(selected_indices)

        if n_selected == 0:
            return ScanPredictionResult(
                phase_names=np.empty((0, k), dtype=object),
                probabilities=np.empty((0, k), dtype=np.float32),
                confidence=np.empty(0, dtype=np.float32),
                eds_contribution=np.empty(0, dtype=np.float32),
                scan_shape=scan_shape,
                k=k,
            )

        # Prepare shared detector encoding
        det_info = detector_info if detector_info is not None else DetectorInfo()
        det_encoded = det_info.encode()

        # Build EDS element index mapping (lightweight, no large alloc)
        eds_element_indices: list[tuple[int, np.ndarray]] = []
        if eds_maps is not None:
            for symbol, values in eds_maps.items():
                vec = eds_dict_to_vector({symbol: 1.0})
                idx = int(np.argmax(vec))
                eds_element_indices.append((idx, values.ravel()))

        k_clamped = min(k, self._model.n_phases)
        phase_name_list = self._model.phase_names

        # Output arrays
        all_phase_names = np.empty((n_selected, k_clamped), dtype=object)
        all_probs = np.zeros((n_selected, k_clamped), dtype=np.float32)
        all_confidence = np.zeros(n_selected, dtype=np.float32)
        all_eds_contrib = np.zeros(n_selected, dtype=np.float32)

        pattern_size = self._model.config.pattern_size

        # Process in batches (patterns indexed lazily from 4D array)
        for start in range(0, n_selected, self.batch_size):
            end = min(start + self.batch_size, n_selected)
            batch_indices = selected_indices[start:end]
            batch_size = end - start

            # Prepare pattern batch — index directly from 4D array
            pattern_batch = torch.zeros(
                batch_size, 1, pattern_size, pattern_size,
                device=self.device,
            )
            for i, pixel_idx in enumerate(batch_indices):
                row, col = divmod(int(pixel_idx), n_cols)
                raw = patterns_4d[row, col]
                normalized, _ = prepare_pattern(raw, target_size=pattern_size)
                pattern_batch[i, 0] = torch.from_numpy(normalized)

            # Prepare EDS batch — build per-batch, avoid full matrix
            eds_chunk = np.zeros(
                (batch_size, NUM_ELEMENTS), dtype=np.float32
            )
            for elem_idx, elem_values in eds_element_indices:
                eds_chunk[:, elem_idx] = elem_values[batch_indices]

            eds_batch = torch.from_numpy(eds_chunk).to(self.device)
            det_batch = torch.from_numpy(
                np.tile(det_encoded, (batch_size, 1))
            ).to(self.device)
            has_eds_batch = torch.from_numpy(
                (eds_chunk.sum(axis=1, keepdims=True) > 0)
                .astype(np.float32)
            ).to(self.device)

            eds_input = EDSEncoder.build_input(
                eds_batch, det_batch, has_eds_batch
            )

            # Forward pass
            logits, alpha = self._model(
                pattern_batch.to(self.device), eds_input
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            alpha_cpu = alpha.cpu().squeeze(-1).numpy()

            # Extract top-k
            top_idx = np.argsort(-probs, axis=1)[:, :k_clamped]
            for i in range(batch_size):
                for j in range(k_clamped):
                    idx = top_idx[i, j]
                    all_phase_names[start + i, j] = phase_name_list[idx]
                    all_probs[start + i, j] = probs[i, idx]
                all_confidence[start + i] = probs[i, top_idx[i, 0]]
                all_eds_contrib[start + i] = 1.0 - alpha_cpu[i]

        return ScanPredictionResult(
            phase_names=all_phase_names,
            probabilities=all_probs,
            confidence=all_confidence,
            eds_contribution=all_eds_contrib,
            scan_shape=scan_shape,
            k=k_clamped,
        )

    # ------------------------------------------------------------------
    # Fallback prediction
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_predict(
        eds_data: dict[str, float] | None,
    ) -> PhasePrediction:
        """Simple heuristic prediction when no ML model is available.

        Returns an empty prediction with zero confidence. If EDS data is
        provided, returns it as a signal that some chemistry was available
        but no model could interpret it.

        Parameters
        ----------
        eds_data : dict or None
            EDS chemistry data.

        Returns
        -------
        PhasePrediction
            With empty top_k and zero confidence.
        """
        return PhasePrediction(
            top_k=[],
            confidence=0.0,
            eds_contribution=0.0,
        )

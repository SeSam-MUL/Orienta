"""Global EDS element color configuration with JSON persistence.

Singleton that manages element-to-color mapping and per-element opacity.
Colors default to MTEX-style pastel palette for ~25 common elements.
Unknown elements get auto-assigned from a color pool.

No GUI dependencies — pure Python + json + pathlib.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# MTEX-style pastel defaults for common EBSD elements
DEFAULT_COLORS: Dict[str, str] = {
    "Fe": "#e8a0a0", "O":  "#a0c8e8", "Si": "#a0e8a0",
    "Al": "#e8d8a0", "C":  "#d0a0e8", "Mn": "#e8c0a0",
    "Cr": "#a0e8d8", "Ni": "#e8a0d0", "Cu": "#e8d0a0",
    "Ti": "#a0a0e8", "Mg": "#c8e8a0", "N":  "#b0d0e8",
    "S":  "#e8e8a0", "P":  "#d0b8a0", "Ca": "#a0d8a0",
    "Zn": "#b8a0e8", "V":  "#a0e8b8", "Mo": "#d8a0b8",
    "W":  "#b8d0a0", "Co": "#e8b0b0", "Nb": "#a0b8d8",
    "B":  "#d8c8a0", "K":  "#c0a0e0", "Na": "#a0c0d0",
}

# Pool for auto-assigning colors to unknown elements
_AUTO_POOL = [
    "#c8b8e8", "#e8b8c8", "#b8e8c8", "#e8c8b8",
    "#b8c8e8", "#c8e8b8", "#d8b8d8", "#b8d8d8",
    "#d8d8b8", "#c0c0e0", "#e0c0c0", "#c0e0c0",
    "#d0c0d0", "#c0d0d0", "#d0d0c0", "#b0b0d0",
]

DEFAULT_GLOBAL_OPACITY = 0.4


class EDSColorConfig:
    """Singleton configuration for EDS element colors and opacities.

    Usage::

        cfg = EDSColorConfig.instance()
        color = cfg.get_color("Fe")  # "#e8a0a0"
        cfg.set_color("Fe", "#ff0000")
    """

    _instance: Optional[EDSColorConfig] = None

    @classmethod
    def instance(cls) -> EDSColorConfig:
        """Return the singleton instance, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._colors: Dict[str, str] = dict(DEFAULT_COLORS)
        self._opacities: Dict[str, float] = {}
        self._global_opacity: float = DEFAULT_GLOBAL_OPACITY
        self._auto_idx: int = 0
        self._config_path: Path = (
            Path.home() / ".kikuchipy_gui" / "eds_element_colors.json"
        )
        self.on_color_changed: List[Callable] = []
        self.on_opacity_changed: List[Callable] = []
        self.load()

    # ── Public API ────────────────────────────────────────────────

    def get_color(self, element: str) -> str:
        """Get hex color for element, auto-assigning if unknown."""
        if element not in self._colors:
            color = _AUTO_POOL[self._auto_idx % len(_AUTO_POOL)]
            self._auto_idx += 1
            self._colors[element] = color
            self.save()
        return self._colors[element]

    def set_color(self, element: str, hex_color: str) -> None:
        """Set hex color for element and save immediately."""
        self._colors[element] = hex_color
        self.save()
        for cb in self.on_color_changed:
            cb(element, hex_color)

    def get_opacity(self, element: str) -> float:
        """Get per-element opacity, falling back to global."""
        return self._opacities.get(element, self._global_opacity)

    def set_opacity(self, element: str, value: float) -> None:
        """Set per-element opacity and save."""
        self._opacities[element] = value
        self.save()
        for cb in self.on_opacity_changed:
            cb(element, value)

    def get_global_opacity(self) -> float:
        return self._global_opacity

    def set_global_opacity(self, value: float) -> None:
        self._global_opacity = value
        self.save()

    def get_all_configured_elements(self) -> List[str]:
        """Return all elements that have a color assigned."""
        return list(self._colors.keys())

    def reset_to_defaults(self) -> None:
        """Reset all colors and opacities to MTEX-pastel defaults."""
        self._colors = dict(DEFAULT_COLORS)
        self._opacities.clear()
        self._global_opacity = DEFAULT_GLOBAL_OPACITY
        self._auto_idx = 0
        self.save()

    # ── Persistence ───────────────────────────────────────────────

    def save(self) -> None:
        """Save current config to JSON file."""
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "colors": self._colors,
                "opacities": self._opacities,
                "global_opacity": self._global_opacity,
            }
            self._config_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except OSError as e:
            logger.warning("Could not save EDS color config: %s", e)

    def load(self) -> None:
        """Load config from JSON file, falling back to defaults."""
        if not self._config_path.exists():
            return
        try:
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
            self._colors = dict(DEFAULT_COLORS)  # start from defaults
            self._colors.update(data.get("colors", {}))
            self._opacities = data.get("opacities", {})
            self._global_opacity = data.get("global_opacity", DEFAULT_GLOBAL_OPACITY)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Malformed EDS color config, using defaults: %s", e)
            self._colors = dict(DEFAULT_COLORS)
            self._opacities = {}
            self._global_opacity = DEFAULT_GLOBAL_OPACITY

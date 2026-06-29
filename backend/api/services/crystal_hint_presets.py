"""Material presets loader for the Crystal Hint feature.

Loads `backend/api/data/crystal_hint_material_presets.json` once at module
import time. Provides query helpers for the API layer.

See docs/superpowers/specs/2026-05-26-crystal-hint-feature-design.md Section 7.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_PRESETS_FILE = (
    Path(__file__).resolve().parent / "crystal_hint_material_presets.json"
)


@dataclass(frozen=True)
class ExpectedPhase:
    name: str
    sg: str
    a_A: float
    c_A: Optional[float] = None
    category: str = "unknown"


@dataclass(frozen=True)
class MaterialPreset:
    key: str
    label: str
    description: str
    elements: tuple[str, ...]
    expected_phases: tuple[ExpectedPhase, ...]
    heat_treatments: tuple[str, ...] = field(default_factory=tuple)


def _parse_preset(key: str, raw: dict) -> MaterialPreset:
    phases = tuple(
        ExpectedPhase(
            name=p["name"],
            sg=p["sg"],
            a_A=float(p["a_A"]),
            c_A=float(p["c_A"]) if "c_A" in p else None,
            category=p.get("category", "unknown"),
        )
        for p in raw.get("expected_phases", [])
    )
    return MaterialPreset(
        key=key,
        label=raw["label"],
        description=raw["description"],
        elements=tuple(raw.get("elements", [])),
        expected_phases=phases,
        heat_treatments=tuple(raw.get("heat_treatments", [])),
    )


def load_presets(path: Optional[Path] = None) -> dict[str, MaterialPreset]:
    """Load all presets from the JSON file. Raises FileNotFoundError / KeyError on bad input."""
    src = path or _PRESETS_FILE
    if not src.exists():
        raise FileNotFoundError(f"Material presets file not found: {src}")
    data = json.loads(src.read_text(encoding="utf-8"))
    if "presets" not in data:
        raise KeyError("Material presets file missing top-level 'presets' key")
    return {key: _parse_preset(key, raw) for key, raw in data["presets"].items()}


_PRESETS_CACHE: Optional[dict[str, MaterialPreset]] = None


def get_presets() -> dict[str, MaterialPreset]:
    """Cached accessor — load once per process."""
    global _PRESETS_CACHE
    if _PRESETS_CACHE is None:
        _PRESETS_CACHE = load_presets()
    return _PRESETS_CACHE


def get_preset(key: str) -> MaterialPreset:
    presets = get_presets()
    if key not in presets:
        raise KeyError(f"Preset '{key}' not in library. Available: {sorted(presets.keys())}")
    return presets[key]


def list_preset_keys() -> list[str]:
    return sorted(get_presets().keys())


def reload_presets() -> dict[str, MaterialPreset]:
    """Force re-read of the JSON file (testing / dev hot-reload)."""
    global _PRESETS_CACHE
    _PRESETS_CACHE = None
    return get_presets()


def boost_score_for_expected_phase(
    preset_key: str,
    phase_name: str,
    base_score: float,
    boost: float = 1.5,
) -> float:
    """If phase is in preset.expected_phases, multiply score by `boost`.

    Used by the aggregator in crystal_hint route to rank "expected for this alloy"
    candidates higher than unrelated cubic matches.
    """
    preset = get_preset(preset_key)
    exp_names_lower = {p.name.lower() for p in preset.expected_phases}
    if phase_name.lower() in exp_names_lower:
        return base_score * boost
    # Token-level match: extract first lex-token of each name (before space/punct)
    # so "Si cF8" matches "Si (eutectic)" via shared "si" token.
    import re

    def first_chem_token(name: str) -> str:
        # Strip greek prefixes, parens, brackets; take first alnum word
        norm = re.sub(r"^(alpha|beta|gamma|delta|theta)-?", "", name.lower())
        m = re.match(r"([a-z0-9]+)", norm)
        return m.group(1) if m else ""

    phase_token = first_chem_token(phase_name)
    if not phase_token:
        return base_score
    for exp_name in exp_names_lower:
        if first_chem_token(exp_name) == phase_token:
            return base_score * boost
    return base_score

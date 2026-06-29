"""
Texture Component Definitions for EBSD Analysis.

Phase-agnostic preset system for ideal texture components (FCC rolling, BCC rolling, etc.).
Each component is defined by Miller indices {hkl}<uvw> or Euler angles.

Reference: MATLAB/MTEX TextureAnalysis_mtex6.m lines 12-25
Architecture: the MTEX-equivalence design notes §2.4
FEAT-12: 14 FCC components (including Rot.Cube ND10 via Euler angles)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TextureComponent:
    """Definition of one ideal texture component.

    A texture component can be defined either by:
    1. Miller indices {hkl}<uvw> with crystal/specimen symmetry
    2. Euler angles (for components like Rot.Cube ND10)

    Attributes:
        name: Component name (e.g., "Brass", "Copper", "Cube")
        hkl: Miller indices for plane normal (None if defined by Euler)
        uvw: Miller indices for rolling direction (None if defined by Euler)
        euler_deg: Euler angles in degrees [phi1, Phi, phi2] (None if defined by Miller)
        symmetry_multiplicity: Number of symmetry equivalents under specimen symmetry
                              (e.g., 2 for Brass, 4 for S, 1 for Goss)
        description: Optional description or reference
    """
    name: str
    hkl: Optional[List[int]] = None
    uvw: Optional[List[int]] = None
    euler_deg: Optional[List[float]] = None
    symmetry_multiplicity: int = 1
    description: str = ""

    def __post_init__(self):
        """Validate that exactly one definition method is provided."""
        has_miller = self.hkl is not None and self.uvw is not None
        has_euler = self.euler_deg is not None

        if has_miller and has_euler:
            raise ValueError(f"{self.name}: Cannot define component with both Miller and Euler")
        if not has_miller and not has_euler:
            raise ValueError(f"{self.name}: Must define component with either Miller or Euler")


# ============================================================================
# FCC Rolling Texture Components (Aluminium, Copper, etc.)
# ============================================================================

FCC_ROLLING_COMPONENTS: Dict[str, TextureComponent] = {
    "Brass": TextureComponent(
        name="Brass",
        hkl=[0, 1, 1],
        uvw=[2, 1, 1],
        symmetry_multiplicity=2,
        description="{011}<211> — Beta-fiber component, typical for FCC rolling"
    ),
    "S": TextureComponent(
        name="S",
        hkl=[1, 2, 3],
        uvw=[6, 3, 4],
        symmetry_multiplicity=4,
        description="{123}<634> — S-component, important for formability"
    ),
    "Copper": TextureComponent(
        name="Copper",
        hkl=[1, 1, 2],
        uvw=[1, 1, 1],
        symmetry_multiplicity=2,
        description="{112}<111> — Cu-component, beta-fiber"
    ),
    "Goss": TextureComponent(
        name="Goss",
        hkl=[0, 1, 1],
        uvw=[1, 0, 0],
        symmetry_multiplicity=1,
        description="{011}<100> — Goss texture, common in Fe-Si"
    ),
    "Inv. Goss": TextureComponent(
        name="Inv. Goss",
        hkl=[0, -1, 1],
        uvw=[0, 1, 1],
        symmetry_multiplicity=1,
        description="{0-11}<011> — Inverted Goss / Rotated Goss"
    ),
    "Rot.Cube ND45": TextureComponent(
        name="Rot.Cube ND45",
        hkl=[1, 0, 0],
        uvw=[0, 1, 1],
        symmetry_multiplicity=1,
        description="{100}<011> — Rotated Cube ND45, PSN or shear origin"
    ),
    "Rot.Cube ND22": TextureComponent(
        name="Rot.Cube ND22",
        hkl=[0, 0, 1],
        uvw=[3, 1, 0],
        symmetry_multiplicity=2,
        description="{001}<310> — Rotated Cube ND22"
    ),
    "Rot.Cube ND10": TextureComponent(
        name="Rot.Cube ND10",
        euler_deg=[10.0, 0.0, 0.0],
        symmetry_multiplicity=2,
        description="Euler [10,0,0] — Rotated Cube ND10 (defined via Euler angles)"
    ),
    "Rot.Cube RD10": TextureComponent(
        name="Rot.Cube RD10",
        euler_deg=[0.0, 10.0, 0.0],
        symmetry_multiplicity=2,
        description="Euler [0,10,0] — Rotated Cube RD10"
    ),
    "Cube": TextureComponent(
        name="Cube",
        hkl=[0, 0, 1],
        uvw=[1, 0, 0],
        symmetry_multiplicity=1,
        description="{001}<100> — Cube texture, recrystallization"
    ),
    "Q": TextureComponent(
        name="Q",
        hkl=[0, 1, 3],
        uvw=[2, -3, 1],
        symmetry_multiplicity=4,
        description="{013}<2-31> — Q-component"
    ),
    "P": TextureComponent(
        name="P",
        hkl=[0, 1, 1],
        uvw=[1, 2, 2],
        symmetry_multiplicity=2,
        description="{011}<122> — P-component"
    ),
    "R": TextureComponent(
        name="R",
        hkl=[1, 2, 4],
        uvw=[2, 1, 1],
        symmetry_multiplicity=4,
        description="{124}<211> — R-component, 4-fold symmetry"
    ),
    "Shear 2": TextureComponent(
        name="Shear 2",
        hkl=[1, 1, 1],
        uvw=[1, 1, 2],
        symmetry_multiplicity=2,
        description="{111}<112> — Shear component"
    ),
}

# ============================================================================
# BCC Rolling Texture Components (Ferrite, Steel, etc.)
# ============================================================================

BCC_ROLLING_COMPONENTS: Dict[str, TextureComponent] = {
    "Goss": TextureComponent(
        name="Goss",
        hkl=[0, 1, 1],
        uvw=[1, 0, 0],
        symmetry_multiplicity=1,
        description="{011}<100> — Goss texture in BCC"
    ),
    "Rot. Goss": TextureComponent(
        name="Rot. Goss",
        hkl=[1, 1, 0],
        uvw=[0, 0, 1],
        symmetry_multiplicity=1,
        description="{110}<001> — Rotated Goss in BCC"
    ),
    # Additional BCC components can be added here:
    # - Alpha fiber: <110>||RD (rolling direction)
    # - Gamma fiber: <111>||ND (normal direction)
}

# ============================================================================
# Preset Collections
# ============================================================================

TEXTURE_PRESETS: Dict[str, Dict[str, TextureComponent]] = {
    "FCC_Rolling": FCC_ROLLING_COMPONENTS,
    "BCC_Rolling": BCC_ROLLING_COMPONENTS,
}


def get_texture_preset(preset_name: str) -> Dict[str, TextureComponent]:
    """Get a texture component preset by name.

    Args:
        preset_name: Name of preset ("FCC_Rolling", "BCC_Rolling")

    Returns:
        Dictionary of {component_name: TextureComponent}

    Raises:
        KeyError: If preset_name not found
    """
    if preset_name not in TEXTURE_PRESETS:
        available = ", ".join(TEXTURE_PRESETS.keys())
        raise KeyError(f"Preset '{preset_name}' not found. Available: {available}")

    return TEXTURE_PRESETS[preset_name]


def list_texture_presets() -> List[str]:
    """List all available texture preset names.

    Returns:
        List of preset names
    """
    return list(TEXTURE_PRESETS.keys())


def get_component_names(preset_name: str) -> List[str]:
    """Get list of component names in a preset.

    Args:
        preset_name: Name of preset

    Returns:
        List of component names in order
    """
    preset = get_texture_preset(preset_name)
    return list(preset.keys())


def validate_component(component: TextureComponent) -> bool:
    """Validate a texture component definition.

    Args:
        component: TextureComponent to validate

    Returns:
        True if valid

    Raises:
        ValueError: If component is invalid
    """
    # Check name
    if not component.name:
        raise ValueError("Component name cannot be empty")

    # Check definition
    has_miller = component.hkl is not None and component.uvw is not None
    has_euler = component.euler_deg is not None

    if not has_miller and not has_euler:
        raise ValueError(f"{component.name}: No definition provided")

    # Validate Miller indices
    if has_miller:
        if len(component.hkl) != 3:
            raise ValueError(f"{component.name}: hkl must have 3 elements")
        if len(component.uvw) != 3:
            raise ValueError(f"{component.name}: uvw must have 3 elements")
        if all(x == 0 for x in component.hkl):
            raise ValueError(f"{component.name}: hkl cannot be [0,0,0]")
        if all(x == 0 for x in component.uvw):
            raise ValueError(f"{component.name}: uvw cannot be [0,0,0]")

    # Validate Euler angles
    if has_euler:
        if len(component.euler_deg) != 3:
            raise ValueError(f"{component.name}: euler_deg must have 3 elements")

    # Validate symmetry multiplicity
    if component.symmetry_multiplicity < 1:
        raise ValueError(f"{component.name}: symmetry_multiplicity must be >= 1")

    return True


def add_custom_component(preset_name: str, component: TextureComponent):
    """Add a custom texture component to a preset.

    Args:
        preset_name: Name of preset to modify
        component: TextureComponent to add

    Raises:
        KeyError: If preset not found
        ValueError: If component is invalid
    """
    validate_component(component)

    if preset_name not in TEXTURE_PRESETS:
        available = ", ".join(TEXTURE_PRESETS.keys())
        raise KeyError(f"Preset '{preset_name}' not found. Available: {available}")

    TEXTURE_PRESETS[preset_name][component.name] = component


def create_custom_preset(preset_name: str, components: Dict[str, TextureComponent]):
    """Create a new custom texture preset.

    Args:
        preset_name: Name for the new preset
        components: Dictionary of {component_name: TextureComponent}

    Raises:
        ValueError: If any component is invalid or preset already exists
    """
    if preset_name in TEXTURE_PRESETS:
        raise ValueError(f"Preset '{preset_name}' already exists")

    # Validate all components
    for comp in components.values():
        validate_component(comp)

    TEXTURE_PRESETS[preset_name] = components

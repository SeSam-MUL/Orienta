"""
EBSD Analysis Module — Python/PyQt5 reimplementation of MTEX post-processing workflows.

This package provides grain reconstruction, texture analysis, deformation analysis,
and recrystallization classification for indexed EBSD data.

Architecture:
- Pure Python (numpy, scipy, orix, sklearn, matplotlib, openpyxl)
- NO PyQt5 imports in this package (GUI code lives in gui/)
- Input: orix.crystal_map.CrystalMap from any indexing method
- Output: Statistical results, Excel reports, matplotlib figures for embedding

Modules:
    ebsd_dataset       - Unified data model wrapper (EBSDDataset + GrainSet)
    grain_analysis     - Grain reconstruction, size analysis
    bc_analysis        - Band contrast histogram + 3-Gaussian GMM
    deformation_analysis - KAM, GB classification, Sphericity
    texture_components - Phase-agnostic texture component presets (FCC/BCC)
    texture_analysis   - ODF, volume fractions, surface plane fractions
    rx_analysis        - Recrystallization classification (GOS/gBC/gKAM)
    smoothing          - Orientation smoothing (HalfQuadratic, Median)
    excel_exporter     - Documentation.xlsx export (14 sheets)
    batch_processor    - Folder → Documentation.xlsx batch processing

Reference:
    - the MTEX-equivalence design notes (basis)
    - matlab_testskripts/ (source of truth)
    - the FEAT-12 design notes (critical gaps + Excel structure)
"""

from analysis.ebsd_dataset import EBSDDataset, GrainSet
from analysis.grain_analysis import (
    reconstruct_grains,
    calculate_grain_properties,
    analyze_grain_size,
    GrainSizeResult,
)
from analysis.bc_analysis import (
    calculate_bc_histogram,
    fit_bc_gmm,
    grain_average_bc,
    BCHistogramResult,
    BCGMMResult,
)
from analysis.deformation_analysis import (
    calculate_kam,
    calculate_kam_histogram,
    classify_grain_boundaries,
    calculate_gb_metrics,
    calculate_gb_segment_histogram,
    calculate_sphericity,
    analyze_deformation,
    DeformationResult,
)
from analysis.texture_components import (
    TextureComponent,
    FCC_ROLLING_COMPONENTS,
    BCC_ROLLING_COMPONENTS,
    TEXTURE_PRESETS,
    get_texture_preset,
    list_texture_presets,
    get_component_names,
    validate_component,
    add_custom_component,
    create_custom_preset,
)
from analysis.texture_analysis import (
    calculate_volume_fraction,
    calculate_texture_component_fractions,
    calculate_surface_plane_fraction,
    analyze_texture,
    TextureResult,
)
from analysis.rx_analysis import (
    calculate_grain_average_kam,
    calculate_area_weighted_histogram,
    classify_rx_grains,
    calculate_rx_fraction,
    analyze_recrystallization,
    RXResult,
)
from analysis.smoothing import (
    median_orientation_filter,
    quality_weighted_mean_filter,
    apply_smoothing,
)
from analysis.excel_exporter import (
    ExcelExporter,
    export_to_excel,
    export_batch_to_excel,
    excel_column,
)
from analysis.batch_processor import (
    BatchProcessor,
    batch_process_folder,
)

__all__ = [
    "EBSDDataset",
    "GrainSet",
    "reconstruct_grains",
    "calculate_grain_properties",
    "analyze_grain_size",
    "GrainSizeResult",
    "calculate_bc_histogram",
    "fit_bc_gmm",
    "grain_average_bc",
    "BCHistogramResult",
    "BCGMMResult",
    "calculate_kam",
    "calculate_kam_histogram",
    "classify_grain_boundaries",
    "calculate_gb_metrics",
    "calculate_gb_segment_histogram",
    "calculate_sphericity",
    "analyze_deformation",
    "DeformationResult",
    "TextureComponent",
    "FCC_ROLLING_COMPONENTS",
    "BCC_ROLLING_COMPONENTS",
    "TEXTURE_PRESETS",
    "get_texture_preset",
    "list_texture_presets",
    "get_component_names",
    "validate_component",
    "add_custom_component",
    "create_custom_preset",
    "calculate_volume_fraction",
    "calculate_texture_component_fractions",
    "calculate_surface_plane_fraction",
    "analyze_texture",
    "TextureResult",
    "calculate_grain_average_kam",
    "calculate_area_weighted_histogram",
    "classify_rx_grains",
    "calculate_rx_fraction",
    "analyze_recrystallization",
    "RXResult",
    "median_orientation_filter",
    "quality_weighted_mean_filter",
    "apply_smoothing",
    "ExcelExporter",
    "export_to_excel",
    "export_batch_to_excel",
    "excel_column",
    "BatchProcessor",
    "batch_process_folder",
]

"""Public API for the GPU dictionary indexer."""
from __future__ import annotations

from .pipeline.indexer import run_dictionary_index as gpu_dictionary_index_patterns

__all__ = ["gpu_dictionary_index_patterns"]

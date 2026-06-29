"""EBSD-specific pipeline stages for GPU spherical indexing.

This package contains our own implementation (preprocessing, detector
geometry, SHT I/O, indexer, refiner, output writer). The math primitives
live in ``backend.spherical_gpu._math`` (vendored from ebsdtorch).
"""

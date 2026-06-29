"""GPU-native EBSD forward model (SP0+SP1).

A PyTorch package that computes the dynamical EBSD master pattern of a
phase from EMsoft crystal data + an (initially EMsoft-derived) Monte-Carlo
energy/depth distribution, parallel to EMsoft (which remains the validation
oracle). See ``docs/superpowers/specs/2026-06-10-forward-sim-sp01-design.md``.
"""

"""Monte-Carlo for the GPU forward model.

- emsoft_mc_input.py: read an EMsoft MC .h5 into MCData (the SP0/SP1 validation input).
- gpu_monte_carlo.py: SP2 GPU-native PyTorch Monte Carlo producing an EMsoft-compatible
  MCData (accum_e/accum_z/lambda_z) — makes the pipeline fully EMsoft-free.
- scattering_mc.py: screened-Rutherford (Joy 1995) elastic + Bethe-Joy-Luo CSDA physics.
- composition.py: crystal -> (mean Z, A, density) MC target.
"""

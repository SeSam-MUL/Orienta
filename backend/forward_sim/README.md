# `backend/forward_sim` — EMsoft-free GPU forward master-pattern model

This package computes a dynamical EBSD **master pattern** for a crystal phase
directly in Python/PyTorch (with CuPy/numba acceleration), without invoking the
EMsoft Fortran binaries. Starting from EMsoft crystal data (`.xtal` / master
`.h5`) and a Monte-Carlo energy/depth distribution, it ports EMsoft's scattering
physics term-for-term — Weickenmeier-Kohl complex scattering factors, the
dynamical scattering/scattering-matrix path, the `CalcLgh` analytic depth
integral, and the `CalcSgh` structure weighting — to build the Lambert-square
master on the GPU. The result is serialised back into the exact EMsoft / EMSphInx
file layouts so it is a drop-in for the rest of Orienta (kikuchipy dictionary
indexing, the SHT spherical-indexing reader, and the simulation "missing master"
scan). EMsoft remains the validation oracle, not a runtime dependency.

## Top-level files

| File | Description |
|------|-------------|
| [`__init__.py`](__init__.py) | Package docstring; marks the GPU-native forward model (SP0+SP1 and beyond). |
| [`runtime.py`](runtime.py) | Device selection. `get_device()` is fail-loud (no silent CPU fallback; raises `ForwardSimError` on a CUDA-less host unless `FORWARD_SIM_DEVICE` is set); `resolve_device_adaptive()` is the hardware-adaptive resolver used by the self-contained "Ours" engine (GPU when present, else CPU, never raises). |

## Subpackages

| Subdir | Description |
|--------|-------------|
| [`crystal/`](crystal/) | **SP0 — crystal data → complex potential coefficients.** Read EMsoft `CrystalData` into a `CrystalStructure`, compute Weickenmeier-Kohl complex scattering factors, build the geometric reflection list and the complex `U_g` table with Bethe partitioning. (See files below.) |
| [`dynamical/`](dynamical/) | **SP1 — dynamical scattering → master pattern.** Assemble the dynamical matrix `A` / scattering matrix `S`, evaluate the `CalcLgh` analytic depth integral, the `CalcSgh` per-atom structure weighting, the point-group symmetry-orbit reduction, and the master-pattern builder (`build_master`). |
| [`mc/`](mc/) | **Monte Carlo.** Read an EMsoft MC `.h5` into `MCData` (the fixed SP0/SP1 input) and the GPU-native PyTorch/CuPy/numba Monte-Carlo engines (screened-Rutherford + Bethe-CSDA, Joy 1995) that make the pipeline fully EMsoft-free. |
| [`io/`](io/) | **Serialisation + integrity checks.** Write our GPU master into EMsoft `master.h5`, EMSphInx `.sht`, and EMsoft MC `.h5`, plus `master_validation.py` to detect the disc-masked-corner artifact. (See files below.) |
| [`validate/`](validate/) | **Validation harnesses vs the EMsoft oracle.** Master-vs-master NCC and projected pattern-vs-pattern / indexing-parity checks. |

### `crystal/`

| File | Description |
|------|-------------|
| [`crystal/xtal_io.py`](crystal/xtal_io.py) | `read_crystal_structure()` reads EMsoft `CrystalData` (`.xtal` or master `.h5`) into a `CrystalStructure`; also exposes centrosymmetry helpers (whether the southern Lambert hemisphere can be derived by inversion). |
| [`crystal/scattering_factors.py`](crystal/scattering_factors.py) | `wk_scattering_factor()` — the complex Weickenmeier-Kohl electron scattering factor `f = f_el + i·f_abs` (Å), parametrised exactly as EMsoft's `WEKO`/`FSCATT`. The most NCC-critical primitive. |
| [`crystal/structure_matrix.py`](crystal/structure_matrix.py) | `reflection_list()` (geometric reflections with `d ≥ dmin`), `compute_Ug_table()` (complex potential coefficients `U_g`), and the Bethe strong/weak partition — with EMsoft `CalcUcg` normalisation constants. |

### `io/`

| File | Description |
|------|-------------|
| [`io/__init__.py`](io/__init__.py) | Re-exports `write_master_h5`, `write_sht`, `write_mc_h5`. |
| [`io/master_h5.py`](io/master_h5.py) | `write_master_h5()` — wraps our master in a faithful EMsoft `EBSDmaster` HDF5 file; a drop-in for `kikuchipy.load(projection=...)` and dictionary indexing. |
| [`io/sht_writer.py`](io/sht_writer.py) | `write_sht()` — forward-transforms our master to spherical-harmonic coefficients and writes the binary EMSphInx `.sht` format (consumed by `backend/spherical_gpu`'s SHT reader). |
| [`io/mc_h5.py`](io/mc_h5.py) | `write_mc_h5()` — serialises our GPU Monte-Carlo output into an EMsoft `EMMCOpenCL` MC `.h5` so `load_mc` / `scan_missing` / the crystal picker recognise the GPU run. |
| [`io/master_validation.py`](io/master_validation.py) | Detects the disc-masked-Lambert-corner bug (stale masters whose square corners were zeroed by an inscribed-disc mask, which corrupt dictionary indexing). `lambert_disc_mask_fraction()`, `is_lambert_disc_masked()`, and `h5_master_is_disc_masked()` let loaders/preview/scan fail loud instead of silently mis-indexing. |

### `dynamical/`

| File | Description |
|------|-------------|
| [`dynamical/scattering_matrix.py`](dynamical/scattering_matrix.py) | `build_A` (dynamical matrix, EMsoft `GetDynMat` mode `'D'` parity) and its exponentiation into the scattering matrix `S(z)`. |
| [`dynamical/depth_integral.py`](dynamical/depth_integral.py) | `Lgh` via the EMsoft `CalcLgh` eigen-decomposition analytic depth integral, weighted by the MC escape-depth distribution. |
| [`dynamical/structure_weight.py`](dynamical/structure_weight.py) | `compute_Sgh` — the direction-independent per-atom back-scatter structure weighting (EMsoft `CalcSgh`). |
| [`dynamical/symmetry_orbit.py`](dynamical/symmetry_orbit.py) | Point-group symmetry-orbit reduction — cuts the per-direction eigensolve count by solving only the irreducible Lambert zone and scattering by symmetry. |
| [`dynamical/master_builder.py`](dynamical/master_builder.py) | `build_master()` — the SP1 capstone assembling the master pattern on the Lambert grid from the validated SP0/SP1 primitives. |

### `mc/`

| File | Description |
|------|-------------|
| [`mc/emsoft_mc_input.py`](mc/emsoft_mc_input.py) | `load_mc()` reads EMsoft `EMData/MCOpenCL` (`accum_e` / `accum_z` / energy axis) into `MCData` — the fixed, validated MC input for SP0/SP1. |
| [`mc/gpu_monte_carlo.py`](mc/gpu_monte_carlo.py) | `run_gpu_mc()` — SP2 GPU-native PyTorch Monte Carlo (step-major, all electrons in parallel) producing an EMsoft-compatible `MCData`. |
| [`mc/cupy_mc_kernel.py`](mc/cupy_mc_kernel.py) | One-thread-per-electron CuPy `RawKernel` MC (each electron's full trajectory in registers, mirroring EMsoft's `EMMC.cl`). |
| [`mc/numba_mc.py`](mc/numba_mc.py) | One-electron-per-thread numba CPU MC — the no-GPU fallback analogue of the CuPy kernel. |
| [`mc/scattering_mc.py`](mc/scattering_mc.py) | Elastic scattering physics: screened-Rutherford (Joy 1995) + Bethe-Joy-Luo CSDA energy loss, selectable via `ElasticModel`. |
| [`mc/composition.py`](mc/composition.py) | Derives the MC target material properties (occupancy-weighted mean `Z`, mean `A`, density `rho`) from a `CrystalStructure`. |

### `validate/`

| File | Description |
|------|-------------|
| [`validate/ncc_vs_emsoft.py`](validate/ncc_vs_emsoft.py) | Builds our master, downsamples and disc-masks the EMsoft oracle master, and reports the zero-mean NCC — the SP0+SP1 acceptance gate. |
| [`validate/pattern_parity.py`](validate/pattern_parity.py) | Projected-pattern parity: compares detector patterns simulated from our master vs the EMsoft master (the quantity a user actually indexes against), plus indexing parity. |

## How it fits into Orienta

Orienta indexes EBSD patterns by correlating experimental patterns against
simulated **master patterns**. Historically those masters came from EMsoft's
Fortran/OpenCL binaries run via WSL (see [`../../simulation/`](../../simulation/)).
This package is the GPU-native, EMsoft-free alternative: it produces masters in
the same on-disk formats, so its outputs flow unchanged into the existing
consumers —

- **kikuchipy dictionary indexing** and the GPU dictionary path
  ([`../dictionary_gpu/`](../dictionary_gpu/)) read the EMsoft `master.h5` written
  by [`io/master_h5.py`](io/master_h5.py);
- **spherical (SHT) indexing** ([`../spherical_gpu/`](../spherical_gpu/)) reads
  the `.sht` written by [`io/sht_writer.py`](io/sht_writer.py);
- the **simulation "missing master" scan** and crystal picker recognise a GPU run
  via the MC `.h5` written by [`io/mc_h5.py`](io/mc_h5.py).

EMsoft is retained as the validation oracle: the `validate/` harnesses gate the
forward model against EMsoft masters and projected patterns rather than calling
EMsoft at runtime.

## Run / use notes

- **Device.** GPU-native. The strict indexing paths use `get_device()` from
  [`runtime.py`](runtime.py), which **fails loud** on a host without CUDA — set
  `FORWARD_SIM_DEVICE=cpu` (or `cuda:1`, etc.) to override. The self-contained
  "Ours" simulation engine uses `resolve_device_adaptive()` instead, which falls
  back to CPU automatically (via the numba / PyTorch CPU MC and CPU master build).
- **Inputs.** Crystal data is read from EMsoft `.xtal` files or master `.h5`
  (both expose the same `CrystalData` group). EMsoft conventions are honoured
  verbatim: lattice `a, b, c` in nm, angles in degrees, Debye-Waller `B` in nm²,
  `AtomData` rows `[x, y, z, occ, B]`.
- **EMsoft-parity.** The scattering physics is ported term-for-term from EMsoft
  source; see the module docstrings and `tasks/forward_sim/lessons.md` for the
  unit reconciliations (notably the WK `S`-variable scaling and the `CalcUcg` 4π
  normalisation), which are the load-bearing details for NCC parity.
- **Outputs are drop-ins, not re-runs.** The `io/` writers never invoke EMsoft;
  they wrap our computed master in the byte/group layout each downstream reader
  expects. Use [`io/master_validation.py`](io/master_validation.py) to reject
  stale disc-masked masters before they reach indexing.

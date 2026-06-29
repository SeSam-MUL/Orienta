# Vendor Source Attribution

These files are vendored from **ebsdtorch** (MIT-licensed) by Zachary Varley.

- Repository: https://github.com/ZacharyVarley/ebsdtorch
- Commit SHA: `ed05cecf29b16aa9745a892216d1c579b9c32fb0`
- Commit date: 2025-02-17
- License: MIT (verbatim copy in `LICENSE.ebsdtorch`)
- Vendored on: 2026-04-30

Why vendor instead of pip-install: ebsdtorch has no stable release at vendor
time. Pinning a commit + checking the files into our repo gives us patch
freedom, byte-stable behaviour across environments, and zero dependency on
upstream availability.

## Files vendored

| Local file | Upstream path |
|---|---|
| `sht.py` | `ebsdtorch/harmonics/sht.py` |
| `sht_cc.py` | `ebsdtorch/harmonics/sht_cc.py` |
| `_wigner_logspace.py` | `ebsdtorch/harmonics/wigner_d_logspace.py` |
| `_wigner_xnum.py` | `ebsdtorch/harmonics/wigner_d_xnum.py` |
| `lambert.py` | `ebsdtorch/s2_and_so3/sphere.py` |
| `LICENSE.ebsdtorch` | `LICENSE` (top-level) |

## Local additions (not from upstream)

| File | Reason |
|---|---|
| `wigner_d.py` | Facade over `_wigner_logspace` + `_wigner_xnum` so callers get a single `wigner_d(j, beta, method="auto")` entry point. |
| `__init__.py` | Re-exports the public surface our pipeline uses. |
| `__SOURCE.md` | This file. |

## Local patches

### Patch 1: rewrite cross-package imports to relative imports (2026-04-30)

- Reason: Upstream `sht.py` and `sht_cc.py` import from `ebsdtorch.harmonics.*`
  and `ebsdtorch.io.*` packages. Since we vendor only `_math/`, those absolute
  imports fail (no `ebsdtorch` installed in our env).
- Files touched:
  - `sht.py` line 23 — `from ebsdtorch.harmonics.wigner_d_logspace import (...)`
    → `from ._wigner_logspace import (...)`
  - `sht_cc.py` line 49 — `from ebsdtorch.harmonics.sht import (...)`
    → `from .sht import (...)`
  - `sht_cc.py` line 50 — `from ebsdtorch.harmonics.wigner_d_logspace import (...)`
    → `from ._wigner_logspace import (...)`
  - `sht_cc.py` line 55 — `from ebsdtorch.io.read_master_pattern import
    read_master_pattern` → replaced with a `NotImplementedError`-raising stub.
    Our own SHT I/O lives at `backend.spherical_gpu.pipeline.sht_io`.
- Search markers: each patched line is annotated with a `# patched 1: was ...`
  comment so the original upstream form is preserved in the source.

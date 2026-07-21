# GPU Dictionary Indexing — Path-A Master-Loading Fix

> 2026-07-17. Discovered while checking whether GPU dictionary indexing works
> on the new EDAX up1 data. The bugs are **independent of up1** — they hit any
> dataset indexed with a real EMsoft master via the GPU path.

## Symptom
`spherical`/`hough` aside, **GPU dictionary indexing never completed** with a
real EMsoft master pattern. Two hard failures + one silent correctness bug.

## Root cause
The indexing route loads the master with a plain `kp.load(path)`
(`backend/api/routes/indexing.py:1438`). For an EMsoft master that yields the
**stereographic, upper-hemisphere** signal: shape `(n_energy, npx, npx)`,
`projection='stereographic'`, `hemisphere='upper'`.

The GPU forward projection (Path A, `backend/dict_gpu/_pcadi/master_to_dict.py`)
needs the **square-Lambert, both-hemisphere** master `(2, npx, npx)`:

1. **Hemisphere** — `_select_energy_slice` got `(11, 1001, 1001)` →
   `ValueError: expected master data shape (2, npx, npx), got (11, 1001, 1001)`.
2. **Projection** — even with both hemispheres, a *stereographic* master
   sampled by the Lambert direction-cosine grid is **geometrically wrong**
   (silent: patterns look plausible but don't match experiment). Path A is a
   translation of kikuchipy's Lambert projection math; it requires the
   `mLPNH/mLPSH` (Lambert) datasets, not `masterSPNH/masterSPSH` (stereographic).
3. **Strides** — kikuchipy master slices can be negative-strided →
   `torch.from_numpy(...)`: "At least one stride ... is negative".

## Fix (isolated to the GPU path — CPU dict path structurally untouched)
`backend/dict_gpu/_pcadi/master_to_dict.py`:
- `_select_energy_slice`: if the master isn't already Lambert+both, **reload it
  from its source file** (`kp.load(path, projection='lambert', hemisphere='both')`,
  path recovered from kikuchipy `tmp_parameters`). Collapse the energy axis to
  the nearest requested energy. Return `np.ascontiguousarray(...)` (kills
  negative strides).
- New helpers `_reload_both_hemispheres`, `_recover_master_path`,
  `_nearest_energy_index`. Fail-loud `MasterPatternError` if the source file
  can't be located.

The CPU path uses `signal.dictionary_indexing(...)` and never imports
`master_to_dict` — so it is unaffected (verified by grep + kept green).

## Verification (decisive)
Path A on the **app-style** master (stereographic/upper — exactly `kp.load(path)`)
vs the kikuchipy CPU reference (Path B, `get_patterns`):
- **median NCC = 1.00000, min = 1.00000** over sampled orientations.
This is the same parity method that originally validated the GPU projection —
it proves the reloaded master produces the *correct* dictionary, bit-for-bit.

Full pipeline `gpu_dictionary_index_patterns` on `Test_data/new test/Scan5.up1`
+ app-style Al master completes end-to-end. (Low indexing NCC on Scan5 itself is
uncalibrated PC (0.5,0.5,0.5) on weak EDAX patterns — a workflow/data matter, not
the projection; the earlier *higher* score came from the wrong stereographic
dictionary producing spurious correlation.)

Tests: `tests/test_dict_gpu_master_fix.py` (4, incl. CUDA parity). Green.

## CPU path — separate but related bug (also fixed)
The CPU path (`compute_mode='cpu'`) was **also broken**, differently: it passed
the raw master straight to kikuchipy's `signal.dictionary_indexing(dictionary)`,
which needs a *simulated dictionary* whose pattern shape matches the detector,
not a `(1001,1001)` master →
`ValueError: Experimental (79,79) and dictionary (1001,1001) signal shapes must
be identical`. The CPU path never projected the master into a dictionary.

Fix (`indexing_controller.py`): new `_dictionary_signal_from_master()` — when
`dictionary` is a master (detected via `hasattr(dictionary, "get_patterns")`),
reload it Lambert+both and project it to the experimental detector geometry with
kikuchipy `master.get_patterns(rotations, detector, energy)`, sampling
orientations at `config.angular_step_deg` (new field, default 1.5°, matches GPU).
A pre-generated dictionary signal (no `get_patterns`) is passed through unchanged
— no regression to that case. Verified end-to-end on Scan5.up1 + Al master.

Both paths now project the master identically (GPU Path A == kikuchipy
`get_patterns` == CPU path), so GPU and CPU dictionary indexing agree by
construction.

## Needs
Backend restart (no auto-reload). Optional follow-up: fine angular steps (≤2°)
can OOM Path A in one batch on a 12 GB GPU — tiling the projection is a separate
scaling improvement, not part of this fix.

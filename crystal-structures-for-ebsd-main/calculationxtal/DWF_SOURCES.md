# Debye–Waller Factors (DWF.xlsx) — sources & provenance

`DWF.xlsx` maps an element symbol to its **Debye–Waller factor B at 300 K**,
used during CIF → `.xtal` conversion (`convert-cif-to-xtal`). The value is the
isotropic B-factor in **nm²** (B = 8π²⟨u²⟩); B[nm²] = B[Å²] × 0.01.
Example: Al = 0.007993 nm² = 0.7993 Å².

If an element is **not** in the table, the converter falls back to a generic
`0.005` nm² — see `Xtal_Generator_GUI.py` / `database.py`
(`dwf_map.get(symbol, 0.005)`). Add missing elements with a real, cited value.

## Source of the current values

All values are **B at 300 K from Peng et al. (1996), Table 1**, transcribed
from the deposited supplement (IUCr SUP82472, article zh0008):

> **Peng, L.-M., Ren, G., Dudarev, S. L. & Whelan, M. J. (1996).**
> *Debye–Waller factors and absorptive scattering factors of elemental
> crystals.* Acta Cryst. **A52**, 456–470. doi:10.1107/S0108767396005382

Peng Table 1 covers **44 elemental crystals** over 1–1000 K (from phonon DOS,
2–3 % accuracy). 38 of them have a solid value at 300 K and are loaded here;
the four noble gases (Ne, Ar, Kr, Xe) have no 300 K solid value and are omitted.

The six pre-existing legacy values (Al, Si, Mg, Ni, Fe, Mn) were checked
against Peng during this work:
- **Al, Si, Mg, Ni** — exact match to Peng @ 300 K (confirmed provenance).
- **Fe** — the legacy value `0.5771` matched **neither** Peng phase
  (Fe-BCC 0.3328, Fe-FCC 0.5710) and was an error. Corrected to the
  **BCC** value **0.3328** (α-ferrite is the room-temperature stable phase;
  the FCC value at 300 K is an extrapolation of a non-stable phase). For Fe in
  intermetallics the true B is site-specific, but BCC is the RT-grounded
  reference and the DWF only weakly affects the simulated pattern.
- **Mn** — **not present in Peng 1996** (it has no Mn or Co), and not in
  Butt, Bashir, Willis & Heger (1988), Acta Cryst. **A44**, 396 either (22
  cubic elements, no Mn). Set on 2026-06-08 to **0.0037 nm²** (B ≈ 0.37 Å² at
  300 K), read from a published B(T) plot for Mn in the La2/3Ba1/3MnO3
  perovskite — an oxide environment, but consistent with a Debye-model
  estimate for metallic Mn (Θ_D = 410 K → 0.39 Å²) and with the 3d neighbours
  in Peng (Cr 0.26, Fe 0.33, Ni 0.36). The Reference column names that plot;
  the value is the one non-Peng entry in the table.

### Elements with more than one row (structure variants)
Peng lists some elements once per crystal structure. Such an element has one
row per structure, the structure in brackets (`Fe (BCC)`, `Fe (FCC)`), and the
converter picks the row like this (`backend/api/services/dwf_table.py`,
`build_dwf_map`):
- a phase that is the element's **own crystal** (one element only, space group
  Fm-3m → `FCC`, Im-3m → `BCC`, P6₃/mmc → `HCP`, Fd-3m → `DIA`) takes the row
  carrying that structure's label, if there is one;
- every other phase (Fe in a compound or an alloy, an unlabelled structure)
  takes the **first** row of the element, which is the room-temperature stable
  structure. For Fe that is `Fe (BCC)`, the value used before the FCC row
  existed. For Fe in an intermetallic the true B is site-specific; the table
  has no better value for it.

Current variants: `Fe (BCC)` 0.3328 Å² and `Fe (FCC)` 0.5710 Å² (both Peng
Table 1, 300 K; checked against the deposited supplement, page 3). Not added
(no phase in the library needs it yet): Ca (BCC) @ 300 K = 2.6615 Å²; the `Ca`
row is Peng's FCC value 2.0443 Å².

**Alloyed austenite keeps the BCC iron value.** The FCC row is chosen only for
an ELEMENTAL crystal (one element in the phase). A future alloyed austenite --
an Fe-Ni, Fe-C or Fe-Cr-Ni CIF in Fm-3m -- has more than one element, so its iron
takes the first row, `Fe (BCC)` 0.3328 Å², although the structure is FCC. This
is the rule working as written, not an oversight: Peng's FCC iron value is for
pure FCC iron, and the table has no value for iron in a solid solution. If such
a phase is added, give it its own value through `dwf_overrides` in the
conversion request (or an explicit table row) and cite where it comes from.

## How to add / edit entries

Use the **Debye-Waller Factors** dialog in the Crystal Database page (inline-
editable value + Reference, plus a “+ Add” row). It upserts via
`PATCH /api/database/dwf`. Always fill the `Reference` column with the citation
so every value keeps its provenance.

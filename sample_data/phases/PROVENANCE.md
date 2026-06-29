# Sample crystal structures — provenance & license

Every CIF in this folder comes from the **Crystallography Open Database (COD)**,
where all data are dedicated to the **public domain under CC0**
(<https://www.crystallography.net/cod/>). They may be freely redistributed and
reused. Please cite the original publication when you use a structure.

| File | Phase | COD ID | Space group | Formula | Original source |
|------|-------|--------|-------------|---------|-----------------|
| `Al.cif` | Aluminium (fcc matrix) | [9008460](https://www.crystallography.net/cod/9008460.cif) | Fm-3m | Al | *Crystal Structures* (Wyckoff), 1963 |
| `Al7Cu2Fe.cif` | Al₇Cu₂Fe | [7223749](https://www.crystallography.net/cod/7223749.cif) | P4/mnc | Al₇Cu₂Fe | *RSC Adv.* 6 (2016) |
| `Al2CuMg.cif` | Al₂CuMg ("S-phase") | [7222567](https://www.crystallography.net/cod/7222567.cif) | Cmcm | Al₂CuMg | Heying et al., *Z. Naturforsch. B* 60 (2005) 491 |
| `alpha-AlMnSi.cif` | cubic α-(Al,Mn,Si) phase | [2007194](https://www.crystallography.net/cod/2007194.cif) | Pm-3 | Al₄Mn₁.₀Si₀.₇ | *Acta Cryst. C* (1998) |

## Why these (and not the originals in `Database/CIF_Library/`)
These open (CC0) structures were chosen specifically so the sample phases can be
**redistributed**. Earlier library entries for the same phases came from
**proprietary** sources (ICSD / FIZ Karlsruhe, SpringerMaterials / Pauling File)
whose CIF files may **not** be redistributed — so they are not shipped here.
The crystal structures themselves are scientific facts; only the specific
proprietary files are restricted.

## Coverage for the sample datasets
`Al` + `Al7Cu2Fe` + `Al2CuMg` + `alpha-AlMnSi` index the **Al matrix** (the
dominant fraction) and the major **Fe/Cu/Mn-bearing intermetallic particles** in
the SampleB, Scan 1 and AA7050 datasets. Phases not in this set — e.g. **MgZn₂
(η)** in 7050 or **Mg₂Si** — will remain unindexed; add their structures if you
need them.

## Regenerating `.xtal` / master / SHT
The `.xtal`, dynamical master pattern and `.sht` files are **derived** from these
CIFs (via the simulation pipeline / EMsoft). Generate them from these open CIFs so
the derived files inherit the same clean CC0 provenance.

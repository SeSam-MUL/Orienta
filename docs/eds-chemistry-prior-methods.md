# EDS-Guided Phase Disambiguation at Indexing — Methods

*Paper-ready methods writeup. Orienta EBSD software. 2026-07-31.*

## 1. Motivation

Kikuchi-pattern indexing assigns each map pixel a crystallographic phase by
correlating its measured pattern against phase references — simulated master
patterns (spherical harmonic / dictionary) or band geometry (Hough). Phases that
are **crystallographically near-identical but chemically distinct** produce
near-identical Kikuchi-band geometries, so pattern-only indexing ranks them
almost equally and the winner is effectively noise. The canonical case in
aluminium alloys is **fcc Al (cF4, m‑3m)** vs **diamond‑cubic Si (cF8, m‑3m)**:
both are cubic with the same Laue class, so Si particles embedded in an Al matrix
are frequently mislabelled as Al. Energy‑dispersive X‑ray spectroscopy (EDS),
acquired simultaneously on the same map grid, separates the two trivially by
composition.

This method couples the two modalities: the per‑pixel EDS composition biases the
phase competition toward the chemically‑consistent phase, decisively for
degenerate ties, **without overriding a clear pattern match**.

## 2. Soft per-phase chemistry prior

For candidate phase *p* and pixel *i*, define a multiplicative weight

> **w_p(i) = (1 − s_p) + s_p · f(c_i, e_p)**

- **c_i** — measured EDS composition at pixel *i* in atomic percent, obtained by
  the standard counts → weight‑% → atomic‑% quantification of the per‑element
  window integrals.
- **e_p** — expected atomic‑percent composition of phase *p*, parsed from its
  stoichiometric formula (e.g. `Al` → 100 % Al; `Al5FeSi` → 71.4/14.3/14.3 %),
  with an optional per‑phase manual override for elements EDS mis‑quantifies.
- **f(·,·) ∈ [0, 1]** — a chemistry‑consistency score (Section 3).
- **s_p ∈ {0, 1}** — whether chemistry participates. **s_p = 0 ⇒ w_p ≡ 1 ⇒ no
  effect** (indexing is bit‑identical to the pattern‑only result).

> **Revision, 2026‑08‑05 — s_p is no longer per‑phase, and no longer continuous.**
> A per‑phase strength is *not* a symmetric prior: because w_p ≡ 1 exactly at
> s_p = 0, raising the strength on a subset of phases does not weight chemistry,
> it handicaps the raised phases while the untouched ones remain immune and
> absorb every ambiguous pixel. On the validation scan this inflated
> α‑Al(Fe,Mn)Si from 1.5 % to 10.1 % of the map when the strength was raised on
> Al and Si only (Section 5). Controlled full‑map runs found only two useful
> states — off, and on for all phases — so the parameter is now a single binary
> switch applied uniformly. The continuum had existed only so that one value
> (0.75) reproduced the interactive Phase Test's fixed damping; that is an
> implementation‑continuity property, not a physical one.

The per‑phase pattern score (a higher‑is‑better confidence: SO(3) correlation
peak for spherical, top‑1 NCC for dictionary, confidence index for Hough) is
multiplied by w_p(i) **before** the per‑pixel winner is selected
(`argmax_p`). Because w_p(i) ∈ [1 − s_p, 1], the prior can only *shrink* a
phase's effective score toward zero: a large pattern‑score margin still wins (a
clear match is preserved), while a near‑tie — precisely the Al/Si degeneracy —
flips to the chemically‑consistent phase. The strength s_p tunes how aggressively
chemistry acts, per phase, so a well‑quantified discriminator (e.g. Si) can be
weighted strongly while a noisier element is left neutral.

Setting **s_p = 0.75** recovers the fixed damping `score·(0.25 + 0.75·f)` used by
the software's interactive single‑pixel Phase Test — i.e. the full‑map prior is
the per‑pixel generalisation of an already‑validated interactive tool, so the
notion of "chemical consistency" is identical across the interactive and batch
workflows.

## 3. Chemistry-consistency function f

f maps a measured composition and an expected composition to [0, 1] (1 = perfect
match, 0 = disjoint):

1. **Restrict + renormalise.** Drop non‑diagnostic light elements (O, C — surface
   oxide / carbon coating) and renormalise each side to fractions summing to 1.
2. **L1 similarity.** `score = 1 − ½·Σ_e |c_e − e_e|` over the union of elements
   (L1 distance ∈ [0, 2]).
3. **Missing‑major‑element veto.** If the phase *requires* an element at ≥ 5 at%
   (nominal) that the pixel measures at < 1.2 at%, cap `score ≤ 0.05`. This is
   what suppresses an Fe‑Al‑Si intermetallic on a pure‑Al pixel (which the
   symmetric L1 would otherwise reward for the shared Al), and Al on a pure‑Si
   pixel.

   > **Both thresholds were corrected on 2026‑08‑05 (were 15 at% and 2 at%).**
   > The *requires* threshold at 15 at% never fired for a phase whose defining
   > elements are minor by stoichiometry: α‑Al(Fe,Mn)Si is defined by Fe 7.5 +
   > Mn 7.5 at%, both below the bar, so their absence went unpunished. Because
   > that phase's nominal Al/Si ratio lies *on* the Al↔Si mixing line, any
   > interaction‑volume mixture pixel resembles it by construction: on rim
   > pixels carrying 0.2 at% Fe it scored **0.596** against Si's **0.235**, i.e.
   > the chemistry actively preferred the impossible phase. The *absent*
   > threshold at 2 at% then sat *inside* the Mn distribution of a genuine
   > α particle (Mn 1.8–2.2 at% throughout), so counting noise perforated the
   > grain. Both are now placed in the empty gap between matrix level
   > (Mn ≈ 0.3 at%) and particle level (Mn ≈ 2.1 at%) rather than at the edge
   > of a measured distribution. A threshold sweep is given in Section 5.
4. **Present‑but‑missing penalty.** Damp the score by the fraction of significant
   (≥ 5 at%) measured composition the phase cannot account for.
5. **Fail‑soft.** Return the neutral value 1.0 when either composition is
   unavailable (no EDS, dead pixel, unparseable formula) — the prior then
   degrades to pattern‑only for that pixel/phase rather than corrupting it.

## 4. Per-method integration

All three indexing methods converge on the same operation — *reweight each
phase's per‑pixel score by w_p(i), then take the per‑pixel argmax* — but differ in
where the competition lives:

- **Spherical (SHT harmonic correlation).** The multi‑phase backend stacks the
  per‑phase SO(3)‑correlation peaks into a (P × N) matrix and picks the argmax; the
  (P × N) weight matrix multiplies these peaks before the argmax. The *reported*
  confidence (CI) is the **raw** peak, not the weighted value — the prior only
  steers which phase wins, so downstream quality maps remain physically
  meaningful.
- **Dictionary** and **Hough.** These are indexed per phase and merged by
  per‑pixel score; the weight multiplies each phase's score map before the merge
  argmax. Hough competes phases internally in the band‑voting stage, so a
  chemistry‑active multi‑phase Hough run is split into per‑phase single‑CIF runs
  to expose a reweightable per‑phase confidence.

**Off ⇒ bit‑identical.** When every strength is 0, or no EDS is present, the
weight matrix is `None` and every method executes the exact pattern‑only code
path (verified against source and by regression tests at all entry points).

## 4a. Pre-flight admissibility check

Before a run, each candidate phase is tested against the measured chemistry and
the result reported to the operator. For phase *p*, its **defining elements** are
those with nominal fraction ≥ 5 at% (the same rule the veto applies), and the
reported quantity is the fraction of the map on which *all* of them are measured
at ≥ 1.2 at%:

> **A_max(p) = |{ i : c_i,e ≥ 1.2 at% ∀ e ∈ D(p) }| / N**

A_max is an **upper bound on the phase's area fraction, not an estimate** — it is
the complement of the set the veto rejects, so a phase reported at 0 % will be
vetoed at every pixel, and the two can never disagree. For single‑element phases
the bound is trivially weak (trace Al is ubiquitous) and is flagged as such.

On the validation scan this bounds α‑Al(Fe,Mn)Si at **0.92 %** and Al₇FeCu₂ at
**0.32 %**, which would have exposed the 10.08 % that α obtained in run B before
any indexing time was spent. The check additionally verifies that the EDS and
EBSD grids are congruent — a mismatch pairs every pattern with a different
pixel's chemistry and is treated as blocking rather than as a warning.

## 5. Validation

On a real scan of an extruded aluminium alloy (402 × 301 px, EDS and EBSD
co‑registered on the same grid):

- **Chemistry discrimination.** At a probed Si‑rich pixel (Si ≈ 93.6 at%, Al ≈
  2.6 at%), the Si‑phase weight is **0.944** versus the Al‑phase weight **0.0015**
  at full strength: the missing‑major‑element veto (the Al phase requires ~100 %
  Al, the pixel measures 2.6 %) correctly collapses Al, so the chemistry breaks
  the Al/Si tie decisively toward Si.
- **Full‑map effect.** A multi‑phase Spherical index of a 6840‑pixel region
  (phases Al + Si) with the prior active **reassigned 425 pixels** relative to the
  pattern‑only baseline, resolving Si particles inside the Al matrix in the phase
  map (they were otherwise absorbed into the matrix as Al). The Hough path
  reproduces the behaviour after per‑phase splitting.

### 5.1 Controlled full‑map experiment (2026‑08‑05)

Five complete indexing runs of the same scan (**121 002 px**, 301 × 402, spherical
GPU backend, bandwidth 88, six phases: Al, Si, Al₇FeCu₂, Al₆Fe, α‑Al(Fe,Mn)Si,
Al₄FeSi). All parameters identical; only the prior configuration varied. Run **B**
reproduced an independently produced operator result digit for digit, which
establishes the comparison as controlled.

| run | prior | Al | Si | α | α median CI | Si particles with an α rim | real α grain recognised |
|---|---|---|---|---|---|---|---|
| B | s = 1 on Al + Si only | 87.22 | 1.17 | **10.08** | 0.397 | **52 / 58** | 99.9 % |
| A | off | 98.02 | **0.04** | 1.54 | 0.634 | 1 / 31 | 99.8 % |
| C | s = 1 all phases | 89.32 | 4.41 | 5.88 | 0.403 | 127 / 168 | — |
| D | C + `requires` 5 at% | 93.03 | 5.50 | 1.01 | 0.636 | **0 / 199** | 75.5 % |
| **E** | **D + `absent` 1.2 at%** | **92.62** | **5.50** | **1.43** | **0.635** | **0 / 199** | **99.8 %** |

Readings:

- **The asymmetric configuration is the failure mode**, not a mis‑tuning: B
  inflates α by 6.5× while its median CI *falls* to 0.397, i.e. the added pixels
  are poor matches.
- **Switching the prior off is not a remedy** — Si collapses to 0.04 %. On this
  material the pattern alone does not separate Al (cF4) from Si (cF8); the two
  are FCC‑derived with similar band geometry. Chemistry is genuinely required.
- **Symmetry alone is insufficient** (run C): 127 of 168 Si particles keep an
  α rim, because at those pixels the *chemistry itself* points the wrong way
  (f = 0.596 for α vs 0.235 for Si) for the structural reason given in §3.
- **Run E agrees with three independent measurements.** Si at 5.50 % against an
  EDS‑measured Si‑rich area of 5.3 % (Si > 50 at%); α at 1.43 % against an
  Fe/Mn‑rich area of 1.78 % (Fe > 3 at%) and against 1.54 % from the
  pattern‑only run; α median CI 0.635, identical to the pattern‑only value.
- **Per‑phase chemistry is self‑consistent in E** (median at% within each
  assignment): Si‑assigned pixels carry Si 69.5 at%; Al₇FeCu₂‑assigned pixels
  carry the map's highest Cu (10.2 at%); α‑assigned pixels carry the highest Mn
  (2.10 at%) and Fe 3.90 at%, against a matrix at Fe 0.19 / Mn 0.26 at%.
  96.5 % of the α area lies in one connected grain; 0.6 % is isolated speckle.

**Threshold sweep** for the *absent* level, evaluated on the same scan against
three requirements simultaneously (retain a genuine α grain; veto α on Si
boundaries; veto α in the matrix):

| `absent` | α grain retained | α vetoed on Si rims | α vetoed in matrix |
|---|---|---|---|
| 2.0 at% | 79.8 % | 100 % | 100 % |
| 1.5 at% | 99.2 % | 100 % | 100 % |
| **1.2 at%** | **100 %** | **100 %** | **100 %** |
| 1.0 at% | 100 % | 100 % | 99.8 % |
| 0.5 at% | 100 % | 99.4 % | 99.2 % |

A wide operating window (1.0–1.5 at%) satisfies all three, confirming that the
value is not finely tuned to this dataset; the previous 2.0 at% lay at the edge
of the window.

## 6. Limitations

- **EDS spatial resolution.** The X‑ray interaction volume is larger than the
  EBSD source region, so at particle/grain boundaries the measured composition is
  a mixture. Measured on the validation scan at 1.0 µm step: the Si 10–90 % edge
  width is **2.72 ± 0.04 µm**, against 1.39 µm for band contrast and 0.82 µm for
  the indexing confidence — the chemistry is 2–3× blurrier than the diffraction,
  and 18.6 % of the map is a chemically intermediate mixture (2‑endmember linear
  unmixing, R² = 0.9989). The prior remains bounded rather than a hard gate for
  this reason.
- **Absolute at% must not be compared to nominal stoichiometry.** With
  standardless Cliff–Lorimer quantification, heavy elements on this instrument
  read low by a factor ≈ 0.42 (a 1600 µm² α particle, far larger than the
  interaction volume, measures Fe+Mn 6.25 at% against 15.0 nominal) while light
  elements read at 0.9–1.05. No pixel in the scan reaches even 70 % of the
  nominal composition of any of the four intermetallics present. The veto is
  therefore formulated as *presence versus absence* with thresholds placed in
  the measured gap, never as agreement with the stoichiometric vector.
- **The consistency metric is dominated by the major element.** f is an L1
  distance over the full normalised composition vector, so an Al‑rich
  intermetallic in an Al matrix is only weakly separated from Al itself
  (measured at an α grain core: f = 0.91 for α against 0.71 for Al). The
  discriminating power presently resides in the veto rather than in the metric.
  A formulation based on *enrichment relative to the local matrix baseline*, or
  on a named ratio in the manner of the (I_Fe + I_Mn)/I_Si criterion of Kuijpers
  et al., would address this directly and is the natural next step.
- **A multiplicative prior can overturn a decisive pattern verdict.** Because the
  weight multiplies the score, a veto (cap 0.05) suppresses any pattern score
  however large its margin. Al (cF4, 4 atoms) and α (cI168, 168 atoms) are not
  degenerate — at the pixels concerned the pattern scored 0.635 for α against
  0.354 for Al — yet a mis‑placed threshold reassigned them. Summing calibrated
  log‑likelihoods instead, so that the chemistry term cannot exceed the pattern
  term's margin, would make this structurally impossible; the present thresholds
  treat the symptom.
- **Stoichiometric assumption.** Expected compositions assume ideal occupancy and
  are parsed from the phase's *file name*, which can disagree with the structure
  actually simulated (observed in this library for four of six phases; in one
  case a file labelled `Al4FeSi` carries a Si‑free Al₆Fe structure). A
  name↔structure consistency check is required before such an assignment is
  reported.
- **Tie‑breaker, not an oracle.** The chemistry biases *among plausible pattern
  candidates*; it cannot rescue a phase whose Kikuchi pattern is genuinely absent,
  only resolve degeneracies the pattern cannot.
- **Phase‑list composition dominates everything above.** A phase omitted from the
  list cannot be assigned, and its pixels are absorbed by the nearest surviving
  candidate: removing Si from the list on this scan returns α to 3.01 % with its
  median CI falling to 0.49. The pre‑flight check (§4a) exists to make this
  visible before a run rather than after.

## 7. Reproducibility (implementation map)

| Component | Location |
|---|---|
| Weight service (w_p(i), expected/measured composition) | `backend/api/services/eds_indexing_prior.py` |
| Consistency function f + formula parsing | `backend/api/services/crystal_hint_phase_fit.py` (`chemistry_fit`, `phase_nominal_at_pct`) |
| EDS quantification (counts→wt%→at%) | `eds_utils.py` |
| Spherical in‑backend reweight | `backend/spherical_gpu/backend.py` |
| Dictionary/Hough merge reweight | `indexing_controller.py` (`compute_comparison_maps`) |
| Route wiring + per‑phase Hough split | `backend/api/routes/indexing.py` |
| Pre‑flight admissibility check (§4a) | `backend/api/services/eds_preflight.py`, `POST /api/indexing/eds-preflight` |
| EDS↔EBSD grid guard (`EdsGridMismatch`) | `backend/api/services/eds_indexing_prior.py` |
| On/off switch + pre‑flight panel UI | `frontend/src/components/Indexing/EdsPreflightPanel.jsx`, `edsPriorParams.js` |
| Threshold + veto tests | `tests/test_eds_preflight_and_veto.py` |

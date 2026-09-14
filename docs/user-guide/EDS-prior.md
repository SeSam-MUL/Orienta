# The EDS chemistry prior at indexing

*Written 2026-09-12. Goes into `docs/user-guide/EDS.md` once the working copy of
that file is free; it is kept separate here because another session has it open.*

## What the switch does

On the Indexing page, **Use EDS chemistry** multiplies each phase's pattern
score by how well that pixel's measured composition matches the phase, before
the winner is chosen:

    weight = (1 - strength) + strength x chemistry_fit(measured, expected)

There is **one switch, not one slider per phase**, and it is deliberate. The
weight above leaves a phase at exactly 1.0 while its strength is 0, so raising
the strength on some phases and not others penalises only the ones you raised.
Measured on ProbeB in August 2026: turning it on for Al and Si alone inflated a
third phase from 1.5 % to 10 % of the map. Off is off for everything; on is on
for everything.

A phase the chemistry rules out keeps a weight of 0.05 — a 20x penalty, close
to a hard filter. The prior is not a tie-breaker at that strength; it decides.

## What "expected composition" means

In order of authority:

1. **What you typed for that phase.** A per-phase expected composition overrides
   everything below. Use it when you know your alloy.
2. **The structure the master pattern was simulated from** — the refined atom
   sites of its CIF — when it names the same elements as the file name.
3. **The formula in the file name.**

(2) is there because a file name is a label. `alpha-Al(Fe,Mn)Si` ships as
`Mn0.5Fe0.5Al5Si0.68`, which is the Pauling File's prototype name for the
*structure type* and splits the shared Fe/Mn site 1:1. The CIF behind it refines
that site as 0.812 Fe / 0.188 Mn, and SampleB measures Fe:Mn = 3.79. Fe and Mn
substitute for each other on one crystallographic site, so **their split is a
property of your alloy, not of the phase** — which is also why (1) exists.

If the file name and the structure disagree about which *elements* a phase
contains, the app keeps the file name and writes both compositions into the log.
That is a broken library entry, not a refinement, and it should be fixed in the
library rather than resolved silently. One such entry is shipped today:
`Al4FeSi (beta-AlFeSi)` was simulated from a CIF that contains no silicon.

## What it cannot do

* **Below about 2 at%, a number is not a measurement of that element.** The
  vendor's window integrals include the continuum under the window, so an
  aluminium matrix that dissolves at most ~0.03 at% Fe reads 1.7-1.8 at%. The
  scorer treats an element below 2 at% as not detected, and you should read the
  displayed at% the same way.
* **Cu and Zn are less reliable than Al, Si, Fe and Mn.** Across four scans of
  the same sample the K-line majors reproduce to a few per cent while Cu spreads
  36-58 % and Zn 16-39 %; the Zn window is additionally contaminated by Cu L
  emission. Treat a phase decided by Cu or Zn alone as unconfirmed.
* **A particle smaller than the interaction volume is measured together with the
  matrix around it.** At 20 kV that volume is 2-3 um across, so on a 0.15 um step
  a whole particle can read as a mixture. The prior compares renormalised
  compositions, which absorbs part of this, but an at% from a small particle is
  systematically pulled toward the matrix. Element *ratios* between two
  non-matrix elements survive the dilution far better than absolute values —
  compare Fe/Mn, not Fe.
* **It cannot separate two phases with the same chemistry.** That is what the
  pattern is for.

## Small particles the pattern cannot see: the orientation rescue

Some phase pairs the pattern cannot separate either. Al and Si share the fcc
band geometry; measured on Probe B, the correlation preferred the Al master on
every pixel of the scan, including pixels with 93 at% Si, and the Al fit and
the Si fit of the same pattern land on the same orientation. On such a pair the
phase decision is the prior's alone, and the prior says Si only when more Si
than Al is measured — about 48 at%. A Si particle smaller than the EDS volume
reads 20-40 at% and stays Al.

When the prior is on, a second pass therefore runs after the phase decision,
for every pair of phases in the run that share a point group and that the
prior actually weighed (a phase with EDS influence 0 takes no part). It takes the
element the particle phase has and the matrix phase lacks (Si for Al/Si; it is
read from the simulated structures, nothing is hard-coded), finds connected
blobs where that element is at least 15 at% and five times the matrix
background, and decides each pixel in a blob by **orientation continuity**:

* a pixel the run called matrix whose orientation matches **no** matrix pixel
  within three pixels around the blob is not matrix — it becomes the particle
  phase, keeping its measured orientation;
* a pixel the run called particle that sits at the rim, matches a **direct**
  matrix neighbour and is not part of the particle's own crystal goes back to
  the matrix phase. That is the beam standing on matrix beside the particle
  while the EDS volume still sees it.

Where continuity says nothing, nothing moves: a particle whose crystal happens
to share the matrix orientation (one on Probe B does), the interior of a large
particle with no matrix in reach, blobs under three pixels, and pixels the
EDS did not measure. The progress log reports the counts
(`EDS particle rescue by orientation continuity: Si: N px -> Si, M px -> Al`),
and the result carries them under `metadata["eds_particle_rescue"]` together
with the changed rows and their previous phase ids. With the prior off the pass
does not run and the result is unchanged.

On the Probe B crop this moved 94 pixels to Si (the five missed particles and
the rims of the large ones that continue the particle crystal) and 11 rim
pixels at 48-57 at% Si back to Al. It is only as good as the orientations: a
pixel with a bad pattern has a random orientation, and a lone such pixel with
enriched Si would pass the forward rule — the three-pixel minimum blob size is
what keeps that out. A pixel the pass moves keeps the confidence value the run
gave it, which was computed against the other phase's master; on Al/Si the two
differ by a few hundredths, and the CI layer does not know which one it shows.

## When to leave it off

* when the EDS map and the EBSD scan are on different grids (the app blocks it);
* when the counts are thin (the pre-flight panel says so);
* when the phases you are indexing differ only in structure, not in chemistry —
  then the prior can only add noise.

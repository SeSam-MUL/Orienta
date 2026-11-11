# Changelog

All notable changes to Orienta, newest first. While Orienta is pre-1.0, the
minor number (0.**x**.0) marks a release that changes how you work — a new
module, a reworked page — and the patch number (0.x.**y**) everything else:
fixes, and refinements to features that are already there.

You can see which version you are running under **Settings → About Orienta**.

---

## v0.4.8 — 2026-11-11

This release changes numbers you may have saved. Grain reconstruction, the
texture analysis (components, surface planes, index and entropy), the
recrystallization analysis and the EDS composition were reworked or corrected;
recompute the results you want to compare with earlier ones. The same goes for
Hough indexing of hexagonal, trigonal, monoclinic and triclinic phases, whose
reflector lists were corrected. It also adds EDAX APEX loading, hexagonal scans,
a Find PC search, NLPAR denoising, and fixes the "Connection to the backend
lost" message after start-up.

### Added

#### Loading and EDAX/hexagonal scans
- EDAX APEX measurements can be loaded with their metadata: open the APEX `.h5`
  file and choose a measurement, or open a `.up1`/`.up2` pattern file with its
  APEX file in the same folder. The grid (square or hexagonal), the step sizes,
  the pattern centre and the camera angles come from the metadata, and nothing
  is resampled. A pattern file is matched to its measurement by the ID in its
  header, not by its name (a file whose header has no ID can only be assigned by
  hand, and only if its pattern count equals the grid). Hexagonal-grid `.h5`
  files written by OIM Analysis (not APEX) are still resampled to a square grid
  when loaded, as before.
- A hexagonal scan is drawn as the hexagons lie on the sample, with true
  proportions and scale bar, in the EBSD Viewer overview, on the Phase Map page
  and in the navigation map of the Indexing page. Hover, click and the
  selections of the EBSD Viewer (and so a crop), and hover, click, the
  rectangle statistics and the line scan of the Phase Map page, act on the
  measurement point under the pointer. This is a display only: nothing is
  resampled, and every measurement point keeps exactly one value. The files do
  not say which way the rows are offset; the convention Orienta assumes is
  stated in the EBSD Viewer guide.
- On a hexagonal scan, the evaluations that compare a point with its neighbours
  use the six neighbours of the lattice: the phase-map cleanup, the
  grain-boundary layer, the pseudo-symmetry unification, the local-anomaly
  layer and the grain tools of Pattern Match and Phase Verification, among
  others (the complete list is in the EBSD Viewer guide). **Frame Average**
  averages each pattern with the points within the same number of steps on the
  sample, not with a square window of the index grid; its limits are in the
  guide.
- Results of a hexagonal scan can be saved. `.ang` (HexGrid header) and Orienta
  `.h5` keep the hexagonal grid and load back as hexagonal; `.ctf` and the new
  square `.ang` option are resampled onto a square grid (nearest measured
  point, as OIM Analysis does it) and say so. The batch and Save crop accept
  hexagonal scans. Where the row offset direction is assumed, each file says
  so.
- **A dialog chooses the measurement and the pattern source.** An HDF5 file
  that offers more than one now opens a dialog instead of loading the first
  measurement as processed patterns: one row per measurement, one choice among
  the pattern sources, with facts and a small preview of each; the last choice
  is preselected. A file with a single variant loads as before. For an EDAX
  APEX file the dialog lists the measurements with their overview image,
  hexagonal or square grid, step, the date the scan ended and the state of the
  pattern file. A row can be ticked only when it can be opened, and the reason
  is shown next to the greyed box. Pattern files in the folder that fit a row by
  pattern count can be assigned to it, and all rows can be searched in another
  folder at once. For a hexagonal measurement the dialog says plainly what is
  assumed (every second row shifted by half a step, which the file does not
  record) and what is not available yet. The chosen variant is remembered per
  file, and edited copies are kept per file and variant, so work on processed
  patterns is never restored into the unprocessed ones. Results and exports
  record which patterns were indexed, and activating a result reloads the
  variant it ran on. The Batch page says that it indexes the processed patterns
  of the first measurement of each file, whichever variant the dialog chose.
- NORDIF calibration patterns open as a scan of one row. Every loader now says
  where the pattern centre of a file comes from (the file header, an `.osc`, the
  APEX metadata, an Orienta export) or that the file has none, which is the case
  for NORDIF `.dat`/`.txt`, Oxford `.ebsp` and loose pattern images. Before,
  such files registered as having a measured pattern centre while the
  placeholder (0.5, 0.5, 0.5) was in use.

#### Pattern centre
- **Find PC** searches the pattern centre from the scan's own patterns, without
  a starting value. It picks nine calibration patterns from the scan (one from
  each ninth) when fewer than five are on the list, then searches one joint
  cost of all calibration patterns, moves its search box when the result sits on
  an edge and tries a second box when the result is unclear. The page says how
  the search ended, in one line: "Search ended normally." or one warning (edge of
  the searched area, no lowest point, two about equally good positions, a PC the
  store did not take). It also gives the PC with the number of patterns, the PC
  in the open file's own convention, the fit contrast and how far the patterns
  disagree (range per axis, in pixels). A result at an edge or without a lowest
  point shows the file's pattern with the bands drawn from the PC and the
  choices "Find PC again with other patterns" (it leaves out the positions
  already used), "Keep this PC anyway" and the step back. Cancel stops the run
  in every step.
- **Global PC Refine** is the same search started at the current PC. Cancel now
  stops the run; before, it only hid the progress bar, and the run went on and
  later wrote its PC unseen.
- **Type the pattern centre in the numbers you have.** The PC fields of the
  Detector Settings take Bruker (Orienta), EDAX/TSL or Oxford numbers; the
  selector starts at the vendor of the open file. Every conversion is made by
  the backend (kikuchipy's), and a live line shows the Orienta (Bruker) numbers.
  The numbers apply only when you press **Use these numbers** (Enter in a field
  presses it). Typing no longer applies a PC: before, a change in a field was
  sent after 500 ms and moved the crosshair and the forward preview. A PC that
  was refused is said in words.
- **Saved calibrations.** A PC can be saved under a name as a calibration, with
  the acquisition settings it is valid for (pattern size and binning, tilts
  within 0.1°, working distance within 0.1 mm when both files state one), the
  file it was made on, the quality of the search and every earlier version.
  Calibrations are stored as one file each in the calibrations folder next to
  the phase library. A PC is a fraction of the pattern, so it carries over to
  another binning of the same pattern shape but not to another height-to-width
  ratio, and the page says so.
  - For the open file, the PC page offers the calibrations that fit: "Use it" for
    matching settings, "Use anyway" with the differences shown for others, each
    with what it would replace and the size of the change in pixels.
  - **Apply to other files** lists the files that fit, with working distance,
    their old PC with version and the new PC, and ticks those that would change;
    a file that does not fit can be opened with "Find PC for it". After applying,
    each file shows what became of it and can be put back.
  - After a search at a file that is on a saved calibration, the page asks first
    whether the other files on that version should follow, with the change and
    the fit contrast before and after. Nothing is passed on unless you say so.
  - **Take PC from...** reads the PC of another file or a calibration.
  - A result remembers the calibration version it was indexed with, and stored
    results computed with an older version are listed.
- **Put the PC back one step.** Every write of a PC (refine, Find PC, typed,
  Apply, a calibration) keeps the previous state, per file family; one step, and
  nothing is written to the file on disk.
- The PC page names where the PC in use comes from (file, Find PC, refine,
  typed, grid calibration, a named calibration with its version) and says when a
  file has no PC. With a per-pixel PC map it says that the fields show the mean
  of the map, that each pattern is indexed at the PC of its own pixel, and what a
  typed PC or a refine did to the map.

#### Analysis page
- Grain Analysis has two new settings. **Close holes (alpha)** sets how wide a
  not-indexed gap inside a grain may be and still be closed into that grain
  (default 3.1 px; 0 closes nothing). **Fill holes and scratches in maps**
  shows the closed pixels with the orientation of their grain, so IPF maps look
  continuous. The orientation given to a filled pixel is for display only and
  never enters a number; the area of closed pixels does count in the sizes that
  use the cell area (see **ECD from**).
- Grain Analysis shows and sets how the grain-size statistics are made.
  **Edge grains** chooses which grains are left out because they touch the
  border of the map: the rule of the MATLAB reference script
  (`matlab_testskripts/GrainSizeAnalysis_mtex6.m`; a strip of two pixel rows
  or columns, the default), MTEX's rule (grains that touch the border or a
  not-indexed area open to it), the outermost pixel row only, or none. A rule
  keeps all grains instead when it would leave fewer than 50 grains or less
  than 15 % of them (the outermost-row rule: when it would leave out more than
  85 %), and the page says so when that happens. **ECD from** chooses whether
  the equivalent circle diameter comes from the area of the grain's measurement
  cells (closed holes included, the default, as MTEX) or from the measured
  pixels times the step size squared (as the script). **Phase** restricts the
  statistics to one phase of the map, or, in a batch, to the phase with the
  most measured pixels of each file. The statistics block names the number of
  grains that entered the means, and the step size. The settings are
  recomputed as soon as you change them and are recorded in the workbook
  (Sheet1). After a reconstruction the page also shows the mean, median and
  standard deviation of the equivalent circle diameter and the mean aspect
  ratio, not only the number of grains.
- The Deformation tab sends its three parameters: **KAM order** (1 to 5,
  default 2), **KAM threshold** (default 7°) and **HAGB from** (default 15°;
  boundaries from 3° up to it are low-angle). The KAM map, the boundary
  lengths, the mean KAM, the RX analysis's KAM, the Excel export and the batch
  use them, and the page prints the values the numbers were made with. Until
  now the three inputs showed 1, 0.55° and 10° while the analysis always used
  2, 7° and 15°.
- The Texture and Recrystallization tabs have a **Phase** selection (default:
  the phase with the most indexed pixels of the map, as the reference script
  analyses one phase, the matrix). Each result names the phase it was made
  with, and the workbook records it in Sheet1 (Texture: row 67). A batch uses
  the phase with the most indexed pixels of each file. In the texture, the
  pixels of every other phase and the not-indexed pixels are no longer compared
  with the components or counted in the denominators; the texture components
  are cubic, so a phase that is not cubic is refused with the cubic phases of
  the map named. In the recrystallization analysis, the classification, the RX
  fraction and its denominator use the grains of that phase only; grains of
  other phases are not recrystallized and are drawn in a neutral colour on the
  RX map.
- The Batch tab shows the file that is being processed. It takes the grain
  reconstruction parameters from the Grain tab and checks them (threshold above
  0, a whole minimum grain size of 1 or more, alpha 0 to 20) before it starts.
  Reconstruction parameters typed on the Grain tab are kept when you switch
  tabs, until a reconstruction has used them.
- The scalar maps (band contrast, KAM, GOS, ECD) show a colour bar under the
  map with the smallest and largest value and the unit; the ECD map also names
  its basis.
- After a grain reconstruction, Grain Analysis shows **Citations for these
  grains**, the panel already known from the Phase Map page: a methods
  paragraph that states the reconstruction and statistics settings, a reference
  list and BibTeX. It cites the grain-detection method of Bachmann, Hielscher
  and Schaeben (Ultramicroscopy, 2011) and their MTEX paper (Solid State
  Phenomena, 2010), with Orienta, kikuchipy and orix. The workbook of the page
  and of the batch has the same texts on a new `Citations` sheet; the other
  eleven sheets are unchanged.

#### EDS
- **Composition from the spectra.** When an EDS file holds the spectrum of
  every pixel (a spectrum cube, as in an AZtec `.h5oina`), the EDS pages
  quantify from a fitted model of those spectra instead of AZtec's window
  integrals. The window integrals are raw sums over a band of 45 to 90 eV
  around each line and include the background; the model fits the lines, the
  escape peaks and the background, with energy offset, gain, peak width and
  absorption fitted once per file on the summed spectrum. On a reference scan
  with known phases, a silicon particle read 93.2 at% Si from the windows and
  96.6 at% from the spectra, the iron reading in the matrix fell from 0.78 to
  0.21 at%, and the largest deviation of an Al7Cu2Fe region from its
  stoichiometry fell from 6.35 to 2.09 at%. This is the only path for a file
  with a spectrum cube; a file without spectra uses the window path, unchanged.
- The fit runs in the background in its own process. While it runs, the
  quantifying pages say that the spectra are being evaluated and fill in by
  themselves afterwards. Opening a 29 GB file until the result was ready took
  38.5 s; for a 10 800-pixel scan the per-pixel projection takes 0.30 s and
  opening to result 3.2 s. If you close a file and open it again within 2 s,
  the running fit is stopped and starts again.
- **Matrix correction.** On the spectrum path, composition is corrected for
  absorption, atomic number and fluorescence effects with a phi(rho z) model
  (XPP, with specimen tilt and fluorescence), using the acquisition geometry in
  the file header (beam energy, detector elevation, specimen tilt). The tilt is
  handled by scaling the depth of the ionisation curve; this was checked only
  for a 70° specimen tilt with the specimen tilted towards the detector (exit
  angle 81°). Without geometry in the header, with a specimen tilt outside 0 to
  89.5°, or with a shadowed detector, the uncorrected spectrum result is used
  and a warning says why. Copper
  is quantified over its K lines on the corrected path (over L on the
  uncorrected spectrum path). Carbon is left out of the composition, because
  its signal is the contamination layer on the surface; its map stays. Lines of
  Mg, Ni and Cr have no reference measurement behind their response and are
  named in a warning, and oxygen-bearing areas were not tested. The
  composition note and the export caveat state the correction, tilt, exit angle
  and reference set that were used.
- How well this agrees with AZtec was measured on one detector, one geometry
  and one alloy family. AZtec's own quantification is standardless, so what was
  measured is agreement with it (and with the nominal Al7Cu2Fe stoichiometry),
  not accuracy against a true composition. Mean absolute difference per element
  before and after the correction: small silicon particle 3.47 to 0.63 at%,
  Al7Cu2Fe box 1.20 to 0.46 at%, a whole-map sum 0.92 to 0.76 at%, and the
  matrix and alpha regions that carry the calibration 0.51 to 0.13 and 0.53 to
  0.15 at%. With each reference region left out of its own calibration (28
  values from 7 regions), the deviation from the reference had a median of
  0.5 at%, about nine in ten values at or below 1.9 at% (25 of 28) and a
  maximum of 2.5 at%. Other alloys will deviate by amounts that were not
  measured; on two scans of a 7050 alloy, the corrected zinc reads 6.57 and
  6.75 wt% instead of 3.71 and 3.73, and magnesium 2.21 instead of 1.49 wt%
  (limits published for alloy 7050: Zn 5.7 to 6.7, Mg 1.9 to 2.6 wt%, quoted from the
  literature, not checked against a specification), and the scanned areas hold particles.
- Regions (Inspector, region statistics, magic wand, particle export) are
  quantified from the summed net counts of the region instead of the mean of
  the clipped per-pixel at%, which was biased upward for weak elements.
- Elements that the spectrum shows but AZtec did not export are not added to
  the composition. They can be added as a net-count map.
- The note under every composition now says, without being opened, that the
  composition is not calibrated on reference standards; opened, it names the
  detector the instrument response was measured on. The Wt.% tooltip and the
  export summary line describe the path that was used instead of saying "no
  ZAF correction" for every result. The EDS user guide has a new section, *How
  the composition is calculated*, with the equations of both paths, what the
  instrument response was measured on, and what the measured agreement does
  and does not tell you.

#### Indexing
- NLPAR pattern denoising in the EBSD Viewer (PyEBSDIndex): each pattern is
  replaced by a weighted average of similar patterns in its neighbourhood, with
  the strength estimated from the scan itself. It is meant for very noisy scans
  whose raw patterns do not index: on a public nickel scan series at the
  highest noise level, the share of pixels within 2° of a low-noise reference
  rose from 9 % to 80 % with dictionary indexing. On scans that already index
  well it brings nothing and can make results worse near grain boundaries and
  small particles (by several percentage points on one aluminium-alloy scan).
  It runs on the CPU and needs room for a second copy of the patterns, so crop
  large scans first. Before the run, the panel estimates from a sample of the
  scan whether NLPAR may help or may blur fine structures.

#### Desktop app and backend
- Every end of the backend is logged in `logs/orienta.log` with its reason, its
  run time, the last request, the requests still running and its memory use; the
  desktop app adds how the process ended (exit code or signal, and whether the
  app stopped it) to `logs/backend-console.log`. Python stacks of a native
  crash go to `logs/backend-fault.log`. A request that holds the backend for
  more than 2 s is logged with where each thread was.
- When the backend ends without the app having stopped it, the desktop app
  shows a notice on the page: how it ended, what it was working on and whether
  it is being restarted. A stop by the app itself (quit, removing the data) is
  never restarted.
- Open dialogs (Browse for a scan, a CrystalMap, a batch folder) start in the
  folder of the last file or folder opened instead of Downloads. A save does
  not move this folder, so an export does not open among the raw data.

#### Simulation and library
- CIF import fills the reference dialog from the CIF (authors, journal, DOI;
  TeX and HTML accents are read as letters). A CIF stored in space group P 1
  whose atoms have more symmetry is detected with spglib over tolerances from
  0.01 to 0.1 Å; where the group changes along that range you choose the
  tolerance, and no copy is written before you do. A header whose space-group
  number and symbol contradict each other is reported. Text passed on to EMsoft
  is written as ASCII and capped in bytes.

### Changed

#### Analysis page
- Grain reconstruction on the Analysis page, in batch processing and in the
  Excel export was rewritten, and its results change. Each measurement is now a
  Voronoi cell (the method of Bachmann et al., 2011). Holes and thin
  preparation scratches inside a grain no longer split it, and a grain never
  spans two phases. Grain areas,
  equivalent circle diameters, perimeters, aspect ratios, GOS, grain-boundary
  lengths and KAM by grain are computed from these cells. The aspect ratio is
  now the long over the short principal axis of the grain's shape (it used to
  be the width over the height of the bounding box). On three test maps the
  grains and these values agree with MTEX for Python 1.0.0 to better than
  1e-5; before, one of them gave 96 grains where MTEX gives 176, and on a
  three-phase map 62 of 90 grains held more than one phase. Grain counts,
  grain-size statistics, recrystallized fractions and the Excel sheets
  therefore differ from earlier versions; recompute results you want to
  compare. Reconstruction is also faster: about 12 s instead of 34 min on an
  805 × 602 map with 65 % not-indexed pixels. A map with many not-indexed
  pixels needs about 2 GB of memory per million pixels; if that is not
  available, the reconstruction stops with a message instead of running out of
  memory.
- The texture is computed on one phase (see **Phase** above). The default is
  the phase with the most indexed pixels of the map; before, it was the lowest
  phase number, which is not the matrix when the matrix is listed second, and
  the page did not say which phase it used. Component fractions, surface-plane
  fractions, the component map and the rolled-versus-recrystallized split
  change on maps with more than one phase.
- The **Texture Components** layer drew an empty (black) image. It now shows
  each pixel of the texture phase in the colour of the component it is closest
  to, grey where no component is within the tolerance and black for other
  phases and not-indexed pixels, with a legend. The ECD, GOS and similar grain
  maps no longer count pixels without a grain as 0: they are left out of the
  colour scale, so the scale starts at the smallest grain value.
- Each scalar map is drawn with its colours stretched between its own smallest
  and largest value, so a change of a few percent in some grains (for example
  after switching the ECD basis) was invisible; the colour bar now shows the
  range.
- The Batch tab keeps its own edge rule, ECD basis and phase choice when you
  switch tabs. The minimum intercept is only recorded in the workbook: leaving
  the field empty no longer stops a reconstruction or a batch, and the workbook
  cell stays empty instead of showing 1.5.
- The README listed the recrystallization statistics `low_gkam_fraction` and
  `low_kam_fraction` as placeholders reported as 0.0. Both are computed (they
  already were in v0.4.7; 0.0 only for a map without grains), and the README
  no longer lists them.

#### EDS
- EDS compositions of files with a spectrum cube change (see Added), and so do
  the phase maps and presets built from them. Presets and phase maps remember
  which source they were built on; loading one built on the window integrals
  shows a notice, and its thresholds stay as they are. Recompute what you want
  to compare with earlier results.
- Automatic classification, the plausibility report and phase suggestion use
  the spectrum numbers. On the two scans with a reference they were better or
  equal: the Al7Cu2Fe region is now named Al7FeCu2 instead of Al2Cu, and the
  adjusted Rand index of one scan rose from 0.752 to 0.773.
- The EDS prior of indexing, the pixel chemistry of the phase test and Crystal
  Hint, particle rescue and the grain-based phase assignment stay on the window
  integrals. With spectrum numbers the correct phase ranked first for 74.55 %
  of the alpha-phase pixels of one scan, against 89.37 % with the windows,
  until these scorers are recalibrated.

#### Pattern centre
- **A file without a pattern centre refuses indexing.** An EDAX `.up1`/`.up2`
  without its `.osc` used to be indexed at kikuchipy's placeholder (0.5, 0.5,
  0.5), which gave confident, wrong orientations (measured on a copy without its
  `.osc`). Hough, Dictionary, Spherical and EMSphInx runs, the batches and the
  phase test now refuse with a message that tells you to calibrate the PC or to
  provide the `.osc` or metadata file; a batch request that brings its own PC is
  let through. A dictionary is no longer built at the placeholder. The `.ang`
  export of such a file writes 0 for x/y/z-star and a comment line that the PC
  was never calibrated.
- **A file's PC belongs to its full path.** Two files with the same name in
  different folders no longer share one pattern centre on the PC page, in batch
  runs or when a PC is copied to other files. A PC copied to other files follows
  rules that keep the copies of one measurement together and ask before it
  reaches other files; a PC typed on one measurement no longer lands on the
  batch data of another.
- **Every result records the pattern centre, its source and the detector tilts
  it was indexed with**, for all methods (a per-pixel map is recorded as its
  mean and flagged). The `.ang`, the `.h5` exports and the batch exports write
  that record instead of whatever is set when the export is made; before, a
  result indexed at one PC and exported after another was typed carried the new
  PC. A result made before this version has no record: its `.h5` says "not
  recorded" and its `.ang` has zero x/y/z-star and a note. The single PC of a
  cropped map is the mean of the cut map. An Oxford rich export saved as `.h5`
  is read by the Oxford reader again, with its pattern-centre map and camera
  tilt.
- The "PC Drift Analysis" buttons on the PC Refinement page are removed: they
  always answered with the same placeholder sentence, which reads as "no drift".
  AZtec's per-pixel PC map and the grid calibration's map are unchanged.

#### Indexing
- The experimental ML Hub page and the "Embedding" indexing option are no
  longer shown; neither worked in installed copies.
- **Hough reflector lists come from correct structure factors in every crystal
  system.** For cells with non-orthogonal axes (hexagonal, trigonal, monoclinic,
  triclinic) the atom sites were moved when the unit cell was completed, so the
  structure factors and the default Hough list were wrong; for hcp Mg, {2-1-11}
  had a non-zero structure factor although it is extinct. The default list, the
  reflector family table, the extinction lock for a hand-added family and the
  Kikuchi-line overlays now use the structure factors of the real cell. Cubic,
  tetragonal and orthorhombic cells without duplicate sites keep their list row
  for row; other phases get a different list, so recompute Hough results you
  want to compare. The default list is a function of the phase and the rule
  (d >= 1 Å, |F| above a tenth of the strongest, up to 70 rows, no repeated rows)
  and never of free memory; a limit of N rows keeps the N strongest rows.
- A Hough library that does not fit the memory budget is refused with a
  message on the Indexing and PC Refinement pages that names the phase, its
  rows, the predicted library and the budget. A button "Use the largest list that
  fits (N rows)" is offered, except for monoclinic and triclinic phases (point
  group m included), where a shorter list was measured to lose orientations, and
  never below 40 rows: on the S phase of a real 7050 map, 32 rows gave 0.3 % of
  the reference pixels within 5° and 40 rows gave 38.0 %. Every result records
  the list it was indexed with (rule, families, number of rows, predicted
  library and memory budget), exports carry that record, and the methods text
  states a list that is not the full default one.
- Trigonal phases of space groups 149, 151, 153, 157, 159, 162 and 163 are
  handled in their true point-group setting: the reflector families (without it
  a family held reflections that differ physically; 48 to 60 of 284 reflections
  had a wrong structure factor, and Cr2N had 42 of 212), the multiplicities, a
  hand selection of families, and the setting kept through a save and load of a
  project or orix file. A Hough run on a trigonal or hexagonal phase carries a
  note that variants can be confused. The m-3m grain segmentation refuses
  trigonal phases. Orientation grids and pseudo-symmetry operators of Dictionary
  and Spherical indexing are not changed.

#### Simulation and library
- The master-pattern sphere in the Database Browser is computed in a second
  Python process. Memory used by a high-resolution sphere is returned to the
  system when it is done, a sphere that fails no longer risks Orienta itself,
  and a **Cancel** button next to the progress indicator stops a sphere you no
  longer need. The first sphere after starting Orienta can take noticeably
  longer, especially with a virus scanner, while the process loads its
  libraries. Elapsed seconds and a note while the process is still starting are
  shown; a Cancel is shown as "Cancelled." with a Retry button. The sphere
  viewer is no longer stuck: reconstructions run one at a time (six
  simultaneous High requests took the process from 3 GB to 13.7 GB of RAM, three
  Max requests to 28.5 GB), identical requests share one reconstruction, a
  request the viewer has left is dropped, and a failed one can be retried. When
  memory runs out the sphere answers with a short message instead of a trace; on
  a computer without a GPU, a sphere the RAM cannot hold is refused beforehand
  instead of the machine ceasing to answer.
- On computers with a GPU, the EMsoft route computes the master pattern on the
  CPU. The result of EMsoft's GPU master program differs from the CPU
  program's, and Dictionary Indexing cannot load it. The Monte-Carlo step still
  runs on the GPU. Master patterns that an earlier version wrote with the GPU
  program stay listed as not loadable. The Orienta Engine is not affected.

#### Desktop app and backend
- Faster start: the phase lists after the second start of the backend come from
  a cache of what was read from the CIF files (in `~/.orienta/cache`), reused
  only for the same file, size, modification time and reader version. Phase
  discovery took 11.8 s and the Crystal Hint index 11.8 s (46 CIFs) on every
  start; on the second start they take 0.1 s and 2.0 s.
- **Release GPU** also ends the idle compute process, which keeps its own CUDA
  memory (433 to 702 MiB measured) for 180 s after its last job.
- A development start (`npm run electron:dev`) shows the waiting page and loads
  the dev server once the backend answers; if the dev server does not answer,
  the window says what to start and to press Ctrl+R.

### Fixed

#### Analysis page
- **The four RX criteria on the Recrystallization tab had no effect.** The page
  never sent them, so every classification used the script's values (GOS 1.25°,
  gBC 0.7, gKAM 0.55°, radius 0.4 µm) whatever the fields showed. **Classify**
  now sends them (GOS 0 to 20°, gBC 0 to 1, gKAM 0 to 10°, radius 0 to 100 µm)
  and prints the criteria next to the numbers; the RX map, the Excel export,
  the batch and **Run Complete Analysis** use the same ones. Red on the RX map
  is exactly the set of grains the numbers count. The RX analysis needs the
  band contrast of the measurement: on a map without it the page says so and
  draws no red. Results made before with other values than the script's were
  computed with the script's values.
- **The preset and the tolerance angle on the Texture tab had no effect.** The
  page never sent them, so every run used FCC rolling and 10°. They are now
  sent with the run, remembered with the phase, and used by the component
  fractions, the Texture Components map, the rolled-versus-recrystallized
  comparison, the Excel export and the batch; the workbook records them
  (Sheet 1, rows 68 and 69). Runs made with another preset or tolerance than
  the defaults were still computed with the defaults.
- **Texture-component fractions written by earlier versions were wrong;
  recompute them.** The ideal orientations were built as the smallest rotation
  that aligns one crystal direction with one sample direction, not as
  `{hkl}<uvw>`: Inv. Goss came out equal to Cube and Goss equal to Rotated Cube
  ND45, the other components were 18° to 45° away from their definitions, and
  the sample symmetry (mmm) was applied on the wrong side, so every component
  had a single variant. Each component is now `{hkl}<uvw>` (plane normal along
  ND, a crystal equivalent of `<uvw>` perpendicular to it along RD) or its Euler
  angles, times the sample symmetry, and a fraction is counted as the reference
  MATLAB script counts it: the shares of pixels within the tolerance of each
  distinct variant are added, so a pixel near two variants of one component
  counts twice. On an aluminium map the 14 fractions equal those MTEX computes
  for the same orientations (largest difference 2e-16 percent at 5°, 10° and
  15°). MTEX for Python's own .h5oina loader uses another specimen frame, so
  its numbers for the same file differ in some components. This changes the
  component columns of the Documentation workbook, the rolled-versus-
  recrystallized comparison sheet, the texture index and entropy, and the
  texture component map. In the map, a pixel that two components claim equally
  goes to the first one, and a pixel without an orientation to none.
- **Texture index and entropy are now the ODF-based values of the reference
  script**, not approximations from the component fractions: `norm(odf)^2` and
  `entropy(odf)` of a kernel density of the orientations (5° half-width). A
  texture without preferred orientation has index 1 and entropy 0. On two
  aluminium test maps the index agrees with MTEX's to about five digits
  (25.5737 against 25.5733 on one, 4.27622 against 4.27620 on the other), and
  the entropy to 0.003 or better (-2.1834 against -2.1805 on the first). Values
  written by earlier versions (between 0 and 1) are not comparable; recompute.
  Computing them adds about 8 seconds on a map of 170 000 pixels.
- The surface-plane fractions {111}, {100} and {110} were computed with the
  inverse of the crystal orientation, so they were wrong for every texture
  except a cube one (a Shear 2 `{111}<112>` orientation gave 0 % for {111}
  instead of 100 %). They are now computed as the MATLAB script computes them,
  and the page and the workbook also give the share of pixels with **any plane
  of the family** in the surface, which does not depend on which symmetric
  equivalent the file stores. On AZtec data the script's number for {100} and
  {110} comes out near 0 %; the family value is the one to read.
- The BCC preset's "Rot. Goss" was the same orientation as "Goss"
  (`{110}<001>`). It is now the rotated Goss `{110}<1-10>`, Goss turned by 90° about
  ND.
- **The Excel workbook is never written into a file that exists.** If the
  chosen name is taken, the export goes to `<name>_2.xlsx` (then `_3`, ...) and
  says which file it wrote; an export that stops half-way keeps its unfinished
  file as `<name>.incomplete.xlsx`. The GOS and gKAM sheets now give percent of
  the total grain area (as MTEX's grain histogram), not pixel counts. An
  analysis step that could not run for a file (for example the RX analysis on a
  map without band contrast) is reported as a warning in the export and in the
  batch result instead of leaving empty cells without a word.
- **Run Complete Analysis** and **Download Excel** follow what is set on the
  tabs: the Grain tab's parameters and statistics settings, the Deformation
  parameters, the Texture tab's preset, tolerance and phase, and the RX
  criteria and phase. The export used to take the values the backend had last
  been sent; it now applies a choice that was made on the Deformation, Texture
  or Recrystallization tab and not run yet before it exports, and says so.
- The Texture tab's **Analyze Texture** no longer waits for a grain
  reconstruction: the texture is computed from the pixels of the chosen phase
  and needs no grains. A texture choice that was changed and not run yet is
  marked on the tab, and the result names the preset and tolerance it was made
  with.
- The results shown on the Deformation, Texture and Recrystallization tabs, and
  the check marks of the tab bar, stayed on screen after another map was loaded
  elsewhere; they now go with the map.
- The quality filter on the Data tab leaves the grains that were made before it
  as they are; the page now says so and draws the maps again.
- A scan just opened in the EBSD Viewer met "No indexed data loaded" on the
  Analysis page and had to be browsed for a second time. The Data tab now fills
  the path field with the viewer's file when the field is empty, and offers it
  as one button while no map is loaded there or the map loaded there is another
  file. It is offered, not loaded on its own, because after an indexing run the
  map to analyse may be the result, not the vendor's map.
- A failed grain reconstruction, or one the backend no longer knew after a
  restart, left the progress indicator running; it now stops with a message.
- Starting a grain reconstruction with an empty number field replaced the page
  with an error screen; it now shows a message.

#### Pattern centre
- On the PC Refinement page: a refused or cancelled action is shown in words
  (Remove, a typed PC or tilt and Add are refused while a refine runs); the
  forward preview never borrows another phase's master (a phase without a
  master of its own showed a preview against the first master of the list and a
  red NCC of 0.016, which reads as a bad PC); a phase simulated at several beam
  energies keeps its master; the pattern list and Remove stay reachable on a
  small window; every calibration pattern of a detector with a circular aperture
  is shown round (patterns added after background removal were square with noisy
  corners); after a refine the CI and the preview belong to the pattern and the
  PC shown; and after a processed copy of the data the page shows the new
  dataset's empty session instead of the old patterns and "Phase loaded". The
  count on "Open PC Refinement" follows the list after a removal, and indexing a
  single pattern on demand no longer fails on a second click.
- A refine or a grid calibration no longer writes into a file that was loaded
  under the same name while it ran, and a PC written from the page no longer
  collapses a per-pixel PC map unseen.
- The batch's **Set PC manually** dialog now sets the pattern centre: the typed
  PC never reached the backend, so the list said "PC set" while the batch kept
  refusing the file for a missing PC. The dialog takes the numbers in Bruker,
  EDAX/TSL or Oxford convention (preselected to the file's vendor), converts
  them with the same conversion as the PC page, and closes only when the backend
  has taken the PC.

#### Indexing
- Switching the Indexing method in a batch clears the picked phases; before, a
  phase picked for Spherical kept its master path under Hough and the batch
  indexed against the wrong candidate without any error.
- **Indexing batch:** the `_indexed.h5` file that the batch of the Indexing page
  writes for each scan now holds grid-shaped arrays and loads back into Orienta
  (it could not be read back before), for square and hexagonal scans alike.
- Switching the Indexing method no longer sends a Spherical master file to the
  Hough reflector lookup, where the backend tried to read the binary file as a
  CIF and logged errors. The reflector controls now show only for a CIF.
- **Apply to grain with Refine no longer runs out of GPU memory or holds the
  backend.** On a real file the refine held the backend for 45 s, ran out of
  GPU memory, left the backend at 32 to 42 GiB of memory and was skipped every
  time. Its Newton step ran at a bandwidth that needs about 62 GiB. It now runs
  at the largest bandwidth that fits 60 % of the free memory (about 72 on a
  free 12 GB card) and says which, stops after 60 s and counts the pixels not
  reached, hands its memory back at the end, and runs in a worker thread. If
  another request changed the data meanwhile, it answers with a conflict and
  writes nothing.

#### Desktop app and backend
- **"Backend: Disconnected" and "Connection to the backend lost" right after
  start-up.** About 35 s after connecting, the page showed "Backend:
  Disconnected" for about 25 s, then "Connection to the backend lost", and the
  settings answered only after up to 60 s, although the backend was running.
  Three start-up requests held it: the phase file discovery (33.8 s), the
  system status waiting for the WSL check (21.7 s) and the server folder check
  (5.1 s), measured on a real page load. They now run in threads, the system
  check once for the three requests the page makes. On the same page load the
  page then showed "Connected" throughout, with no stall longer than 2 s from
  these routes.
- **Installer data folder.** "change..." in the setup no longer adds an
  Orienta level per click (.../Orienta/Orienta/Orienta), and a reinstall is no
  longer refused because of empty folders of such a chain. A chain of Orienta
  folders is followed to the installation at its end, a recorded data folder
  that was deleted is settled at start-up, and a folder that cannot be used is
  refused before the pointer to the working one is rewritten, so choosing a
  folder no longer strands an existing installation. A refusal names the folder
  and what is in the way; where it holds the crystal library it tells you to
  check and back it up instead of advising to delete it, and the uninstaller
  recognises Orienta's leftovers by their shape.
- The exit of a backend that the app had already replaced no longer clears the
  state of the current one.

#### Simulation and library
- **A simulated master that kikuchipy cannot load is no longer offered for
  Dictionary indexing.** EMsoft's GPU master program writes such files. The
  phase library lists them by name, a simulation that produced one ends with a
  warning naming the file and the program, and "Simulate All Missing" reports a
  phase whose only master is such a file instead of saying that nothing is
  missing. To get a usable master, delete the file and simulate the phase
  again; the simulation now uses the CPU master program. A new run does not
  replace an existing file.
- **EMsoft simulation:** with no central database configured (blank paths),
  results are no longer copied into the working directory.
- **Simulation file names.** Every element is written by its symbol (Mg
  appeared as `Z12`, N as `Z7`), phases whose main metal is magnesium are filed
  under `Mg` in the central database, and the Pearson symbol is counted from
  the structure (Mg and alpha-Ti are `hP2`, Si is `cF8`). Files already in a
  library keep their names; re-simulating such a phase writes a file with the
  corrected name beside the old one, and the phase card shows the symbol of the
  newest file.
- **Crystal Database:** the CIF info panel lists the atom sites as the CIF
  writes them (asymmetric unit, coordinates in the file's own setting) with a
  displacement parameter only when the file states one, and it no longer blocks
  the application while parsing. DOI and reference are read by one reader
  everywhere, typed references come back as typed, and a DOI edit reaches the
  tag that is read. The Materials Project download writes them with the same
  writer. New `.xtal` files store the bare DOI in `useReference` instead of a
  link.
- **Debye-Waller table:** iron has `Fe (BCC)` 0.3328 Å² and `Fe (FCC)` 0.5710 Å²
  (Peng et al. 1996, 300 K). Converting an elemental iron CIF in Fm-3m uses the
  FCC value; iron in compounds and alloys keeps the BCC value. Existing `.xtal`
  files are unchanged.

### Known limitations
- Hexagonal scans, not yet available and each refused with a message that says
  so: NLPAR denoising, averaging neighbouring patterns in Crystal Hint (a
  radius of 0 works), the pixel-wise pattern-centre calibration, the joint
  orientation/pattern-centre refinement, the automatic grain-based phase
  assignment from EDS chemistry, and the Analysis page (grain reconstruction,
  grain size, deformation analysis and its maps). The Analysis page draws its
  maps on the index grid, which would put every second row half a step off;
  using the hexagonal lattice there is open work.
- On a hexagonal scan, one rule inside the pseudo-symmetry unification, the one
  that tells salt-and-pepper variants from coherent twin domains, still counts
  four neighbours of the index grid, and its thresholds have not been tested on
  a hexagonal scan; check the result of a unification there. Loading saved
  hexagonal files into OIM Analysis is untested.
- EDS matrix correction: checked for one detector position and a 70° specimen
  tilt only; another geometry needs its own check before the numbers are
  trusted. The deviation figures above are not an uncertainty for other
  specimens.

---

## v0.4.7 — 2026-10-10

This release corrects "BG Static", which imprinted the bands of one pattern on
every pattern of a scan; a detector-distance error with kikuchipy 0.12 and
newer; the atom positions Hough indexing read for silicon and MgCu₂; the
symmetry used for monoclinic phases; the step size, coordinates and lattice
constants of `.ang` exports and the reference frame of `.ctf` exports; and
several settings on the PC refinement page that had no effect. It adds control
over the Hough reflector list, pattern-centre refinement with several phases,
and adding a phase by its file path. The installers move to kikuchipy 0.13.1,
orix 0.15.0 and PyEBSDIndex 0.3.10.1.

*The open points of the Phase Library page that v0.4.6 planned for this release
(groups, keyboard use and some panels) are not part of it; they are planned for a
later release.*

### Fixed
- **BG Static subtracted the pattern at scan position (0, 0) from every pattern.**
  "BG Static" and the first step of the recommended pipeline used that single
  pattern as the background, which imprinted its bands on every pattern. They now
  subtract the average of all patterns. A static background stored in the file is
  not used: for Oxford files the loaded patterns are already processed by the
  acquisition software, and the stored background belongs to the unprocessed ones.
  Batch processing with static background removal (Hough and Dictionary) had the
  same default and is fixed as well.
- **Detector distance too large by the camera binning factor (8× for 156×128 px
  modes) with kikuchipy ≥ 0.12 on Oxford files.** The newer Oxford reader reports
  the binning, and Orienta applied it a second time to a pixel size that was
  already per binned pixel. This affected spherical indexing (the built-in GPU
  indexer and EMSphInx) and the pattern-simulation tools; Hough and dictionary
  indexing were not affected. Under kikuchipy 0.11.3 only Oxford files whose
  header contains `Camera Binning Mode` could be affected. When such an exported
  result is imported, its detector geometry is corrected for the
  pattern-simulation tools where this is unambiguous (otherwise the log reports
  it). The orientations stored in it are not changed;
  re-index.
- **Hough indexing read CIFs written in origin choice 2 of a two-origin space
  group with the wrong atom positions.** For the 24 space groups with two origin
  choices, the CIF reader that Hough indexing uses expanded the atoms with the
  operators of origin choice 1, whatever the file was written in. Silicon (Fd-3m,
  written in origin choice 2) came out with 16 atoms in the cell instead of 8, and
  MgCu₂ as Mg32Cu8 instead of Mg8Cu16. The atoms decide the structure factors that
  select Hough's reflectors: for silicon the list held the diamond-forbidden {222}
  and lacked {220} and {224}. Hough now reads these files with their origin
  choice, as the simulation path does since v0.4.6; a file whose origin choice
  cannot be decided is refused with a message (a non-standard setting named only
  by its symbol is read with a warning). Of the 36 CIF files we checked, only
  silicon and MgCu₂ were affected.
- **Monoclinic phases with unique axis b were compared with the wrong two-fold
  axis.** In Orienta's crystal frame (x ∥ a, z ∥ c*) the two-fold axis of such a
  phase (for example Al13Fe4 or β-AlFeSi) lies along y, but the symmetry used to
  compare and reduce orientations had it along z. Two orientations related by the
  real two-fold axis were treated as 180° apart, so pixels of one grain could be
  split into different grains, boundaries could be drawn inside a grain, KAM could
  leave out neighbours, and the two forms of one orientation got different IPF
  colours. Stored orientations from spherical and Hough indexing are not affected
  (checked for spherical indexing with patterns rendered from the Al13Fe4 master;
  for Hough this follows from the code and was not measured). Dictionary indexing
  sampled its orientation grid for the wrong axis, which did not cover every
  orientation of these phases, so dictionary results can hold
  wrong orientations. Grain reconstruction, KAM, grain boundaries, IPF colours,
  pole figures and the dictionary orientation grid now use the real axis, and the
  default Hough reflector list of these phases is expanded with it; Hough results
  of these phases may differ from earlier versions, and the size of the change
  was not measured. Phases with unique axis c and all other crystal systems are
  unchanged. The corrected calculations were checked on synthetic orientation
  maps. Nothing has yet been compared on a measured Al13Fe4 map.
- **PC refinement settings had no effect.** Minimum d-spacing, structure-factor
  threshold, maximum reflectors and number of bands on the PC refinement page never
  reached the indexer. The first three are replaced by the per-phase reflector-family
  table; the number of bands is now applied.
- **PC refinement: the Kikuchi-line overlay was drawn shrunk into the top-left corner**
  after a calibration pattern was picked from the list, because the pattern image was
  delivered at the size of a rendered figure instead of its own pixel size.
- **PC refinement: "Global CI" showed the CI of a single pattern** after a global
  refinement. It is now the mean over all calibration patterns at the current pattern
  centre, re-computed after a refinement, and shows N/A after a manual change of the
  pattern centre until the patterns are indexed again.
- **The PC refinement page modified a phase's structure on every call**, which
  changed the overlay reflectors from the second pattern on (observed for MgZn2 and
  π-Al8FeMg3Si6).
- **EDAX UP1/UP2 files on a hexagonal grid were read as if the grid were square.**
  They are now refused with an error when the version-3 header or the `.osc` file
  shows a hexagonal grid (resampling is implemented for EDAX H5 only). A version-1
  file without its `.osc` cannot be checked; keep the `.osc` next to it, or export
  hexagonal scans as H5. Detection was tested with synthetic headers only.
- **`.ang` export: wrong step size, coordinates, lattice constants and symmetry
  codes.** Results whose map was kept on the pixel grid (seen for spherical
  indexing and for Hough indexing with the EDS chemistry prior) were written with a
  step of 1.0 and x/y in pixels, so lengths and areas came out wrong by the step
  size and its square. Lattice constants were written as 1.000 for spherical
  results and in batch exports, and for monoclinic phases the symmetry field held
  112, which is not a TSL code. Step and coordinates are now in micrometres (a
  cropped map starts at 0, with its origin in the header), lattice constants come
  from the phase's structure, CIF or master file (the export stops with a message
  if none is known), and monoclinic phases with unique axis b get TSL code 20.
  Re-export affected `.ang` files.
- **Batch `.ctf` export: Euler angles in the wrong reference frame, placeholder
  lattice constants.** For scans from Oxford systems the Euler angles were written
  in the EDAX TSL frame although `.ctf` is an Oxford format, so φ1 differed by 90°
  from the source system's own angles and from the light `.h5`; lattice constants
  were written as 1.000. The angles are now written in the frame of the source
  file, the same as in the light `.h5`, with the lattice constants of the phase's
  structure, CIF or master file. Scans from EDAX systems keep the EDAX frame, which
  is their own. Re-export `.ctf` files.
- Exporting an `.ang` over an existing file failed or kept the old file; batch `.ang`
  files wrote Bruker pattern-centre values under the TSL names.
- **Hough indexing from a source installation with pyopencl failed after about 55
  calls in one session** with an OpenCL out-of-memory error, because PyEBSDIndex's
  OpenCL kernels were rebuilt on every call and their memory was not freed (about
  250 MB per call; measured on one GPU and driver under Windows, the number of
  calls depends on the machine). Orienta now reuses one OpenCL context per GPU for
  these calls. The installers use the CPU band detector and were not affected.
- The page now detects a connection to the backend that has silently died,
  reconnects with increasing wait times, and re-reads its state afterwards.
- The group headings "Pure elements" and "Other" in the phase picker were shown in
  German in every language.

### Added
- Hough reflector families per phase, editable on the Indexing and PC refinement pages.
- PC refinement with several phases; the phase of each calibration pattern is shown.
- Add a phase on the Indexing page by typing its file path (Hough, Spherical, Dictionary).
- Light/rich `.h5` format 1.4: for Oxford `.h5oina` sources the scan geometry from the
  file header (Scanning Rotation Angle,
  Specimen Orientation Euler, Tilt Angle; not applied to the Euler angles), space
  group, lattice parameters and crystal reference frame per phase, and whether the
  EDS chemistry prior was used. `.ang` and `.ctf` headers state the reference frame,
  the crop origin, the scan geometry (for Oxford sources) and the
  phase-assignment method.
- Updating an installed Orienta also updates its Python packages to the versions the release ships.
- Settings → About → Show log files.
- Tooltips for every Hough parameter.

### Changed
- PyEBSDIndex is cited in the README, CITATION.cff, NOTICE and the About page, and
  in the methods paragraph of every result computed with this version that used
  it: Hough indexing, the Hough anchor of spherical indexing, and a pattern centre
  from PC refinement. Reference: Rowenhorst, D. J., Callahan, P. G. & Ånes, H. W.,
  *J. Appl. Cryst.* **57**, 3–19 (2024), doi:10.1107/S1600576723010221.
- A source install now requires kikuchipy 0.13.1, orix 0.15.0 and PyEBSDIndex
  0.3.10.1 (each below its next minor version), and the installers ship exactly
  these instead of 0.11.3 / 0.14.1 / 0.3.9.1. An installation updated from
  v0.4.6 brings its Python packages to these versions the first time it starts
  (a few megabytes to download). If that cannot finish, for example without an
  internet connection, Orienta says so once, starts with the packages it has,
  and tries again at a later start; set `ORIENTA_SKIP_PACKAGE_SYNC=1` to keep
  the current packages. Updating a source checkout installs the new versions
  into the environment Orienta runs in. Both sets were compared on real data:
  spherical indexing agreed to within 3e-5° and dictionary indexing was
  identical. Hough indexing in the installed app uses PyEBSDIndex's CPU band
  detection, and its results change: on a nickel map the orientations moved by
  0.07° (median) and 99.8 % of the pixels by less than 1°; on a two-phase map of
  an aluminium alloy (Al and α-Al(Fe,Mn)Si, 10,800 pixels) 98 % of the pixels
  kept their phase, and of the α-Al(Fe,Mn)Si pixels that kept it, 92 % moved by
  less than 1° and most of the others by about 72°, a pseudo-symmetric variant
  of that phase. On the nickel map the new CPU band detection gives the same
  orientations as the OpenCL band detection of source installations with
  pyopencl (median difference below 0.0001°; 0.085° with PyEBSDIndex 0.3.9.1).
  On the two-phase map the two band detectors disagree for about a third of the
  α-Al(Fe,Mn)Si pixels, with the old and the new version alike.
- kikuchipy 0.12 reversed the meaning of the detector's azimuthal angle (the
  tilt of the detector about its vertical axis). Orienta keeps the earlier
  meaning, which is also EMsoft's, on every path that uses the angle: where it
  hands a detector to kikuchipy 0.12.1 or newer, it converts the sign, so CPU and
  GPU dictionary indexing and generation agree on both kikuchipy versions. Whether
  this matches the vendor's definition of the header value (the EDAX field `Camera
  Azimuthal Angle`) is not documented and has not been measured; every file we
  tested reads 0. Hough and spherical indexing do not use the angle, and with
  kikuchipy 0.11.3 neither does the Kikuchi-band overlay in PC refinement. When a
  dataset has a non-zero azimuthal angle, the run log says so.
- `start_app.py` refuses to start when port 8000 is already in use.
- The README listed EDAX hexagonal-grid scans as unsupported, although v0.4.6
  already reads them: an EDAX H5 file whose header says `Grid Type = HexGrid` is
  resampled to a square grid, each square pixel taking the pattern of the nearest
  measured point (patterns are not interpolated). The README now describes this
  and its limits.

---

## v0.4.6 — 2026-09-27

You can install Orienta now — on Windows, on a Mac and on Linux. Add-ons have
a place in the interface. Every result can say what to cite, EDS data can
leave the program per particle, spherical indexing got more accurate on
several phase types, and anyone who installed from source since March has
been computing on the processor without being told.

*Numbered 0.4.6 rather than 0.5.0 because the testers already run 0.4.5 and this
repository was on 0.3.0; by the rule above this would be a 0.x.0.*

**If you cloned, downloaded or browsed this repository between v0.2.6 and
v0.3.0, read "The Materials Project API key was readable in this repository"
below — that key must be treated as compromised.**

**Read the next section before you trust an old result on silicon or MgCu₂.**

### Two crystal structures in the library were wrong, and so was everything
built from them

A structure read from a CIF was expanded with the wrong set of symmetry
operations. For the 24 space groups that have two possible origins, the
library Orienta uses publishes its operators for the first origin while
structure databases publish coordinates for the second — so those structures
came out with **atoms on the wrong sites**. Silicon arrived as 16 atoms in the
unit cell instead of 8, and its density as 4.66 g/cm³ against a literature
value of 2.329.

That is not cosmetic. The atom basis sets the electron density for the Monte
Carlo step and the crystal potential for the master pattern, so a master built
from such a structure is wrong throughout, not slightly off.

**Reading for simulation is fixed.** A CIF is now expanded on the correct
orbit, `.xtal` files carry and honour their origin setting, the 3-D structure
view and the master writer use it, and a file whose setting cannot be verified
says so when it is read. Hough indexing reads CIF files through a separate
reader that this fix does not cover: in this version it still expands silicon
to 16 atoms and MgCu₂ to Mg32Cu8, and the effect on Hough results has not been
established.

**Two files in the library were not repaired by that fix**, because their
error was written into them at conversion time rather than made when reading
them: `Si.xtal` and `sd_1816951.xtal` (MgCu₂). Everything computed from them
carries it — their master patterns, their Monte-Carlo caches, and the Si
dictionary. **If you indexed against either of those phases, treat the result
as unverified.** `Al3Fe2Si_mp-1190708_symmetrized.xtal` is in the same space
group but was written correctly: it has the atom count and density of its CIF.
An earlier version of this entry listed it as affected. The other 32 structures
in the library are in single-origin space groups and are unaffected.

**Both have now been rebuilt.** The converter was fixed first, because
rebuilding with it would have written a different wrong thing: it used to take
the origin from an unrelated quantity, and the choice now comes from what a CIF
says about its own atoms — its operators, the multiplicity of its sites, its
formula — with a refusal instead of a guess when those contradict each other or
say nothing. Then both were regenerated from their CIFs, Monte-Carlo step and
master pattern included: silicon 16 → **8 atoms**, density 4.66 →
**2.329 g/cm³**; MgCu₂ Mg32Cu8 → **Mg8Cu16**, 6.145 → **5.787 g/cm³**.
β-AlFeSi had a different fault: the library used a CIF that models every mixed
Al/Si site as Al, so the cell contained no silicon. The phase now comes from the
silicon-bearing CIF it should have used. An audit across the
whole library afterwards found **34 files and no remaining deviation** (one fewer
than before: the silicon-free β-AlFeSi entry was retired). The
superseded files were moved aside rather than deleted.

**No structure file shipped in this repository is affected.** The four structures
under `sample_data/phases/` were checked against their CIFs with Orienta's own
audit: all four come back with the atom count and density they should have, and
none of them is in one of the 24 space groups that have two origins. The wrong
files — `Si` and MgCu₂ with the origin fault, β-AlFeSi built from a silicon-free
cell — were in the maintainers' own crystal library, which this repository has
never contained; if you built masters from your own copies of any of them, those
are the ones to rebuild.

**So if you indexed silicon or MgCu₂ with Dictionary or Spherical indexing in an
earlier version, index those maps again.** On silicon the new master is a
different answer, not a nudge: at the silicon pixels of a real scan it renders
at 0.714 where the old one managed 0.574 and aluminium 0.646 — so the old
master lost to aluminium on pixels that are silicon. Which phase won on that
particular scan moved only slightly (113 → 116 pixels), because the chemistry
prior was already carrying the decision there; the pattern now agrees with it
instead of contradicting it.

This also corrects something we wrote on 14 September: master patterns **are**
affected. The note then said they were not.

### Add-ons

- **Another group's analysis can now run inside Orienta.** An add-on is a
  folder with a `orienta-addon.toml` next to its Python: it declares what it
  computes, which settings it takes and what it should be cited as. Orienta
  builds the settings form from that declaration, runs the analysis against an
  indexing result you choose, and shows what came back — numbers with their
  units, tables you can copy or save as CSV, and maps you can put straight onto
  the Phase Maps page as a layer, with their own value scale.
- **What it computed is written into the result, and so is its citation.** The
  methods paragraph under **Phase Maps → Citations** gains the add-on's own
  sentence with your settings in it, and its work joins the bibliography —
  the same panel and the same exported `.h5` as Orienta's own steps. Run it
  again with different settings and everything follows: the numbers, the layer
  and the sentence all describe the run you are looking at, not the one before.
- **Add-ons live in `~/.orienta/addons`** (on Windows,
  `C:\Users\<you>\.orienta\addons`). Orienta creates that folder at startup and
  names it on the Add-ons page when it is empty, so there is nothing to look
  up. Drop a folder in, press **Rescan**, and it appears — with the path it was
  found at.
- **Nothing runs until you say so.** The first time you enable an add-on,
  Orienta shows you who wrote it, which version, which DOI and where it sits on
  disk, and states plainly that it will run in this process with your
  permissions. It asks again if the add-on under that name has changed since
  you allowed it. An add-on that fails three times is switched off, and the
  list says whether that was you or Orienta.
- **A worked example ships with the program**, in `examples/orienta-addon-bc-gmm`:
  Orienta's own band-contrast mixture model, rebuilt as an add-on. The
  documented way to write one is to copy it and change it; its README lists the
  rules a manifest has to follow, and `CHANGELOG-ADDON-API.md` records what
  this first version of the interface cannot do yet.

### Installing Orienta

- **There is a Windows installer.** One `.exe`, no Python, no git, no command
  line. On first start it sets up a private Python for Orienta, all packages,
  and — if the machine has a suitable NVIDIA card — the GPU versions of them.
  (EMsoft and WSL stay a separate step inside the program, as before.)
- **The first screen asks for your language** (English, German, Japanese,
  Chinese), and the application then starts in it.
- **You choose where Orienta puts its own files** — the private Python, the
  program files and the crystal library, together 2.5 to 8 GB. That can be a
  different drive from the one the shortcut goes on. Your measurement files stay
  wherever you keep them; Orienta only reads them.
- **A first install works even when GitHub cannot be reached:** the installer
  carries a copy of the program inside itself, and it also accepts a package
  file placed next to it. It still needs the internet for Python itself and for
  the packages, so a completely offline machine is not covered yet.
- **Updates come from GitHub.** Orienta replaces its own program files and
  **never writes to or deletes your crystal library** — that is enforced in
  the update code, against nine ways of spelling the path.
- **A desktop shortcut is offered, not forced** — a checkbox on the last page.
- **Uninstalling asks about your data, in two steps.** First: also remove the
  downloaded Python and program files (2.5–8 GB)? That one is preselected
  **Yes**. Only if you say yes does it ask the second: also delete your crystal
  library? That one is preselected **No**. Files you put into the folder
  yourself are left alone.
- **Windows will warn you** that the publisher is unverified, because the
  installer is not signed. The release page shows the two clicks that get past
  it ("More info" → "Run anyway") and publishes the SHA-256 checksum of the
  file, so you can check you got what we built.

### Citing your results

- **Every result can now say what to cite.** A "Citations for this result"
  panel on the Phase Maps page gives you a BibTeX file, a ready methods
  paragraph and a plain reference list — for the method you actually ran, not a
  generic list.
- **The citations travel with the file.** They are written into the exported
  result and read back when it is imported, together with the steps that
  changed the numbers. Orienta's own entry comes from its citation file, so it
  stays correct across versions.

### EDS

- **Your EDS analysis can leave the program.** Per particle: equivalent
  circular diameter, largest and smallest Feret diameter with its angle,
  perimeter, circularity, solidity, whether the particle touches the edge, and
  the composition twice — over all its pixels and over an eroded core, which
  tells you how much the surrounding matrix diluted the reading.
- **Settings you use again and again can be saved as presets**, and every
  export carries the settings it was made with, checked by the server rather
  than taken on trust.
- **A plausibility report** compares the measured composition of each region
  with the nominal composition of the phase it was matched to, and says plainly
  when they disagree by more than a factor of two. It changes no assignment.

### The phase library

There is a new page that shows what is actually in your crystal library: one card
per phase with its formula, space group, structure type, the elements it contains,
and which of the three indexing methods can use it — a CIF alone is enough for
Hough, spherical needs a simulated master. You can search it, filter by element,
give a phase your own name or a synonym without renaming files on disk, and drag
phases into groups. Where a phase carries a source, the card offers it for
citation.

Groups can have one level of subgroups. Choose **Use this group for the next run**,
and the Phase Tester, Indexing and the EDS phase map start from that group's phases:
Indexing lists only those, the Phase Tester and the EDS phase map pre-select them, and
**Show all phases** brings back the rest. The toolbar shows which group is active.
Groups are plain JSON files in `Database/Collections/`, so they travel with your
library.

*This page is not finished, and 0.4.6 ships it anyway because what is there is
usable on its own. The open points — mostly around groups, keyboard use and a few
panels — are planned for 0.4.7. Nothing on this page deletes or moves a crystal file
(deleting a group removes only that group's own file), so the worst an unfinished
corner can cost you is a second attempt.*

### Indexing

- **Spherical indexing is now used directly for phases where Orienta used to
  fall back to Hough.** Cubic m-3/23 phases (such as α-Al(Fe,Mn)Si) and
  orthorhombic mmm phases (such as the S-phase Al₂CuMg) were re-indexed with
  Hough first, a detour that came from a decoding error fixed in v0.3.0.
  Measured against known orientations: 0.17° median for the m-3 α-phase and
  0.26° for the S-phase, all within 2°. A SampleB run also loses about
  24 seconds of re-indexing.
  - **-43m phases keep the old detour** (for example Mg17Al12): there the
    sphere alone is right only 58 % of the time, the rest is 90° off. The
    "Orientation from Hough" note on those results is intended.
  - **Orientations change slightly against v0.3.0** on the affected phases:
    same phases everywhere, median change 0.0°, but 42 of 6712 α pixels and 33
    of 595 S-phase pixels move by more than 3°.
- **Master patterns that Orienta's own GPU simulation wrote carried a wrong
  symmetry number for 15 space groups** — among them Pm-3m (221), P-43m (215),
  P432 (207), P23 (195), P4/mmm (123) and P6/mmm (191), so B2 and many
  intermetallics. The value is also used to decide which harmonics are stored,
  so such a file is not merely mislabelled. Files in the shipped library and
  anything simulated through EMsoft are unaffected. If you built such a master
  in Orienta, build it again and re-index the maps that used it.
- **"Index Pattern" on the Pattern Centre page no longer breaks indexing for
  the rest of the session.** A safety cap on the Hough index could get stuck,
  and every later attempt then answered with an error until the program was
  restarted. It showed up most often right after a fresh installation.
- **The Pattern Centre preview shows the phase you loaded.** With several
  simulated phases in the library it used to take the first file it found, so
  calibrating on nickel could show an aluminium simulation next to your pattern
  and report a good pattern centre as a bad one.
- **Importing crystal files takes several at once** in one dialog, on the
  Crystal Database page.

### If you installed from source

- **`requirements.txt` handed every new user a processor-only PyTorch.** Since
  March, a fresh install from source got a version without CUDA support, and
  nothing said so, because the other GPU library kept working. Measured on this
  machine: 0.71 instead of 11.2 trillion operations per second, a factor of 16.
  The requirements now pin the GPU version, and the installer does this for you.
- **Package versions are capped** (kikuchipy below 0.12, orix below 0.15,
  pyebsdindex below 0.3.10). The newer versions were measured against real
  data: orientations, dictionary indexing and phase assignment come out the
  same, but on Oxford files the newer kikuchipy reports the camera binning
  correctly for the first time, and Orienta's own detector geometry then counts
  it twice and puts the detector eight times too far away. That is our error,
  not theirs, and it will be corrected as a unit fix rather than days before a
  release. Newer pyebsdindex also measured 1.4× slower on Hough indexing.

### Larger files

- **Large Oxford files can load slowly on environments with a newer
  kikuchipy.** Orienta's workaround for a camera-setting bug no longer applied
  there, and loading silently fell back to the slow path — for a 27 GB file
  that meant ten minutes and 16 GB of memory instead of seconds. Fresh
  installations of this version get the older kikuchipy and were never affected; if
  you keep your own environment, this fix restores fast loading.

### macOS and Linux

- **There is a `.dmg` for Apple Silicon and an `.AppImage` for 64-bit Linux.**
  Both bring their own Python and set themselves up on first start, exactly as
  the Windows installer does. Your measurement files stay where they are.
- **The first start is different on each, and neither is obvious.** On a Mac a
  double-click only says the app cannot be opened. On macOS 15 and later, click
  **Done**, then open System Settings → Privacy & Security and click **Open
  Anyway**; on macOS 14, right-click → **Open**, then **Open** again. Orienta is
  signed, but not with a paid Apple certificate — there is no Apple developer
  account behind this project. On Linux the AppImage needs `chmod +x` first;
  without it a double-click does nothing at all, with no message.
- **What is not on offer there:** EMsoft simulations, which Orienta reaches
  through WSL, a Windows feature; and graphics-card acceleration, which needs
  an NVIDIA card. Orienta Engine, the built-in simulator, runs on all three
  systems and needs neither. The settings page says so instead of showing red
  errors for things those machines cannot have.
- **Both have now been started by a person.** On Arch Linux with an NVIDIA
  card: the window came up, a phase test over 32 phases ran, and a simulation
  completed. On a MacBook Pro (M5 Pro, macOS 26): setup in under a minute, a
  file loaded in about 4 seconds, Hough over 441 patterns in about 2, and a
  master pattern for aluminium computed in 228 seconds on the processor.

### Four faults that only showed up on another machine

Building for a second and third system found these. Each of them was in the
Windows program too, and none of them was visible there.

- **Orienta could eat its own memory until the system killed it.** A log
  message that could not be delivered produced a warning, which was itself
  logged, which produced another warning. On a machine with plenty of memory
  this looked like "the application is getting slow" and ended with the
  process disappearing. Measured on the Linux runner: 931 MB to 7.8 GB, then
  gone, with 280 752 of 842 567 log lines being that one warning.
- **A phase name taken from a file path could contain the whole path**,
  including your user name, on a Mac or under Linux — and from there it could
  reach an exported methods paragraph. On Windows the same code returned just
  the name.
- **Closing the application could leave the backend running on a Mac.** The
  last-resort cleanup asked the system who owned the port, which macOS answers
  only for an administrator, so it quietly gave up; the next start then
  attached itself to the old process.
- **The setup could advise a Mac user to update an NVIDIA driver**, because
  the function deciding what to recommend asked which system it was running
  on instead of which system it had been asked about.

Underneath, from the preparation:

- **The likely cause of the crash a Mac user reported is addressed.** On a Mac
  several packages each brought their own parallel-computing library, and the
  second one to start aborted Orienta about 30 seconds after launch ("OMP:
  Error #15"). There is now one environment file that builds the whole stack
  from a single source with one such library, plus a macOS section in
  `INSTALL.md`. **We could not try it on a Mac** — we have none — so please
  report back.
- **Hough indexing on a Mac uses the same processor-based band detector as a
  Windows installation**, instead of Apple's discontinued OpenCL.
- **Updating inside the app uses conda on a Mac**, so the update cannot
  reintroduce the second library, and says what to do when conda is missing.

### Smaller things you may notice

- The program's window frame was updated to a current version; two save buttons
  on the Analysis page had lost their suggested folder and have it back.
- An installed copy knows its own version and can update itself; before it read
  "unknown" and refused.
- A half-finished installation now opens the setup assistant instead of a blank
  browser error page, and says "Orienta" during the 30 to 40 seconds a cold
  start takes.
- When something goes wrong at startup, the page shows where to get a working
  version, and links open in your browser instead of stranding the app.

### What the tester rounds changed

- **"Check for updates" no longer tells an installed copy that it cannot
  update itself.** It asks, names the new version, links it, and says that
  your data folder, your settings and the installed environment stay where
  they are. It takes effect from this version on: a copy of 0.4.5 or older
  still shows the old sentence.
- **The built-in simulator has one name.** It is called Orienta Engine
  everywhere a person reads it, instead of a different phrase on each page.
- **The interface speaks one language at a time.** The status bar, the layer
  names, the EDS pre-check, the warnings and the loading screen follow the
  language you chose, and the German text uses *Sie* throughout.
- **Every "export PNG" button goes through the export dialog**, so a picture
  saved from any page can carry a scale bar and be enlarged past the
  measurement grid. Before, some of them wrote the raw measurement resolution
  — a 21 × 21 pixel file, with no scale — which is unusable in a report.
- **The window is no longer closed out from under a long job.** The watchdog
  that shuts the backend down when the interface disappears now trusts a
  declared window owner, instead of guessing from a connection count that a
  half-closed browser tab kept above zero.
- **The database browser stops flickering** while it polls, and a machine
  without a graphics card sees processor labels rather than GPU ones.

### The Materials Project API key was readable in this repository

The v0.2.6 entry below announced that the key had stopped being a literal in
`cif_database_builder.py`. That was true when it was written. The v0.3.0 release
port then copied the file from the development tree, which still carried the
literal, and put it back without anyone noticing.

**Anyone who cloned, downloaded or browsed this repository between those two
releases could read that key. Treat it as compromised; it has since been
revoked, so it no longer opens anything.** It is out of the source
again: the key comes from **Settings → API keys** or from the `MP_API_KEY`
environment variable, and when none is set the database builder simply works
offline from your local CIF files.

So that this cannot happen a third time, a test now fails if any
credential-shaped literal appears in any tracked file, and a second test pins
that the builder passes only the resolved key to the API.

### Two further corrections the same port had undone

- A comment in `backend/spherical_gpu/pipeline/detector.py` called kikuchipy
  MIT-licensed again. kikuchipy is GPL-3.0-or-later, which is what `NOTICE.md`
  has said all along.
- `sample_data/README.md` describes the Zenodo archive but had lost its DOI, so
  the page that tells you the example datasets exist did not say where. The DOI
  is back.

### Third-party attribution in the MATLAB reference scripts

`matlab_testskripts/` is reference material, not part of the running
application, and it was shipped without attribution: `freezeColors.m` is
another author's work, and the scripts call MTEX throughout. Both are now
credited, and two comments that named people who are not authors of this
project have been removed.

### Licences and attribution

- **NOTICE.md names micromamba**, which the macOS package downloads and runs,
  and its licence text ships in `licenses/`.
- The public repository was audited against the private tree; what it should
  never have carried is out of it, and a guard keeps those files from being
  ported again.

---

## v0.3.0 — 2026-09-14

**If you have indexed anything with an earlier version, read the first section.**
A decoding error that had been in every spherical run since the module was written
is fixed, and orientations change because of it.

### Every spherical orientation was about 1.6° wrong, on every phase

The step that turns the correlation volume into Bunge Euler angles carried two
defects, neither of them in the correlation itself.

- A **constant crystal-frame operator**, a half turn about ⟨1 -1 0⟩. That operator
  is a symmetry of every phase the indexer had ever been checked against, so on
  cubic m-3m and tetragonal 4/mmm masters it was invisible. On m-3 it is not, and
  that is where it surfaced: α-Al(Fe,Mn)Si came out 88.7° from the reference.
- A **half-bin origin error**. The α and γ axes carry a half-turn offset that was
  written as a whole number of bins, but the grid has an odd number of them. Every
  α was 1.03° too large and every γ 1.03° too small, at every bandwidth.

Measured against EMSphInx on the same patterns and the same master: the raw
spherical answer went from 88.67° to **0.145°** median disorientation, and from
0 % to **99.89 %** of pixels within 5°. Aluminium and Al₇Cu₂Fe were **not**
unaffected: 1.101° and 1.992° → **0.119°** and **0.117°**. The forward render of
the raw answer scores 0.810 against EMSphInx's 0.806, so for the first time the
unrefined result matches the reference it is measured against.

**What this means for you.** Saved `.h5` results, exported `.ang`/`.ctf` files and
any figure made from them are wrong by roughly this amount on every phase. Re-index
rather than re-interpret. Confidence indices are unaffected, which is exactly why
this was hard to see: the number that is supposed to tell you something is wrong
did not move.

### Silicon read correctly, iron read three times too low

The EDS quantification had an element-wise error. The k-factor table carried no
overvoltage term, and for iron on a 20 kV beam that is not a detail: iron read
**4.4 at% where Aztec reads 14.0** on the same data. Three further defects in the
same area:

- A window that names its line could skip the overvoltage gate.
- Two windows of the same element collapsed into one, silently.
- One unusable window cost the whole scan its composition instead of being dropped.

Everything downstream moves with this: the chemistry prior, the phase suggestions,
the EDS phase map.

### "Not measured" is no longer read as "not there"

The chemistry prior treated an element nobody could quantify as an element that is
absent, which vetoed exactly the phases that element identifies. It now goes off
when a window cannot be priced, and says so rather than failing quietly. The prior
also reads the composition of the **structure that was simulated**, not the one in
the file name; on α-(Al,Mn)Si the two differ by up to 4.8 at%, and the phase went
from 10 % to 89 % correctly assigned.

### New: particles the pattern cannot see

Silicon and aluminium are both face-centred cubic. Their Kikuchi patterns are
degenerate, and measured on real data the correlation prefers aluminium on **every**
pixel, even at 93 at% silicon. The chemistry decides instead, but the EDS
interaction volume is 2 to 3 µm, so a particle smaller than that reads as a dilute
mixture and stays below the decision threshold.

A post-pass now settles those pixels by **orientation continuity**: an enriched blob
whose orientation does not continue the surrounding matrix is its own crystal, and a
rim pixel that does continue the matrix goes back to it. It runs only when the EDS
prior is on; with the prior off the run is bit-identical to before.

Measured on a 301 × 402 map: 2408 pixels moved to silicon, 159 back to aluminium,
and **63 blobs that had no silicon pixel at all** became particles. 97 % of the
moved pixels are orientation-coherent with a silicon neighbour.

### The installation no longer trusts whoever is calling

The backend treated every caller as the user. A POST carrying only query parameters
is a CORS "simple request" and needs no preflight, WebSockets have no CORS at all,
and DNS rebinding makes a foreign page same-origin. Measured before the fix: a page
on another origin could reach the endpoint that unregisters a WSL distribution, and
the log socket accepted foreign origins and streamed the root logger.

There is now an origin and host guard on every writing route and every socket
upgrade. Local callers are unaffected: they send no `Origin` header.

Three defects in the install wizard are fixed with it. The worst wrote **the sudo
password itself** into `/etc/fstab` and the OpenCL vendor files, because a shell
helper piped the password into stdin and thereby replaced the stdin that
`echo <path> | sudo tee <file>` was relying on. That is the origin of the February report that the ICD files
held a stray number instead of a library path: what they held was the password.

### What we checked and did not change

A Hough run was 4.2° from the spherical result on every phase, which looked like a
convention difference between the methods. It was **ours**: the stored run had been
made with the camera elevation left at zero. Re-run with the scan's own 4.31° it
lands at 0.25° / 0.04° / 0.63°, and the phase map is 98.5 % identical. No fix was
needed; a guard test now pins that the elevation reaches PyEBSDIndex. Worth carrying
home: without it every Hough orientation is rigidly wrong by exactly that angle
while the confidence index does not move at all, so only a forward render sees it.

### Notes

- The crystal database is not part of this repository and is not shipped with it.
  Library CIFs that had been stored without their symmetry operators were repaired
  with a maintenance script that is not part of this repository either; on one
  phase that moved the share of pixels within 5° of the spherical answer from
  87 % to 97 %.
- Tests that need binary reference data which the repository does not carry are not
  part of the shipped suite.

---

## v0.2.6 — 2026-09-08

Nothing in the app behaves differently. This release is about what the project
says about itself — because from this version on, Orienta is public, and an
archived copy of it carries a DOI that people will cite.

### The README now matches the code

An audit compared every claim in the README against what is actually wired up,
and four of them were wrong.

- **"texture/ODF" was overstated.** The texture index and entropy come from
  texture-component volume-fraction bins, not from an orientation distribution
  function. Useful for comparing your own datasets against each other; not
  comparable with MTEX numbers. Said plainly now.
- **"Spherical (EMSphInx via WSL)" was out of date.** The default is the
  built-in GPU indexer; EMSphInx is the alternative, not the main path.
- **"EDAX H5" needed a limit.** Hexagonal-grid scans are not detected on the
  load path, so they open sheared instead of being refused. Until that is
  fixed, do not load HexGrid data — this is the one place where you can get a
  plausible-looking result that is wrong.
- **The status line was too pessimistic**, claiming parts of ML and refinement
  were hidden. Both are ordinary sidebar pages and have been for a long time.

There is now a **Known limitations** section carrying those, plus the ones
already known: the simulation engine cannot start because its automation module
is not part of this repository, the "Texture Components" map layer returns an
empty map, two recrystallisation statistics are placeholders, the frontend build
needs Node 20.19+, `requirements.txt` installs the CUDA runtime even without an
NVIDIA card, Python 3.12 is blocked by one leftover dependency, and a ZIP
download reports its version as "unknown".

### Attribution that was missing

- **EMsoft and SHTfile are now credited, with their licence texts.** Three files
  say of themselves that they carry transcribed EMsoft content — the WEKO
  elastic coefficients for Z = 1..98, the LFSR113 random-number stream, and two
  230-entry tables from the SHTfile reference implementation. Both upstreams are
  BSD-3-Clause and require their notice to travel with the source; neither
  licence text was in the repository. Both are now under `licenses/`.
- **The description of what ships was wrong in NOTICE.md.** It said runtime
  dependencies are "not redistributed", while three.js sits in
  `frontend/public/vendor/` and is loaded directly. It is now listed as
  redistributed, and vanta.js — whose minified bundle carries no copyright text
  at all — has the notice MIT requires beside it.
- Three more locations with derived code were undeclared, and one comment called
  kikuchipy MIT-licensed when it is GPL-3.0. (That comment came back with the
  v0.3.0 port and is corrected again; see *Unreleased*.)

### The Materials Project key no longer lives in the source

> **Correction.** The v0.3.0 release port put the literal back. It is out
> again; see *Unreleased* at the top of this file.

It was a literal in `cif_database_builder.py`. It now comes from your own
settings, the same place the rest of the app already read it from, with the
`MP_API_KEY` environment variable as a fallback. Unconfigured means the online
lookups are skipped and the database builder works offline from your local CIF
files. **If you used that builder's online lookup, enter your own key under
Settings → API keys.**

### Citation metadata

`CITATION.cff` said version 0.1.0 and pointed at the DOI of that first archive.
It now tracks the current release and cites the **concept DOI**, which always
resolves to the newest archived version, and a `.zenodo.json` gives the archive
its full description and the funding acknowledgement.

---

## v0.2.5 — 2026-09-08

Figures in series, and a phase check that judges fairly. **If you have run
"Check phases" before, read the second section: its verdicts can differ now,
and the old ones were sometimes wrong.**

### Every element map as its own file

- **The EDS page can write one file per map.** Right-click a tile and choose
  *Export each map as its own file…*: the ordinary export dialog opens on the
  first map, you set resolution, border, format and scale bar once and see
  them on a real map, then one click writes the whole series. The folder is
  chosen once, not once per file.
- **Each file carries its own map's name** in the corner and in the file name,
  so a panel of element maps is not a set of pictures that all say the same
  thing. A switch in the dialog turns that off if you would rather caption the
  sample than the map.
- **Each file keeps its own micrometres per pixel.** The tiles do not share a
  raster — the electron images sit on the SEM raster and the element maps on
  the scan raster, on a real file 10.6x apart — so one shared number would
  mis-scale half the series.
- **The IPF colour key can be exported on its own**, on a white plate or
  transparent to lay onto a figure of your own. Right-click the key panel; it
  was the only picture on the phase map page without an export.

### Colour keys beside a map are no longer cut off

- **The border could never be wider than half the map.** A three-phase IPF key
  is 2.75x as tall as it is wide, and beside a wide, short map it needs about
  1.4 map heights above and below — so it was sliced across the top and the
  bottom, and the file came out at exactly twice the map height whatever you
  asked for. The border now takes the room the figure needs; the only limits
  left are what the browser can encode, and the dialog says when it had to
  trim.
- **The size the dialog announces is the size it writes.** It ignored a key
  that had grown since the border was last fitted.

### Checking phases got fairer — and it can change what you see

- **A phase is now judged at a fair orientation.** It used to be scored at its
  stored orientation while its rivals were scored at their best, which is not
  a comparison. Measured on a real map: **8 of 24 island findings were
  artefacts**, and the per-grain stage flipped a whole MgCuAl2 particle to Al
  (0.207 stored against 0.420 at a fair orientation).
- **The score no longer chases noise.** It compared a noisy measured pattern
  against a noise-free simulation across the full frequency range; limiting
  both sides to the band that carries the signal takes the median on a real
  7050 map from **0.196 to 0.417**. That is what had made every grain a
  suspect.
- **A second stage finds wrong pixels enclosed inside a grain** — islands
  below grain size (a grain being 9 pixels or more), taking the orientation
  from the grain around them. Outcomes are named: reassigned, rescued, or
  undecided — and undecided is never applied.

### Hough: the reflector budget is yours to set

- **A CIF written as P 1 asked for 44.7 GiB** for its band-triplet library and
  took the machine to its commit limit. Orienta now works out the cost before
  allocating anything, refuses what does not fit, and lets you set the number
  of reflector families per phase — on the Indexing page and in the phase
  verification panel. Your choice is remembered between sessions.
- **Trimming reflectors automatically was measured and rejected**: on Ni, 24
  families put 192 of 800 patterns 120 degrees off and 16 families gave a
  median fit of 180 degrees, silently. A refusal you can see beats a wrong
  answer you cannot.

### Fixes

- **The cleanup sliders and "Apply cleanup" now refresh the IPF and BC
  layers.** They had been showing the state before the cleanup.
- **A diagnostic layer that paints only its findings no longer decides how
  large the map is drawn.** A 39x136 scan collapsed onto the ~20 flagged
  pixels, drawn with a 100 nm scale bar instead of 2 um.
- **Pattern match refuses a render grid the graphics card cannot hold**
  instead of failing inside the driver.
- **The changelog listed v0.2.3 twice**, above v0.2.4.

---

## v0.2.4 — 2026-08-28

Scale bars, and a set of crashes that took a whole page down. If you exported
a figure from Orienta before today, please check its scale bar against this
release before the figure goes anywhere.

### Crashes that a single click could trigger

- **Switching to the Wand paint mode killed the entire EDS page.** One click
  was enough; no data needed to be loaded.
- **Opening the Phase Map from the Batch page killed the Phase Maps page**,
  every time — and retrying crashed again.
- **When a map export failed, the message about it crashed the page.** The
  notice meant to tell you what went wrong was the thing that went wrong.

### When the backend dies

- **The app now says so within seconds instead of appearing frozen.** Its
  health check shared the five-minute timeout used for long operations, so a
  backend struggling with memory — which keeps its connection open but answers
  nothing — left the app reporting "connected" while nothing worked. Measured
  against a backend that answers nothing: reported after 17 seconds, where
  before the first sign could not arrive for five minutes.
- **A batch run no longer spins forever** when the backend is gone; after three
  missed replies it says the connection was lost.
- **Running out of memory says so.** It used to end in "Indexing failed:" with
  no reason at all, because that particular error carries no message of its own.

### Saved results

- **A crash can no longer leave half a result file behind.** Results are now
  built alongside their final name and moved into place only once finished —
  before, a run killed midway left a file that opened cleanly, looked
  complete, and held only part of the data. A failed re-export also leaves the
  previous result intact instead of destroying it.

- **The scale bar on an exported image was wrong** wherever the picture was not
  the raster its measurement came from. On the test scans the error was between
  1.3x and 8.5x, and which one depended on the file. It affected the EDS
  overlay export, single-layer exports from the Phase Maps page, and the
  "all maps on one sheet" export. A bar labelled "2 um" could be 17 um long.
  Every export now takes its scale from the picture actually being written, so
  a map and an electron image exported from the same file carry their own
  correct bars.
- **A montage no longer offers a scale bar at all.** Its panels are scaled to a
  fixed cell width and separated by gaps, so no single bar can be true for the
  sheet.
- **The scale bar in the pattern-match figure works.** It could never show
  micrometres, and the pixel bar it fell back to drew one length while its
  label claimed another; dragging the bar's handles changed its length and left
  the label alone. Its length now follows the map it measures, and where a scan
  carries no step size it measures in map pixels instead of inventing a
  micrometre value.
- **The scale bar on screen follows the map after you resize the window.** It
  used to keep measuring against the width the map had before.
- **A map with no step size says so** instead of drawing a bar from a
  placeholder value of 1 um per pixel.
- **A step size of exactly 1 um is shown in the status bar again.** It was
  being hidden, on the assumption that it meant "unknown".
- **Right-clicking a single layer in the Phase Maps tile view offers that
  layer's export again.** It was offering the composed map instead.
- **The IPF-X and IPF-Y buttons on the EDS page work.** They were drawn,
  enabled and did nothing when pressed - which reads as a problem with your
  data rather than as an unconnected button.
- **"Download All" in the Database Browser now downloads what it says.** It
  opened a window headed "Sync Upload" that listed your *local* files and
  greyed out anything you did not already have - and then downloaded.
  Categories that exist only on the server could not be selected at all. The
  count shown after any completed sync, in either direction, was also missing.

---

## v0.2.3 — 2026-08-27

Fixes for four things users reported, three of which read as "it just does
nothing".

- **The Phase Maps page crashed every time it was opened.** It has been
  unusable since v0.2.0 — three separate reports were this one fault.
- **Detector width no longer starts at an impossible value.** It was preset to
  0.1 mm, which is not a detector at any pattern size; that value quietly
  skewed the pattern centre and everything computed from it. It is now read
  from the measurement file, and where the file does not carry it, estimated
  and clearly labelled as an estimate. A value no real detector could have is
  now called out — this matters because a wrong detector geometry leaves the
  maps looking perfectly correct.
- **Choosing a map layer now shows that layer.** Until now the map only
  reloaded when you pressed "Refresh", so picking "Grain boundaries" appeared
  to do nothing at all. When a layer cannot be drawn yet, the reason is shown
  as a warning instead of a quiet status line.
- **IPF maps no longer offer a colormap.** IPF colours are a fixed code for
  crystal directions, not a value range; the setting never had any effect on
  them and only suggested otherwise. Colormaps remain where they belong, on
  Band Contrast, KAM, GOS and the other scalar maps.
- The status bar shows the grid size as "150×201" again, instead of the
  raw escape sequence it had been printing.
- "Load CIF" no longer fails in browser mode.

---

## v0.2.2 — 2026-08-27

Problem reports become easier to file and easier for us to act on.

- **"⚠ Report a problem" is now in the toolbar at the top of every page.** You
  no longer have to leave the page where something went wrong in order to
  report it — which also means the record of what happened is more complete.
- **A picture of the window is included automatically** (desktop app only),
  taken the moment you open the dialog, so you no longer need to make a
  screenshot yourself. You can leave it out if the screen shows something you
  would rather not share.
- **Every report carries a short error id.** The same fault produces the same
  id on every machine, so if two people run into one problem it is visible as
  one problem instead of two unrelated reports. It is shown in the dialog and
  written into the report.
- **"Open a GitHub issue"** opens a new issue with your description, the
  version, the error id and the preceding events already filled in — you only
  drag the report file into it. (GitHub does not allow programs to attach
  files to an issue, so that last step stays manual.)

---

## v0.2.1 — 2026-08-27

Fixes to the update check itself, found by running it against a real
installation.

- **Says when it cannot sign in.** A private repository answers "not found"
  rather than "access denied" when credentials are missing, so the check used
  to report "server unreachable" and send you looking for a network problem.
  It now tells you to sign in once instead.
- **The check never opens a credential window on its own.** It runs in the
  background at start-up, where a prompt nobody asked for would either hang or
  appear out of nowhere.
- **"Check for updates" now says what it found** — up to date, not signed in,
  or unreachable — instead of only ever reporting "up to date".
- Release notes with dashes and umlauts are no longer garbled on Windows.

---

## v0.2.0 — 2026-08-27

The first release since v0.1.0, and a large one: 164 changes. The headline
items are a rebuilt EDS phase map, cropping a scan to a selection, and a
one-click problem report.

### Staying up to date

- **Orienta now checks for a new release when it starts** and offers to install
  it: it fetches the new version, installs any changed packages, rebuilds the
  interface and restarts. If any step fails, your working version stays exactly
  as it was.
- Only **released versions** are offered, never work in progress.
- You can skip a version, postpone the question, or switch the check off
  entirely under Settings → About Orienta.
- Installations set up from a downloaded archive (rather than a git clone)
  cannot update themselves; they are told so, with instructions, instead of
  being offered a button that cannot work.

### Reporting problems

- **Every install now has a version.** Settings → About Orienta shows it;
  quote it in bug reports.
- **"Report a problem…"** in Settings and on the crash screen packages your
  description, the app version, environment details and the log files into one
  zip to attach. It contains no measurement data and is never sent
  automatically. See [the guide](docs/user-guide/ReportingProblems.md).
- **Orienta now keeps log files** (`logs/`), so the cause of a failure survives
  closing the window — including crashes that happen before the interface even
  appears.

### EDS phase map

- **The map is grouped into regions by chemistry first**, and phases are
  assigned to those regions afterwards — instead of deciding pixel by pixel.
  Regions can be merged, split, grown, shrunk and snapped to chemical edges.
- **Region inspector** below the map: full composition with spread, enrichment
  against the map background, neighbours, and candidate phases with their
  deviation.
- **Magic-wand selection** — click a feature to select all of it, with live
  preview, a readout of what is selected, undo, and map-wide phase replacement.
- **Rules that decide which phases may compete** for a region (e.g. "silicon at
  least twice the background"), so ambiguous EDS signals stop pulling the wrong
  phase in.
- **Phase colours you pick**, shared with the EBSD phase map, and an overview
  that folds many regions into few phases.
- The phase map has **its own tab**, can be **zoomed with the scroll wheel**,
  **overlaid on an element map or the SEM image**, and **exported by right-click**.
- **EDS-only files** (Aztec element-distribution exports without EBSD data) now
  open directly on the EDS page.

### Cropping a scan

- **Crop to a rectangle, ellipse or lasso selection** in the EBSD viewer —
  patterns, EDS maps, band contrast and per-pixel data all follow, without loss.
- **Export the crop as a standalone file** (a 546 MB source became a 42 MB file
  in testing, patterns bit-identical).
- Indexing, including batch runs, honours a non-rectangular selection.
- Electron images follow the crop window through their own micrometre geometry.

### Indexing & phase identification

- **Render-verified phase check and grain reassignment** — find and correct
  regions where a phase was assigned that the pattern does not support.
- **Pseudo-symmetry tools**: map-wide variant unification, grain flip with coset
  snap and undo, a variant gallery with neighbour-grain and reference-pixel
  candidates.
- **Grain-stabilised IPF colouring** removes the speckle caused by colour-key
  discontinuities, and IPF maps can be **filtered to a single phase**.
- **EDAX .up1/.up2 and .osc files** can be loaded.
- Hough indexing works for CIFs written in a non-standard space-group setting
  (previously it tried to allocate over a terabyte and failed).
- Spherical indexing falls back to the CPU on machines without an NVIDIA GPU
  instead of crashing.
- Indexing speed is reported as patterns/second for **all three methods**, so
  they can be compared.

### Maps, images and export

- **Image export everywhere** — right-click a pattern, map or preview for a
  dialog with crop, resolution, format, border, scale bar and caption.
- **Scale bars and colour keys are movable objects** in the exported figure,
  not fixed columns.
- **Grain boundaries** as a map layer, classified by misorientation into three
  freely adjustable bands.
- **Layer picker with search** that only offers layers which can actually be
  drawn, and says what is missing for the rest.
- **Value-scale legends** for every layer whose colours mean a number.
- **3D crystal-structure viewer** for CIF and XTAL files in the database browser.

### Fixes worth knowing about

- **Closing the app now really stops the backend.** A surviving backend used to
  hold port 8000 and, after a drive was unplugged, served dead file handles —
  every file load then failed with a cryptic error.
- Band contrast comes from a single service with a clear order of preference
  (native → stored → computed) and is named accordingly, instead of silently
  substituting a different measure.
- Saved results re-import with the geometry they were computed with, so a
  pattern match after reloading matches again.
- Batch indexing with Hough or Dictionary no longer crashes at startup.

### New in the interface

- New wordmark, logo and favicon.
- The About section carries the licence notice, the version and the problem
  report button.

---

## v0.1.0 — 2026-06-29

Initial release for beta testers.

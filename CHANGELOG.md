# Changelog

All notable changes to Orienta, newest first. While Orienta is pre-1.0, the
minor number (0.**x**.0) marks a release that changes how you work — a new
module, a reworked page — and the patch number (0.x.**y**) everything else:
fixes, and refinements to features that are already there.

You can see which version you are running under **Settings → About Orienta**.

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
  of these phases may therefore differ slightly from earlier versions (not
  measured). Phases with unique axis c and all other crystal systems are
  unchanged. That the spherical indexer stores orientations in this frame was
  checked with patterns rendered from the Al13Fe4 master; the corrected
  calculations were checked on synthetic orientation maps. Nothing has yet been
  compared on a measured Al13Fe4 map.
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
  these instead of 0.11.3 / 0.14.1 / 0.3.9.1. An installation updated from v0.4.6
  brings its Python packages to these versions the first time it starts (a few
  megabytes to download). If that cannot finish, for example without an internet
  connection, Orienta says so once, starts with the packages it has, and tries
  again at a later start; set `ORIENTA_SKIP_PACKAGE_SYNC=1` to keep the current
  packages.
  Updating a source checkout installs the new versions into the environment
  Orienta runs in. Both sets were compared on real data: spherical indexing agreed
  to within 3e-5° and dictionary indexing was identical. Hough indexing in the
  installed app uses PyEBSDIndex's CPU band detection, and its results change: on
  a nickel map the orientations moved by 0.07° (median) and 99.8 % of the pixels
  by less than 1°; on a two-phase map of an aluminium alloy (Al and α-Al(Fe,Mn)Si,
  10,800 pixels) 98 % of the pixels kept their phase, and of the α-Al(Fe,Mn)Si
  pixels that kept it, 92 % moved by less than 1° and most of the others by about
  72°, a pseudo-symmetric variant of that phase. On the nickel map the new CPU
  band detection gives the same orientations as the OpenCL band detection of
  source installations with pyopencl (median difference below 0.0001°; 0.085° with
  PyEBSDIndex 0.3.9.1). On the two-phase map the two band detectors disagree for
  about a third of the α-Al(Fe,Mn)Si pixels, with the old and the new version
  alike.
- kikuchipy 0.12 reversed the meaning of the detector's azimuthal angle (the
  rotation of the detector about its optical axis). Orienta keeps the earlier
  meaning, which is also EMsoft's, on every path that uses the angle: where it
  hands a detector to kikuchipy 0.12.1 or newer, it converts the sign, so CPU and
  GPU dictionary indexing and generation agree on both kikuchipy versions. Whether
  this matches the vendor's definition of the header value (the EDAX field `Camera
  Azimuthal Angle`) is not documented and has not been measured; every file we
  tested reads 0. Hough and spherical indexing do not use the angle, and with
  kikuchipy 0.11.3 neither does the Kikuchi-band overlay in PC refinement. When a
  dataset has a non-zero azimuthal angle, the run log says so.
- `start_app.py` refuses to start when port 8000 is already in use.

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

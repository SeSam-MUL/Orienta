# Tilt distortion of .h5oina images and maps

## What it does

For EBSD the sample is tilted, typically by 70°. AZtec nevertheless stores the
electron images and the maps with square pixels (`X Step` = `Y Step`). On the
sample surface one image direction is then foreshortened by the factor
`k = 1 / cos(tilt)`, which is **2.92 at 70°**. A length, a grain size, a particle
shape or a scale bar read off such an image is wrong by that factor in one
direction.

When you load an `.h5oina`, Orienta checks every electron image and map of the
file and shows the result in the status bar:

| Status bar | Meaning |
|---|---|
| *(nothing)* | Not tilted and no measurable distortion, or not an `.h5oina`. |
| **Tilt distortion: X foreshortened ×2.92** | Header tilt and image measurement agree. Hover for the true map step and the correction. |
| **Tilt distortion unclear** | The header says the sample was tilted, but the image does not confirm it. Check manually. |

The check only reports. It changes nothing in the file or the data.

## How the direction is found

The header says *whether* a distortion is to be expected and *how large* it should
be. It does not reliably say *which* direction is foreshortened: on the data this
was developed on, `Tilt Axis` was 0 while the foreshortening lay in **X**.
`Tilt Axis`, `Scan Rotation` and the per-point detector distances are therefore
recorded but never decisive. The direction is measured on the image:

1. **Header.** `Tilt Angle` (holder pre-tilt) plus `Stage Position/Tilt` gives the
   effective tilt and `k = 1/cos(tilt)`. Below 10° the image counts as untilted
   (k ≈ 1), and the measurement still runs as a control.
2. **Measurement** on the SE image (the sharpest; FSD only as a supplement):
   subtract a Gaussian low-pass (σ = 15 px), cut the image into 4 × 4 tiles, compute
   each tile's autocorrelation by FFT, and take the shift at which it falls to 0.5
   along X (`Lx`) and along Y (`Ly`). The median of `Lx/Ly` over the 8 tiles with
   the most texture is the result. If the sample texture is isotropic on average, a
   foreshortened direction has the shorter correlation length.

| Median Lx/Ly | Reading |
|---|---|
| below 0.6 | X foreshortened (expected ≈ cos(tilt), 0.34 at 70°) |
| 0.6 to 0.75 | probably X foreshortened (blur lowers the contrast) |
| 0.75 to 1.3 | no measurable foreshortening |
| 1.3 to 1.7 | unclear |
| above 1.7 | Y foreshortened (expected ≈ 1/cos(tilt)) |

3. **Header and measurement together:**

| Header tilt | Measurement | Result |
|---|---|---|
| ≥ 10° | Lx/Ly ≈ cos(tilt) | X foreshortened, factor k confirmed |
| ≥ 10° | Lx/Ly ≈ 1 | **Contradiction:** a tilt correction may be active, or the tilt is only in the header. Reported as unclear. |
| < 10° | Lx/Ly ≈ 1 | undistorted |
| < 10° | Lx/Ly clearly below 0.6 | **Contradiction:** reported as unclear, with the measured factor 1/(Lx/Ly) |

An unclear image is never corrected by a guess: its correction factor is 1.

Things that limit the measurement:

- Blur, for example from the shallow depth of field of a tilted view, widens
  structures in both directions equally and pulls the ratio towards 1. Sharp
  tiles carry the most weight.
- FSD images can give 1.1 to 2.5 without any tilt (scan artefacts). The file is
  never judged on FSD alone.
- A very smooth image (a polished section with little texture) gives too few
  usable tiles and is reported as unclear.
- The visual impression of an oblique view (blurred background, structures
  standing up) is no evidence of foreshortening.

## Maps

For every EBSD and EDS map, the size of one map step in the electron image is
computed: `Relative Size[0] × image X Cells / map X Cells` and the same in Y. If the
two agree within 15 %, the map lies in the image's coordinate system and
**inherits its distortion**: the stated `X Step` is only right in the
undistorted direction, and in the foreshortened one the true step is
`X Step × k`. This applies to element maps, phase maps, layered images, and grain
sizes and grain shapes derived from them. The EBSD `Hit Rate` is reported as
well; at 0 % a phase map says nothing.

## True aspect ratio on the EDS page

With a confirmed finding (X or Y foreshortened), the EDS Analysis page draws the
electron images, the element maps and the composite overlay with the sample's
true aspect ratio: X (or Y) k times longer. The switch **True aspect ratio
(X × 2.92)** in the page header turns it off and on; it is on by default.

Only the display changes. The file, the bitmaps and the pixel grid stay as they
are, and a click, a region or the crosshair still lands on the same map pixel. A
stretched electron image is smoothed; maps keep nearest neighbour so the raster
stays visible. Maps the check found on a raster of their own are left as they
are. When the check could not tell (no `Relative Size`), they are stretched
with the image, which the page already draws over the whole map. An unclear
finding is never corrected.

Exports from the EDS page (**Export image…** and the right-click menu: one map,
every map, the overlay, the sheet of all maps) follow the switch: with the
correction on, the written picture is stretched the same way and its file name
ends in `_true-aspect`. The scale bar stays true: each exported pixel is the
stored step size in both directions again. The phase and region map exports are
not stretched yet.

## Correcting an image

If an X foreshortening is confirmed: stretch X by k and leave Y. A 1024 × 768 px
image at k = 2.92 becomes 2994 × 768 px. Interpolate images with Lanczos and maps
with nearest neighbour, so the measuring raster stays visible. After the stretch,
`X Step` is the µm per pixel in both directions; label the scale bar as valid
horizontally. The correction is strictly valid for the base plane only: with
strong relief, structures that stand up are drawn too wide.

## Checking files from the command line

```bash
python scripts/h5oina_tilt_check.py "Messungen/*.h5oina" --summary
python scripts/h5oina_tilt_check.py file.h5oina          # full JSON
```

The JSON has one object per file (German field names, as the check was specified):

```json
{
  "datei": "…AB6….h5oina",
  "kippung_header_grad": 70.0,
  "tischkippung_grad": 0.0,
  "faktor_k_theoretisch": 2.92,
  "bilder": [{"name": "SE/Elektronenbild 33 (Input1)", "pixel": [1024, 768],
              "step_nominal_um": 0.0621, "lx_ly_median": 0.34, "befund": "X gestaucht",
              "faktor_gemessen": 2.9, "korrektur": "X × 2.92, Y × 1",
              "um_pro_px_nach_korrektur": [0.0621, 0.0621], "zielgroesse_px": [2994, 768]}],
  "karten": [{"typ": "EBSD", "raster": [12, 9], "step_nominal_um": 0.658,
              "px_pro_schritt_im_bild": [9.83, 8.66], "erbt_bildverzerrung": true,
              "wahrer_schritt_um": [1.92, 0.658], "hit_rate_prozent": 15.7}],
  "hinweise": ["…"],
  "befund": "X gestaucht",
  "zusammenfassung": "Verzerrt: ja, X gestaucht, Faktor 2,92."
}
```

The same answer comes from `GET /api/ebsd/tilt-check` for the open file, and as
`tilt_check` in the answer of `POST /api/ebsd/load`.

## Reference cases

Measured on the data set "Masterarbeit_Schlegl_2026_07" (SE images). The decision
rules are tested against these values (`tests/test_tilt_distortion.py`); the
measurement itself is tested on synthetic images with a known foreshortening, as
the files are not distributed.

| File | Header tilt | Lx/Ly (SE) | Result |
|---|---|---|---|
| 6m_EBSD_TEST AB6, Elektronenbild 33 | 70° | 0.34 | X foreshortened, k = 2.92 |
| 6m_EBSD_TEST AB10, Elektronenbild 38 | 70° | 0.39 | X foreshortened, k = 2.92 |
| 6m_EBSD_TEST AB11, Elektronenbild 39 | 70° | 0.29 | X foreshortened, k = 2.92 |
| 6m_EBSD_TEST AB3, Elektronenbild 30 | 70° | 0.41 | X foreshortened, k = 2.92 |
| 6m_EBSD_TEST AB5, Elektronenbild 32 | 70° | 0.38 | X foreshortened, k = 2.92 |
| 6m_EBSD_TEST AB4, Elektronenbild 31 | 0° | 1.01 | undistorted |
| 6m_EBSD_TEST AB7, Elektronenbild 35 | 0° | 1.01 | undistorted (looks oblique) |
| 6_m AB4, Elektronenbild 18 | 0° | 1.04 | undistorted |
| Projekt 1, Probe 1, Elektronenbild 1 | 70° | 1.63 | unclear (contradiction, little texture) |
| Projekt 1, Probe 1, Elektronenbild 2 | 70° | 1.02 | unclear (contradiction) |

AB6 map raster: 12 × 9 points, 0.658 µm, 9.83 × 8.66 px per step in the image. The
map inherits the distortion; the true step in X is about 1.92 µm.

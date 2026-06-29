# Pole Figure

## What it does

The Pole Figure tool plots the **crystallographic texture** of an indexing result:
for a chosen phase and a set of crystal pole families (`{hkl}`), it projects the
symmetry-equivalent poles of every indexed orientation into the **sample
reference frame** and shows them as a stereographic/equal-area pole figure. This
reveals preferred orientations (texture) — clusters of poles indicate a texture
component, an even spread indicates a random texture.

For each requested pole family it draws one panel containing a **scatter** of poles
and/or a smoothed **density** contour. The projection, hemisphere, and axis
convention come from the shared coordinate-system frame, so the pole figure matches
the IPF colours on the [Phase Map](PhaseMap.md) and exports.

It is rendered server-side (matplotlib + orix symmetry) via the backend
`/api/pole-figure` route and stays live by polling the backend's state version.

## When to use it

Use the Pole Figure **after indexing**, when you want to characterise **texture**
rather than the spatial map. It is the standard companion to the IPF map for
reporting rolling/recrystallisation textures and verifying that a phase's
orientation distribution makes physical sense.

## How to open it

The Pole Figure is **not a sidebar entry**. Open it from the **Phase Map** page:
in the left panel (near the Coordinate System panel) click **Open Pole Figure**.
This launches a **detached window**:

- In the Electron desktop app it opens as a separate native window.
- In a browser it opens as a separate same-origin popup.

The window reads everything from the backend over REST and updates itself live, so
re-indexing or changing the coordinate frame is reflected automatically.

## How to use it — step by step

1. Click **Open Pole Figure** on the Phase Map page. The detached window appears
   with a controls column on the left and the figure on the right.
2. Choose the **Phase** from the dropdown (it lists the phases of the active
   indexing result; the first phase is selected automatically).
3. Enter the **pole families** in the `{hkl}` field as a comma-separated list
   (e.g. `100,110,111`). Each family becomes one panel; up to four are allowed.
   Compact single-digit form (`111`) or spaced form (`1 1 1`) both work. The figure
   refreshes when you leave the field.
4. Pick the **Style** dropdown: **Both** (density contour + scatter), **Density**
   (smoothed contour only), or **Scatter** (points only).
5. Open the **Coordinate System** panel (same control as on the Phase Map page) to
   set the orientation reference frame and the **projection** (equal-area /
   stereographic), **hemisphere** (upper/lower), **X direction**, and the
   *Z into plane* option. Changes re-render the figure immediately and stay in sync
   with the Phase Map window.

## Inputs & outputs

- **Inputs:**
  - The **active indexing result** (CrystalMap) in backend memory — the same result
    shown on the Phase Map.
  - The selected phase, the `{hkl}` pole families, the plot style, and the shared
    coordinate-system frame.
- **Outputs:**
  - The pole-figure image (rendered PNG shown in the window). There is **no
    dedicated export button** in the window; use a screenshot, or compose figures on
    the Phase Map page. The figure is plotted on a white background suitable for
    pasting into documents.

## Tips & notes

- **Default pole families** depend on crystal system if you leave the field at its
  defaults: cubic → `{100} {110} {111}`, hexagonal/trigonal → basal/prism/pyramidal.
- **Sensible `{hkl}` per system.** Entering cubic families for a hexagonal phase is
  allowed but not physically meaningful; pick families appropriate to the phase's
  symmetry.
- **Live and shared.** The window polls the backend, so it tracks the active result
  and the coordinate frame without a manual refresh. Editing the frame here also
  updates the Phase Map (and vice versa).
- **Performance.** For very large maps the scatter is subsampled (a fixed random
  subset of orientations) to keep rendering fast; the density contour still reflects
  the full distribution within that subsample.
- **No active result, no figure.** If no indexing result is active the window shows
  "no result"; index a scan first (and keep *Send to Phase Map* on).
- **The frame is non-destructive.** The coordinate-system rotation is applied only
  for display/plotting; it does not alter the stored orientations.

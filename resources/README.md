# `resources/` — Application Icons

This directory holds the **Orienta application icon** in the two raster formats the
desktop shell needs: a `256×256` PNG used as the live window/taskbar icon, and a
multi-resolution Windows `.ico` used when the app is packaged and installed. Both
files are *generated artifacts* — they are produced from a single script so the icon
stays consistent across formats and sizes — and they are the only assets this folder
contains.

## Files

| File | Format / sizes | Description |
|------|----------------|-------------|
| [`icon.png`](icon.png) | PNG, `256×256`, RGBA | The primary app icon. Loaded by the Electron main process as the runtime window and taskbar icon. |
| [`icon.ico`](icon.ico) | Windows ICO, 6 embedded sizes (`16`, `32`, `48`, `64`, `128`, `256`) | Multi-resolution Windows icon for packaging/installer use (Explorer, shortcuts, the executable). |

### The icon design

The icon is an **IPF (inverse pole figure) orientation triangle** — a barycentric
RGB blend (red top, green bottom-right, blue bottom-left) on a dark rounded-square
card. This mirrors the IPF colouring that EBSD orientation maps use throughout the
app, so the icon visually signals what Orienta does.

## How it fits the overall architecture

Orienta runs as an **Electron shell → React frontend → FastAPI backend** stack
(see the top-level project docs). This directory serves only the Electron shell
layer: when the desktop window is created, the main process points its window
`icon` at [`icon.png`](icon.png) (see [`electron/main.js`](../electron/main.js),
where the `BrowserWindow` is configured with
`icon: path.join(__dirname, '..', 'resources', 'icon.png')`). The `.ico` is the
Windows-native equivalent consumed by the packaging/installer step.

> **Note on UI theme/styling:** the visual theme (colours, layout, component
> styling) is **not** stored here — it lives with the React frontend as CSS under
> [`frontend/src/`](../frontend/src). This folder is strictly the binary app-icon
> assets.

## Use / regeneration notes

- **Do not hand-edit** `icon.png` / `icon.ico`. They are regenerated from
  [`branding/make_logo.py`](../branding/make_logo.py), which draws the triangle,
  applies the rounded-square mask, and writes both files plus a larger
  `512×512` web/preview copy under [`branding/`](../branding). Running that script
  is the canonical way to update the icon; edit the script (vertex colours, card
  background, corner radius), then re-run it.
- **Paths are referenced relatively** (resolved from the project root via
  `__dirname`/`Path`), so the assets work regardless of where the project is
  checked out — keep the filenames `icon.png` and `icon.ico` stable so the
  Electron and packaging references continue to resolve.
- Related branding assets (wordmarks, logo sheets, the `512×512` icon) live in the
  sibling [`branding/`](../branding) directory, not here.

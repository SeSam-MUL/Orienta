# Orienta — Brand

**Name:** Orienta
**Tagline:** Making EBSD indexing accessible
**Idea:** EBSD indexing determines crystal **orientation** — the icon is the
**IPF orientation triangle** (the field's instantly-recognizable signature).
Name ↔ logo ↔ function all line up.

## Logo files
| File | Use |
|------|-----|
| `branding/icon.svg` | App icon — IPF triangle in a dark rounded square (vector) |
| `frontend/public/favicon.svg` | Browser-tab icon — same geometry, white outline dropped (it muddies the apex at 16–32 px) |
| `resources/icon.png` (256) · `resources/icon.ico` | Electron window + Windows installer icon |
| `branding/orienta-icon-512.png` | README / Zenodo / web (raster) |
| `branding/wordmark-dark.svg` | Wordmark `orıenta` (i-dot = IPF triangle) on dark |
| `branding/wordmark-light.svg` | Wordmark on light / paper |
| `branding/wordmark-mono.svg` | Wordmark, one colour (teal) — stamps, watermarks, single-ink print |
| `branding/icon-mono.svg` | App icon, one colour (teal outline) |
| `branding/mark.svg` | The triangle alone — no plate, cropped to the shape, transparent |
| `branding/export.html` | **Export page** — rasterises any of the above to PNG at any width |
| `branding/export/*.png` | Ready-made high-res PNGs (transparent), for slides and print |
| `branding/final.html` | Logo sheet — **loads the real SVGs**, so it cannot drift from what ships |
| `branding/orienta-final.png` | Rendered logo sheet (open `final.html` and re-shoot it to update) |
| `branding/make_logo.py` | Regenerates the PNG/ICO (true barycentric IPF triangle) |

## Colours
| Token | Hex | Use |
|-------|-----|-----|
| IPF Red | `#ff3b3b` | triangle top corner ([001]) |
| IPF Green | `#22dd66` | triangle bottom-right ([101]) |
| IPF Blue | `#2f7bff` | triangle bottom-left ([111]) |
| Accent Teal | `#3fb6c4` | UI accent, mono logo |
| Background | `#0d1117` / card `#0f141b` | dark UI |
| Ink / Muted | `#e6edf3` / `#8b98a5` | text |

## Putting the logo in a slide / paper / poster

**Use the SVG.** PowerPoint (2016+), Word, Illustrator and Inkscape all place SVG
directly and it stays sharp at any size. Never pull a logo out of
`orienta-final.png` — that file is a screenshot of the overview sheet, so it is
only as sharp as the sheet was.

Where SVG is not an option, take a PNG from `branding/export/` (transparent
background, `wordmark-*` at 4096 px, icons at 2048 px). For any other size open
`branding/export.html` over a local server:

```bash
python -m http.server 8000          # from the project root
# then open localhost:8000/branding/export.html, set a width, hit download
```

The page rasterises the real SVGs through the browser, so the PNG matches the
vector exactly — `cairosvg`/`rsvg` do not implement the `mix-blend-mode` the IPF
triangle is built from and render it wrong.

## Usage
- Keep clear space ≥ the triangle's height around the icon.
- Wordmark is lowercase `orıenta`; the i-dot is the IPF triangle (don't add a normal dot).
- Mono contexts (stamps, watermarks): teal triangle outline, single colour.
- Don't recolour the IPF triangle arbitrarily — the red/green/blue corners are the EBSD signature.
- **The triangle sits on the `ı`, not next to it.** Its x is measured from the rendered
  glyph (Segoe UI 680, letter-spacing −1): centre `84.5` at font-size 58, `50.4` at 40.
  Change the font, the size or the letter-spacing → re-measure, don't eyeball. Verified
  centre-of-mass offset is ≤0.35 px across the mark.

## Naming / trademark note
Preliminary screening (web): no direct software/scientific product named
"Orienta" found → low infringement risk, usable. But "Orienta" is a generic
word (Spanish/Italian "it orients"; some unrelated HR/education services use it),
so it is a **weak/descriptive mark** — fine to *use* for this research tool, not
easily *ownable* as an exclusive trademark. Bare handles (`orienta.com`, PyPI
`orienta`) are likely taken → qualify them for repo/package (e.g. `orienta-ebsd`)
while the **display name stays "Orienta"**. Binding clearance = institutional
legal/tech-transfer via EUIPO eSearch + USPTO before public release.

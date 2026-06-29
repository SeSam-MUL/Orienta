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
| `resources/icon.png` (256) · `resources/icon.ico` | Electron window + Windows installer icon |
| `branding/orienta-icon-512.png` | README / Zenodo / web (raster) |
| `branding/wordmark-dark.svg` | Wordmark `orıenta` (i-dot = IPF triangle) on dark |
| `branding/wordmark-light.svg` | Wordmark on light / paper |
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

## Usage
- Keep clear space ≥ the triangle's height around the icon.
- Wordmark is lowercase `orıenta`; the i-dot is the IPF triangle (don't add a normal dot).
- Mono contexts (stamps, watermarks): teal triangle outline, single colour.
- Don't recolour the IPF triangle arbitrarily — the red/green/blue corners are the EBSD signature.

## Naming / trademark note
Preliminary screening (web): no direct software/scientific product named
"Orienta" found → low infringement risk, usable. But "Orienta" is a generic
word (Spanish/Italian "it orients"; some unrelated HR/education services use it),
so it is a **weak/descriptive mark** — fine to *use* for this research tool, not
easily *ownable* as an exclusive trademark. Bare handles (`orienta.com`, PyPI
`orienta`) are likely taken → qualify them for repo/package (e.g. `orienta-ebsd`)
while the **display name stays "Orienta"**. Binding clearance = institutional
legal/tech-transfer via EUIPO eSearch + USPTO before public release.

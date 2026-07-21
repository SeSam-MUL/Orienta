# EDAX UP1/UP2 + OSC — verifizierte Format-Fakten

> Reverse-engineered + gegen die echten Dateien in `Test_data/new test/` verifiziert.
> **Kein Raten** — jedes Feld unten ist an ≥2 Dateien bestätigt. Datum: 2026-07-17.

## Testdaten (Fakten)
| Datei | up1-Ver | Pattern (sx×sy) | Grid (nc×nr) | npoints | Phase |
|---|---|---|---|---|---|
| Scan5.up1/.osc | v1 | 79×79 | 151×151 | 22801 | Aluminum |
| Scan57.up1/.osc | v1 | 79×79 | 151×151 | 22801 | Aluminum |
| map…cropped.up1/.osc | v3 | 114×114 | 123×91 | 11193 | Aluminum (+350 nicht indiziert) |

## UP1/UP2 (die Patterns) — autoritativ aus kikuchipy `edax_binary/_api.py`
Byte-Layout (little-endian, uint32):
- `@0` version, `@4` sx (Pattern-Breite), `@8` sy (Pattern-Höhe), `@12` pattern_offset
- **v1**: Header = 16 Bytes. Grid NICHT enthalten → Reader liefert `nx=npat, ny=1` (flach).
  `npat = (filesize - pattern_offset) / (sx*sy*bytes)`. Schrittweite NICHT enthalten.
- **v3+**: `@16` 1 Skip-Byte, `@17` nx (uint32), `@21` ny (uint32), `@25` is_hex (uint8),
  `@26` dx (float64), `@34` dy (float64), Patterns ab `@42`. **Aber dx=dy=1.0 in unseren Dateien**
  (Platzhalter — echte Schrittweite steht auch bei v3 nur in der .osc).
- dtype: **up1 = uint8, up2 = uint16**. Reader validiert `nx*ny==npat` (fail-loud).
- Hexagonales Grid (`is_hex=True`) → nur 1 Nav-Dim, egal was übergeben wird.
- kikuchipy akzeptiert `kp.load(path, lazy=True, nav_shape=(nrows,ncols))` → reshape für v1. Verifiziert.

## OSC — Grid-Mini-Header (fixe Offsets, beide Dateien bestätigt)
- `@16` uint32 = **ncols-1**, `@20` uint32 = **nrows-1**, `@24` uint32 = **npoints**
- Konsistenz-Check: `(o16+1)*(o20+1) == o24`. Scan5: 151·151=22801 ✓; map: 123·91=11193 ✓
- Cross-validiert gegen kikuchipys unabhängigen v3-Header-Parse (123,91). Rock solid.

## OSC — OIM-Hough-Ergebnisse (Magic-verankert, alle 3 Dateien bestätigt)
Spec-Quelle: MTEX `loadEBSD_osc.m`, korrigiert um die **Alignment-Falle** (Block ist NICHT
4-Byte-alignt zum Dateianfang — ab `k+24` in BYTES lesen, nicht via 4-alignten float-view!).

- Magic (8 Bytes): `B9 0B EF FF 02 00 00 00` bei Byte `k` (genau 1×).
- `k+8`  uint32 = **Byte-Offset, wo die Punktdaten enden** (Scan5: 1282518 = exakt gemessenes Ende ✓)
- `k+12` uint32 = 0
- `k+16` float32 = **Xstep** (µm), `k+20` float32 = **Ystep** (µm). Unsere Dateien: **1.0 µm** (real).
- `k+24` … : **npoints Records à 14 float32, row-major**:
  `[phi1, Phi, phi2, x, y, IQ, CI, Phase, SEM, Fit, u10, u11, u12, u13]`
  - phi1,phi2 ∈ [0,2π], Phi ∈ [0,π] — **Radiant, Bunge-Euler** (OIM-Konvention).
  - x,y in µm (Raster: x 0→(nc-1)·Xstep pro Zeile, y +Ystep pro Zeile). Letzter Punkt = (150,150)/(122,90) ✓
  - IQ = Image Quality (roh, ~30k–84k), CI = Confidence Index ∈ [0,1], Phase = 0-basierter Index,
    Fit = mean angular deviation (deg), SEM = SE-Signal.
  - **Nicht indizierte Pixel**: phi=4π (12.566), CI=-1, Phase=-1, Fit=180 (map: 350 Px).
  - u11/u12/u13 = konstante Footer-Werte (ignorieren); u10 variiert nur in v3-map (Zusatzmetrik).
- Danach großer Trailer (Scan5 ~8.65 MB): Hough-Peak-/Pattern-Quality-Daten. **Nicht benötigt.**

## OSC — Pattern Center (KRITISCH fürs Indexing)
- **Fixer Offset `@1860`**: 3× float32 = **xstar, ystar, zstar** (EDAX/TSL-Konvention).
  - Scan5/Scan57: `(0.5499, 0.5026, 0.7020)` · map: `(0.5758, 0.6487, 0.6435)`.
- up1/up2 selbst tragen KEINEN PC → kikuchipy hängt Platzhalter `(0.5,0.5,0.5)` an → Indexing
  komplett daneben. Der echte PC steckt nur hier in der `.osc`.
- **Verifiziert korrekt**: ein Joint-Orientation+PC-Refine (kikuchipy) konvergiert von diesem
  Wert praktisch nicht weg (`0.550/0.503/0.702` → `0.555/0.496/0.718`), und der `.osc`-PC gibt
  **~2.5× die Match-NCC** ggü. Default (refined 0.30 vs 0.12; GPU-Dict 3°: 0.16 vs ~0.09).
- Konvention = **tsl** (kikuchipy `EBSDDetector(pc=(x,y,z), convention='tsl')`); tsl↔native
  spiegelt nur ystar (`native_y = 1 - ystar`), Betrag hier ~0.503↔0.497.
- Validierungs-Lektion: absolute NCC ist durch **Gitter-Kappung** niedrig (Render-NCC-Peak ~2°
  schmal) — NICHT als „PC falsch" fehldeuten; per Refine/feinem Gitter prüfen. Die OIM-
  Orientierungen aus der `.osc` rendern übrigens NICHT direkt (separate OIM-Frame-Konvention —
  fürs eigene Indexing irrelevant, deshalb bewusst nicht genutzt).

## Phasen-Tabelle
- ASCII-Phasenname im Header (`"Aluminum\0"`), Raumgruppe per Index in externer
  `…\TexSEM\SpaceGroupsV3.bin` referenziert (nicht inline). → Für Import: Name + Phasenindex
  je Punkt genügen; Kristallographie kommt aus unserer DB (Al.cif/Al.xtal vorhanden).

## Konsequenz für die Umsetzung
up1/up2 + .osc zusammen liefern: **Patterns + korrektes 2D-Grid + echte Schrittweite +
fertige OIM-Orientierungen/CI/IQ/Phase**. Damit ist ein up1-Load OHNE eigenes Indizieren
sofort als Karte darstellbar (und als Ground-Truth-Vergleich nutzbar).

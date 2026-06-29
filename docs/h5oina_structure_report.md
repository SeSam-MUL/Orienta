# H5OINA Datenstruktur - Oxford Instruments Aztec

## Datei
`EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 Elementverteilungsdaten 1.h5oina`
Größe: ~521 MB

## Root-Struktur
```
/1/                          (Haupt-Messgruppe)
/Format Version/
/Index/
/Manufacturer/
/Software Version/
```

## EBSD-Daten: /1/EBSD/

### Data-Datasets: /1/EBSD/Data/
**10800 Messpunkte** (wahrscheinlich 90x120 oder ähnliches Grid)

#### Qualitäts-Parameter:
- `Band Contrast` (10800,) uint8 - Bandkontrast
- `Band Slope` (10800,) uint8 - Bandsteigung
- `Bands` (10800,) uint8 - Anzahl detektierter Bänder
- `Error` (10800,) uint8 - Fehler-Flags
- `Mean Angular Deviation` (10800,) float32 - MAD-Wert
- `Pattern Quality` (10800,) float32 - Pattern-Qualität

#### Orientierungsdaten:
- `Euler` (10800, 3) float32 - **Euler-Winkel (φ1, Φ, φ2)**
- `Phase` (10800,) uint8 - Phasen-ID

#### Pattern-Daten:
- **`Processed Patterns` (10800, 128, 156) uint8** - Bearbeitete EBSD-Patterns (~214 MB)
- **`Unprocessed Patterns` (10800, 128, 156) int16** - Rohe EBSD-Patterns (~428 MB)

#### Geometrie:
- `Beam Position X/Y` (10800,) float32 - Strahl-Position
- `X` (10800,) float32 - Scan X-Koordinate
- `Y` (10800,) float32 - Scan Y-Koordinate
- `Detector Distance` (10800,) float32 - Detektorabstand

#### Pattern Center:
- `Pattern Center X` (10800,) float32 - PC X-Koordinate
- `Pattern Center Y` (10800,) float32 - PC Y-Koordinate

---

### Header: /1/EBSD/Header/

#### Acquisition-Parameter:
- `Acquired Pattern Height/Width` - Original Pattern-Größe
- `Pattern Height/Width` (1,) int32 - **128 x 156 Pixel**
- `Acquisition Date/Time` - Aufnahme-Zeitstempel
- `Acquisition Speed` (1,) float32
- `Number Frames Averaged` (1,) int32

#### Mikroskop-Parameter:
- `Beam Voltage` (1,) float32 - Beschleunigungsspannung
- `Working Distance` (1,) float32 - Arbeitsabstand
- `Magnification` (1,) float32
- `Tilt Angle/Axis` (1,) float32 - **Proben-Tilt**

#### Detektor-Parameter:
- `Detector Insertion Distance` (1,) float32
- `Detector Orientation Euler` (1, 3) float32 - **Detektor-Orientierung**
- `Camera Exposure Time/Gain/Mode` - Kamera-Einstellungen

#### Pattern Center & Band Detection:
- `Band Detection Circle Center X/Y/Radius` - Hough-Parameter
- `Hough Resolution` (1,) int32
- `Number Bands Detected` (1,) int32

#### Background-Korrektur:
- `Processed Static Background` (128, 156) uint8 - Bearbeiteter Hintergrund
- `Unprocessed Static Background` (128, 156) int16 - Roher Hintergrund
- `Static Background Correction` (1,) uint8 - Flag
- `Auto Background Correction` (1,) uint8

#### Grid-Parameter:
- `X Cells/Y Cells` (1,) int32 - **Grid-Größe**
- `X Step/Y Step` (1,) float32 - **Schrittweite in µm**
- `Bounding Box Size` (2,) float32
- `Scanning Rotation Angle` (1,) float32

#### Stage Position: /1/EBSD/Header/Stage Position/
- `X/Y/Z` (1,) float32 - Stage-Koordinaten
- `Tilt/Rotation` (1,) float32 - Stage-Winkel

#### Phasen-Information: /1/EBSD/Header/Phases/1/
- `Phase Name` (1,) object - Phasenname
- `Phase Id` (1,) int32
- `Lattice Dimensions` (1, 3) float32 - **Gitterkonstanten (a, b, c)**
- `Lattice Angles` (1, 3) float32 - **Gitterwinkel (α, β, γ)**
- `Space Group` (1,) int32 - Raumgruppe
- `Laue Group` (1,) int32
- `Number Reflectors` (1,) int32 - Anzahl Reflektoren
- `Color` (1, 3) uint8 - RGB-Farbe für Visualisierung
- `Reference` (1,) object - Datenbank-Referenz

---

## EDS-Daten: /1/EDS/

### Data: /1/EDS/Data/
- `Live Time` (10800,) float32
- `Real Time` (10800,) float32
- `Spectrum` (10800, 2048) int32 - **EDS-Spektren für jeden Punkt**
- `Window Integral/` - Element-Mappings

---

## Wichtige Erkenntnisse

1. **Pattern-Daten vorhanden**:
   - Sowohl prozessierte als auch unprozessierte Patterns
   - 128x156 Pixel pro Pattern
   - 10800 Messpunkte

2. **Pattern Center bereits gespeichert**:
   - PCX und PCY für jeden Messpunkt
   - Kann als Ausgangswert verwendet werden

3. **Vollständige Metadaten**:
   - Alle Mikroskop-Parameter
   - Detektor-Geometrie
   - Phasen-Information

4. **Kombinierte EBSD+EDS Messung**:
   - EDS-Spektren für jeden EBSD-Punkt
   - Ermöglicht korrelierte Analyse

## Für Kikuchipy-Import wichtig

- **Patterns**: `/1/EBSD/Data/Processed Patterns` oder `Unprocessed Patterns`
- **Shape**: (10800, 128, 156) → muss zu (ny, nx, 128, 156) reshapen
- **Grid**: Aus `X Cells` und `Y Cells` Header ermitteln
- **Detector**: Aus `Detector Orientation Euler`, Pattern Center, etc.
- **Phase**: Aus `/1/EBSD/Header/Phases/1/` laden

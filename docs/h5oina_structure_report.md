# H5OINA Data Structure - Oxford Instruments Aztec

## File
`EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 Elementverteilungsdaten 1.h5oina`
Size: ~521 MB

## Root Structure
```
/1/                          (main measurement group)
/Format Version/
/Index/
/Manufacturer/
/Software Version/
```

## EBSD Data: /1/EBSD/

### Data Datasets: /1/EBSD/Data/
**10800 measurement points** (likely a 90x120 or similar grid)

#### Quality Parameters:
- `Band Contrast` (10800,) uint8 - band contrast
- `Band Slope` (10800,) uint8 - band slope
- `Bands` (10800,) uint8 - number of detected bands
- `Error` (10800,) uint8 - error flags
- `Mean Angular Deviation` (10800,) float32 - MAD value
- `Pattern Quality` (10800,) float32 - pattern quality

#### Orientation Data:
- `Euler` (10800, 3) float32 - **Euler angles (φ1, Φ, φ2)**
- `Phase` (10800,) uint8 - phase ID

#### Pattern Data:
- **`Processed Patterns` (10800, 128, 156) uint8** - processed EBSD patterns (~214 MB)
- **`Unprocessed Patterns` (10800, 128, 156) int16** - raw EBSD patterns (~428 MB)

#### Geometry:
- `Beam Position X/Y` (10800,) float32 - beam position
- `X` (10800,) float32 - scan X coordinate
- `Y` (10800,) float32 - scan Y coordinate
- `Detector Distance` (10800,) float32 - detector distance

#### Pattern Center:
- `Pattern Center X` (10800,) float32 - PC X coordinate
- `Pattern Center Y` (10800,) float32 - PC Y coordinate

---

### Header: /1/EBSD/Header/

#### Acquisition Parameters:
- `Acquired Pattern Height/Width` - original pattern size
- `Pattern Height/Width` (1,) int32 - **128 x 156 pixels**
- `Acquisition Date/Time` - acquisition timestamp
- `Acquisition Speed` (1,) float32
- `Number Frames Averaged` (1,) int32

#### Microscope Parameters:
- `Beam Voltage` (1,) float32 - accelerating voltage
- `Working Distance` (1,) float32 - working distance
- `Magnification` (1,) float32
- `Tilt Angle/Axis` (1,) float32 - **sample tilt**

#### Detector Parameters:
- `Detector Insertion Distance` (1,) float32
- `Detector Orientation Euler` (1, 3) float32 - **detector orientation**
- `Camera Exposure Time/Gain/Mode` - camera settings

#### Pattern Center & Band Detection:
- `Band Detection Circle Center X/Y/Radius` - Hough parameters
- `Hough Resolution` (1,) int32
- `Number Bands Detected` (1,) int32

#### Background Correction:
- `Processed Static Background` (128, 156) uint8 - processed background
- `Unprocessed Static Background` (128, 156) int16 - raw background
- `Static Background Correction` (1,) uint8 - flag
- `Auto Background Correction` (1,) uint8

#### Grid Parameters:
- `X Cells/Y Cells` (1,) int32 - **grid size**
- `X Step/Y Step` (1,) float32 - **step size in µm**
- `Bounding Box Size` (2,) float32
- `Scanning Rotation Angle` (1,) float32

#### Stage Position: /1/EBSD/Header/Stage Position/
- `X/Y/Z` (1,) float32 - stage coordinates
- `Tilt/Rotation` (1,) float32 - stage angles

#### Phase Information: /1/EBSD/Header/Phases/1/
- `Phase Name` (1,) object - phase name
- `Phase Id` (1,) int32
- `Lattice Dimensions` (1, 3) float32 - **lattice constants (a, b, c)**
- `Lattice Angles` (1, 3) float32 - **lattice angles (α, β, γ)**
- `Space Group` (1,) int32 - space group
- `Laue Group` (1,) int32
- `Number Reflectors` (1,) int32 - number of reflectors
- `Color` (1, 3) uint8 - RGB color for visualization
- `Reference` (1,) object - database reference

---

## EDS Data: /1/EDS/

### Data: /1/EDS/Data/
- `Live Time` (10800,) float32
- `Real Time` (10800,) float32
- `Spectrum` (10800, 2048) int32 - **EDS spectra for each point**
- `Window Integral/` - element mappings

---

## Key Findings

1. **Pattern data present**:
   - Both processed and unprocessed patterns
   - 128x156 pixels per pattern
   - 10800 measurement points

2. **Pattern Center already stored**:
   - PCX and PCY for each measurement point
   - Can be used as an initial value

3. **Complete metadata**:
   - All microscope parameters
   - Detector geometry
   - Phase information

4. **Combined EBSD+EDS measurement**:
   - EDS spectra for each EBSD point
   - Enables correlated analysis

## Important for Kikuchipy Import

- **Patterns**: `/1/EBSD/Data/Processed Patterns` or `Unprocessed Patterns`
- **Shape**: (10800, 128, 156) → must be reshaped to (ny, nx, 128, 156)
- **Grid**: Determine from `X Cells` and `Y Cells` headers
- **Detector**: From `Detector Orientation Euler`, Pattern Center, etc.
- **Phase**: Load from `/1/EBSD/Header/Phases/1/`

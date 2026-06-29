# Option A: Learned Kikuchi-Pattern Embedding mit FAISS-Retrieval

## Detailplan für ein ML-basiertes EBSD-Indexierungssystem

---

## 1. Überblick und Grundidee

### Kernkonzept
Ein CNN-Encoder $f_\theta$ lernt eine Abbildung:

$$f_\theta: \text{Kikuchi-Pattern} \rightarrow \mathbb{R}^d \quad (d = 128\text{–}256)$$

sodass Patterns mit ähnlicher Orientierung nahe beieinander im Embedding-Raum liegen. Zur Indexierung wird ein experimentelles Pattern encodiert und per Approximate Nearest Neighbor (ANN) gegen eine vorberechnete Datenbank von Referenz-Embeddings abgeglichen.

### Erwartete Vorteile gegenüber DI und SI
- **Geschwindigkeit**: ~1000x schneller als DI, ~10–50x schneller als SI bei Inferenz
- **Robustheit**: Durch Training mit augmentierten Daten intrinsisch tolerant gegen Rauschen, PC-Fehler, Strain
- **Skalierbarkeit**: FAISS-Index skaliert sublinear mit der Anzahl der Orientierungen
- **Flexibilität**: Erweiterbar auf Multi-Phase, Strain-Quantifizierung, Confidence-Schätzung

---

## 2. Architektur des Gesamtsystems

### 2.1 Systemkomponenten

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                            │
│                                                             │
│  Dynamical Simulation ──► Augmentation ──► CNN-Encoder      │
│  (EMsoft/kikuchipy)       Pipeline          Training         │
│                                                             │
│  Output: Trainiertes Modell f_θ                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 DICTIONARY BUILD PHASE                       │
│                                                             │
│  Orientierungsgrid ──► Pattern-Simulation ──► f_θ ──► FAISS │
│  (cubochoric/SO(3))    (Master Pattern)               Index │
│                                                             │
│  Output: FAISS-Index mit (embedding, orientation) Paaren    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE                           │
│                                                             │
│  Exp. Pattern ──► Preprocessing ──► f_θ ──► FAISS Query     │
│                                             │               │
│                                             ▼               │
│                                     Top-k Orientierungen    │
│                                             │               │
│                                             ▼               │
│                                     Lokale Verfeinerung     │
│                                     (optional)              │
│                                             │               │
│                                             ▼               │
│                                     Finale Orientierung     │
│                                     + Confidence Score      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 CNN-Encoder Architektur

#### Basis-Architektur: Modified ResNet-18

```
Input: (1, H, W) — Grayscale Kikuchi-Pattern (z.B. 60×60 oder 120×120 px)

├── Conv2d(1, 64, 7×7, stride=2, padding=3) + BN + ReLU
├── MaxPool(3×3, stride=2)
├── ResBlock(64, 64) × 2
├── ResBlock(64, 128, stride=2) × 2
├── ResBlock(128, 256, stride=2) × 2
├── ResBlock(256, 512, stride=2) × 2
├── AdaptiveAvgPool2d(1,1)
├── Flatten
├── Linear(512, d)           # Projection Head
├── L2-Normalisierung        # Embedding liegt auf Einheitskugel S^(d-1)

Output: Embedding ∈ S^(d-1) ⊂ ℝ^d
```

**Warum ResNet-18 und nicht tiefer?**
- Kikuchi-Patterns sind relativ strukturiert (Bandmuster) — ein zu tiefes Netz overfittet
- 60×60 px Input braucht nicht die Kapazität eines ResNet-50
- RTX 4070 (8 GB VRAM) limitiert die Batch-Größe bei größeren Netzen
- Schnellere Inferenz = schnelleres Indexieren großer Maps

**Alternative: EfficientNet-B0 oder MobileNetV3**
- Noch schneller bei Inferenz
- Sinnvoll wenn Geschwindigkeit höchste Priorität hat
- Etwas weniger Kapazität für feine Unterschiede bei ähnlichen Orientierungen

#### Embedding-Dimension d

| d | Vorteile | Nachteile |
|---|----------|-----------|
| 64 | Sehr schnell, kleiner Index | Möglicherweise zu wenig Kapazität für viele Phasen |
| 128 | Guter Kompromiss | Standard-Wahl |
| 256 | Mehr Kapazität für Multi-Phase | Größerer Index, langsamer |
| 512 | Maximale Trennfähigkeit | Diminishing Returns, FAISS langsamer |

**Empfehlung**: Start mit d=128, dann ablaten.

---

## 3. Trainingsdaten und Simulation

### 3.1 Master-Pattern-Generierung

Für jede Phase wird ein **Master Pattern** benötigt (dynamische Simulation):

- **Tool**: EMsoft (`EMEBSDmaster`) oder kikuchipy mit diffsims
- **Parameter**:
  - Beschleunigungsspannung: typisch 10–30 kV (phasenabhängig optimal)
  - Maximaler Beugungsordnung: hoch genug für realistische Patterns
  - Energiebins: für energiegewichtete Patterns

**Phasen-Coverage** (typisch für Al-Legierungen):
- α-Al (FCC, Fm-3m)
- Al₂Cu (θ, I4/mcm)
- Al₃Fe (C2/m)
- Mg₂Si (Fm-3m, andere Gitterkonstante)
- Al₆Mn (Cmcm)
- Si (Fd-3m)
- Weitere je nach Legierungssystem

### 3.2 Orientierungssampling

#### Gleichmäßiges Sampling auf SO(3)

Verwendung von **cubochoric Koordinaten** (Rosca et al., 2014) für gleichmäßiges Sampling:

```
Auflösung    | Anzahl Orientierungen | Mittlerer Winkelabstand
-------------|----------------------|------------------------
N_cubochoric = 50  | ~130.000            | ~2.5°
N_cubochoric = 80  | ~530.000            | ~1.6°
N_cubochoric = 100 | ~1.030.000          | ~1.2°
N_cubochoric = 150 | ~3.500.000          | ~0.8°
```

**Empfehlung**: 
- Training: N=100 (~1 Mio. Orientierungen pro Phase)
- Inference-Dictionary: N=150 (~3.5 Mio. pro Phase) oder dichter

#### Symmetriereduktion
- Nur den **fundamentalen Bereich** der jeweiligen Kristallsymmetrie samplen
- Für kubisch (Oh): Reduktion um Faktor 24 → ~43.000 unabhängige Orientierungen bei N=100
- Für niedrigere Symmetrien: weniger Reduktion, mehr Orientierungen nötig

### 3.3 Pattern-Generierung aus Master Patterns

Für jede Orientierung g ∈ SO(3):
1. Master Pattern rotieren nach g
2. Gnomonic Projection mit gegebenen PC-Koordinaten (PC_x, PC_y, PC_z)
3. Binning auf Zielauflösung (z.B. 60×60)

**Wichtig**: Die Projektion hängt von (PC_x, PC_y, PC_z) ab — siehe Abschnitt 5 für PC-Toleranz.

---

## 4. Training

### 4.1 Loss-Funktion: Contrastive Learning auf SO(3)

#### Grundproblem
Standard Contrastive Learning (z.B. SimCLR) behandelt alle negativen Paare gleich. Aber auf SO(3) gibt es **Abstufungen der Ähnlichkeit** — eine Orientierung, die 2° entfernt ist, ist "ähnlicher" als eine, die 40° entfernt ist.

#### Lösung: Angular-Margin-Aware Contrastive Loss

Definiere die **Misorientation** zwischen zwei Orientierungen:

$$\Delta\omega(g_1, g_2) = \min_{s \in S} \arccos\left(\frac{\text{tr}(s \cdot g_1^{-1} \cdot g_2) - 1}{2}\right)$$

wobei S die Symmetriegruppe des Kristalls ist.

**Soft Contrastive Loss**:

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_{k} w_k \cdot \exp(\text{sim}(z_i, z_k) / \tau)}$$

mit Gewichten $w_k$ basierend auf dem Misorientierungswinkel:

$$w_k = \begin{cases} 0 & \text{wenn } \Delta\omega < \omega_{\text{pos}} \quad (\text{zu ähnlich, kein negativer Push}) \\ 1 & \text{wenn } \Delta\omega > \omega_{\text{neg}} \quad (\text{klar negativ}) \\ \text{linear interpoliert} & \text{dazwischen} \end{cases}$$

**Typische Werte**: $\omega_{\text{pos}} = 2°$, $\omega_{\text{neg}} = 10°$, $\tau = 0.07$

#### Alternative: Geodesic Triplet Loss

$$\mathcal{L} = \max(0, \|z_a - z_p\|_2 - \|z_a - z_n\|_2 + m(\Delta\omega))$$

mit orientierungsabhängiger Margin $m(\Delta\omega)$, die mit der Misorientation wächst.

### 4.2 Augmentation Pipeline

Dies ist der **kritischste Teil** des gesamten Systems. Die Augmentierungen bestimmen, gegen welche experimentellen Variationen das Modell robust wird.

```
Simuliertes Pattern
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              AUGMENTATION PIPELINE                    │
│                                                       │
│  1. PC-Perturbation (Abschnitt 5)                    │
│     PC_x += N(0, σ_x),  σ_x ∈ [0.005, 0.02]        │
│     PC_y += N(0, σ_y),  σ_y ∈ [0.005, 0.02]        │
│     PC_z += N(0, σ_z),  σ_z ∈ [0.005, 0.02]        │
│                                                       │
│  2. Poisson-Rauschen (Kamera-Statistik)              │
│     I_noisy = Poisson(I_clean × gain) / gain          │
│     gain ∈ [50, 500] (simuliert versch. Belichtung)  │
│                                                       │
│  3. Gauß-Rauschen (Elektronik)                       │
│     I += N(0, σ_gauss),  σ_gauss ∈ [0.01, 0.1]     │
│                                                       │
│  4. Helligkeitsgradient (Hintergrund)                │
│     I += a·x + b·y + c  (lineare Hintergrundrampe)  │
│     a, b ~ U(-0.3, 0.3)                             │
│                                                       │
│  5. Gauss-Blurring (Strain-Simulation, Abschnitt 7) │
│     Kernel σ ∈ [0, 2.0] px                           │
│                                                       │
│  6. Kontrast-Variation                               │
│     I = α·I + β,  α ∈ [0.7, 1.3], β ∈ [-0.1, 0.1] │
│                                                       │
│  7. Partielle Verdeckung (Dead Pixels, Schatten)     │
│     Random Rectangles/Circles ausblenden (5–15% Area)│
│                                                       │
│  8. Bandpass-Filter-Variation                        │
│     (Simuliert unterschiedliche Preprocessing-Stufen) │
│                                                       │
│  9. Normalisierung                                   │
│     Zero-Mean, Unit-Variance pro Pattern             │
└─────────────────────────────────────────────────────┘
```

### 4.3 Training Hyperparameter

```yaml
# Training Configuration
optimizer: AdamW
learning_rate: 3e-4
weight_decay: 1e-4
scheduler: CosineAnnealingWarmRestarts
  T_0: 10 epochs
  T_mult: 2
batch_size: 256          # ~4 GB VRAM bei 60×60 Input
epochs: 100–200
embedding_dim: 128
temperature: 0.07        # für Contrastive Loss

# Hardware
gpu: RTX 4070 (8 GB VRAM)
mixed_precision: True    # FP16 Training für 2x Speedup
num_workers: 8           # DataLoader

# Daten
patterns_per_phase: ~1.000.000
augmentations_per_pattern: 5–10 (on-the-fly)
```

### 4.4 Training-Strategie

#### Phase 1: Single-Phase Pre-Training (je ~2–4h auf RTX 4070)
- Trainiere einen Encoder pro Phase
- Verifiziere, dass der Embedding-Raum sinnvolle Orientierungstopologie hat
- Evaluiere: Winkelgenauigkeit auf Validierungsdaten

#### Phase 2: Multi-Phase Joint Training
- Gemeinsamer Encoder für alle Phasen
- Patterns verschiedener Phasen müssen im Embedding-Raum **getrennte Cluster** bilden
- Zusätzlicher **Phase-Classification Head** (optional, siehe Abschnitt 6)

#### Phase 3: Fine-Tuning auf experimentellen Daten (optional)
- Wenn gelabelte experimentelle Daten vorhanden (z.B. aus konventionellem DI)
- Semi-supervised: nur wenige gelabelte experimentelle Patterns nötig
- Schließt die Domain-Gap zwischen Simulation und Experiment

---

## 5. Pattern-Center-Toleranz (PC-Robustheit)

### 5.1 Das Problem

Das Pattern Center (PC_x, PC_y, PC_z) definiert die gnomische Projektion:
- PC_x, PC_y: Position des Durchstoßpunkts auf dem Detektor
- PC_z: Abstand Probe–Detektor (in Detektorbreiten)

**Typische Unsicherheiten**:
- Gut kalibriert: ±0.005 (in Bruker-Konvention)
- Mäßig kalibriert: ±0.01–0.02
- Schlecht/unbekannt: ±0.03+

**Auswirkung**: Ein PC-Fehler von 0.01 verzerrt das Pattern geometrisch — Bänder verschieben sich, Zonensachen wandern. Bei DI/SI führt das direkt zu Fehlindizierungen.

### 5.2 Lösungsstrategie: Multi-PC Training

#### Ansatz A: PC als Augmentation (empfohlen als Start)

Während des Trainings wird jedes Pattern mit **zufällig perturbiertem PC** generiert:

```python
def augment_pc(pc_nominal, sigma_pc=0.015):
    """Perturbiere PC für robustes Training."""
    pc_x = pc_nominal[0] + np.random.normal(0, sigma_pc)
    pc_y = pc_nominal[1] + np.random.normal(0, sigma_pc)
    pc_z = pc_nominal[2] + np.random.normal(0, sigma_pc)
    return (pc_x, pc_y, pc_z)
```

**Effekt**: Der Encoder lernt, Orientierungsinformation zu extrahieren, die **invariant** gegenüber kleinen PC-Variationen ist. Er lernt implizit, die geometrische Verzerrung zu ignorieren.

**Grenze**: Bei sehr großen PC-Fehlern (>0.03) reicht das nicht — die Patterns sehen fundamental anders aus.

#### Ansatz B: PC als Zusatz-Input (für höhere Genauigkeit)

```
Input: (Pattern, PC) → Encoder → Embedding

Konkret:
  Pattern → CNN → Feature-Vektor (512-dim)
  PC      → MLP(3 → 64 → 128)  → PC-Feature (128-dim)
  
  Concat(Feature, PC-Feature) → MLP(640 → 256 → 128) → Embedding
```

**Vorteil**: Das Modell kann die PC-Information nutzen, um die Projektion intern zu "entzerren"
**Nachteil**: PC muss zur Inferenzzeit bekannt sein (ist es meistens, zumindest approximativ)

#### Ansatz C: PC-Estimation als Hilfsaufgabe (Multi-Task)

```
Pattern → CNN-Backbone → Feature-Vektor
                            │
                    ┌───────┴────────┐
                    │                │
              Orientation       PC Estimation
              Embedding         Head (→ PC_x, PC_y, PC_z)
              (Hauptaufgabe)    (Hilfsaufgabe)
```

**Vorteil**: Das Netz lernt explizit, den PC zu schätzen → kann zur Nachkalibrierung genutzt werden
**Multi-Task Loss**: $\mathcal{L} = \mathcal{L}_{\text{contrastive}} + \lambda \cdot \mathcal{L}_{\text{PC}}$, mit $\lambda = 0.1$

### 5.3 Empfehlung

**Start mit Ansatz A** (PC-Augmentation mit σ=0.015). Wenn die Genauigkeit nicht reicht, auf **Ansatz C** (Multi-Task mit PC-Estimation) erweitern. Ansatz B nur wenn PC wirklich gut bekannt ist und maximale Genauigkeit gebraucht wird.

### 5.4 Validierung der PC-Robustheit

Teste systematisch mit kontrolliert falschen PCs:

```
PC-Fehler:  0.000  0.005  0.010  0.015  0.020  0.030  0.050
            ──────────────────────────────────────────────────
Winkel-     < 0.5° < 0.5° < 1.0° < 1.5° < 2.0°  ???   ???
fehler:     (Ziel)
```

---

## 6. Multi-Phase-Indexierung

### 6.1 Das Problem

In einer realen EBSD-Messung können an jedem Punkt verschiedene Phasen vorliegen:
- Phase muss **identifiziert** werden
- Orientierung muss **innerhalb der Phasen-Symmetrie** bestimmt werden
- Phasen mit ähnlicher Struktur (z.B. zwei FCC-Phasen mit unterschiedlicher Gitterkonstante) sind schwer zu trennen

### 6.2 Architektur-Optionen

#### Option 1: Unified Embedding Space (empfohlen)

Alle Phasen teilen sich einen Embedding-Raum, aber bilden **getrennte Cluster**:

```
Pattern → Gemeinsamer Encoder → Embedding ∈ ℝ^128
                                     │
                              ┌──────┴──────┐
                              │             │
                         FAISS-Index    Phase-Cluster
                         (alle Phasen)  Zuordnung
```

**Training**: Contrastive Loss, wobei Patterns verschiedener Phasen immer als Negative behandelt werden, egal wie ähnlich die Orientierung ist.

**Vorteile**:
- Ein Encoder, ein FAISS-Index
- Phase-ID ergibt sich automatisch aus dem nächsten Nachbarn
- Einfach zu implementieren und zu skalieren

**Nachteile**:
- Bei sehr ähnlichen Phasen (z.B. BCC Fe vs. BCC Cr) kann Trennung schwierig sein
- Embedding-Kapazität muss für alle Phasen reichen

#### Option 2: Hierarchisch (Phase-Classifier + Per-Phase Encoder)

```
Pattern → Phase-Classifier → Phase-ID
               │
               ▼
          Phase-spezifischer Encoder → Phase-spezifisches Embedding
               │
               ▼
          Phase-spezifischer FAISS-Index → Orientierung
```

**Vorteile**:
- Jeder Encoder ist spezialisiert
- Besser für schwer trennbare Phasen

**Nachteile**:
- Mehr Modelle zu trainieren und zu pflegen
- Phase-Classifier-Fehler propagieren
- Kein "Zweifel" zwischen Phasen möglich

#### Option 3: Hybrid mit Confidence

```
Pattern → Gemeinsamer Encoder → Embedding
                                    │
                              ┌─────┴─────┐
                              │           │
                         FAISS Query  FAISS Query
                         (Phase A)   (Phase B)
                              │           │
                              ▼           ▼
                         Score_A      Score_B
                              │           │
                              └─────┬─────┘
                                    │
                              Confidence:
                              max(S_A, S_B) / (S_A + S_B)
```

**Empfehlung**: **Option 1 (Unified)** für den Start, mit Confidence-basierter Phase-Zuordnung. Wenn spezifische Phasenpaare Probleme machen, Option 3 für diese Paare.

### 6.3 Schwierige Phasen-Szenarien

| Szenario | Problem | Lösung |
|----------|---------|--------|
| FCC α-Al vs. FCC Si | Ähnliche Symmetrie, unterschiedliche Gitterkonstante → sehr ähnliche Patterns | Zusätzlichen EDS-Input integrieren (Dein multimodales Konzept aus ebsd-ai!) |
| Sehr feine Ausscheidungen | Pattern ist Überlagerung von Matrix + Ausscheidung | Confidence-Score niedrig → flaggen, nicht falsch indizieren |
| Unbekannte Phase | Phase nicht im Dictionary | Embedding liegt weit von allen Clustern → Anomalie-Erkennung |
| Amorphe/nano-kristalline Bereiche | Kein definiertes Pattern | Spezielles Training mit "Klasse: nicht-indizierbar" |

### 6.4 Integration von EDS-Daten (Multimodal)

Für schwer trennbare Phasen kann **chemische Information** den Unterschied machen:

```
Kikuchi-Pattern → CNN → Pattern-Feature (256-dim)
                                │
                                ├── Concat
                                │
EDS-Spektrum/Counts → MLP → Chemistry-Feature (64-dim)
                                │
                                ▼
                        Fusion MLP → Embedding (128-dim)
```

Dies entspricht konzeptionell dem Ansatz aus Deinem ebsd-ai Paket und würde hier direkt Synergien schaffen.

---

## 7. Strain-Toleranz (Verspannte Kristalle)

### 7.1 Das Problem

Elastische Gitterverzerrung bewirkt:
- **Bandverbreiterung**: Kikuchi-Bänder werden diffuser
- **Bandverschiebung**: Leichte Verschiebung der Bandpositionen
- **Asymmetrische Profile**: Bänder werden auf einer Seite schärfer
- **Intensitätsvariation**: Lokale Intensitätsänderungen

**Größenordnung**:
- Typische elastische Dehnung in Metallen: ε ≈ 0.001–0.01
- Resultierende Bandverbreiterung: ~0.05–0.5° (je nach Beugungsordnung)
- Bei hoher Spannung (nahe Fließgrenze): ε ≈ 0.01–0.05

### 7.2 Simulation von Strain-Effekten

#### Methode A: Geometrische Approximation (einfach, schnell)

Strain verbreitert Bänder primär durch geometrisches Blurring:

```python
def simulate_strain_blur(pattern, strain_level):
    """
    Approximiere Strain-Effekt durch anisotropes Gauss-Blurring.
    strain_level: 0.0 (kein Strain) bis 1.0 (stark verformt)
    """
    sigma = strain_level * 2.0  # Pixel, anpassbar
    return gaussian_filter(pattern, sigma=sigma)
```

**Limitierung**: Nur isotrope Verbreiterung, keine richtungsabhängigen Effekte

#### Methode B: Richtungsabhängige Verbreiterung (realistischer)

Strain ist ein Tensor — die Verbreiterung hängt von der Richtung des Beugungsvektors relativ zur Dehnungsrichtung ab:

```python
def simulate_anisotropic_strain(pattern, strain_tensor):
    """
    Anisotrope Bandverbreiterung basierend auf dem Strain-Tensor.
    Vereinfachung: Elliptisches Blurring entlang der Hauptdehnungsrichtungen.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(strain_tensor)
    # Blurring-Kernel entlang der Hauptachsen skaliert
    sigma_1 = abs(eigenvalues[0]) * scale_factor
    sigma_2 = abs(eigenvalues[1]) * scale_factor
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
    # Anisotropes Gauss-Filter anwenden
    kernel = anisotropic_gaussian_kernel(sigma_1, sigma_2, angle)
    return convolve2d(pattern, kernel)
```

#### Methode C: Physikalische Simulation mit Versetzungsdichte (gold standard)

Berücksichtigung der tatsächlichen Versetzungsstruktur:
- GND (Geometrisch Notwendige Versetzungen) → Gitterkrümmung → Bandshift
- SSD (Statistisch Gespeicherte Versetzungen) → Mosaikstruktur → Bandverbreiterung

**Implementierung**: Modifizierte dynamische Simulation mit perturbiertem Kristallgitter
**Problem**: Extrem rechenintensiv, nicht praktikabel für Millionen Trainingsdaten

### 7.3 Trainings-Strategie für Strain-Robustheit

```yaml
# Strain Augmentation Config
strain_augmentation:
  enabled: true
  
  # Isotrope Verbreiterung (immer an)
  isotropic_blur:
    probability: 0.5         # 50% der Patterns bekommen Blur
    sigma_range: [0.0, 2.5]  # Pixel
  
  # Anisotrope Verbreiterung (optional, Phase 2)
  anisotropic_blur:
    probability: 0.3
    sigma_range: [0.0, 2.0]
    angle_range: [0, 180]    # Zufällige Orientierung der Anisotropie
  
  # Intensitäts-Modulation (Strain-induziert)
  intensity_modulation:
    probability: 0.3
    amplitude: 0.1           # Relative Intensitätsvariation
```

### 7.4 Strain-Quantifizierung als Zusatzaufgabe

Das Modell kann nicht nur trotz Strain korrekt indexieren, sondern den **Grad der Verzerrung** auch quantifizieren:

```
Pattern → CNN → Feature-Vektor
                     │
             ┌───────┴────────┐
             │                │
        Orientation      Strain Indicator
        Embedding        Head
        (128-dim)        (→ σ_strain oder
                          qualitativ: low/med/high)
```

**Nutzen**: 
- Map mit "Strain-Confidence" als Quality-Metrik
- Korrelation mit KAM (Kernel Average Misorientation)
- Identifikation von stark verformten Bereichen

### 7.5 Bekannte Limitierungen bei hohem Strain

| Strain-Level | Erwartetes Verhalten |
|--------------|---------------------|
| ε < 0.005 | Kein messbarer Effekt auf Indexierung |
| 0.005 < ε < 0.02 | Leichte Verschlechterung, Augmentation gleicht aus |
| 0.02 < ε < 0.05 | Merkbare Bandverbreiterung, ~0.5–2° Genauigkeitsverlust möglich |
| ε > 0.05 | Patterns stark degradiert, EBSD generell problematisch |
| Nano-kristallin / hochverformt | Patterns nicht mehr indizierbar → Anomalie-Erkennung nötig |

---

## 8. Schlechte Pattern-Qualität

### 8.1 Ursachen schlechter Patterns

1. **Niedrige Beschleunigungsspannung** → weniger Kikuchi-Kontrast
2. **Schlechte Probenpräparation** → Oberflächenschäden, amorphe Schicht
3. **Hoher Strahlstrom bei kleiner Schrittweite** → Aufladung, Drift
4. **Korngrenzen** → Überlappung zweier Orientierungen
5. **Nanostruktur** → Diffuse Patterns
6. **Oxidation/Kontamination** → Signalverlust

### 8.2 Quality-Aware Training

#### Confidence-Estimation Head

```
Pattern → CNN → Feature
                  │
          ┌───────┴────────┐
          │                │
     Orientation      Confidence
     Embedding        Score (0–1)
     (128-dim)        
```

**Training des Confidence-Heads**:
- Rausch-freie Patterns → Confidence nahe 1.0
- Stark augmentierte/degradierte Patterns → Confidence nahe 0.0
- Loss: $\mathcal{L}_{\text{conf}} = \text{BCE}(\hat{c}, c_{\text{target}})$

Wobei $c_{\text{target}}$ geschätzt wird als:

$$c_{\text{target}} = \exp\left(-\frac{\Delta\omega_{\text{pred}}^2}{2\sigma_c^2}\right)$$

(d.h., Confidence sinkt wenn die Vorhersage fehlerhaft wird)

#### Degradation-Aware Augmentation

```python
class DegradationPipeline:
    """Simuliert verschiedene Stufen der Pattern-Degradation."""
    
    def __call__(self, pattern, degradation_level):
        # Level 0: Perfekt (nur leichtes Rauschen)
        # Level 1: Moderate Qualität
        # Level 2: Schlechte Qualität
        # Level 3: Sehr schlecht / kaum indizierbar
        
        if degradation_level >= 1:
            pattern = self.add_poisson_noise(pattern, gain=100)
            pattern = self.add_background_gradient(pattern)
        
        if degradation_level >= 2:
            pattern = self.add_gaussian_blur(pattern, sigma=1.5)
            pattern = self.reduce_contrast(pattern, factor=0.5)
            pattern = self.add_hot_pixels(pattern, density=0.01)
        
        if degradation_level >= 3:
            pattern = self.heavy_blur(pattern, sigma=3.0)
            pattern = self.add_structured_noise(pattern)
            pattern = self.partial_occlusion(pattern, ratio=0.2)
        
        return pattern
```

### 8.3 Grain-Boundary-Patterns (Überlappung)

An Korngrenzen überlagern sich Patterns von zwei (oder mehr) Orientierungen:

$$P_{\text{boundary}} = \alpha \cdot P_{\text{grain1}} + (1-\alpha) \cdot P_{\text{grain2}}, \quad \alpha \in [0, 1]$$

**Strategie 1: Erkennung (binär)**
- Trainiere einen Classifier: "Ist dieses Pattern eine Überlappung?"
- Training: Synthetische Überlagerungen mit verschiedenen α

**Strategie 2: Dekomposition (fortgeschritten)**
- Netzwerk gibt Top-2 Embeddings aus
- Zuordnung über FAISS zu zwei Orientierungen
- Zusätzlicher α-Output für das Mischungsverhältnis

**Strategie 3: Pragmatisch (empfohlen für Start)**
- Überlappungsmuster haben niedrige Confidence → Flaggen
- Keine explizite Dekomposition, sondern Ausschluss aus der Indexierung
- Korrelation mit den Nachbarpixeln für nachträgliche Zuordnung

---

## 9. FAISS-Index und Retrieval

### 9.1 Index-Aufbau

```python
import faiss

# Konfiguration
d = 128                          # Embedding-Dimension
n_orientations = 3_500_000       # Pro Phase
n_phases = 5                     # Typisch für Al-Legierungen
n_total = n_orientations * n_phases  # ~17.5 Mio. Einträge

# Index-Typ wählen
# Option A: Flat (exakt, aber langsam bei >1M Einträgen)
index = faiss.IndexFlatIP(d)     # Inner Product für cosine similarity

# Option B: IVF + PQ (empfohlen für Produktion)
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFPQ(
    quantizer,
    d,                           # Dimension
    n_lists=4096,                # Anzahl Voronoi-Zellen
    m=16,                        # PQ Subquantizer
    nbits=8                      # Bits pro Subquantizer
)

# Training des Index (mit einem Subset der Embeddings)
index.train(training_embeddings)  # ~100k–500k Samples

# Alle Embeddings einfügen
index.add(all_embeddings)

# GPU-beschleunigt (RTX 4070)
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
```

### 9.2 Retrieval-Geschwindigkeit (geschätzt)

| Index-Typ | Einträge | Query-Zeit (single) | Batch (1000) | Recall@1 |
|-----------|----------|---------------------|-------------|----------|
| Flat (exakt) | 3.5M | ~5 ms | ~200 ms | 100% |
| IVF4096,PQ16 | 3.5M | ~0.1 ms | ~5 ms | ~95–98% |
| IVF4096,PQ16 (GPU) | 3.5M | ~0.01 ms | ~0.5 ms | ~95–98% |
| HNSW | 3.5M | ~0.05 ms | ~3 ms | ~98–99% |

**Für eine typische EBSD-Map (500×500 = 250.000 Pixel)**:
- Flat (CPU): ~20 min (zu langsam)
- IVF+PQ (CPU): ~25 Sekunden
- IVF+PQ (GPU): ~2.5 Sekunden
- HNSW (CPU): ~12 Sekunden

**Plus CNN-Inferenz** (~0.01 ms/Pattern auf GPU mit Batching):
- 250.000 Patterns: ~2.5 Sekunden

**Gesamtzeit für 250k-Map**: ~5–30 Sekunden (je nach Index-Typ)

### 9.3 Index-Metadaten

Jeder Eintrag im Index braucht zugeordnete Metadaten:

```python
# Metadaten-Array (parallel zum FAISS-Index)
metadata = {
    'orientations': np.array([...]),  # (N, 4) Quaternionen
    'phases': np.array([...]),         # (N,) Phase-IDs
    'euler_angles': np.array([...]),   # (N, 3) Optional, für schnelle Ausgabe
}
```

---

## 10. Lokale Verfeinerung (Post-Retrieval)

### 10.1 Warum Verfeinerung?

FAISS liefert die nächste Orientierung im **diskreten** Grid. Für Sub-Grid-Genauigkeit (<0.5°) braucht es einen zusätzlichen Schritt.

### 10.2 Methoden

#### Methode A: Gewichtete Interpolation der Top-k Nachbarn

```python
def refine_orientation(query_embedding, top_k_embeddings, top_k_orientations, top_k_scores):
    """
    Interpoliere die finale Orientierung aus den Top-k FAISS-Ergebnissen.
    """
    # Scores als Gewichte (softmax-normalisiert)
    weights = softmax(top_k_scores / temperature)
    
    # Quaternion-Interpolation (gewichtet)
    q_refined = quaternion_weighted_average(top_k_orientations, weights)
    
    return q_refined
```

**Problem**: Quaternion-Mittelung ist nicht trivial auf SO(3) — braucht iterative Methoden (z.B. Markley et al., 2007).

#### Methode B: Lokale Optimierung (Nelder-Mead / L-BFGS)

```python
def refine_local(pattern, initial_orientation, master_pattern, pc):
    """
    Lokale Optimierung des NCC (Normalized Cross Correlation)
    im Orientierungsraum um die FAISS-initialisierte Orientierung.
    """
    def objective(params):
        # params: 3 Euler-Winkel (oder Rodrigues)
        orientation = euler_to_rotation(params)
        simulated = simulate_pattern(master_pattern, orientation, pc)
        return -normalized_dot_product(pattern, simulated)
    
    result = minimize(
        objective,
        x0=orientation_to_euler(initial_orientation),
        method='Nelder-Mead',
        options={'xatol': 0.01, 'fatol': 1e-6}  # ~0.01° Genauigkeit
    )
    return euler_to_rotation(result.x)
```

**Geschwindigkeit**: ~1–5 ms pro Pattern (akzeptabel als optionaler Schritt)

#### Methode C: Learned Refinement (Neural)

Trainiere ein kleines MLP, das aus dem FAISS-Top-k-Ergebnis eine Korrektur vorhersagt:

```
(query_embedding, top5_embeddings, top5_scores) → MLP → Δq (Quaternion-Korrektur)

q_final = q_coarse ⊗ Δq   (Quaternion-Komposition)
```

**Vorteil**: Sehr schnell (~0.01 ms), lernbar
**Nachteil**: Braucht eigenes Training, weniger interpretierbar

### 10.3 Empfehlung

**Methode A** für die meisten Fälle (schnell, einfach, ~0.3° Verbesserung). **Methode B** optional für Fälle, wo Sub-0.5° Genauigkeit kritisch ist.

---

## 11. Evaluierung und Benchmarking

### 11.1 Metriken

| Metrik | Beschreibung | Zielwert |
|--------|-------------|----------|
| Mean Angular Error (MAE) | Mittlere Misorientation zur Ground-Truth | < 1.0° |
| Median Angular Error | Robuster als MAE | < 0.5° |
| 95th Percentile Error | Worst-Case-Abschätzung | < 3.0° |
| Phase-ID Accuracy | Korrekte Phasenidentifikation | > 98% |
| Indexing Rate | Anteil erfolgreich indizierter Patterns | > 95% |
| Throughput | Patterns/Sekunde | > 10.000 |
| Confidence Calibration | Korrelation Confidence ↔ tatsächlicher Fehler | Monoton |

### 11.2 Benchmarking-Protokoll

```
Test-Datensätze:
1. Simuliert, perfekt    → Baseline-Genauigkeit
2. Simuliert + Rauschen  → Robustheit
3. Simuliert + PC-Fehler → PC-Toleranz
4. Simuliert + Strain    → Strain-Robustheit
5. Experimentell (Si)    → Real-World Single-Phase
6. Experimentell (Al-Cu) → Real-World Multi-Phase
7. Experimentell (stark verformt) → Extrembedingung

Vergleich gegen:
- Hough-basiertes Indexing (EDAX OIM / Oxford AZtec)
- Dictionary Indexing (EMsoft / kikuchipy)
- Spherical Indexing (EMSphInx)
```

### 11.3 Ablationsstudien

Was passiert wenn man einzelne Komponenten weglässt?

1. Ohne PC-Augmentation → Wie stark leidet PC-Robustheit?
2. Ohne Strain-Augmentation → Genauigkeit auf verformten Proben?
3. Ohne Confidence-Head → Kann man schlechte Patterns noch erkennen?
4. Embedding-Dimension: 64 vs. 128 vs. 256
5. Encoder-Architektur: ResNet-18 vs. EfficientNet vs. MobileNet
6. FAISS-Index-Typ: Flat vs. IVF+PQ vs. HNSW
7. Mit vs. ohne lokale Verfeinerung

---

## 12. Implementierungsplan (Meilensteine)

### Phase 1: Foundation (2–3 Wochen)

```
□ Master-Pattern-Generierung für α-Al (FCC) mit EMsoft/kikuchipy
□ Orientierungs-Sampling (cubochoric, N=100)
□ Pattern-Generierung Pipeline (Master → Detektor-Pattern)
□ Basis-Augmentation (Rauschen, Kontrast, Hintergrund)
□ ResNet-18 Encoder implementieren
□ Contrastive Loss (SO(3)-aware) implementieren
□ Single-Phase Training Loop
□ Erste Evaluierung: MAE auf simulierten Daten
```

### Phase 2: Robustheit (2–3 Wochen)

```
□ PC-Augmentation implementieren und testen
□ Strain-Augmentation (Gauss-Blur) implementieren
□ Confidence-Head hinzufügen
□ Degradation-Pipeline (verschiedene Qualitätsstufen)
□ FAISS-Index aufbauen und testen
□ Lokale Verfeinerung (Methode A: gewichtete Interpolation)
□ Systematische Ablation: PC-Robustheit, Strain-Robustheit
```

### Phase 3: Multi-Phase (2–3 Wochen)

```
□ Master Patterns für weitere Phasen (Al₂Cu, Mg₂Si, Si, ...)
□ Unified Embedding Space Training (alle Phasen)
□ Phase-Separation im Embedding-Raum verifizieren
□ Multi-Phase FAISS-Index
□ Phase-ID Accuracy evaluieren
□ Optional: EDS-Integration (multimodal)
```

### Phase 4: Validation & Integration (2–3 Wochen)

```
□ Benchmarking gegen DI, SI, Hough
□ Test auf experimentellen Daten
□ Integration in kikuchipy_GUI als Indexierungs-Backend
□ Performance-Optimierung (Batch-Inferenz, GPU-FAISS)
□ Dokumentation und API-Design
□ Paper-Outline / Datenvisualisierungen
```

### Geschätzter Gesamtzeitraum: 8–12 Wochen

---

## 13. Risiken und Mitigationen

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Sim-to-Real Gap zu groß | Mittel | Hoch | Fine-Tuning auf exp. Daten, Bridge-Augmentation |
| PC-Robustheit nicht ausreichend | Niedrig-Mittel | Hoch | Multi-Task PC-Estimation (Ansatz C) |
| Phasen-Verwechslung | Mittel | Mittel | EDS-Integration, hierarchisches Indexing |
| VRAM reicht nicht für Training | Niedrig | Mittel | Gradient Accumulation, kleineres Batch |
| FAISS-Recall zu niedrig | Niedrig | Mittel | HNSW statt IVF+PQ, oder dichteren Index |
| Methode nicht besser als SI | Niedrig | Hoch | Genauigkeit/Speed-Tradeoff klar kommunizieren |
| Orientierungs-Topologie im Embedding nicht korrekt | Mittel | Hoch | Ablation des Losses, Visualisierung via UMAP |

---

## 14. Technische Abhängigkeiten

```
Python >= 3.10
PyTorch >= 2.0 (CUDA 12.x)
faiss-gpu >= 1.7.4
kikuchipy >= 0.9
diffsims >= 0.6
orix >= 0.12
numpy, scipy, matplotlib
einops (optional, für Tensor-Operationen)
wandb oder tensorboard (Logging)
```

---

## 15. Potenzial für Publikation

Dieser Ansatz wäre publishable in:
- **Ultramicroscopy** (methodisch, Vergleich mit DI/SI)
- **Acta Materialia** (wenn mit Materials-Science-Anwendung kombiniert)
- **npj Computational Materials** (wenn ML-Aspekt stark betont)

**USPs gegenüber existierender Literatur**:
1. Erste systematische Analyse der PC-Robustheit von Embedding-basiertem Indexing
2. SO(3)-aware Contrastive Learning (vs. naive Classification)
3. Multi-Phase in einem unified Embedding Space
4. Strain-Quantifizierung als Nebenprodukt
5. Integration von EDS (multimodal) für schwer trennbare Phasen
6. Open-Source-Integration in kikuchipy-Ökosystem

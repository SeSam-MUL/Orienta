# Option A: Learned Kikuchi-Pattern Embedding with FAISS Retrieval

## Detailed plan for an ML-based EBSD indexing system

---

## 1. Overview and Core Idea

### Core concept
A CNN encoder $f_\theta$ learns a mapping:

$$f_\theta: \text{Kikuchi pattern} \rightarrow \mathbb{R}^d \quad (d = 128\text{–}256)$$

such that patterns with similar orientations lie close together in the embedding space. For indexing, an experimental pattern is encoded and matched via Approximate Nearest Neighbor (ANN) against a precomputed database of reference embeddings.

### Expected advantages over DI and SI
- **Speed**: ~1000x faster than DI, ~10–50x faster than SI at inference
- **Robustness**: Intrinsically tolerant to noise, PC errors, and strain thanks to training with augmented data
- **Scalability**: The FAISS index scales sublinearly with the number of orientations
- **Flexibility**: Extensible to multi-phase, strain quantification, and confidence estimation

---

## 2. Architecture of the Overall System

### 2.1 System components

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                            │
│                                                             │
│  Dynamical Simulation ──► Augmentation ──► CNN-Encoder      │
│  (EMsoft/kikuchipy)       Pipeline          Training         │
│                                                             │
│  Output: Trained model f_θ                                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 DICTIONARY BUILD PHASE                       │
│                                                             │
│  Orientation grid ──► Pattern simulation ──► f_θ ──► FAISS │
│  (cubochoric/SO(3))    (Master Pattern)               Index │
│                                                             │
│  Output: FAISS index with (embedding, orientation) pairs    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE                           │
│                                                             │
│  Exp. Pattern ──► Preprocessing ──► f_θ ──► FAISS Query     │
│                                             │               │
│                                             ▼               │
│                                     Top-k orientations      │
│                                             │               │
│                                             ▼               │
│                                     Local refinement        │
│                                     (optional)              │
│                                             │               │
│                                             ▼               │
│                                     Final orientation       │
│                                     + Confidence Score      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 CNN Encoder Architecture

#### Base architecture: Modified ResNet-18

```
Input: (1, H, W) — Grayscale Kikuchi pattern (e.g. 60×60 or 120×120 px)

├── Conv2d(1, 64, 7×7, stride=2, padding=3) + BN + ReLU
├── MaxPool(3×3, stride=2)
├── ResBlock(64, 64) × 2
├── ResBlock(64, 128, stride=2) × 2
├── ResBlock(128, 256, stride=2) × 2
├── ResBlock(256, 512, stride=2) × 2
├── AdaptiveAvgPool2d(1,1)
├── Flatten
├── Linear(512, d)           # Projection Head
├── L2 normalization         # Embedding lies on the unit sphere S^(d-1)

Output: Embedding ∈ S^(d-1) ⊂ ℝ^d
```

**Why ResNet-18 and not deeper?**
- Kikuchi patterns are relatively structured (band patterns) — too deep a network overfits
- A 60×60 px input does not need the capacity of a ResNet-50
- The RTX 4070 (8 GB VRAM) limits the batch size for larger networks
- Faster inference = faster indexing of large maps

**Alternative: EfficientNet-B0 or MobileNetV3**
- Even faster at inference
- Useful when speed is the top priority
- Slightly less capacity for fine distinctions between similar orientations

#### Embedding dimension d

| d | Advantages | Disadvantages |
|---|----------|-----------|
| 64 | Very fast, small index | Possibly too little capacity for many phases |
| 128 | Good compromise | Standard choice |
| 256 | More capacity for multi-phase | Larger index, slower |
| 512 | Maximum separability | Diminishing returns, FAISS slower |

**Recommendation**: Start with d=128, then ablate.

---

## 3. Training Data and Simulation

### 3.1 Master Pattern Generation

A **master pattern** (dynamical simulation) is required for each phase:

- **Tool**: EMsoft (`EMEBSDmaster`) or kikuchipy with diffsims
- **Parameters**:
  - Accelerating voltage: typically 10–30 kV (phase-dependent optimum)
  - Maximum diffraction order: high enough for realistic patterns
  - Energy bins: for energy-weighted patterns

**Phase coverage** (typical for Al alloys):
- α-Al (FCC, Fm-3m)
- Al₂Cu (θ, I4/mcm)
- Al₃Fe (C2/m)
- Mg₂Si (Fm-3m, different lattice constant)
- Al₆Mn (Cmcm)
- Si (Fd-3m)
- Others depending on the alloy system

### 3.2 Orientation Sampling

#### Uniform sampling on SO(3)

Use of **cubochoric coordinates** (Rosca et al., 2014) for uniform sampling:

```
Resolution   | Number of orientations | Mean angular spacing
-------------|----------------------|------------------------
N_cubochoric = 50  | ~130,000            | ~2.5°
N_cubochoric = 80  | ~530,000            | ~1.6°
N_cubochoric = 100 | ~1,030,000          | ~1.2°
N_cubochoric = 150 | ~3,500,000          | ~0.8°
```

**Recommendation**:
- Training: N=100 (~1 million orientations per phase)
- Inference dictionary: N=150 (~3.5 million per phase) or denser

#### Symmetry reduction
- Sample only the **fundamental zone** of the respective crystal symmetry
- For cubic (Oh): reduction by a factor of 24 → ~43,000 independent orientations at N=100
- For lower symmetries: less reduction, more orientations needed

### 3.3 Pattern Generation from Master Patterns

For each orientation g ∈ SO(3):
1. Rotate the master pattern by g
2. Gnomonic projection with given PC coordinates (PC_x, PC_y, PC_z)
3. Binning to the target resolution (e.g. 60×60)

**Important**: The projection depends on (PC_x, PC_y, PC_z) — see Section 5 for PC tolerance.

---

## 4. Training

### 4.1 Loss Function: Contrastive Learning on SO(3)

#### Core problem
Standard contrastive learning (e.g. SimCLR) treats all negative pairs equally. But on SO(3) there are **gradations of similarity** — an orientation that is 2° away is "more similar" than one that is 40° away.

#### Solution: Angular-Margin-Aware Contrastive Loss

Define the **misorientation** between two orientations:

$$\Delta\omega(g_1, g_2) = \min_{s \in S} \arccos\left(\frac{\text{tr}(s \cdot g_1^{-1} \cdot g_2) - 1}{2}\right)$$

where S is the symmetry group of the crystal.

**Soft Contrastive Loss**:

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_{k} w_k \cdot \exp(\text{sim}(z_i, z_k) / \tau)}$$

with weights $w_k$ based on the misorientation angle:

$$w_k = \begin{cases} 0 & \text{if } \Delta\omega < \omega_{\text{pos}} \quad (\text{too similar, no negative push}) \\ 1 & \text{if } \Delta\omega > \omega_{\text{neg}} \quad (\text{clearly negative}) \\ \text{linearly interpolated} & \text{in between} \end{cases}$$

**Typical values**: $\omega_{\text{pos}} = 2°$, $\omega_{\text{neg}} = 10°$, $\tau = 0.07$

#### Alternative: Geodesic Triplet Loss

$$\mathcal{L} = \max(0, \|z_a - z_p\|_2 - \|z_a - z_n\|_2 + m(\Delta\omega))$$

with an orientation-dependent margin $m(\Delta\omega)$ that grows with the misorientation.

### 4.2 Augmentation Pipeline

This is the **most critical part** of the entire system. The augmentations determine which experimental variations the model becomes robust to.

```
Simulated pattern
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              AUGMENTATION PIPELINE                    │
│                                                       │
│  1. PC perturbation (Section 5)                      │
│     PC_x += N(0, σ_x),  σ_x ∈ [0.005, 0.02]        │
│     PC_y += N(0, σ_y),  σ_y ∈ [0.005, 0.02]        │
│     PC_z += N(0, σ_z),  σ_z ∈ [0.005, 0.02]        │
│                                                       │
│  2. Poisson noise (camera statistics)                │
│     I_noisy = Poisson(I_clean × gain) / gain          │
│     gain ∈ [50, 500] (simulates varying exposure)    │
│                                                       │
│  3. Gaussian noise (electronics)                     │
│     I += N(0, σ_gauss),  σ_gauss ∈ [0.01, 0.1]     │
│                                                       │
│  4. Brightness gradient (background)                 │
│     I += a·x + b·y + c  (linear background ramp)    │
│     a, b ~ U(-0.3, 0.3)                             │
│                                                       │
│  5. Gaussian blurring (strain simulation, Section 7)│
│     Kernel σ ∈ [0, 2.0] px                           │
│                                                       │
│  6. Contrast variation                               │
│     I = α·I + β,  α ∈ [0.7, 1.3], β ∈ [-0.1, 0.1] │
│                                                       │
│  7. Partial occlusion (dead pixels, shadows)         │
│     Mask out random rectangles/circles (5–15% area)  │
│                                                       │
│  8. Bandpass-filter variation                        │
│     (Simulates different preprocessing stages)        │
│                                                       │
│  9. Normalization                                    │
│     Zero-mean, unit-variance per pattern             │
└─────────────────────────────────────────────────────┘
```

### 4.3 Training Hyperparameters

```yaml
# Training Configuration
optimizer: AdamW
learning_rate: 3e-4
weight_decay: 1e-4
scheduler: CosineAnnealingWarmRestarts
  T_0: 10 epochs
  T_mult: 2
batch_size: 256          # ~4 GB VRAM at 60×60 input
epochs: 100–200
embedding_dim: 128
temperature: 0.07        # for contrastive loss

# Hardware
gpu: RTX 4070 (8 GB VRAM)
mixed_precision: True    # FP16 training for 2x speedup
num_workers: 8           # DataLoader

# Data
patterns_per_phase: ~1,000,000
augmentations_per_pattern: 5–10 (on-the-fly)
```

### 4.4 Training Strategy

#### Phase 1: Single-Phase Pre-Training (~2–4 h each on RTX 4070)
- Train one encoder per phase
- Verify that the embedding space has a meaningful orientation topology
- Evaluate: angular accuracy on validation data

#### Phase 2: Multi-Phase Joint Training
- Shared encoder for all phases
- Patterns of different phases must form **separate clusters** in the embedding space
- Additional **phase classification head** (optional, see Section 6)

#### Phase 3: Fine-Tuning on experimental data (optional)
- When labeled experimental data are available (e.g. from conventional DI)
- Semi-supervised: only a few labeled experimental patterns needed
- Closes the domain gap between simulation and experiment

---

## 5. Pattern Center Tolerance (PC Robustness)

### 5.1 The Problem

The pattern center (PC_x, PC_y, PC_z) defines the gnomonic projection:
- PC_x, PC_y: position of the pierce point on the detector
- PC_z: sample–detector distance (in detector widths)

**Typical uncertainties**:
- Well calibrated: ±0.005 (in Bruker convention)
- Moderately calibrated: ±0.01–0.02
- Poor/unknown: ±0.03+

**Effect**: A PC error of 0.01 geometrically distorts the pattern — bands shift, zone axes wander. In DI/SI this leads directly to misindexing.

### 5.2 Solution Strategy: Multi-PC Training

#### Approach A: PC as augmentation (recommended as a starting point)

During training, each pattern is generated with a **randomly perturbed PC**:

```python
def augment_pc(pc_nominal, sigma_pc=0.015):
    """Perturb PC for robust training."""
    pc_x = pc_nominal[0] + np.random.normal(0, sigma_pc)
    pc_y = pc_nominal[1] + np.random.normal(0, sigma_pc)
    pc_z = pc_nominal[2] + np.random.normal(0, sigma_pc)
    return (pc_x, pc_y, pc_z)
```

**Effect**: The encoder learns to extract orientation information that is **invariant** to small PC variations. It implicitly learns to ignore the geometric distortion.

**Limit**: For very large PC errors (>0.03) this is not enough — the patterns look fundamentally different.

#### Approach B: PC as additional input (for higher accuracy)

```
Input: (Pattern, PC) → Encoder → Embedding

Concretely:
  Pattern → CNN → feature vector (512-dim)
  PC      → MLP(3 → 64 → 128)  → PC feature (128-dim)
  
  Concat(feature, PC feature) → MLP(640 → 256 → 128) → Embedding
```

**Advantage**: The model can use the PC information to internally "de-distort" the projection
**Disadvantage**: The PC must be known at inference time (it usually is, at least approximately)

#### Approach C: PC estimation as an auxiliary task (multi-task)

```
Pattern → CNN backbone → feature vector
                            │
                    ┌───────┴────────┐
                    │                │
              Orientation       PC Estimation
              Embedding         Head (→ PC_x, PC_y, PC_z)
              (main task)       (auxiliary task)
```

**Advantage**: The network explicitly learns to estimate the PC → can be used for recalibration
**Multi-Task Loss**: $\mathcal{L} = \mathcal{L}_{\text{contrastive}} + \lambda \cdot \mathcal{L}_{\text{PC}}$, with $\lambda = 0.1$

### 5.3 Recommendation

**Start with Approach A** (PC augmentation with σ=0.015). If the accuracy is insufficient, extend to **Approach C** (multi-task with PC estimation). Use Approach B only when the PC is genuinely well known and maximum accuracy is required.

### 5.4 Validation of PC Robustness

Test systematically with controlled, deliberately wrong PCs:

```
PC error:   0.000  0.005  0.010  0.015  0.020  0.030  0.050
            ──────────────────────────────────────────────────
Angular     < 0.5° < 0.5° < 1.0° < 1.5° < 2.0°  ???   ???
error:      (target)
```

---

## 6. Multi-Phase Indexing

### 6.1 The Problem

In a real EBSD measurement, different phases may be present at each point:
- The phase must be **identified**
- The orientation must be determined **within the phase symmetry**
- Phases with similar structure (e.g. two FCC phases with different lattice constants) are hard to separate

### 6.2 Architecture Options

#### Option 1: Unified Embedding Space (recommended)

All phases share one embedding space but form **separate clusters**:

```
Pattern → Shared encoder → Embedding ∈ ℝ^128
                                     │
                              ┌──────┴──────┐
                              │             │
                         FAISS index    Phase-cluster
                         (all phases)   assignment
```

**Training**: Contrastive loss, where patterns of different phases are always treated as negatives, no matter how similar the orientation.

**Advantages**:
- One encoder, one FAISS index
- Phase ID follows automatically from the nearest neighbor
- Easy to implement and scale

**Disadvantages**:
- For very similar phases (e.g. BCC Fe vs. BCC Cr), separation can be difficult
- The embedding capacity must suffice for all phases

#### Option 2: Hierarchical (Phase classifier + per-phase encoder)

```
Pattern → Phase classifier → Phase ID
               │
               ▼
          Phase-specific encoder → Phase-specific embedding
               │
               ▼
          Phase-specific FAISS index → Orientation
```

**Advantages**:
- Each encoder is specialized
- Better for hard-to-separate phases

**Disadvantages**:
- More models to train and maintain
- Phase classifier errors propagate
- No "doubt" between phases possible

#### Option 3: Hybrid with confidence

```
Pattern → Shared encoder → Embedding
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

**Recommendation**: **Option 1 (Unified)** for the start, with confidence-based phase assignment. If specific phase pairs cause problems, use Option 3 for those pairs.

### 6.3 Difficult Phase Scenarios

| Scenario | Problem | Solution |
|----------|---------|--------|
| FCC α-Al vs. FCC Si | Similar symmetry, different lattice constant → very similar patterns | Integrate an additional EDS input (your multimodal concept from ebsd-ai!) |
| Very fine precipitates | Pattern is a superposition of matrix + precipitate | Low confidence score → flag, do not misindex |
| Unknown phase | Phase not in the dictionary | Embedding lies far from all clusters → anomaly detection |
| Amorphous/nano-crystalline regions | No defined pattern | Special training with "class: non-indexable" |

### 6.4 Integration of EDS Data (Multimodal)

For hard-to-separate phases, **chemical information** can make the difference:

```
Kikuchi pattern → CNN → pattern feature (256-dim)
                                │
                                ├── Concat
                                │
EDS spectrum/counts → MLP → chemistry feature (64-dim)
                                │
                                ▼
                        Fusion MLP → Embedding (128-dim)
```

Conceptually this corresponds to the approach from your ebsd-ai package and would create direct synergies here.

---

## 7. Strain Tolerance (Strained Crystals)

### 7.1 The Problem

Elastic lattice distortion causes:
- **Band broadening**: Kikuchi bands become more diffuse
- **Band shift**: Slight shift of the band positions
- **Asymmetric profiles**: Bands become sharper on one side
- **Intensity variation**: Local intensity changes

**Order of magnitude**:
- Typical elastic strain in metals: ε ≈ 0.001–0.01
- Resulting band broadening: ~0.05–0.5° (depending on the diffraction order)
- At high stress (near the yield point): ε ≈ 0.01–0.05

### 7.2 Simulation of Strain Effects

#### Method A: Geometric approximation (simple, fast)

Strain broadens bands primarily through geometric blurring:

```python
def simulate_strain_blur(pattern, strain_level):
    """
    Approximate the strain effect with anisotropic Gaussian blurring.
    strain_level: 0.0 (no strain) to 1.0 (heavily deformed)
    """
    sigma = strain_level * 2.0  # pixels, adjustable
    return gaussian_filter(pattern, sigma=sigma)
```

**Limitation**: Only isotropic broadening, no direction-dependent effects

#### Method B: Direction-dependent broadening (more realistic)

Strain is a tensor — the broadening depends on the direction of the diffraction vector relative to the strain direction:

```python
def simulate_anisotropic_strain(pattern, strain_tensor):
    """
    Anisotropic band broadening based on the strain tensor.
    Simplification: elliptical blurring along the principal strain directions.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(strain_tensor)
    # Blurring kernel scaled along the principal axes
    sigma_1 = abs(eigenvalues[0]) * scale_factor
    sigma_2 = abs(eigenvalues[1]) * scale_factor
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
    # Apply the anisotropic Gaussian filter
    kernel = anisotropic_gaussian_kernel(sigma_1, sigma_2, angle)
    return convolve2d(pattern, kernel)
```

#### Method C: Physical simulation with dislocation density (gold standard)

Accounting for the actual dislocation structure:
- GND (Geometrically Necessary Dislocations) → lattice curvature → band shift
- SSD (Statistically Stored Dislocations) → mosaic structure → band broadening

**Implementation**: Modified dynamical simulation with a perturbed crystal lattice
**Problem**: Extremely compute-intensive, not practical for millions of training samples

### 7.3 Training Strategy for Strain Robustness

```yaml
# Strain Augmentation Config
strain_augmentation:
  enabled: true
  
  # Isotropic broadening (always on)
  isotropic_blur:
    probability: 0.5         # 50% of the patterns get blur
    sigma_range: [0.0, 2.5]  # pixels
  
  # Anisotropic broadening (optional, Phase 2)
  anisotropic_blur:
    probability: 0.3
    sigma_range: [0.0, 2.0]
    angle_range: [0, 180]    # Random orientation of the anisotropy
  
  # Intensity modulation (strain-induced)
  intensity_modulation:
    probability: 0.3
    amplitude: 0.1           # Relative intensity variation
```

### 7.4 Strain Quantification as an Auxiliary Task

The model cannot only index correctly despite strain, but can also quantify the **degree of distortion**:

```
Pattern → CNN → feature vector
                     │
             ┌───────┴────────┐
             │                │
        Orientation      Strain Indicator
        Embedding        Head
        (128-dim)        (→ σ_strain or
                          qualitative: low/med/high)
```

**Benefit**:
- Map with "strain confidence" as a quality metric
- Correlation with KAM (Kernel Average Misorientation)
- Identification of heavily deformed regions

### 7.5 Known Limitations at High Strain

| Strain level | Expected behavior |
|--------------|---------------------|
| ε < 0.005 | No measurable effect on indexing |
| 0.005 < ε < 0.02 | Slight degradation, augmentation compensates |
| 0.02 < ε < 0.05 | Noticeable band broadening, ~0.5–2° accuracy loss possible |
| ε > 0.05 | Patterns severely degraded, EBSD generally problematic |
| Nano-crystalline / heavily deformed | Patterns no longer indexable → anomaly detection needed |

---

## 8. Poor Pattern Quality

### 8.1 Causes of poor patterns

1. **Low accelerating voltage** → less Kikuchi contrast
2. **Poor sample preparation** → surface damage, amorphous layer
3. **High beam current at small step size** → charging, drift
4. **Grain boundaries** → overlap of two orientations
5. **Nanostructure** → diffuse patterns
6. **Oxidation/contamination** → signal loss

### 8.2 Quality-Aware Training

#### Confidence estimation head

```
Pattern → CNN → Feature
                  │
          ┌───────┴────────┐
          │                │
     Orientation      Confidence
     Embedding        Score (0–1)
     (128-dim)        
```

**Training of the confidence head**:
- Noise-free patterns → confidence near 1.0
- Heavily augmented/degraded patterns → confidence near 0.0
- Loss: $\mathcal{L}_{\text{conf}} = \text{BCE}(\hat{c}, c_{\text{target}})$

where $c_{\text{target}}$ is estimated as:

$$c_{\text{target}} = \exp\left(-\frac{\Delta\omega_{\text{pred}}^2}{2\sigma_c^2}\right)$$

(i.e., confidence drops when the prediction becomes erroneous)

#### Degradation-Aware Augmentation

```python
class DegradationPipeline:
    """Simulates various levels of pattern degradation."""
    
    def __call__(self, pattern, degradation_level):
        # Level 0: Perfect (only slight noise)
        # Level 1: Moderate quality
        # Level 2: Poor quality
        # Level 3: Very poor / barely indexable
        
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

### 8.3 Grain-Boundary Patterns (Overlap)

At grain boundaries, patterns from two (or more) orientations are superimposed:

$$P_{\text{boundary}} = \alpha \cdot P_{\text{grain1}} + (1-\alpha) \cdot P_{\text{grain2}}, \quad \alpha \in [0, 1]$$

**Strategy 1: Detection (binary)**
- Train a classifier: "Is this pattern an overlap?"
- Training: synthetic superpositions with various α

**Strategy 2: Decomposition (advanced)**
- The network outputs the top-2 embeddings
- Assignment via FAISS to two orientations
- Additional α output for the mixing ratio

**Strategy 3: Pragmatic (recommended for the start)**
- Overlap patterns have low confidence → flag
- No explicit decomposition, but exclusion from indexing
- Correlation with the neighboring pixels for subsequent assignment

---

## 9. FAISS Index and Retrieval

### 9.1 Index construction

```python
import faiss

# Configuration
d = 128                          # Embedding dimension
n_orientations = 3_500_000       # Per phase
n_phases = 5                     # Typical for Al alloys
n_total = n_orientations * n_phases  # ~17.5 million entries

# Choose the index type
# Option A: Flat (exact, but slow at >1M entries)
index = faiss.IndexFlatIP(d)     # Inner product for cosine similarity

# Option B: IVF + PQ (recommended for production)
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFPQ(
    quantizer,
    d,                           # Dimension
    n_lists=4096,                # Number of Voronoi cells
    m=16,                        # PQ subquantizer
    nbits=8                      # Bits per subquantizer
)

# Train the index (with a subset of the embeddings)
index.train(training_embeddings)  # ~100k–500k samples

# Insert all embeddings
index.add(all_embeddings)

# GPU-accelerated (RTX 4070)
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
```

### 9.2 Retrieval Speed (estimated)

| Index type | Entries | Query time (single) | Batch (1000) | Recall@1 |
|-----------|----------|---------------------|-------------|----------|
| Flat (exact) | 3.5M | ~5 ms | ~200 ms | 100% |
| IVF4096,PQ16 | 3.5M | ~0.1 ms | ~5 ms | ~95–98% |
| IVF4096,PQ16 (GPU) | 3.5M | ~0.01 ms | ~0.5 ms | ~95–98% |
| HNSW | 3.5M | ~0.05 ms | ~3 ms | ~98–99% |

**For a typical EBSD map (500×500 = 250,000 pixels)**:
- Flat (CPU): ~20 min (too slow)
- IVF+PQ (CPU): ~25 seconds
- IVF+PQ (GPU): ~2.5 seconds
- HNSW (CPU): ~12 seconds

**Plus CNN inference** (~0.01 ms/pattern on GPU with batching):
- 250,000 patterns: ~2.5 seconds

**Total time for a 250k map**: ~5–30 seconds (depending on the index type)

### 9.3 Index Metadata

Each entry in the index needs associated metadata:

```python
# Metadata array (parallel to the FAISS index)
metadata = {
    'orientations': np.array([...]),  # (N, 4) quaternions
    'phases': np.array([...]),         # (N,) phase IDs
    'euler_angles': np.array([...]),   # (N, 3) optional, for fast output
}
```

---

## 10. Local Refinement (Post-Retrieval)

### 10.1 Why refinement?

FAISS returns the nearest orientation in the **discrete** grid. For sub-grid accuracy (<0.5°) an additional step is required.

### 10.2 Methods

#### Method A: Weighted interpolation of the top-k neighbors

```python
def refine_orientation(query_embedding, top_k_embeddings, top_k_orientations, top_k_scores):
    """
    Interpolate the final orientation from the top-k FAISS results.
    """
    # Scores as weights (softmax-normalized)
    weights = softmax(top_k_scores / temperature)
    
    # Quaternion interpolation (weighted)
    q_refined = quaternion_weighted_average(top_k_orientations, weights)
    
    return q_refined
```

**Problem**: Quaternion averaging is not trivial on SO(3) — it needs iterative methods (e.g. Markley et al., 2007).

#### Method B: Local optimization (Nelder-Mead / L-BFGS)

```python
def refine_local(pattern, initial_orientation, master_pattern, pc):
    """
    Local optimization of the NCC (Normalized Cross Correlation)
    in orientation space around the FAISS-initialized orientation.
    """
    def objective(params):
        # params: 3 Euler angles (or Rodrigues)
        orientation = euler_to_rotation(params)
        simulated = simulate_pattern(master_pattern, orientation, pc)
        return -normalized_dot_product(pattern, simulated)
    
    result = minimize(
        objective,
        x0=orientation_to_euler(initial_orientation),
        method='Nelder-Mead',
        options={'xatol': 0.01, 'fatol': 1e-6}  # ~0.01° accuracy
    )
    return euler_to_rotation(result.x)
```

**Speed**: ~1–5 ms per pattern (acceptable as an optional step)

#### Method C: Learned Refinement (Neural)

Train a small MLP that predicts a correction from the FAISS top-k result:

```
(query_embedding, top5_embeddings, top5_scores) → MLP → Δq (quaternion correction)

q_final = q_coarse ⊗ Δq   (quaternion composition)
```

**Advantage**: Very fast (~0.01 ms), learnable
**Disadvantage**: Needs its own training, less interpretable

### 10.3 Recommendation

**Method A** for most cases (fast, simple, ~0.3° improvement). **Method B** optionally for cases where sub-0.5° accuracy is critical.

---

## 11. Evaluation and Benchmarking

### 11.1 Metrics

| Metric | Description | Target value |
|--------|-------------|----------|
| Mean Angular Error (MAE) | Mean misorientation to ground truth | < 1.0° |
| Median Angular Error | More robust than MAE | < 0.5° |
| 95th Percentile Error | Worst-case estimate | < 3.0° |
| Phase-ID Accuracy | Correct phase identification | > 98% |
| Indexing Rate | Fraction of successfully indexed patterns | > 95% |
| Throughput | Patterns/second | > 10,000 |
| Confidence Calibration | Correlation confidence ↔ actual error | Monotonic |

### 11.2 Benchmarking Protocol

```
Test datasets:
1. Simulated, perfect    → Baseline accuracy
2. Simulated + noise     → Robustness
3. Simulated + PC error  → PC tolerance
4. Simulated + strain    → Strain robustness
5. Experimental (Si)     → Real-world single-phase
6. Experimental (Al-Cu)  → Real-world multi-phase
7. Experimental (heavily deformed) → Extreme condition

Comparison against:
- Hough-based indexing (EDAX OIM / Oxford AZtec)
- Dictionary Indexing (EMsoft / kikuchipy)
- Spherical Indexing (EMSphInx)
```

### 11.3 Ablation Studies

What happens when individual components are removed?

1. Without PC augmentation → How much does PC robustness suffer?
2. Without strain augmentation → Accuracy on deformed samples?
3. Without the confidence head → Can poor patterns still be detected?
4. Embedding dimension: 64 vs. 128 vs. 256
5. Encoder architecture: ResNet-18 vs. EfficientNet vs. MobileNet
6. FAISS index type: Flat vs. IVF+PQ vs. HNSW
7. With vs. without local refinement

---

## 12. Implementation Plan (Milestones)

### Phase 1: Foundation (2–3 weeks)

```
□ Master-pattern generation for α-Al (FCC) with EMsoft/kikuchipy
□ Orientation sampling (cubochoric, N=100)
□ Pattern-generation pipeline (master → detector pattern)
□ Basic augmentation (noise, contrast, background)
□ Implement ResNet-18 encoder
□ Implement contrastive loss (SO(3)-aware)
□ Single-phase training loop
□ First evaluation: MAE on simulated data
```

### Phase 2: Robustness (2–3 weeks)

```
□ Implement and test PC augmentation
□ Implement strain augmentation (Gaussian blur)
□ Add a confidence head
□ Degradation pipeline (various quality levels)
□ Build and test the FAISS index
□ Local refinement (Method A: weighted interpolation)
□ Systematic ablation: PC robustness, strain robustness
```

### Phase 3: Multi-Phase (2–3 weeks)

```
□ Master patterns for further phases (Al₂Cu, Mg₂Si, Si, ...)
□ Unified embedding-space training (all phases)
□ Verify phase separation in the embedding space
□ Multi-phase FAISS index
□ Evaluate phase-ID accuracy
□ Optional: EDS integration (multimodal)
```

### Phase 4: Validation & Integration (2–3 weeks)

```
□ Benchmarking against DI, SI, Hough
□ Test on experimental data
□ Integration into kikuchipy_GUI as an indexing backend
□ Performance optimization (batch inference, GPU FAISS)
□ Documentation and API design
□ Paper outline / data visualizations
```

### Estimated total time frame: 8–12 weeks

---

## 13. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Sim-to-real gap too large | Medium | High | Fine-tuning on exp. data, bridge augmentation |
| PC robustness insufficient | Low-Medium | High | Multi-task PC estimation (Approach C) |
| Phase confusion | Medium | Medium | EDS integration, hierarchical indexing |
| VRAM insufficient for training | Low | Medium | Gradient accumulation, smaller batch |
| FAISS recall too low | Low | Medium | HNSW instead of IVF+PQ, or a denser index |
| Method not better than SI | Low | High | Clearly communicate the accuracy/speed tradeoff |
| Orientation topology in the embedding not correct | Medium | High | Ablation of the loss, visualization via UMAP |

---

## 14. Technical Dependencies

```
Python >= 3.10
PyTorch >= 2.0 (CUDA 12.x)
faiss-gpu >= 1.7.4
kikuchipy >= 0.9
diffsims >= 0.6
orix >= 0.12
numpy, scipy, matplotlib
einops (optional, for tensor operations)
wandb or tensorboard (logging)
```

---

## 15. Potential for Publication

This approach would be publishable in:
- **Ultramicroscopy** (methodological, comparison with DI/SI)
- **Acta Materialia** (if combined with a materials-science application)
- **npj Computational Materials** (if the ML aspect is strongly emphasized)

**USPs compared to existing literature**:
1. First systematic analysis of the PC robustness of embedding-based indexing
2. SO(3)-aware contrastive learning (vs. naive classification)
3. Multi-phase in a unified embedding space
4. Strain quantification as a byproduct
5. Integration of EDS (multimodal) for hard-to-separate phases
6. Open-source integration into the kikuchipy ecosystem

> **⚠️ Status: work in progress — to be finished.** This module ships with the
> Orienta source release but is not yet complete; its APIs and results may change.

# EBSD-AI: Phase Prediction for EBSD Diffraction Patterns

A Python program that uses artificial intelligence to automatically identify crystal phases in materials.

---

## What does this program do?

This program analyzes special images of materials captured with an electron microscope (so-called "EBSD patterns") and predicts which crystal structure (phase) the material has at that location.

### Explained simply

When you examine a piece of metal or another crystalline material under a special microscope, characteristic patterns appear (like the material's fingerprints). This program "learns" to recognize these patterns and can then automatically say: "This is probably iron in this form" or "This is steel with this structure".

### What is EBSD?

EBSD stands for "Electron Backscatter Diffraction". This is a technique in which:
- An electron microscope shoots electrons at a material sample
- These electrons are backscattered by the crystal lattice
- This produces characteristic stripe patterns (like barcodes)
- Every crystal phase has its own typical pattern

### What are crystal phases?

Many materials (metals in particular) can exist in different crystal forms, for example:
- Iron can occur as "ferrite" (cubic atomic arrangement)
- Or as "austenite" (cubes packed differently)
- Steel can have various mixtures and phases

These different arrangements are called "phases", and they have different properties (hardness, magnetism, etc.).

### How does the AI work?

The program uses a "neural network" - a kind of artificial brain that learns from examples:
1. You show the program many EBSD patterns with labels ("this is phase A", "this is phase B")
2. The program learns the typical features of each phase
3. For new, unlabeled patterns it can then predict which phase it is

In addition, the program can also use chemical information (EDS data) - that is, which chemical elements are present in the material.

---

## Requirements

Before you begin, you need:

1. **A computer** running Windows, Linux, or macOS
2. **Python version 3.10 or newer** ([download here](https://www.python.org/downloads/))
   - Check your Python version in the command line: `python --version`
3. **Basic command-line knowledge** (Terminal/CMD)
   - How to open a folder
   - How to type commands and confirm them with Enter

### What is the command line?

- **Windows**: The "CMD" or "PowerShell" program (search in the Start menu)
- **Mac**: The "Terminal" program (in Applications > Utilities)
- **Linux**: Terminal application (usually Ctrl+Alt+T)

---

## Installation

### Step 1: Download the project

Download this project to your computer:

```bash
# If you have Git installed:
git clone https://github.com/IHR-BENUTZERNAME/Ai_Ml.git
cd Ai_Ml

# OR: Download the ZIP file and extract it,
# then open the command line in that folder
```

### Step 2: Create a virtual environment

A "virtual environment" is like a separate room for this project, so that it does not collide with other Python programs on your computer.

```bash
# Create the virtual environment (only needed once):
python -m venv .venv
```

**What happens here?**
- Python creates a folder named `.venv`
- All program libraries for this project are stored in it

### Step 3: Activate the virtual environment

Every time you work with the project, you have to activate the environment:

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

**Successful?**
- You now see `(.venv)` at the start of your command line
- This means: the virtual environment is active

### Step 4: Install the program

Now we install the program and all required libraries:

```bash
pip install -e .
```

**What happens here?**
- `pip` is the Python package manager (downloads libraries)
- `-e` means "editable" - you can edit the code and the changes take effect immediately
- `.` means "install the project in the current folder"

**Installed libraries:**
- `torch` - The AI library (PyTorch)
- `numpy` - Math and arrays
- `h5py` - Stores large amounts of data efficiently
- `scikit-image` - Image processing
- `scipy` - Scientific computations
- `matplotlib` - Creates plots

**Note:** The installation can take 5-10 minutes; PyTorch in particular is large (several GB).

### Step 5: Verify the installation

Test whether everything works:

```bash
python -c "import ebsd_ai; print('Installation successful!')"
```

**Expected output:**
```
Installation successful!
```

**If you see an error message:**
- See the "Troubleshooting" section below

---

## First Steps: Simple Examples

### Example 1: Save training data

This program needs training data in order to learn. Here is how you save EBSD patterns:

```python
# File: my_first_example.py
import numpy as np
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.config import DetectorInfo, DataSource

# 1. Create a data store
store = TrainingStore(local_path="./my_data", samples_per_shard=1000)

# 2. Create an example EBSD pattern (normally you load real data)
example_pattern = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

# 3. Example EDS data (chemical composition)
eds_data = {
    "Fe": 70.0,  # 70% iron
    "C": 2.0,    # 2% carbon
    "Cr": 18.0,  # 18% chromium
    "Ni": 10.0,  # 10% nickel
}

# 4. Information about the detector
detector = DetectorInfo(
    manufacturer="Oxford",
    convention="TSL",
    pixel_count=307200,
)

# 5. Save the sample
store.add_sample(
    pattern=example_pattern,
    eds_dict=eds_data,
    detector=detector,
    confirmed_phase="Austenite",  # The known phase
    orientation=(1.0, 0.0, 0.0, 0.0),  # Quaternion
    confidence=0.85,  # How certain is the label? (0-1)
    source=DataSource.MANUAL_INDEXING,
    source_file="example_scan_01.h5",
)

print(f"Saved! Statistics: {store.get_dataset_stats()}")
```

**What happens here?**
1. We create a folder `./my_data` for the training data
2. An EBSD pattern is saved (randomly generated here - normally you load real data)
3. EDS data specify the chemical composition (which elements, what percentage)
4. We tell the program that this is "austenite" (for training)
5. Everything is stored compressed in an HDF5 file

**Run it:**
```bash
python my_first_example.py
```

**Expected output:**
```
Saved! Statistics: {'total_samples': 1, 'phases': ['Austenite'], 'samples_with_eds': 1}
```

### Example 2: Create an AI model

Here is how you create a model that can predict phases:

```python
# File: create_model.py
import torch
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.config import ModelConfig

# 1. Configuration: which phases should the model be able to distinguish?
phases = ["Ferrite", "Austenite", "Martensite", "Pearlite"]

config = ModelConfig(
    n_phases=len(phases),         # Number of phases
    pattern_size=128,             # Image size (128x128 pixels)
    pattern_feature_dim=256,      # How many features to extract from the image
    eds_feature_dim=64,           # How many features from EDS data
    fused_feature_dim=128,        # Combined features
    dropout=0.2,                  # Prevents overfitting (20%)
)

# 2. Create the model
model = PhaseClassifier(config=config, phase_names=phases)

# 3. Show the model structure
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
print(f"Phases: {model.phase_names}")

# 4. Save the model
torch.save(model.get_save_dict(), "my_model.pt")
print("Model saved to: my_model.pt")
```

**What happens here?**
1. We define 4 phases that the model should distinguish
2. `ModelConfig` sets the architecture (how large the neural network is)
3. `PhaseClassifier` creates the AI model
4. The model is saved as a file (to load it later)

**Run it:**
```bash
python create_model.py
```

**Expected output:**
```
Model created with 2847940 parameters
Phases: ['Ferrite', 'Austenite', 'Martensite', 'Pearlite']
Model saved to: my_model.pt
```

**What are parameters?**
- These are the "weights" in the neural network that are adjusted during training
- About 2.8 million parameters - the model learns from data what values these should have

### Example 3: Train a model

The training pipeline is now fully implemented. Here is how you train a model on your data:

```python
# File: train_model.py
import torch
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.config import ModelConfig, TrainingConfig
from ebsd_ai.data.dataset import EBSDPhaseDataset
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.training.trainer import Trainer

# 1. Load the data
#    (Make sure you have saved data beforehand - see Example 1)
store = TrainingStore(local_path="./my_data")
dataset = EBSDPhaseDataset(store)

# 2. Create the model
phases = dataset.phase_names  # Phases are detected automatically from the data
config = ModelConfig(n_phases=len(phases))
model = PhaseClassifier(config=config, phase_names=phases)

# 3. Configure the training
training_config = TrainingConfig(
    learning_rate=0.001,         # Learning rate (how fast the model learns)
    batch_size=64,               # How many patterns to process at once
    epochs=50,                   # How often all data is passed through
    early_stopping_patience=10,  # Stop if no improvement for 10 epochs
    mixed_precision=True,        # Faster training on GPU (automatically off on CPU)
    val_fraction=0.2,            # Use 20% of the data for validation
)

# 4. Create the trainer and start training
trainer = Trainer(
    model=model,
    config=training_config,
    output_dir="checkpoints",    # Intermediate model states are saved here
    device="auto",               # Uses GPU if available, otherwise CPU
)

# 5. Start training
result = trainer.train(dataset)

# 6. Show the results
print(f"Training completed after {result.epochs_completed} epochs")
print(f"Best epoch: {result.best_epoch + 1}")
print(f"Best validation loss: {result.best_val_loss:.4f}")
print(f"Early stopped: {'Yes' if result.early_stopped else 'No'}")
print(f"Duration: {result.elapsed_seconds:.1f} seconds")
```

**What happens here?**
1. We load the saved training data from the TrainingStore
2. A new AI model is created - the phases are detected automatically
3. `TrainingConfig` determines how the training proceeds:
   - **Learning rate**: How strongly the model is adjusted at each step
   - **Batch size**: How many patterns are processed at once
   - **Epochs**: How often the entire dataset is passed through
   - **Early stopping**: Stops automatically when the model stops improving
4. The `Trainer` handles the training fully automatically:
   - It splits the data into training (80%) and validation (20%)
   - It automatically saves the best model as `checkpoints/best.pt`
   - It saves the latest state as `checkpoints/last.pt`
   - On a GPU, "Mixed Precision" is used automatically (faster)
5. At the end you receive a `TrainingResult` with all statistics

**Run it:**
```bash
python train_model.py
```

**Expected output (example):**
```
Training completed after 35 epochs
Best epoch: 25
Best validation loss: 0.3421
Early stopped: Yes
Duration: 142.3 seconds
```

**Important note:** You need enough training data (at least a few hundred patterns per phase) for the model to learn meaningfully.

### Example 4: Resume training

If training was interrupted or you want to train for longer:

```python
# File: resume_training.py
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.config import TrainingConfig
from ebsd_ai.data.dataset import EBSDPhaseDataset
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.training.trainer import Trainer

# 1. Load the data
store = TrainingStore(local_path="./my_data")
dataset = EBSDPhaseDataset(store)

# 2. Load the model from the last checkpoint
model = PhaseClassifier.from_save_dict(
    __import__("torch").load("checkpoints/last.pt", weights_only=False)["model"]
)

# 3. Create the trainer
training_config = TrainingConfig(epochs=100)  # Now 100 epochs in total
trainer = Trainer(model=model, config=training_config, output_dir="checkpoints")

# 4. Resume from the last checkpoint
trainer.resume("checkpoints/last.pt")

# 5. Continue training
result = trainer.train(dataset)
print(f"Resumed and trained {result.epochs_completed} more epochs")
```

**What happens here?**
1. We load the model and the training state from the last checkpoint
2. `resume()` restores the optimizer state (learning rate, momentum, etc.)
3. Training continues where it left off

### Example 5: Monitor progress

You can pass a callback function to watch the training progress live:

```python
# File: training_with_progress.py
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.config import ModelConfig, TrainingConfig
from ebsd_ai.data.dataset import EBSDPhaseDataset
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.training.trainer import Trainer

def show_progress(epoch, total, train_loss, val_loss):
    """Called after every epoch."""
    bar = "#" * int(20 * (epoch + 1) / total)
    empty = "-" * (20 - len(bar))
    print(
        f"[{bar}{empty}] Epoch {epoch+1}/{total} | "
        f"Training: {train_loss:.4f} | Validation: {val_loss:.4f}"
    )

# Prepare data and model (as in Example 3)
store = TrainingStore(local_path="./my_data")
dataset = EBSDPhaseDataset(store)
phases = dataset.phase_names
model = PhaseClassifier(
    config=ModelConfig(n_phases=len(phases)), phase_names=phases
)

# Create the trainer with a callback
trainer = Trainer(
    model=model,
    config=TrainingConfig(epochs=20),
    output_dir="checkpoints",
    progress_callback=show_progress,  # <-- Pass it here
)

result = trainer.train(dataset)
```

**Expected output (example):**
```
[#-------------------] Epoch 1/20  | Training: 1.3862 | Validation: 1.3845
[##------------------] Epoch 2/20  | Training: 1.2501 | Validation: 1.2130
[###-----------------] Epoch 3/20  | Training: 1.0843 | Validation: 1.0612
...
[####################] Epoch 20/20 | Training: 0.2145 | Validation: 0.3102
```

### Example 6: Evaluate a model (check quality)

After training you should check how good your model really is. The evaluator computes various quality metrics:

```python
# File: evaluate_model.py
import torch
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.data.dataset import EBSDPhaseDataset, train_val_test_split
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.training.evaluator import Evaluator, format_classification_report

# 1. Load the data and split it into train/val/test
store = TrainingStore(local_path="./my_data")
dataset = EBSDPhaseDataset(store, eds_dropout_rate=0.0)

# Split: 70% training, 15% validation, 15% test
train_ds, val_ds, test_ds = train_val_test_split(
    dataset, val_fraction=0.15, test_fraction=0.15
)

# 2. Load the trained model
save_dict = torch.load("checkpoints/best.pt", weights_only=False)["model"]
model = PhaseClassifier.from_save_dict(save_dict)

# 3. Create the evaluator
evaluator = Evaluator(
    model=model,
    device="auto",       # GPU if available, otherwise CPU
    k=3,                 # Compute top-3 accuracy
    n_calibration_bins=10,  # Calibration analysis with 10 bins
    batch_size=64,
)

# 4. Evaluate on the test data
result = evaluator.evaluate(test_ds)

# 5. Show the report
print(format_classification_report(result))

# 6. Query individual metrics
print(f"\nTop-1 accuracy: {result.top1_accuracy * 100:.1f}%")
print(f"Top-3 accuracy: {result.topk_accuracy * 100:.1f}%")
print(f"Mean confidence: {result.mean_confidence * 100:.1f}%")
print(f"Attention Alpha:    {result.mean_attention_alpha:.2f}")
print(f"Number of test samples:  {result.n_samples}")

# 7. Show the confusion matrix (which phases are confused?)
print("\nConfusion Matrix (rows = true phase, columns = predicted phase):")
print(f"{'':>15}", end="")
for name in result.phase_names:
    print(f"{name:>12}", end="")
print()
for i, name in enumerate(result.phase_names):
    print(f"{name:>15}", end="")
    for j in range(len(result.phase_names)):
        print(f"{result.confusion_matrix[i, j]:>12d}", end="")
    print()
```

**What happens here?**
1. The data is split into three parts: training, validation, and test
2. The trained model is loaded from the best checkpoint
3. The `Evaluator` automatically computes all the important quality metrics
4. `format_classification_report()` produces a clear text report

**Run it:**
```bash
python evaluate_model.py
```

**Expected output (example):**
```
Phase                Precision     Recall         F1    Support
--------------------------------------------------------------
Ferrite                 0.8500     0.8947     0.8718         19
Austenite               0.9091     0.8333     0.8696         12
Martensite              0.7500     0.8571     0.8000         14
--------------------------------------------------------------
Top-1 Accuracy:  0.8444
Top-3 Accuracy:  0.9778
Mean Confidence: 0.7234
Mean Alpha:      0.6812
Samples:         45

Top-1 accuracy: 84.4%
Top-3 accuracy: 97.8%
Mean confidence: 72.3%
Attention Alpha:    0.68
Number of test samples:  45

Confusion Matrix (rows = true phase, columns = predicted phase):
                     Ferrite    Austenite   Martensite
       Ferrite            17            1            1
      Austenite            1           10            1
     Martensite            2            0           12
```

**The metrics explained:**
- **Top-1 accuracy**: How often the top prediction is correct (e.g. 84.4% = correct for 84 out of 100 samples)
- **Top-3 accuracy**: How often the correct phase is among the 3 most probable (important for use as a pre-filter)
- **Precision**: When the model says "ferrite", how often is that correct?
- **Recall** (hit rate): Of all the true ferrite samples, how many did the model find?
- **F1 score**: Combines precision and recall into a single number (1.0 = perfect, 0.0 = poor)
- **Confusion matrix**: Shows exactly which phases are confused with one another
- **Calibration**: Checks whether a model that says "80% confident" is actually correct in 80% of cases
- **Attention Alpha**: How strongly the model trusts the EBSD pattern (1.0 = pattern only, 0.0 = EDS only)

### Example 7: Make a prediction (PhasePredictor)

The `PhasePredictor` is the recommended interface for predictions. It automatically handles loading the model, preprocessing the data, and converting it into the right format.

```python
# File: prediction.py
import numpy as np
from ebsd_ai.inference.predictor import PhasePredictor

# 1. Load the predictor from a checkpoint (best model from training)
predictor = PhasePredictor("checkpoints/best.pt", device="auto")

# 2. Check whether the model was loaded
print(f"Model loaded: {predictor.has_model}")
print(f"Phases: {predictor.phase_names}")

# 3. Example EBSD pattern (any size - automatically scaled)
pattern = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

# 4. EDS data (chemical composition) - simply as a dictionary
eds_data = {
    "Fe": 70.0,  # 70% iron
    "C": 2.0,    # 2% carbon
    "Cr": 18.0,  # 18% chromium
    "Ni": 10.0,  # 10% nickel
}

# 5. Make a prediction - a single line!
result = predictor.predict_phase(
    pattern=pattern,
    eds_data=eds_data,
    k=3,  # Top-3 predictions
)

# 6. Show the results
print("\nTop-3 predictions:")
for phase, probability in result.top_k:
    print(f"  {phase}: {probability*100:.1f}%")

print(f"\nConfidence: {result.confidence*100:.1f}%")
print(f"EDS contribution: {result.eds_contribution*100:.1f}%")
```

**What happens here?**
1. `PhasePredictor` loads the model from the checkpoint -- you do not have to worry about `torch.load`, `eval()`, or tensors
2. Your EBSD pattern can be any size -- it is automatically scaled to 128x128 and normalized
3. You simply pass EDS data as a dictionary (element name and percentage)
4. `predict_phase()` does everything: preprocessing, tensor conversion, prediction
5. The result is a `PhasePrediction` object with the top-k phases, confidence, and EDS contribution

**Run it:**
```bash
python prediction.py
```

**Expected output (example):**
```
Model loaded: True
Phases: ['Ferrite', 'Austenite', 'Martensite', 'Pearlite']

Top-3 predictions:
  Ferrite: 72.3%
  Austenite: 18.5%
  Martensite: 6.1%

Confidence: 72.3%
EDS contribution: 27.0%
```

**Interpretation:**
- The model thinks with 72.3% probability that it is ferrite
- 27% of the result is based on the EDS data, 73% on the EBSD pattern

**Good to know:** The PhasePredictor also works when only part of the data is available:
```python
# Only EBSD pattern (without EDS data)
result = predictor.predict_phase(pattern=pattern)

# Only EDS data (without EBSD pattern)
result = predictor.predict_phase(eds_data=eds_data)
```

### Example 8: Predict an entire EBSD scan (batch)

If you have a complete EBSD scan (many measurement points across an area), you can predict all points at once:

```python
# File: scan_prediction.py
import numpy as np
from ebsd_ai.inference.predictor import PhasePredictor

# 1. Load the predictor
predictor = PhasePredictor("checkpoints/best.pt", device="auto")

# 2. Load the EBSD scan (4D array: rows x columns x height x width)
#    Example: 100x200 measurement points, each pattern 480x640 pixels
n_rows, n_cols = 100, 200
scan = np.random.randint(0, 255, (n_rows, n_cols, 480, 640), dtype=np.uint8)

# 3. EDS maps (optional): element -> flat array with one value per measurement point
eds_maps = {
    "Fe": np.random.uniform(60, 80, n_rows * n_cols),
    "Cr": np.random.uniform(10, 20, n_rows * n_cols),
    "Ni": np.random.uniform(5, 15, n_rows * n_cols),
}

# 4. Optional: only predict certain pixels (e.g. with sufficient confidence)
mask = np.ones((n_rows, n_cols), dtype=bool)  # All pixels
# mask[0:10, :] = False  # Skip the first 10 rows

# 5. Start the batch prediction
scan_result = predictor.predict_scan(
    patterns_4d=scan,
    eds_maps=eds_maps,
    selection_mask=mask,
    k=3,
)

# 6. Show the results
print(f"Scan size: {scan_result.scan_shape}")
print(f"Predicted pixels: {scan_result.n_pixels}")
print(f"Phases: {scan_result.phase_names}")
print(f"Probabilities shape: {scan_result.probabilities.shape}")
print(f"Confidence shape: {scan_result.confidence.shape}")
print(f"EDS contribution shape: {scan_result.eds_contribution.shape}")
```

**What happens here?**
1. The scan is a 4D array: (rows, columns, pattern height, pattern width)
2. EDS maps give the chemical composition at each measurement point
3. The `selection_mask` determines which pixels are to be predicted (saves compute time)
4. `predict_scan()` processes all measurement points efficiently in batches
5. The result contains probabilities, confidence, and EDS contribution for each measurement point

**Run it:**
```bash
python scan_prediction.py
```

**Expected output (example):**
```
Scan size: (100, 200)
Predicted pixels: 20000
Phases: ['Ferrite', 'Austenite', 'Martensite', 'Pearlite']
Probabilities shape: (20000, 4)
Confidence shape: (20000,)
EDS contribution shape: (20000,)
```

### Example 9: Continuously improve the model (online learning)

If you regularly receive new indexing results, the model can improve itself automatically without being completely retrained:

```python
# File: online_learning.py
import torch
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.training.online_learner import OnlineLearner, OnlineLearnerConfig

# 1. Load the existing model and training data
store = TrainingStore(local_path="./my_data")
save_dict = torch.load("checkpoints/best.pt", weights_only=False)["model"]
model = PhaseClassifier.from_save_dict(save_dict)

# 2. Configure the online learner
config = OnlineLearnerConfig(
    learning_rate=0.0001,     # Small learning rate (cautious update)
    steps_per_update=50,      # 50 gradient steps per update
    batch_size=32,            # Mini-batch size
    replay_fraction=0.5,      # Mix in 50% old data (against forgetting)
    ewc_lambda=0.5,           # EWC strength (protects important weights)
)

# 3. Create the learner
learner = OnlineLearner(
    model=model,
    store=store,
    config=config,
    device="auto",
    checkpoint_dir="checkpoints",  # Saves after every update
)

# 4. Run an update (e.g. after new indexing results)
result = learner.update()

print(f"Steps: {result.steps_completed}")
print(f"Mean loss: {result.mean_loss:.4f}")
print(f"New phases discovered: {result.new_phases_added}")
print(f"Samples processed: {result.samples_used}")
```

**What happens here?**
1. The existing trained model is loaded
2. The `OnlineLearner` performs a careful update:
   - **Experience replay**: Mixes new and old data so that the model does not forget old phases
   - **EWC (Elastic Weight Consolidation)**: Protects important model weights from excessively large changes
   - **Phase expansion**: When new phases appear in the data, the model is automatically extended
3. A checkpoint is saved (`checkpoints/online_latest.pt`)

**Typical use:** After every EBSD scan with high confidence (CI > 0.3), you save the results in the TrainingStore and then call `learner.update()`. This way the model gets better with every measurement.

### Example 10: Enhance EBSD patterns (pattern enhancement)

Noisy EBSD patterns can be improved with the PatternEnhancer before they are classified or indexed:

```python
# File: enhance_pattern.py
import numpy as np
import torch
from ebsd_ai.models.pattern_enhancer import PatternEnhancer, EnhancerConfig
from ebsd_ai.config import DetectorInfo, DetectorConvention, DetectorManufacturer

# 1. Create the enhancer (or load from a checkpoint)
config = EnhancerConfig(
    base_channels=32,   # Channel count in the first block (32 -> 64 -> 128 -> 256)
    depth=4,             # 4 encoder/decoder stages
    use_film=True,       # Use detector metadata for conditioning
)
enhancer = PatternEnhancer(config=config)

# 2. Enhance a single pattern (any size)
noisy_pattern = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

# Without detector information:
enhanced = enhancer.enhance(noisy_pattern)
print(f"Output shape: {enhanced.shape}")  # (128, 128)

# With detector information (for better results):
detector = DetectorInfo(
    manufacturer=DetectorManufacturer.OXFORD,
    pc=(0.5, 0.2, 0.6),
    pc_convention=DetectorConvention.OXFORD,
    kv=20.0,
    working_distance=15.0,
    sample_tilt=70.0,
)
enhanced = enhancer.enhance(noisy_pattern, detector_info=detector)

# 3. Enhance a whole scan
n_rows, n_cols = 10, 20
scan = np.random.randint(0, 255, (n_rows, n_cols, 120, 160), dtype=np.uint8)

enhanced_scan = enhancer.enhance_scan(
    patterns_4d=scan,
    detector_info=detector,
    batch_size=32,  # How many patterns to process at once
)
print(f"Scan shape: {enhanced_scan.shape}")  # (10, 20, 128, 128)

# 4. Optional: only enhance certain regions
mask = np.zeros((n_rows, n_cols), dtype=bool)
mask[2:8, 5:15] = True  # Only enhance a section
partial_scan = enhancer.enhance_scan(scan, detector_info=detector, selection_mask=mask)

# 5. Save/load the model
torch.save(enhancer.get_save_dict(), "enhancer.pt")

# Load later:
save_dict = torch.load("enhancer.pt", weights_only=False)
loaded_enhancer = PatternEnhancer.from_save_dict(save_dict)
```

**What happens here?**
1. The `PatternEnhancer` is a U-Net (encoder-decoder network with skip connections)
2. It takes a noisy EBSD pattern and returns an enhanced 128x128 pattern
3. Optionally it can be conditioned on detector metadata (kV, PC, detector type) via FiLM layers
4. `enhance_scan()` processes whole scans efficiently in batches
5. The enhanced patterns can then be used for better Dictionary/Spherical indexing

**Important:** The enhancer must first be trained on paired data (noisy → clean) before it produces meaningful results.

### Example 11: Synchronize training data (server sync)

If you work as a team, you can share training data over a shared network drive:

```python
# File: synchronize_data.py
from ebsd_ai.sync.server_sync import ServerSync

# 1. Create the ServerSync (local folder + shared server path)
sync = ServerSync(
    local_dir="./my_data",               # Your local TrainingStore folder
    server_dir="//server/share/ebsd_ml/", # Shared network drive path
)

# 2. Upload your own data to the server (push)
push_result = sync.push()
print(f"Uploaded: {push_result.n_copied} files")
print(f"Skipped: {push_result.n_skipped} (already present)")

# 3. Download other users' data (pull)
pull_result = sync.pull()
print(f"Downloaded: {pull_result.n_copied} files")

# 4. Or both at once (bidirectional)
push_res, pull_res = sync.full_sync()

# 5. Show statistics
print(f"Local shards: {sync.local_stats()['n_shards']}")
print(f"Server shards: {sync.server_stats()['n_shards']}")

# 6. With a progress display
def progress(current, total, filename):
    print(f"  [{current}/{total}] {filename}")

sync.push(progress_callback=progress)
```

**What happens here?**
1. `ServerSync` connects a local TrainingStore folder with a network drive
2. `push()` copies new local shard files to the server (existing ones are skipped)
3. `pull()` copies new server files into the local folder
4. The copies are **atomic**: a file is first written under a temporary name and then renamed, so an interruption never leads to a corrupted file
5. Each file is copied independently — if one fails, the others are still transferred

**Run it:**
```bash
python synchronize_data.py
```

**Expected output (example):**
```
Uploaded: 2 files
Skipped: 1 (already present)
Downloaded: 3 files
Local shards: 6
Server shards: 6
```

### Example 12: Training via the command line (CLI)

Instead of writing Python code, you can start training directly from the command line:

```bash
# Simple training (default settings)
.venv/bin/python scripts/train.py --data ./my_data

# With customized settings
.venv/bin/python scripts/train.py \
    --data ./my_data \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0005 \
    --patience 15 \
    --output ./my_model

# Resume training (from the last checkpoint)
.venv/bin/python scripts/train.py \
    --data ./my_data \
    --resume ./my_model/last.pt
```

**Available options:**

| Option | Default | Description |
|--------|----------|-------------|
| `--data` | (required) | Path to the TrainingStore folder |
| `--epochs` | 50 | Maximum number of training epochs |
| `--batch-size` | 64 | Mini-batch size |
| `--lr` | 0.001 | Learning rate |
| `--patience` | 10 | Early-stopping patience (epochs) |
| `--val-fraction` | 0.15 | Fraction of validation data |
| `--eds-dropout` | 0.3 | EDS dropout rate |
| `--output` | checkpoints | Output folder for checkpoints |
| `--device` | auto | Device: auto, cpu, or cuda |
| `--resume` | — | Checkpoint to resume from |
| `--top-k` | 3 | Top-k for evaluation |

**What happens?**
1. The script loads the training data and displays statistics
2. A model is created (or loaded from a checkpoint)
3. Training runs with a progress display
4. At the end the model is evaluated on the validation data
5. The results (precision, recall, F1, confusion matrix) are displayed
6. Checkpoints are saved as `best.pt` and `last.pt`

**Expected output (example):**
```
Loaded TrainingStore: 500 samples from ./my_data
  Phases: {'Ferrite': 200, 'Austenite': 180, 'Martensite': 120}
  Samples with EDS: 300
  Samples without EDS: 200

Created model: 3 phases, 3,010,340 parameters

Training (max 50 epochs, patience 10)...
  Device: cuda
  Output: checkpoints

  [##------------------] Epoch   5/50 | train_loss=0.82 | val_loss=0.91
  ...
  [####################] Epoch  35/50 | train_loss=0.21 | val_loss=0.34

Training complete:
  Epochs completed: 35
  Best epoch:       25
  Best val loss:    0.3102
  Early stopped:    Yes
  Duration:         142.3s

Phase                 Precision     Recall         F1    Support
--------------------------------------------------------------
Ferrite                  0.8500     0.8947     0.8718         19
Austenite                0.9091     0.8333     0.8696         12
Martensite               0.7500     0.8571     0.8000         14
--------------------------------------------------------------
Top-1 Accuracy:  0.8444
Top-3 Accuracy:  0.9778
```

### Example 13: Evaluate a model via the command line

Here is how you assess a trained model directly from the command line:

```bash
# Evaluate the model on all data
.venv/bin/python scripts/evaluate.py \
    --model checkpoints/best.pt \
    --data ./my_data

# Evaluate on only 20% of the data (train/test split)
.venv/bin/python scripts/evaluate.py \
    --model checkpoints/best.pt \
    --data ./my_data \
    --split 0.2

# With top-5 accuracy and 20 calibration bins
.venv/bin/python scripts/evaluate.py \
    --model checkpoints/best.pt \
    --data ./my_data \
    --top-k 5 \
    --calibration-bins 20
```

**Available options:**

| Option | Default | Description |
|--------|----------|-------------|
| `--model` | (required) | Path to the model checkpoint |
| `--data` | (required) | Path to the TrainingStore folder |
| `--top-k` | 3 | Top-k for accuracy calculation |
| `--batch-size` | 64 | Batch size |
| `--device` | auto | Device: auto, cpu, or cuda |
| `--split` | 0 | Fraction for evaluation (0 = all data) |
| `--calibration-bins` | 10 | Number of calibration bins |

**What is displayed?**
- Classification report with precision, recall, F1 per phase
- Absolute and normalized confusion matrix
- Calibration analysis (confidence vs. actual accuracy)

### Example 14: Import EBSD scan files (CLI)

Instead of saving training data manually in Python code, you can import EBSD scan files into the TrainingStore directly from the command line:

```bash
# Import a single file
.venv/bin/python scripts/import_data.py scan.h5oina --store ./my_data

# Multiple files at once
.venv/bin/python scripts/import_data.py *.ctf --store ./my_data

# With a customized CI threshold (default: 0.3)
.venv/bin/python scripts/import_data.py scan.ang scan.ctf \
    --store ./my_data --ci 0.4

# Check in advance what would be imported (without writing data)
.venv/bin/python scripts/import_data.py scan.h5oina \
    --store ./my_data --dry-run

# Verbose output with phase details
.venv/bin/python scripts/import_data.py scan.h5oina \
    --store ./my_data --ci 0.0 --verbose
```

**Available options:**

| Option | Default | Description |
|--------|----------|-------------|
| `files` | (required) | One or more EBSD scan files (.ang, .ctf, .h5, .h5oina) |
| `--store` | (required) | Path to the TrainingStore folder |
| `--ci` | 0.3 | Minimum confidence index for import |
| `--target-size` | 128 | Normalized pattern size in pixels |
| `--dry-run` | — | Only analyze files, save nothing |
| `--verbose` / `-v` | — | Detailed output with warnings and phases |

**Supported file formats:**
- **.ang** — EDAX/OIM (indexing results only, no patterns → skipped)
- **.ctf** — Oxford/HKL Channel Text File (indexing results only, no patterns → skipped)
- **.h5oina** — Oxford Instruments HDF5 (with patterns, if present)
- **.h5** — kikuchipy or EMsoft HDF5 (automatic detection)

**Expected output (example):**
```
Import target: ./my_data
CI threshold:  0.3
Pattern size:  128x128
Files:         2

  scan1.h5oina ... 1850/2000 points (H5OINA)
  scan2.h5oina ... 920/1000 points (H5OINA)

--- Summary ---
Files processed: 2
Files with errors: 0
Total scan points: 3000
Imported: 2770

Phase breakdown:
  Ferrite: 1500
  Austenite: 870
  Martensite: 400
```

**What happens here?**
1. The script automatically detects the file format of each file
2. Each file is parsed and converted into a unified `ScanData` format
3. Only measurement points with a confidence index >= threshold are imported
4. Detector metadata (kV, manufacturer) are read automatically from the files
5. The results are stored compressed in the TrainingStore (HDF5)
6. `--dry-run` is useful to check beforehand how many data points would be imported

**Note:** Only files that contain EBSD patterns (typically HDF5 formats) can actually be imported. Pure text formats (.ang, .ctf) contain only indexing results without patterns and are skipped with a warning.

### Example 15: Import scan files in Python code

You can also control the import directly from Python code:

```python
# File: import_scan.py
from ebsd_ai.data.scan_import import parse_scan, import_to_store
from ebsd_ai.data.training_store import TrainingStore

# 1. Parse the scan file (format is detected automatically)
scan = parse_scan("my_scan.h5oina")

print(f"Format: {scan.source_format.value}")
print(f"Measurement points: {scan.n_points}")
print(f"Phases: {scan.phase_names}")
print(f"Has patterns: {scan.has_patterns}")
print(f"Has EDS: {scan.has_eds}")

# 2. Import into the TrainingStore (with CI filtering)
store = TrainingStore(local_path="./my_data")
result = import_to_store("my_scan.h5oina", store, ci_threshold=0.3)

print(f"\nImported: {result.n_points_imported}/{result.n_points_total}")
print(f"Phases: {result.phase_counts}")
if result.warnings:
    print(f"Warnings: {result.warnings}")
```

---

## Project Structure

If you want to understand or adapt the code:

```
Ai_Ml/
├── ebsd_ai/                      # Main package (source code)
│   ├── config.py                 # Central configuration
│   ├── data/                     # Data processing
│   │   ├── pattern_io.py         # Loading and normalizing EBSD patterns
│   │   ├── training_store.py     # Storing training data (HDF5)
│   │   ├── augmentation.py       # Data augmentation (more variance)
│   │   ├── dataset.py            # PyTorch Dataset with sampling and splits
│   │   ├── scan_import.py        # Import infrastructure (ScanData, Euler conversion, format detection)
│   │   ├── ang_parser.py         # Parser for EDAX/OIM .ang files
│   │   ├── ctf_parser.py         # Parser for Oxford/HKL .ctf files
│   │   └── hdf5_parser.py        # Parser for HDF5-based EBSD formats (H5OINA, kikuchipy, EMsoft)
│   ├── features/                 # Feature extraction
│   │   ├── pattern_encoder.py    # CNN for EBSD patterns
│   │   ├── eds_encoder.py        # MLP for EDS data
│   │   └── fusion.py             # Combines both data sources
│   ├── models/                   # AI models
│   │   ├── phase_classifier.py   # Main model: phase classification
│   │   ├── pattern_enhancer.py   # U-Net for pattern denoising/enhancement
│   │   └── losses.py             # Loss functions for training
│   ├── training/                 # Training pipeline
│   │   ├── trainer.py            # Training loop with checkpointing
│   │   ├── evaluator.py          # Evaluation (metrics, confusion matrix)
│   │   └── online_learner.py     # Continuous learning from new data
│   ├── inference/                # Prediction pipeline
│   │   └── predictor.py          # PhasePredictor for single and scan predictions
│   └── sync/                     # Network synchronization
│       └── server_sync.py        # Bidirectional data exchange over network drives
├── scripts/                      # Command-line scripts
│   ├── train.py                 # Training via the command line
│   ├── evaluate.py              # Evaluation via the command line
│   └── import_data.py           # Import EBSD scan files
├── tests/                        # Automated tests
├── pyproject.toml                # Project configuration
└── README.md                     # This file
```

**What is what?**

- **config.py**: Central settings (which elements exist, model sizes, etc.)
- **pattern_io.py**: Loads EBSD images, scales them to 128x128, normalizes brightness
- **training_store.py**: Stores thousands of EBSD patterns efficiently in HDF5 files
- **augmentation.py**: Creates variations (rotation, noise) for better training
- **dataset.py**: PyTorch Dataset for efficient data loading, with phase-balanced sampling and train/val/test splits
- **pattern_encoder.py**: CNN (Convolutional Neural Network) extracts features from EBSD patterns
- **eds_encoder.py**: MLP (Multi-Layer Perceptron) processes chemical data
- **fusion.py**: Combines EBSD + EDS intelligently with an attention mechanism
- **phase_classifier.py**: The main model that brings everything together
- **pattern_enhancer.py**: U-Net for pattern denoising -- enhances noisy EBSD patterns with optional detector conditioning via FiLM
- **losses.py**: Loss functions (measure how wrong the predictions are, so the model can learn)
- **trainer.py**: The complete training loop with automatic saving and early stopping
- **evaluator.py**: Assesses a trained model with accuracy, precision, recall, F1, confusion matrix, and calibration analysis
- **online_learner.py**: Continuous learning -- incrementally updates an existing model with new data without forgetting old phases
- **predictor.py**: The public prediction interface -- loads a model and makes predictions for individual patterns or entire scans
- **server_sync.py**: Synchronizes training data between the local machine and a network drive -- atomic file operations, progress callback, bidirectional
- **scan_import.py**: Import infrastructure for EBSD scan data -- detects file formats (ANG, CTF, HDF5), converts Euler angles to quaternions, provides a unified `ScanData` container
- **ang_parser.py**: Parser for EDAX/OIM .ang files -- reads the header (phases, grid size, detector metadata) and the data columns (Euler angles, CI, phase ID)
- **ctf_parser.py**: Parser for Oxford/HKL .ctf files -- reads the Channel Text File header (phases with lattice parameters, step sizes) and the data columns (Euler angles in degrees, MAD, Band Contrast)
- **hdf5_parser.py**: Parser for HDF5-based EBSD formats -- supports H5OINA (Oxford Instruments), kikuchipy (open source), and EMsoft (simulation software) with automatic format detection and quaternion-to-Euler conversion
- **import_data.py** (scripts): CLI script for importing EBSD scan files -- detects formats automatically, filters by confidence index, and stores the data in the TrainingStore

---

## Key Concepts

### 1. EBSD patterns

- **Input format**: Grayscale images, typically 480x640 or 1024x1024 pixels
- **Processing**: Scaled to 128x128 and normalized (mean 0, standard deviation 1)
- **Why?** The neural network needs uniform inputs

### 2. EDS data (chemistry)

- **Format**: A vector of 92 numbers (one value per chemical element)
- **Meaning**: Atomic percentage of each element (e.g. Fe=70.0 means 70% iron)
- **Optional**: If no EDS data are available, zeros are used

### 3. Phases

- **What are phases?** Different crystal structures of the same material
- **Examples**: Ferrite (body-centered cubic), austenite (face-centered cubic)
- **Classification**: The model predicts: "This pattern belongs to phase X with Y% probability"

### 4. Training vs. prediction

- **Training**: The model learns from labeled data (can take hours)
- **Prediction**: The trained model analyzes new, unlabeled data (seconds)

### 5. Multimodal fusion

- **Problem**: We have two data sources (EBSD pattern + EDS chemistry)
- **Solution**: An attention mechanism learns how strongly each source is weighted
- **Advantage**: Works even when only EBSD or only EDS is available

### 6. Training pipeline

The training pipeline automates the entire training process:

- **Optimizer (Adam)**: Adjusts the model parameters to reduce errors
- **Learning-rate scheduler (OneCycleLR)**: Changes the learning rate during training - it starts small, grows larger, then small again. This often leads to better results.
- **Early stopping**: Automatically stops training when the model stops improving. This prevents "overfitting" (the model memorizes the training data instead of learning general patterns).
- **Checkpointing**: Saves the model state regularly:
  - `best.pt` - The best model so far (lowest validation loss)
  - `last.pt` - The latest state (for resuming)
- **Mixed precision**: On NVIDIA GPUs, computation is done automatically at half precision. This is roughly twice as fast with the same result.
- **Class balancing**: If you have more data for one phase than for another, the rarer phases are automatically weighted more strongly.

### 7. Evaluation (quality assessment)

After training it is important to check how well the model actually works. The evaluator provides:

- **Top-1 accuracy**: How often the top prediction is correct
- **Top-k accuracy**: How often the correct answer is among the k most probable (default: top-3). Especially important when the model is used as a pre-filter.
- **Precision, recall, F1**: Detailed per-phase metrics - they show where the model is strong or weak
- **Confusion matrix**: A table that shows which phases are confused with one another. Rows are the true phases, columns the predicted ones.
- **Calibration**: Checks whether the model's confidence values are trustworthy. If the model says "80% confident", it should also be correct in roughly 80% of cases.

---

## Current Implementation

**Status: February 2026**

### Fully implemented (training, evaluation, and prediction pipeline complete)

- Central configuration (detectors, elements, model parameters)
- EBSD pattern input/output (loading, scaling, normalization)
- Training-data store (HDF5 with compression and sharding)
- Data augmentation (rotation, noise, brightness variation)
- Pattern encoder (ResNet CNN for image features)
- EDS encoder (MLP for chemical data)
- Multimodal fusion (attention-weighted combination)
- Phase classifier (end-to-end model with dynamic phase expansion)
- Model serialization (saving/loading checkpoints)
- PyTorch Dataset (EBSDPhaseDataset with EDS dropout and phase-balanced sampling)
- Loss functions (weighted cross-entropy with class balancing and confidence penalty)
- Training loop (Trainer with checkpointing, early stopping, and mixed precision)
- Evaluation (Evaluator with top-1/top-k accuracy, precision, recall, F1, confusion matrix, and calibration analysis)
- **Predictor API (PhasePredictor for single and batch predictions with automatic preprocessing and graceful degradation)**
- **Online learning (OnlineLearner for continuous learning from new indexing results with experience replay and EWC)**
- **Pattern enhancement (PatternEnhancer U-Net for pattern denoising with FiLM conditioning on detector metadata)**
- **Server synchronization (ServerSync for bidirectional data exchange over network drives with atomic file operations)**
- **Training CLI (scripts/train.py) for training from the command line with all options**
- **Evaluation CLI (scripts/evaluate.py) for model assessment with confusion matrix and calibration**
- **Import infrastructure (data/scan_import.py) with a ScanData container, Euler-to-quaternion conversion, and automatic format detection for ANG, CTF, and HDF5**
- **ANG parser (data/ang_parser.py) for EDAX/OIM .ang files with header parsing, phase detection, and robust error handling**
- **CTF parser (data/ctf_parser.py) for Oxford/HKL .ctf files with Euler-degrees-to-radians conversion and MAD-to-confidence mapping**
- **HDF5 parser (data/hdf5_parser.py) for H5OINA, kikuchipy, and EMsoft HDF5 formats with automatic format detection**
- **Import pipeline (parse_scan + import_to_store) for automatic import of EBSD scan files into the TrainingStore with CI filtering**
- **Import CLI (scripts/import_data.py) for file import from the command line with dry-run mode and multi-file support**
- **NaN/Inf sanitization: automatic detection and cleanup of non-finite values (NaN, Inf) at all data input points, in parsers, in the dataset, and in the trainer. Prevents silent numerical corruption. Includes gradient clipping during training.**
- **Memory-efficient processing of large scans: thread locks (RLock) for safe parallel HDF5 access, lightweight metadata querying (get_metadata/iter_metadata), HDF5 chunking, chunked import (configurable chunk_size), lazy pattern indexing for scan predictions, num_workers=0 in DataLoaders for h5py compatibility**
- **Clean public API: all important classes are importable directly via `from ebsd_ai import ...`. Consistent error messages (RuntimeError with helpful hints when no model is loaded). Configurable MAD-to-confidence mapping (`mad_to_confidence(mad, decay=1.0)`) for flexible adaptation to different materials and detectors.**

**What this means:** All modules of the planned architecture are now implemented. The entire pipeline from training to prediction is functional. You can prepare data, create a model, train it automatically, assess its quality with detailed metrics, and then conveniently make predictions with the `PhasePredictor` -- both for individual patterns and for entire EBSD scans. With the `OnlineLearner`, the model can continuously learn from new data without forgetting previously learned phases. The `PatternEnhancer` can enhance noisy EBSD patterns before they are used for classification or indexing. With `ServerSync`, training data can be shared with other users over a network drive.

---

## Running Tests

This project has automated tests to make sure everything works.

### Run all tests

```bash
pytest
```

**Expected output:**
```
======================= test session starts ========================
collected 743 items

tests/test_augmentation.py .........................         [  5%]
tests/test_config.py ......................                  [ 10%]
tests/test_coverage_gaps.py ..........                      [ 12%]
tests/test_dataset.py ..................................................  [ 23%]
tests/test_eds_encoder.py .......                            [ 24%]
tests/test_evaluator.py ........................................  [ 33%]
tests/test_fusion.py ..............                          [ 36%]
tests/test_integration.py ...............                    [ 39%]
tests/test_losses.py .................................       [ 46%]
tests/test_online_learner.py ..........................      [ 52%]
tests/test_pattern_encoder.py ........                       [ 53%]
tests/test_pattern_enhancer.py .......................................  [ 62%]
tests/test_pattern_io.py .............                       [ 65%]
tests/test_phase_classifier.py ..........................    [ 70%]
tests/test_predictor.py ...........................................  [ 79%]
tests/test_server_sync.py .........................          [ 85%]
tests/test_training_store.py ..............                  [ 88%]
tests/test_ang_parser.py ..............................          [ 94%]
tests/test_ctf_parser.py ..............................          [ 89%]
tests/test_hdf5_parser.py ......................................................  [ 97%]
tests/test_import_pipeline.py ...................................  [ 98%]
tests/test_memory_efficiency.py ....................              [ 98%]
tests/test_nan_sanitization.py ............................      [ 98%]
tests/test_public_api.py ...........................             [ 99%]
tests/test_scan_import.py ....................................................  [ 99%]
tests/test_trainer.py ..................................     [100%]

======================= 743 passed in 75.18s =======================
```

**What does this mean?**
- Each dot (`.`) is a successful test
- `743 passed` means: all 743 tests worked
- If a test fails, you see an `F` instead of `.`

### Run only specific tests

```bash
# Configuration tests only
pytest tests/test_config.py

# Model tests only
pytest tests/test_phase_classifier.py

# Loss-function tests only
pytest tests/test_losses.py

# Evaluation tests only
pytest tests/test_evaluator.py

# Predictor tests only (prediction interface)
pytest tests/test_predictor.py

# Online-learning tests only
pytest tests/test_online_learner.py

# Pattern-enhancement tests only
pytest tests/test_pattern_enhancer.py

# Server-sync tests only
pytest tests/test_server_sync.py

# Import-infrastructure tests (Euler conversion, format detection, ScanData)
pytest tests/test_scan_import.py

# ANG-parser tests (EDAX/OIM .ang files)
pytest tests/test_ang_parser.py

# CTF-parser tests (Oxford/HKL .ctf files)
pytest tests/test_ctf_parser.py

# HDF5-parser tests (H5OINA, kikuchipy, EMsoft)
pytest tests/test_hdf5_parser.py

# Import-pipeline tests (parse_scan, import_to_store, CLI)
pytest tests/test_import_pipeline.py

# NaN/Inf-sanitization tests
pytest tests/test_nan_sanitization.py

# Memory-efficiency tests (thread locks, metadata, chunking)
pytest tests/test_memory_efficiency.py

# Public API and error messages
pytest tests/test_public_api.py

# End-to-end integration test (checks the entire pipeline)
pytest tests/test_integration.py

# Verbose mode (more details)
pytest -v
```

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'ebsd_ai'"

**Cause:** The package was not installed or the virtual environment is not activated.

**Solution:**
```bash
# 1. Activate the virtual environment
source .venv/bin/activate  # Mac/Linux
# OR
.venv\Scripts\activate     # Windows

# 2. Install the package
pip install -e .
```

### Problem: "torch not found" or PyTorch errors

**Cause:** PyTorch was not installed correctly.

**Solution:**
```bash
# Install PyTorch manually
pip install torch torchvision

# Test the installation
python -c "import torch; print(torch.__version__)"
```

**For GPU support (optional, for faster training):**
Visit https://pytorch.org/ and select your system configuration (CUDA version).

### Problem: "RuntimeError: CUDA out of memory"

**Cause:** The GPU does not have enough memory (during GPU training).

**Solution:**
```python
# Reduce the batch size in your configuration
training_config = TrainingConfig(
    batch_size=8,  # Instead of 64 (default)
)

# OR: use CPU instead of GPU
trainer = Trainer(model=model, config=training_config, device="cpu")
```

### Problem: Training stops immediately (after 1-2 epochs)

**Cause:** Early stopping triggers because there is too little data or `early_stopping_patience` is too low.

**Solution:**
```python
# Increase the patience
training_config = TrainingConfig(
    early_stopping_patience=20,  # Wait for 20 epochs without improvement
    epochs=100,                  # Allow more epochs
)
```

### Problem: "RuntimeError: expected scalar type Float but found Half"

**Cause:** Mixed precision is enabled, but your code expects full precision.

**Solution:**
```python
# Turn off mixed precision
training_config = TrainingConfig(
    mixed_precision=False,
)
```

### Problem: Tests fail

**Cause:** Possibly incompatible versions or missing dependencies.

**Solution:**
```bash
# Update all packages
pip install --upgrade -e .[dev]

# Run tests individually to find the problem
pytest tests/test_config.py -v
```

### Problem: HDF5 error when saving data

**Cause:** Corrupted HDF5 file or permission problems.

**Solution:**
```bash
# Delete corrupted files
rm -rf my_data/*.h5

# Make sure the folder is writable
chmod -R u+w my_data  # Linux/Mac
```

### Problem: "ImportError: cannot import name 'X'"

**Cause:** Outdated installation or changes to the code.

**Solution:**
```bash
# Uninstall and reinstall
pip uninstall ebsd-ai
pip install -e .
```

### Problem: Very slow training

**Cause:** The CPU is being used instead of the GPU, or the batch size is too small.

**Solution:**
```python
# Check whether a GPU is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Increase the batch size (if there is enough GPU memory)
training_config = TrainingConfig(
    batch_size=128,  # More data per step = faster
)
```

**Note:** Without an NVIDIA GPU, training is slow. For large datasets we recommend a computer with a GPU (NVIDIA with CUDA support).

### Problem: Evaluation results show 0% accuracy

**Cause:** The model was not trained properly, or you are evaluating on completely different phases than during training.

**Solution:**
```python
# Make sure the phases match
print(f"Model phases: {model.phase_names}")
print(f"Data phases:  {dataset.phase_names}")

# Check whether there was enough training data
# At least a few hundred patterns per phase are recommended
```

### Problem: Calibration shows poor values

**Cause:** The model is "overconfident" (gives high confidence even for wrong predictions).

**Solution:**
- This is normal with small datasets or short training times
- Longer training with more data improves the calibration
- The confidence-penalty loss already helps counter this automatically

---

## Advanced Usage

### Add a new phase

The model can be expanded dynamically when you discover a new phase:

```python
# Load a model with 3 phases
import torch
from ebsd_ai.models.phase_classifier import PhaseClassifier

save_dict = torch.load("checkpoints/best.pt", weights_only=False)["model"]
model = PhaseClassifier.from_save_dict(save_dict)
print(f"Before: {model.phase_names}")

# Add a new phase
model.add_phase("Bainite")
print(f"After: {model.phase_names}")

# Save and then continue training with new data
torch.save(model.get_save_dict(), "model_extended.pt")
```

### Use data augmentation

Create more training data through variations:

```python
import numpy as np
from ebsd_ai.data.augmentation import augment_pattern, augment_eds, AugmentationConfig

# Configuration
config = AugmentationConfig(
    rotation_angle=2.0,       # +/-2 degrees rotation
    brightness_range=0.2,     # +/-20% brightness
    contrast_range=0.2,       # +/-20% contrast
    noise_std=0.02,           # 2% noise
    horizontal_flip_prob=0.5, # 50% chance of flipping
)

# Original data
pattern = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
eds = np.array([70.0, 2.0, 18.0, 10.0] + [0.0]*88)  # Fe, C, Cr, Ni

# Augment (5 variations)
for i in range(5):
    aug_pattern = augment_pattern(pattern, config, rng=np.random.default_rng(i))
    aug_eds = augment_eds(eds, config, rng=np.random.default_rng(i))
    print(f"Variation {i+1}: pattern shape {aug_pattern.shape}, EDS sum {aug_eds.sum():.1f}")
```

### Typical workflow

Here is the recommended workflow from start to finish:

```
1. Collect data       -->  EBSD patterns with known phases
2. Save data          -->  TrainingStore.add_sample() (Example 1)
3. Create model       -->  PhaseClassifier (Example 2)
4. Train model        -->  Trainer.train() (Example 3)
5. Evaluate model     -->  Evaluator.evaluate() (Example 6)
6. Predict            -->  PhasePredictor.predict_phase() (Example 7)
7. Predict scan       -->  PhasePredictor.predict_scan() (Example 8)
8. Keep learning      -->  OnlineLearner.update() (Example 9)
9. Enhance patterns   -->  PatternEnhancer.enhance() (Example 10)
10. Share data        -->  ServerSync.push/pull() (Example 11)
11. New phase?        -->  Detected and added automatically
```

---

## Technical Details

### Model architecture

```
Input: EBSD pattern (128x128) + EDS (92) + metadata (15+1)
    |
PatternEncoder (ResNet-CNN)      EDSEncoder (MLP)
    | (256-dim)                      | (64-dim)
    +----------> MultimodalFusion <--+
              (Attention-gated)
                    | (128-dim)
              Linear Classifier
                    |
    Output: phase logits + attention weights
```

### Training, evaluation, and prediction pipeline

```
EBSDPhaseDataset
    |
    +-- Automatic train/val/test split (stratified by phase)
    |
    +-- Phase-balanced sampling (rare phases are shown more often)
    |
    +-- Data augmentation (rotation, noise, EDS dropout)
    |
    v
Trainer
    |
    +-- Adam optimizer + OneCycleLR scheduler
    |
    +-- CombinedLoss (weighted cross-entropy + confidence penalty)
    |
    +-- Mixed precision (automatic on GPU)
    |
    +-- Early stopping (stops on no improvement)
    |
    +-- Checkpointing (best.pt + last.pt)
    |
    v
TrainingResult (loss curves, best epoch, duration)
    |
    v
Evaluator
    |
    +-- Top-1 and top-k accuracy
    |
    +-- Precision, recall, F1 per phase
    |
    +-- Confusion matrix (absolute and normalized)
    |
    +-- Calibration analysis (confidence vs. actual accuracy)
    |
    +-- Text report via format_classification_report()
    |
    v
EvaluationResult (all metrics in one object)
    |
    v
PhasePredictor (prediction interface)
    |
    +-- predict_phase(): Predict individual patterns
    |
    +-- predict_scan(): Predict whole EBSD scans (batch)
    |
    +-- Automatic preprocessing (scaling, normalization)
    |
    +-- Graceful degradation (works with/without EDS)
    |
    v
PhasePrediction / ScanPredictionResult
    |
    v (results back into TrainingStore)
OnlineLearner (continuous learning)
    |
    +-- Experience replay (mix old + new data)
    |
    +-- EWC (Elastic Weight Consolidation against forgetting)
    |
    +-- Automatic phase expansion
    |
    v
Improved model (back to the PhasePredictor)
```

### Data storage

- **Format**: HDF5 with gzip compression (level 4)
- **Sharding**: Max. 5000 samples per file (configurable)
- **Structure**:
  ```
  shard_00000.h5
  ├── sample_00000000/
  │   ├── pattern_original (uint8/uint16)
  │   ├── pattern_normalized (float32, 128x128)
  │   ├── eds_atomic_pct (float32, 92)
  │   ├── has_eds (bool)
  │   ├── confirmed_phase (string)
  │   └── ... (metadata)
  ├── sample_00000001/
  ...
  ```

### System requirements

- **RAM**: Min. 8 GB (16 GB recommended)
- **Storage**: 10 GB free (for data and models)
- **GPU** (optional): NVIDIA with CUDA for fast training
- **CPU**: Works, but slower

---

## License and Contributing

This project is open source. Contributions are welcome!

**Developer notes:**
- Code style: Follow PEP 8 (checked automatically with `ruff`)
- Type hints: All functions should have types
- Tests: New features need tests
- Documentation: NumPy-style docstrings

**Check the code:**
```bash
# Lint (style check)
ruff check .

# Type checking
mypy ebsd_ai

# Tests with coverage
pytest --cov=ebsd_ai
```

---

## Changelog

### Version 0.1.0 (February 2026)

**Implemented (complete ML pipeline):**
- Central configuration and data structures
- EBSD pattern I/O and normalization
- HDF5-based training-data store
- Data augmentation (rotation, noise, photometry)
- Pattern encoder (ResNet architecture)
- EDS encoder (MLP with batch normalization)
- Multimodal fusion (attention-gated)
- Phase classifier (end-to-end model)
- PyTorch Dataset with EDS dropout, phase-balanced sampling, and stratified splits
- Loss functions (weighted cross-entropy, confidence penalty, combined loss)
- Training loop (Adam + OneCycleLR, checkpointing, early stopping, mixed precision)
- Evaluation (top-1/top-k accuracy, precision, recall, F1, confusion matrix, calibration)
- Predictor API (PhasePredictor for single and batch predictions)
- Online learning (OnlineLearner with experience replay, EWC, and automatic phase expansion)
- Pattern enhancement (PatternEnhancer U-Net with FiLM conditioning for pattern denoising)
- Server synchronization (ServerSync for bidirectional data exchange with atomic file operations)
- Command-line training script (scripts/train.py) with argparse, resume, evaluation
- Command-line evaluation script (scripts/evaluate.py) with confusion matrix and calibration
- End-to-end integration test (checks the entire pipeline from data storage to prediction)
- Full type-checking coverage (mypy: 0 errors across 23 files)
- Full lint coverage (ruff: 0 warnings)
- Test-coverage optimization (99% overall coverage, every module above 96%)
- Import infrastructure (ScanData container, Euler-to-quaternion conversion, automatic format detection)
- ANG parser for EDAX/OIM .ang files (header parsing, phase detection, robust error handling)
- CTF parser for Oxford/HKL .ctf files (Euler-degrees conversion, MAD-to-confidence mapping)
- HDF5 parser for H5OINA, kikuchipy, and EMsoft (automatic format detection, quaternion-Euler conversion)
- Import pipeline (parse_scan + import_to_store) with CI filtering and automatic detector detection
- Import CLI (scripts/import_data.py) for file import from the command line
- NaN/Inf sanitization at all data input points, parsers, dataset, and trainer
- Gradient clipping and NaN-loss detection in the trainer
- Memory-efficient processing of large scans (thread locks, lightweight metadata API, HDF5 chunking, chunked import, lazy pattern indexing)
- Clean public API with __all__ exports and consistent error messages
- Configurable MAD-to-confidence mapping (mad_to_confidence with adjustable decay)
- Complete test suite (743 tests)

**All modules of the planned architecture are implemented.**

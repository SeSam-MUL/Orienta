# ML Hub

## What it does

The ML Hub trains and runs a **machine-learning phase classifier** for EBSD
patterns. Instead of comparing each pattern against a simulated dictionary (as in
classical indexing), a small neural network learns directly from patterns that
have already been indexed, and can then predict the phase of new patterns very
quickly.

The Hub has three parts:

- **Status** — shows whether the ML module is available, whether a model is
  currently loaded, which compute backend/device is in use, how many training
  samples have been collected, and which trained models exist.
- **Training** — builds a training set from your most recent indexing result and
  trains a phase classifier on it, with live progress and a log.
- **Prediction** — runs the loaded model on selected patterns and reports the
  predicted phase, confidence, and score.

It is best understood as an *accelerator* and *cross-check* for phase
identification, not a replacement for indexing: the model can only learn the
phases that appear (with good confidence) in the indexing results you feed it.

## When to use it

Use the ML Hub after you already have at least one good indexing result:

- You have indexed a representative dataset and want a **fast classifier** that
  can label patterns in similar future datasets without re-running a full
  dictionary/spherical index.
- You want a **second opinion** on the phase of specific patterns — predict them
  and compare against the indexed assignment.

It sits at the end of the workflow, after
[Indexing](../README.md): load data → index → (optionally) train a classifier
here → predict.

## How to use it — step by step

### Train a phase classifier

1. First run an indexing job on a loaded dataset (Dictionary, Spherical, or
   Hough). The Hub trains on the **last indexing result** automatically — there
   is no separate "training data path" to set.
2. Open the **Training** tab.
3. Set the hyperparameters:
   - **Epochs** — how many passes over the data (default 50).
   - **Batch** — samples per training step (default 32).
   - **Learning rate** — optimiser step size (default 0.001).
   - **CI threshold** — minimum confidence index a pixel must have to be added to
     the training store. Higher = cleaner labels, fewer samples. The value is
     colour-coded (green ≥ 0.30, orange ≥ 0.15, red below).
4. Click **Start Training**. A progress bar tracks the epoch count and the log
   shows how many samples were added and the per-epoch train/validation loss.
   You can **Stop** watching at any time (this stops the on-screen polling).
5. When training completes, a summary reports the trained phase names and the
   best validation loss, and the Status tab updates to show the model as loaded.
6. **Clear Store** wipes all accumulated training samples — use it to reset after
   a bad/mislabelled run before training a fresh model. You will be prompted to
   type `clear` to confirm.

### Predict phases

1. Open the **Prediction** tab. The model selected on the Status tab is shown.
2. Enter one or more **pattern indices** — flat pixel indices into the loaded
   signal, separated by commas or spaces (e.g. `0, 1, 2`).
3. Click **Predict**. Results appear in a table: pattern index, predicted phase,
   confidence (%), and score. The button is disabled if the ML module is
   unavailable.

### Check status

The **Status** tab is read-only: ML module availability, model-loaded state,
backend/device, training-store statistics (total samples, samples per phase, mean
confidence), and the list of available trained models. Click a model to select it
for prediction.

## Inputs & outputs

**Inputs**

- The **last indexing result** (CrystalMap with phase labels, orientations, and
  confidence scores) plus the corresponding EBSD patterns. Patterns are taken
  from the indexing result if present, otherwise from the active EBSD-Viewer
  signal.
- Hyperparameters and the CI threshold you set in the Training tab.
- For prediction: pattern indices into the currently loaded signal.

**Outputs**

- A trained model checkpoint saved on disk (managed by the ML module), which
  becomes the loaded model.
- A persistent **training store** of labelled samples that accumulates across
  "train" runs until you clear it.
- On-screen prediction results (phase, confidence, score). Nothing is written to
  your dataset automatically.

## Tips & notes

- **Training is GPU/CPU-heavy and runs in the background.** It executes off the
  request thread so the rest of the app stays responsive; expect minutes (longer
  on CPU or with many epochs). The Status tab reports the device actually used.
- **Garbage in, garbage out.** The classifier can only learn phases that appear
  with sufficient confidence in your indexing result. If indexing mislabelled a
  region, raise the **CI threshold** so only high-confidence pixels become
  training labels, or **Clear Store** and retrain.
- **The training store is cumulative.** Each training run first adds the latest
  indexing result's samples to the store, so stale or mislabelled samples from a
  previous run will poison later models — clear the store when in doubt.
- **Prediction needs a loaded model.** If no model is trained/loaded, prediction
  returns an explanatory error per index rather than a phase.
- **Some advanced ML sections are not part of this build.** The page contains
  additional ideas (a pattern *denoiser* and a FAISS *embedding/indexing* method)
  that are **not wired to a backend** and are hidden in the shipped UI. Only the
  phase-classifier train/predict workflow described above is active.

# Ai_Ml — EBSD-AI (experimental ML module)

> ⚠️ **Experimental / work in progress.** This module ships with Orienta but is
> not yet complete; its API and results may change. It is currently exposed in
> the app only through the **ML Hub** page (also work in progress) and is **not
> required** for the core load → index → analyze workflow.

`Ai_Ml` is Orienta's machine-learning component for EBSD. It provides a
neural-network **phase classifier** (predict the crystal phase of a pattern,
optionally fused with EDS chemistry) and a **pattern denoiser / enhancer**. It is
a self-contained Python package (`ebsd_ai`) with accompanying command-line
`scripts/` and `tests/`.

## How it fits into Orienta

This is **part of the Orienta source tree** (the `Ai_Ml/` folder) — you do **not**
clone or install it as a separate project. Within Orienta it is driven from the
application, not as a standalone tool:

- **UI:** the *ML Hub* page (experimental).
- **Backend route:** [`backend/api/routes/ml_hub.py`](../backend/api/routes/ml_hub.py)
- **Controller:** [`../ml_controller.py`](../ml_controller.py)

It uses the same Python environment as the rest of Orienta (the conda `ebsd`
environment from [INSTALL.md](../INSTALL.md)); there is no separate virtual
environment or install step for normal use.

## Optional: using the package / CLI directly

If you want to train or run models outside the app, install the package
**into the existing Orienta environment** (do not create a separate venv):

```bash
conda activate ebsd
pip install -e Ai_Ml      # exposes the `ebsd_ai` package + the scripts/ tools
```

The command-line tools then live under `Ai_Ml/scripts/`:

| Script | Purpose |
|--------|---------|
| `scripts/import_data.py` | Import EBSD/EDS scans into a training dataset |
| `scripts/train.py`       | Train a phase-classifier model |
| `scripts/evaluate.py`    | Evaluate a trained model |
| `scripts/export_onnx.py` | Export a trained model to ONNX |

The GPU is used automatically when a CUDA device is available, otherwise the CPU.

## Layout

```
Ai_Ml/
├── ebsd_ai/        # the package: features/, models/, inference/, config, …
├── scripts/        # command-line tools (import / train / evaluate / export)
├── tests/          # unit tests
└── pyproject.toml  # package metadata (installs as `ebsd_ai`)
```

For the design and architecture of the planned ML-based indexing system, see
[`Detailed_Plan_Ai_integration.md`](Detailed_Plan_Ai_integration.md).

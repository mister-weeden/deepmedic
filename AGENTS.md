# Repository Guidelines

## Project Structure & Module Organization
- `deepmedic/`: Python package with core code.
  - `neuralnet/`: TensorFlow 3D CNN building blocks and training utilities.
  - `dataManagement/`: IO, preprocessing, sampling, augmentation.
  - `frontEnd/`: CLI/session orchestration and config parsing.
  - `routines/`: High-level training/testing flows.
  - `logging/`: Metric/log utilities and TensorBoard hooks.
- `deepMedicRun`: Main CLI entrypoint for train/test.
- `examples/`: Minimal configs and sample data to smoke-test.
- `documentation/`: Installation and usage guide.

## Build, Test, and Development Commands
- Install (editable): `python -m venv .venv && source .venv/bin/activate && pip install -e .`
- CLI help: `./deepMedicRun -h`
- Train tiny example (CPU):
  `./deepMedicRun -model examples/configFiles/deepMedic/model/modelConfig.cfg -train examples/configFiles/deepMedic/train/trainConfig.cfg -dev cpu`
- Test with a trained model:
  `./deepMedicRun -model examples/configFiles/deepMedic/model/modelConfig.cfg -test examples/configFiles/deepMedic/test/testConfig.cfg -dev cpu -load /path/to/checkpoint`
- Plot progress: `python plotTrainingProgress.py /path/to/training-log.txt`

## Coding Style & Naming Conventions
- Language: Python 3; follow PEP 8 (4-space indents, 88–100 cols preferred).
- Names: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_CASE`.
- Docstrings: short, action-focused; include expected shapes/dtypes and config keys.
- Imports: standard lib → third‑party → local; avoid wildcard imports.

## Testing Guidelines
- No formal unit test suite at present. Validate via the tiny example:
  - Run the train command above; confirm logs and checkpoints are created.
  - Optionally enable TensorBoard logging and verify metrics.
- Add tests near modules you touch (e.g., lightweight function tests under `deepmedic/<submodule>/tests/`). Prefer deterministic, small fixtures.

## Commit & Pull Request Guidelines
- Commits: imperative, present tense; focused changes.
  - Example: `fix(sampling): handle empty class safely`.
- PRs: clear description, linked issues, what/why, and how to reproduce.
  - Include affected configs/paths, before/after behavior, and sample logs/screenshots.
  - Keep diffs minimal; update README/config comments when behavior changes.

## Security & Configuration Tips
- Do not commit data, checkpoints, or large artifacts; keep them outside the repo.
- Paths in configs should be absolute or resolved via the CLI; prefer `-dev cpu` for portability.
- Ensure TensorFlow/CUDA/cudnn versions are compatible (see `documentation/README.md`).

## HPC Run Guide (GPU)

This section documents a complete, reproducible run on an HPC node with GPUs: environment setup, preprocessing (examples are preprocessed), training, checkpoints, and inference.

### Prerequisites on HPPC
- GPU node with recent NVIDIA driver compatible with TensorFlow 2.13 (CUDA 11.8 runtime bundled in wheel).
- Module toolchain suggested by the center (example below).
- Internet access from compute node (to install Python wheels), or a mirror/cache.
- Mamba/Conda available.

Example HPPC session bootstrap (adapt to your cluster queue/partitions):

```sh
interactive -p matador -c 8 -g 2 -m 8G
module load gcc/10.1.0 nvhpc/21.3-mpi openmpi/4.1.4-cuda
```

### Create environment
We recommend Python 3.9 with pip for TF 2.13. Use mamba to create an isolated env and install lightweight deps via conda, then TensorFlow via pip.

```sh
mamba create -n deepmedic_env -y python=3.9 pip
mamba activate deepmedic_env

# Optional: faster installs for scientific stack
mamba install -y -c conda-forge numpy=1.23 scipy pandas nibabel

# Install DeepMedic in editable mode (no extra deps pulled)
pip install --no-deps -e .

# Install TensorFlow (GPU-enabled on Linux)
pip install "tensorflow==2.13.*"

# Sanity check TF + GPU visibility
python - << 'PY'
import tensorflow as tf
print('TF:', tf.__version__)
print('Devices:', tf.config.list_physical_devices())
PY
```

Notes:
- Wheels for TF 2.13 bundle CUDA 11.8 and cuDNN 8.6; only the NVIDIA driver must be present on the node.
- If your site pins CUDA differently, load the site’s CUDA modules as advised and stick to a TF version aligned with that toolchain.

### Data and configs
- Example NIfTI data are bundled under `examples/dataForExamples/` (train/validation/test). They are already preprocessed (z‑score normalized within ROI, resampled consistently) for the examples.
- Example configs are under `examples/configFiles/`. For a quick smoke test, use the tiny model; for a fuller run, use the deepMedic configs.

### Train (tiny example, 2 epochs)

```sh
# GPU run using the tiny example (fast)
./deepMedicRun \
  -model examples/configFiles/tinyCnn/model/modelConfig.cfg \
  -train examples/configFiles/tinyCnn/train/trainConfigWithValidation.cfg \
  -dev cuda
```

Outputs created:
- Logs: `examples/output/logs/trainSessionWithValidTiny.txt`
- Checkpoints: `examples/output/saved_models/trainSessionWithValidTiny/` and `examples/output/cnnModels/` (files ending in `.model.ckpt.*` per epoch)
- Validation predictions: `examples/output/predictions/trainSessionWithValidTiny/predictions/`
- TensorBoard (if enabled in train config): `examples/output/tensorboard/trainSessionWithValidTiny/`

Plot training curve (optional):

```sh
python plotTrainingProgress.py examples/output/logs/trainSessionWithValidTiny.txt -d
```

### Inference (using the trained tiny model)

1) Identify the latest checkpoint (path must end with `.model.ckpt`, not the `.index`/`.data` files):

```sh
CKPT_DIR=examples/output/saved_models/trainSessionWithValidTiny
CKPT=$(ls -1t ${CKPT_DIR}/tinyCnn.trainSessionWithValidTiny.*.model.ckpt 2>/dev/null | head -n1)
echo "Using checkpoint: $CKPT"
```

2) Run test config against that checkpoint:

```sh
./deepMedicRun \
  -model examples/configFiles/tinyCnn/model/modelConfig.cfg \
  -test  examples/configFiles/tinyCnn/test/testConfig.cfg \
  -dev cuda \
  -load "$CKPT"
```

Outputs:
- Test predictions: `examples/output/predictions/testSessionTiny/predictions/*.nii.gz`
- Feature maps (if enabled in test config): `examples/output/predictions/testSessionTiny/features/`

### Full DeepMedic config (optional, longer)

```sh
./deepMedicRun \
  -model examples/configFiles/deepMedic/model/modelConfig.cfg \
  -train examples/configFiles/deepMedic/train/trainConfig.cfg \
  -dev cuda
```

You can later resume or fine‑tune with:

```sh
./deepMedicRun \
  -model examples/configFiles/deepMedic/model/modelConfig.cfg \
  -train examples/configFiles/deepMedic/train/trainConfig.cfg \
  -dev cuda \
  -load /path/to/prev/checkpoint  # ends with .model.ckpt
```

### Notes and troubleshooting
- macOS Apple Silicon: creating a TF v1 Session may segfault depending on TF version. Use `tensorflow-macos==2.9.2` + `tensorflow-metal==0.5.*` if running locally on macOS; this does not affect Linux HPPC runs.
- If TF cannot see GPUs, ensure the node has GPUs allocated and the NVIDIA driver is visible; verify with `python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"`.
- Example data are already normalized; for your own data, see `documentation/README.md` Section 1.4 on preprocessing.

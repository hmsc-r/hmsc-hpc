# Hmsc-HPC Project Instructions

Hmsc-HPC is a TensorFlow-based implementation of Hierarchical Modelling of Species Communities (HMSC). It provides a high-performance framework for Joint Species Distribution Modelling (JSDM), extending the capabilities of the original R-based Hmsc package with GPU acceleration.

## Project Overview

- **Core Technology**: TensorFlow 2.x, TensorFlow Probability, NumPy, Pandas, SciPy.
- **Main Goal**: Efficiently fit HMSC models using Gibbs sampling, leveraging GPU acceleration where available.
- **Key Components**:
    - `hmsc/`: The core Python package.
    - `hmsc/gibbs_sampler.py`: Contains the `GibbsSampler` class, which implements the main sampling loop as a `tf.Module` with `@tf.function` for graph execution.
    - `hmsc/updaters/`: Modular implementation of various Gibbs update steps (e.g., `updateBetaLambda`, `updateEta`, `updateSigma`).
    - `hmsc/run_gibbs_sampler.py`: Command-line interface for running the sampler.
    - `hmsc/utils/`: Utility functions for data handling, RDS file interop, and TensorFlow/Linear Algebra helpers.

## Building and Running

### Installation
The project uses `setup.py` for installation. To install in editable mode with development dependencies:
```bash
pip install -e . -r requirements_dev.txt
```

### Running the Sampler
The main entry point is `hmsc/run_gibbs_sampler.py`. It accepts various arguments for sampling parameters and input/output files:
```bash
python hmsc/run_gibbs_sampler.py \
    --input TF-init-obj.rds \
    --output TF-postList-obj.rds \
    --samples 100 \
    --transient 50 \
    --thin 1 \
    --chains 0 1
```
Use `python hmsc/run_gibbs_sampler.py --help` for a full list of options, including precision (`--fp 32` or `--fp 64`) and HMC parameters.

### Running Tests
Tests are located in `hmsc/test/` and use `pytest`.
```bash
pytest
```
CI is configured via GitHub Actions in `.github/workflows/ci.yml`.

## Development Conventions

- **TensorFlow Usage**: 
    - Prefer graph execution for performance. Use `@tf.function` on performance-critical methods like `sampling_routine`.
    - Be mindful of `tf.function` retracing; ensure inputs are consistent or use `TensorSpec` where appropriate.
    - Support for both `float32` and `float64` is required throughout the codebase (controlled by `dtype` arguments).
- **Data Interop**: 
    - The project heavily uses `.rds` files for compatibility with the R ecosystem. Interop is handled via the `rdata` package and `hmsc/utils/rds.py`.
- **Modular Updaters**: 
    - Each Gibbs update step is isolated in `hmsc/updaters/`. When adding or modifying parameters, update the corresponding updater and add/update its test in `hmsc/test/`.
- **Architecture**:
    - `GibbsSampler` inherits from `tf.Module` to manage stateful variables and functions.
    - `sampling_routine` uses `tf.while_loop` (via `tf.range` loop) for efficient execution of the MCMC chains.

## Important Files
- `README.rst`: High-level overview and links to the paper and documentation.
- `docs/csc_install.md`: Detailed installation instructions for high-performance computing (HPC) environments operated by CSC.
- `examples/`: Contains notebooks and scripts for learning how to use the package and benchmarking performance.

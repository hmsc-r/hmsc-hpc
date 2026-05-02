# Installing `hmsc-hpc` (Python) on CSC

This document shows how to install and update the `hmsc-hpc` Python package on CSC (Puhti, Mahti, Roihu), assuming you use CSC’s `tensorflow` module. The document specifically aims to be accessible for new `hmsc-hpc` and CSC cluster users without prior Python package management experience.

If you only need the minimal commands, read the **TL;DR** and skip the rest.

---

## TL;DR – Recommended virtual environment install (CSC)

**Use this when you are doing a fresh install in a new environment.**

```bash
# 0. List your CSC projects and respective directories
csc-workspaces

# 1. Go to a CSC project directory where you want to keep the environment
cd /projappl/<your_project>

# 2. Load CSC’s TensorFlow module
module load tensorflow/2.18

# 3. Create a virtual environment that can see the module’s packages
python3 -m venv --system-site-packages hmsc_tf_env

# 4. Activate the environment
source hmsc_tf_env/bin/activate

# 5. Install hmsc-hpc from GitHub (clean install)
pip install git+https://github.com/hmsc-r/hmsc-hpc.git

# 6. Quick sanity check: TensorFlow + GPU + hmsc-hpc import
python3 -c "import tensorflow as tf, hmsc; print('TF:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

To use later in a shell or in an `sbatch` script:

```bash
module load tensorflow/2.18
source /projappl/<your_project>/hmsc_tf_env/bin/activate
```

---

## Detailed installation guide

Start with making a fresh login to the CSC cluster (Puhti, Mahti, Roihu). You need to get acess to the shell inteface - can be availabe both via `ssh` connection and web interface.

### 1. Choose where to install

Typical CSC filesystem options:

- **Recommended (persistent project area)**:  
  `/projappl/<your_project>` – good default for environments shared within a project.
- **Scratch project area**:  
  `/scratch/<your_project>` – larger but temporary netdisk storage; forced cleaned up regularly.
- **Personal home**:  
  `$HOME` – most private but limited quota; avoid installing lots of packages here.

In the examples below, replace `<your_project>` with your CSC project, typically something like `project_1234567`. Use `csc-workspaces` command to list your available CSC projects.

---

### 2. Option A (recommended): dedicated virtual environment

This is best if you need to run different Python codes with various dependencies. The downside is that you shall keep in mind that you installed to the virtual environment and do not forget to activate it whenever running `hmsc-hpc`.

#### 2.1 Create and activate the environment

```bash
cd /projappl/<your_project>   # or /scratch/<your_project> or $HOME

# Load the CSC TensorFlow module you plan to use
module load tensorflow/2.18

# Create a venv that sees the module’s Python packages
python3 -m venv --system-site-packages hmsc_tf_env

# Activate it
source hmsc_tf_env/bin/activate
```

After the last comand you shall start observing `(hmsc_tf_env)` prefix on the left part of your console line.

**Sanity check 1 – confirm which Python is used:**

```bash
which python
```

The path should end with `.../hmsc_tf_env/bin/python`.

#### 2.2 Install `hmsc-hpc` from GitHub

For a **clean new environment** the simplest is to let `pip` pull the Python‑level dependencies:

```bash
pip install git+https://github.com/hmsc-r/hmsc-hpc.git
```

#### 2.3 Sanity check: TensorFlow + GPU + `hmsc-hpc`

Still inside the activated environment:

```bash
python3 -c "import tensorflow as tf; print('TF Version:', tf.__version__); print('GPU(s):', tf.config.list_physical_devices('GPU'))"
python3 -c "import hmsc"
```

- The first command should print the TensorFlow version and list at least one GPU
  (on GPU nodes).
- The second command should exit without errors.

#### 2.4 Using the environment later

Each time (in an interactive shell or inside an `sbatch` script):

```bash
module load tensorflow/2.18
source /projappl/<your_project>/hmsc_tf_env/bin/activate
```

After activation of virtual environment, `python` and `pip` refer to this environment. See the example batch script: [csc_sbatch_example.sh](csc_sbatch_example.sh). For more advanced `sbatch` scripting check CSC web guides.


---

### 3. Option B: install to a local user directory (no venv)

Use this if `venv` causes problems, or if you prefer a single user‑wide install. Given that almost all dependecies of `hmsc-hpc` are already covered by the `tensorflow` CSC module, this is a perfectly viable strategy. For other software this may lead to exceeding the user disk quota.


#### 3.1 Install `hmsc-hpc` into default directory

```bash
# Load TensorFlow module
module load tensorflow/2.18

# Install hmsc-hpc (and its Python-level dependencies)
pip install --user git+https://github.com/hmsc-r/hmsc-hpc.git
```

**Sanity check:**

```bash
python3 -c "import tensorflow as tf, hmsc; print('TF:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

#### 3.2 Using this installation later

In this case you only need to load the `tensorflow` CSC module whenever you aim to run `hmsc-hpc` (in an interactive shell or inside an `sbatch` script):

```bash
module load tensorflow/2.18
```

---

## Updating `hmsc-hpc`

The update procedure depends on whether you use a venv or a user‑dir install.

### A. Updating inside a virtual environment

1. Load the module and activate the same environment:

   ```bash
   module load tensorflow/2.18
   source /projappl/<your_project>/hmsc_tf_env/bin/activate
   ```

2. Update `hmsc-hpc`:

   ```bash
   pip install --upgrade --force-reinstall --no-deps git+https://github.com/hmsc-r/hmsc-hpc.git
   ```

   - `--upgrade` asks for a newer version if available.
   - `--force-reinstall` makes sure the package is reinstalled **even if the version number did not change** (helpful when a hotfix was pushed without bumping the version).
   - `--no-deps` avoids re‑installing heavy dependencies (such as TensorFlow) that are provided by the CSC module.

3. Quick check:

   ```bash
   python3 -c "import hmsc"
   ```

### B. Updating a user‑dir installation

1. Load the module:

   ```bash
   module load tensorflow/2.18
   ```

2. Run the same update command:

   ```bash
   pip install --upgrade --force-reinstall --no-deps git+https://github.com/hmsc-r/hmsc-hpc.git
   ```

3. Check that `hmsc` still imports correctly:

   ```bash
   python3 -c "import hmsc"
   ```

---

## Common sanity checks (summary)

Use these when debugging or verifying a setup:

```bash
# Which Python am I using?
which python

# Is TensorFlow visible and seeing GPUs? 
python3 -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

# Does hmsc-hpc import?
python3 -c "import hmsc"
```

# Final notes

1. Do not run `hmsc-hpc` in the login nodes! Check how to reserve an interactive session if you just start learning `hmsc-hpc`, need reactivity or not yet ready to proceed to `sbatch` scripting.
2. `hmsc-hpc` is designed for `tensorflow` routines being executed on GPU devices. It is still possible to run it on CPU node (e.g. for debug purpose) but the performance is expected to be much worse.
3. Try to avoid installing major dependencies, e.g. `tensorflow` and rely on the CSC modules whenever possible. There is a fair chance that you will install an incorrect or suboptimal software version. For instance, the CUDA libraries are often problematic. Also CSC strongly disencourages major custom installations in the manner as described above for `hmsc-hpc`.
4. Nowadays, combining CSC manual webpages with some AI assistant can greatly accelerate customisation of the provided generic examples for your specific use case. 
5. Do not hesitate to ask for advice if you feel you need it. Hmsc team can assist with common `hmsc-hpc` issues, and CSC runs weekly general user support sessions.

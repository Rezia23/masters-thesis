# Explainability Framework

A research framework for computing and comparing feature-importance
explanation methods on binary classification models.  

The framework

* implements 14 explanation methods (SHAP, LIME and 12 game-theoretic
  power indices),
* runs them in parallel over several datasets and configurations,
* writes one CSV per method into a results folder, and
* computes pairwise agreement, ranking, stability and useless-feature
  statistics from those CSVs.

## Implemented explanation methods

| Key        | Method                                                |
|------------|-------------------------------------------------------|
| `shap`     | SHAP (`KernelExplainer`)                              |
| `lime`     | LIME                                                  |
| `sv`       | Exact Shapley value (only feasible for ≤ ~10 features)|
| `bi`       | Banzhaf index (positive coalitions)                   |
| `bi_neg`   | Banzhaf index (negative coalitions)                   |
| `bi_full`  | Banzhaf index (combined)                              |
| `ssi`      | Shapley–Shubik index (positive)                       |
| `ssi_neg`  | Shapley–Shubik index (negative)                       |
| `ssi_full` | Shapley–Shubik index (combined)                       |
| `dp`       | Deegan–Packel index                                   |
| `hp`       | Holler–Packel index                                   |
| `ri`       | Responsibility index                                  |
| `rwi`      | Range-weighted index                                  |
| `bww`      | Banzhaf winning width                                 |


## Datasets and trained models

Trained model weights and pre-processed data are stored under
`training/<name>_assets/` and loaded by the matching user script in
`user_scripts/`.

| Workflow script                           | Dataset                            | Model                | # features |
|-------------------------------------------|-----------------------------------|----------------------|-----------|
| `breast_cancer_user_script.py`            | UCI Breast Cancer Wisconsin       | Small NN             | 10        |
| `breast_cancer_noisy_user_script.py`      | Breast Cancer + `Noise_Uniform`   | Small NN             | 11        |
| `german_credit_user_script.py`            | UCI Statlog German Credit         | Small NN             | 15        |
| `german_credit_balanced_user_script.py`   | German Credit (class-balanced)    | Random Forest        | 15        |
| `german_credit_noisy_user_script.py`      | German Credit + `Constant_Zero`   | Small NN             | 16        |
| `german_credit_noisy_balanced_user_script.py` | German Credit noisy + balanced| Random Forest        | 16        |
| `linear_user_script.py`                   | Synthetic linear model            | known weights        | 6         |
| `nonlinear_user_script.py`                | Synthetic nonlinear model         | Random Forest        | 7         |


## Project structure

```
explainer.py                # BackgroundSet + My_Explainer (computes the 14 indices)
background_method.py        # BackgroundMethod value object
helpers.py                  # load_workflow / save_feature_names / get_positive_samples
expl_statistics.py          # Ranking, agreement and stability statistics
run_experiment_with_pool.py # CLI runner (single config or JSON config-file in parallel)
run_stats.py                # CLI for computing aggregate statistics on results

experiments/                # Pre-configured experiment scripts (see below)
training/                   # Trainer scripts and saved model assets
user_scripts/               # One workflow module per dataset/model
results/                    # Output CSVs for each experiment (generated, not committed)
data/                       # Raw datasets (e.g. UCI German Credit)
```

### Experiment scripts

Each module under `experiments/` is a thin wrapper that prepares a list of
configs and forwards them to `run_experiment_with_pool`.  Run any of them
with `python -m experiments.<name>`.

| Script              | Purpose                                                                                   |
|---------------------|-------------------------------------------------------------------------------------------|
| `exp_basic.py`      | Baseline run of all methods on Breast Cancer and German Credit (kmeans:40 background).    |
| `exp_bg_sets.py`    | Sweeps several background-set strategies (`kmeans:N`, `kernel_density:N`, `random:N`, `all`). |
| `exp_pivot.py`      | Compares the `majority` and `veto` pivot rules for the power-index methods.               |
| `exp_stability.py`  | Repeats every method 10 times with re-sampled background sets to measure stability.       |
| `exp_useless.py`    | Adds a known-useless feature (uniform noise / constant 0) and checks that indices suppress it. |
| `exp_synthetic.py`  | Runs all methods on a synthetic linear model with known weights.                          |
| `exp_nonlinear.py`  | Runs all methods on a synthetic nonlinear Random Forest with known feature importances.   |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python ≥ 3.11 is required (developed and tested on 3.13).

## Running experiments

### Option A — pre-configured experiments

```bash
python -m experiments.exp_basic
python -m experiments.exp_bg_sets
python -m experiments.exp_pivot
python -m experiments.exp_stability
python -m experiments.exp_useless
python -m experiments.exp_synthetic
python -m experiments.exp_nonlinear
```

Most experiment scripts ship with `force=False`, so configurations whose
results already exist under `results/` are skipped; set `force=True` in the
script (or delete the target folder) to recompute.

### Option B — single ad-hoc CLI run

```bash
python run_experiment_with_pool.py \
    --workflow user_scripts/breast_cancer_user_script.py \
    --output results/quick_test \
    --background kmeans:10 \
    --pivot majority \
    --indices shap lime bi ssi dp hp ri \
    --max-samples 5 \
    --force
```

| Argument        | Default      | Description                                                                  |
|-----------------|-------------|------------------------------------------------------------------------------|
| `--workflow`    | *(required)*| Path to a user workflow script.                                              |
| `--output`      | *(required)*| Output folder for the result CSVs.                                           |
| `--background`  | `kmeans:10` | Background method: `kmeans:N`, `kernel_density:N`, `random:N`, or `all`.     |
| `--pivot`       | `majority`  | Pivot rule for the DP/HP/RI/BWW family: `majority` or `veto`.                |
| `--indices`     | all 14      | Subset of explanation methods to compute.                                    |
| `--max-samples` | none        | Limit the number of positive samples that get explained.                     |
| `--force`       | false       | Overwrite the output folder if it already exists.                            |
| `--max-workers` | one per cfg | Number of worker processes when used together with `--config-file`.          |
| `--config-file` | none        | JSON file with a list of configs; runs them in parallel.                     |

### Background-method syntax

| String              | Meaning                                                |
|---------------------|--------------------------------------------------------|
| `kmeans:40`         | 40 KMeans cluster centres                              |
| `kernel_density:40` | 40 samples drawn from a KDE fit on the training set    |
| `random:20`         | 20 random training samples drawn each call             |
| `all`               | Full background dataset                                |

## Computing statistics

After an experiment has produced its per-method CSVs, aggregate statistics
are computed by `run_stats.py`.  The script has one positional argument
(the results folder) and one mode flag per experiment family:

```bash
# Basic / pairwise agreement
python run_stats.py results/exp_basic/breast_cancer        --exp-basic
python run_stats.py results/exp_basic/german_credit        --exp-basic
python run_stats.py results/exp_basic/german_credit_rf     --exp-basic

# Background-set sensitivity
python run_stats.py results/exp_bg_sets/breast_cancer      --exp-bg
python run_stats.py results/exp_bg_sets/german_credit      --exp-bg
python run_stats.py results/exp_bg_sets/german_credit_rf   --exp-bg

# Pivot comparison
python run_stats.py results/exp_pivot/breast_cancer        --exp-pivot
python run_stats.py results/exp_pivot/german_credit        --exp-pivot
python run_stats.py results/exp_pivot/german_credit_rf     --exp-pivot

# Stability across 10 runs
python run_stats.py results/exp_stability/breast_cancer    --exp-stability
python run_stats.py results/exp_stability/german_credit    --exp-stability
python run_stats.py results/exp_stability/german_credit_rf --exp-stability

# Useless-feature suppression (pass the known-useless feature name)
python run_stats.py results/exp_useless/breast_cancer_noisy    --exp-useless Noise_Uniform
python run_stats.py results/exp_useless/german_credit_noisy    --exp-useless Constant_Zero
python run_stats.py results/exp_useless/german_credit_noisy_rf --exp-useless Constant_Zero

# Synthetic linear (provide the true weights, in feature order)
python run_stats.py results/exp_synthetic --exp-synthetic 10 5 0.5 0.01 0 -10

# Synthetic nonlinear
python run_stats.py results/exp_nonlinear --exp-nonlinear
```

Per-experiment outputs are written into `<folder>/stats/`.  For
`--exp-pivot`, per-index files are saved under `<folder>/stats/<index>/`
(including `uninformative_row_rate_per_method.csv` for each pivot rule);
combined comparison summaries are written to
`<folder>/stats/pivot_comparison_summary.csv` and
`<folder>/stats/pivot_uninformative_summary.csv`.

## Writing a custom workflow script

A workflow module must expose three functions:

```python
def load_everything() -> dict:
    """Load model + auxiliary data and return them as a single bundle."""
    return {
        "model": model,
        "feature_names": [...],   # optional, used in plots/CSVs
    }

def get_data() -> tuple:
    """Return (X_test, X_background) as numpy arrays."""
    ...
    return X_test, X_background

def predict(bundle: dict, X) -> np.ndarray:
    """Return predicted positive-class probabilities, shape (n_samples,) or (n_samples, 1)."""
    ...
```

## Tie-breaking convention

Whenever two features receive identical scores, the framework ranks them
equally.  Two `scipy.stats.rankdata` conventions are used:

* `method="dense"` for set-membership / placement statistics
  (`aggregate_and_save_rankings`, `topk_percentage`,
  `topk_placement_breakdown`).  Tied features share the same rank and the
  next distinct score gets the immediately following rank, so "place 2"
  always means "the second-best distinct score group".  This matches the
  convention used in the related literature.
* `method="average"` for the Spearman / Kendall correlation functions, which
  is the statistically standard treatment for tied ranks.


## Quick test

```bash
python run_experiment_with_pool.py \
    --workflow user_scripts/breast_cancer_user_script.py \
    --output results/_test \
    --background kmeans:10 \
    --indices bi ssi \
    --max-samples 3 \
    --force
```

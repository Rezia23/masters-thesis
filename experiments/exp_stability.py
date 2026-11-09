"""exp_stability.py - Stability experiment with multiple random runs."""

from run_experiment_with_pool import run_experiment_with_pool, parse_background

N_RUNS = 10
BG_N_SAMPLES_BREAST = 50  # size of the pre-sampled background set per run
BG_N_SAMPLES_GERMAN = 100
INDICES = [
    "shap", "lime", "bi", "ssi", "bi_neg", "ssi_neg",
    "bi_full", "ssi_full", "dp", "hp", "ri", "rwi", "bww"
]
BG = parse_background("all")  # overridden by pre-sampling; kept for clarity
PIVOT = "veto"

DATASETS_EXISTING = [
    ("user_scripts/breast_cancer_user_script.py", "breast_cancer"),
    ("user_scripts/german_credit_user_script.py", "german_credit"),
]

nn_configs = []
for workflow, ds_name in DATASETS_EXISTING:
    for run_idx in range(N_RUNS):
        nn_configs.append(dict(
            workflow_name=workflow,
            output_folder=f"results/exp_stability/{ds_name}/run_{run_idx:02d}",
            background=BG,
            pivot_method=PIVOT,
            indices=INDICES,
            background_n_samples=BG_N_SAMPLES_BREAST if ds_name == "breast_cancer" else BG_N_SAMPLES_GERMAN,
            background_seed=run_idx,
            force=True,
            verbose=False,
        ))

rf_configs = []
for run_idx in range(N_RUNS):
    rf_configs.append(dict(
        workflow_name="user_scripts/german_credit_balanced_user_script.py",
        output_folder=f"results/exp_stability/german_credit_rf/run_{run_idx:02d}",
        background=BG,
        pivot_method=PIVOT,
        indices=INDICES,
        background_n_samples=BG_N_SAMPLES_GERMAN,
        background_seed=run_idx,
        force=True,
        verbose=False,
    ))

if __name__ == "__main__":
    # NN configs run in parallel; RF configs run sequentially (max_workers=1)
    # so sklearn's internal parallelism can use all cores without conflict.
    run_experiment_with_pool(nn_configs, max_workers=4)
    run_experiment_with_pool(rf_configs, max_workers=1)

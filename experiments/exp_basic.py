"""exp_basic.py - Basic experiment comparing explanation methods on standard datasets."""

from run_experiment_with_pool import run_experiment_with_pool, parse_background

INDICES = [
    "shap", "lime", "bi", "ssi", "bi_neg", "ssi_neg",
    "bi_full", "ssi_full", "dp", "hp", "ri", "rwi", "bww"
]
BG = parse_background("kmeans:40")
PIVOT = "majority"

nn_configs = [
    dict(
        workflow_name="user_scripts/breast_cancer_user_script.py",
        output_folder="results/exp_basic/breast_cancer",
        background=BG,
        pivot_method=PIVOT,
        indices=INDICES,
        force=False,
        verbose=False,
    ),
    dict(
        workflow_name="user_scripts/german_credit_user_script.py",
        output_folder="results/exp_basic/german_credit",
        background=BG,
        pivot_method=PIVOT,
        indices=INDICES,
        force=False,
        verbose=False,
    ),
]

rf_configs = [
    dict(
        workflow_name="user_scripts/german_credit_balanced_user_script.py",
        output_folder="results/exp_basic/german_credit_rf",
        background=BG,
        pivot_method=PIVOT,
        indices=INDICES,
        force=True,
        verbose=False,
    ),
]

if __name__ == "__main__":
    # NN configs run in parallel; RF runs alone so sklearn gets all cores.
    run_experiment_with_pool(nn_configs, max_workers=2)
    run_experiment_with_pool(rf_configs, max_workers=1)

"""exp_pivot.py - Experiment comparing majority vs veto pivot methods."""

from run_experiment_with_pool import run_experiment_with_pool, parse_background

# SHAP and LIME are excluded: the majority/veto pivot rule only affects the
# power-index family (DP/HP/RI/BWW and the BI/SSI variants).
INDICES = [
    "bi", "ssi", "bi_neg", "ssi_neg",
    "bi_full", "ssi_full", "dp", "hp", "ri", "rwi", "bww"
]
BG = parse_background("kmeans:40")

DATASETS_EXISTING = [
    ("user_scripts/breast_cancer_user_script.py", "breast_cancer"),
    ("user_scripts/german_credit_user_script.py", "german_credit"),
]

nn_configs = []
for workflow, ds_name in DATASETS_EXISTING:
    for pivot in ["majority", "veto"]:
        nn_configs.append(dict(
            workflow_name=workflow,
            output_folder=f"results/exp_pivot/{ds_name}/{pivot}",
            background=BG,
            pivot_method=pivot,
            indices=INDICES,
            force=False,
            verbose=False,
        ))

rf_configs = []
for pivot in ["majority", "veto"]:
    rf_configs.append(dict(
        workflow_name="user_scripts/german_credit_balanced_user_script.py",
        output_folder=f"results/exp_pivot/german_credit_rf/{pivot}",
        background=BG,
        pivot_method=pivot,
        indices=INDICES,
        force=True,
        verbose=False,
    ))

if __name__ == "__main__":
    # NN configs run in parallel; RF configs run sequentially (max_workers=1)
    # so sklearn's internal parallelism can use all cores without conflict.
    run_experiment_with_pool(nn_configs, max_workers=4)
    run_experiment_with_pool(rf_configs, max_workers=1)

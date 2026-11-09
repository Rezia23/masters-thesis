"""exp_bg_sets.py - Experiment comparing different background set selection methods."""

from run_experiment_with_pool import run_experiment_with_pool, parse_background

INDICES = [
    "shap", "lime", "bi", "ssi", "bi_neg", "ssi_neg",
    "bi_full", "ssi_full", "dp", "hp", "ri", "rwi", "bww"
]
PIVOT = "veto"

BG_COMMON = [
    "kmeans:10", "kmeans:40", "kmeans:80",
    "kernel_density:10", "kernel_density:40", "random:20", "all"
]

DATASETS_NN = [
    ("user_scripts/breast_cancer_user_script.py", "breast_cancer", BG_COMMON),
    ("user_scripts/german_credit_user_script.py", "german_credit", BG_COMMON + ["kmeans:150"]),
]

DATASETS_RF = [
    ("user_scripts/german_credit_balanced_user_script.py", "german_credit_rf", BG_COMMON + ["kmeans:150"]),
]

nn_configs = []
for workflow, ds_name, bg_strings in DATASETS_NN:
    for bg_str in bg_strings:
        bg = parse_background(bg_str)
        label = bg_str.replace(":", "_")
        nn_configs.append(dict(
            workflow_name=workflow,
            output_folder=f"results/exp_bg_sets/{ds_name}/{label}",
            background=bg,
            pivot_method=PIVOT,
            indices=INDICES,
            force=True,
            verbose=False,
        ))

rf_configs = []
for workflow, ds_name, bg_strings in DATASETS_RF:
    for bg_str in bg_strings:
        bg = parse_background(bg_str)
        label = bg_str.replace(":", "_")
        rf_configs.append(dict(
            workflow_name=workflow,
            output_folder=f"results/exp_bg_sets/{ds_name}/{label}",
            background=bg,
            pivot_method=PIVOT,
            indices=INDICES,
            force=True,
            verbose=False,
        ))

if __name__ == "__main__":
    # NN configs run in parallel; RF configs run sequentially so sklearn gets all cores.
    run_experiment_with_pool(nn_configs, max_workers=2)
    run_experiment_with_pool(rf_configs, max_workers=1)

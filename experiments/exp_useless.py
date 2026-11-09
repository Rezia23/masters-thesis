"""exp_useless.py - Test that indices correctly suppress known-useless features.

Datasets:
  breast_cancer_noisy        - 10 real features + Noise_Uniform (random noise)
  german_credit_noisy        - 15 real features + Constant_Zero (zero-variance constant)
  german_credit_noisy_rf     - same as above but Random Forest model
"""

from run_experiment_with_pool import run_experiment_with_pool, parse_background

INDICES = [
    "shap", "lime", "bi", "ssi", "bi_neg", "ssi_neg",
    "bi_full", "ssi_full", "dp", "hp", "ri", "rwi", "bww"
]
BG_BREAST_CANCER = parse_background("kmeans:10")
BG_GERMAN_CREDIT = parse_background("kmeans:40")
PIVOT = "veto"

nn_configs = [
    dict(
        workflow_name="user_scripts/breast_cancer_noisy_user_script.py",
        output_folder="results/exp_useless/breast_cancer_noisy",
        background=BG_BREAST_CANCER,
        pivot_method=PIVOT,
        indices=INDICES,
        force=True,
        verbose=False,
    ),
    dict(
        workflow_name="user_scripts/german_credit_noisy_user_script.py",
        output_folder="results/exp_useless/german_credit_noisy",
        background=BG_GERMAN_CREDIT,
        pivot_method=PIVOT,
        indices=INDICES,
        force=True,
        verbose=False,
    ),
]

rf_configs = [
    dict(
        workflow_name="user_scripts/german_credit_noisy_balanced_user_script.py",
        output_folder="results/exp_useless/german_credit_noisy_rf",
        background=BG_GERMAN_CREDIT,
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

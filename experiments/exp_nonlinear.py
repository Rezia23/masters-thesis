"""exp_nonlinear.py - Experiment on synthetic nonlinear model."""

from run_experiment_with_pool import run_experiment_with_pool, parse_background

INDICES = [
    "shap", "lime", "bi", "ssi", "bi_neg", "ssi_neg",
    "bi_full", "ssi_full", "dp", "hp", "ri", "rwi", "bww"
]
BG = parse_background("kmeans:30")
PIVOT = "veto"

configs = [
    dict(
        workflow_name="user_scripts/nonlinear_user_script.py",
        output_folder="results/exp_nonlinear",
        background=BG,
        pivot_method=PIVOT,
        indices=INDICES,
        force=True,
        verbose=False,
    )
]

if __name__ == "__main__":
    run_experiment_with_pool(configs, max_workers=1)

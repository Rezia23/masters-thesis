"""exp_grid_comparison.py - Compare binary-search vs grid-scan RWI/BWW.

Runs rwi, bww, rwi_grid, bww_grid (plus bi to build the shared DP table) on
the two fastest datasets: synthetic linear and synthetic nonlinear.  Results
land in results/exp_grid_comparison/ for manual or scripted comparison.
"""

from run_experiment_with_pool import run_experiment_with_pool, parse_background

INDICES = ["bi", "rwi", "bww", "rwi_grid", "bww_grid"]
BG = parse_background("kmeans:30")
PIVOT = "veto"

configs = [
    dict(
        workflow_name="user_scripts/linear_user_script.py",
        output_folder="results/exp_grid_comparison/linear",
        background=BG,
        pivot_method=PIVOT,
        indices=INDICES,
        force=True,
        verbose=True,
    ),
    dict(
        workflow_name="user_scripts/nonlinear_user_script.py",
        output_folder="results/exp_grid_comparison/nonlinear",
        background=BG,
        pivot_method=PIVOT,
        indices=INDICES,
        force=True,
        verbose=True,
    ),
    dict(
        workflow_name="user_scripts/breast_cancer_user_script.py",
        output_folder="results/exp_grid_comparison/breast_cancer",
        background=BG,
        pivot_method=PIVOT,
        indices=INDICES,
        force=True,
        verbose=True,
    ),
]

if __name__ == "__main__":
    run_experiment_with_pool(configs, max_workers=1)

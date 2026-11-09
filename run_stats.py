"""Compute aggregate statistics on experiment results.

The script reads per-method CSV files produced by ``run_experiment_with_pool``
and writes summary tables, correlation heatmaps and stability metrics into a
``stats/`` subfolder of the input directory.

Each ``--exp-*`` flag selects a different analysis tailored to the
corresponding experiment under ``experiments/``.  Without any flag the script
falls back to the basic pairwise-agreement summary.

Usage examples
--------------
::

    python run_stats.py --exp-basic     results/exp_basic/breast_cancer
    python run_stats.py --exp-stability results/exp_stability/breast_cancer
    python run_stats.py --exp-bg        results/exp_bg_sets/breast_cancer
    python run_stats.py --exp-pivot     results/exp_pivot/breast_cancer
    python run_stats.py --exp-useless Noise_Uniform \\
        results/exp_useless/breast_cancer_noisy
    python run_stats.py --exp-synthetic 10 5 0.5 0.01 0 -10 \\
        results/exp_synthetic
    python run_stats.py --exp-nonlinear results/exp_nonlinear
"""

import argparse
import json
import os
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from expl_statistics import (
    compare_pivot_methods,
    ground_truth_metrics,
    pairwise_kendalls_tau,
    pairwise_kendalls_tau_nonzero,
    pairwise_ranking_agreement,
    pairwise_ranking_agreement_nonzero,
    rank_inversion_rate,
    spearman_kendall_across_runs,
    topk_jaccard_across_configs,
    topk_jaccard_across_runs,
    topk_jaccard_nonzero,
    topk_percentage,
    topk_placement_breakdown,
    uninformative_row_rate_per_method,
    useless_feature_metrics,
)


def _parse_line(line: str) -> Any:
    """Parse a single CSV line into a Python value."""
    line = line.strip()
    if not line:
        return None

    # Try JSON first
    try:
        return json.loads(line)
    except Exception:
        pass

    s = line
    if s.startswith("array(") and s.endswith(")"):
        s = s[len("array("):-1]
    s = s.strip().lstrip("[(").rstrip(")]\n\r ")
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]

    parts = [p for p in s.replace(",", " ").split() if p]
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except Exception:
            continue

    if nums:
        return nums

    return line  # last resort: raw string


def _load_feature_names(folder: str, n_features: int = 0) -> List[str]:
    """Try to read feature names from folder; fall back to generic names."""
    for fname in ("feature_names.txt", "features.txt"):
        fp = os.path.join(folder, fname)
        if os.path.exists(fp):
            with open(fp, encoding="utf-8") as fh:
                names = [l.strip() for l in fh if l.strip()]
            if names:
                return names
    return [f"Feature {i}" for i in range(n_features)]


def read_results_from_folder(folder: str) -> Dict[str, List[Any]]:
    """Read per-method result CSVs from a folder."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    results: Dict[str, List[Any]] = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".csv"):
            continue
        method = os.path.splitext(fname)[0]
        path = os.path.join(folder, fname)
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        parsed = [v for line in lines if (v := _parse_line(line)) is not None]
        results[method] = parsed

    return results


def compute_and_save_stats(foldername: str) -> None:
    """Compute pairwise agreement statistics for a result folder."""
    results_folder = os.path.join(foldername, "stats")
    os.makedirs(results_folder, exist_ok=True)

    results = read_results_from_folder(foldername)
    if not results:
        print(f"[run_stats] No CSV results found in {foldername}")
        return

    n_features = 0
    for v in results.values():
        if v and isinstance(v[0], (list, tuple)):
            n_features = len(v[0])
            break
    feature_names = _load_feature_names(foldername, n_features)

    pairwise_ranking_agreement(results, results_folder)
    pairwise_kendalls_tau(results, results_folder)
    topk_percentage(results, results_folder)

    print(f"[run_stats] stats saved to {results_folder}/")


def compute_stats_basic(foldername: str, indices: List[str] = None) -> None:
    """Compute exp_basic statistics, handling the high rate of all-zero rows.

    Runs two complementary passes:
    1. **All instances** - existing Spearman / Kendall with neutral mid-rank for
       zeros (gives an overview that includes abstentions).
    2. **Non-zero only** - Spearman, Kendall and top-k Jaccard computed only on
       instances where *both* methods returned a non-zero vector.  This is the
       more honest measure of ranking agreement.

    Additionally reports:
    - Zero/constant-row rate per method (``uninformative_row_rate_per_method.csv``).
    - Top-k percentage (``topk_perc_all.csv`` / ``topk_perc_nonzero.csv``).

    Parameters
    ----------
    foldername : str
        Folder with per-method CSVs.
    indices : list[str], optional
        If given, only these method keys are used in the pairwise and average
        score statistics.  The uninformative-row report always covers all methods
        present in the folder so the full picture is visible.
    """
    results_folder = os.path.join(foldername, "stats")
    os.makedirs(results_folder, exist_ok=True)

    results = read_results_from_folder(foldername)
    if not results:
        print(f"[run_stats] No CSV results found in {foldername}")
        return

    n_features = 0
    for v in results.values():
        if v and isinstance(v[0], (list, tuple)):
            n_features = len(v[0])
            break
    feature_names = _load_feature_names(foldername, n_features)

    # 1. Uninformative-row report - always over ALL methods present
    df_zero = uninformative_row_rate_per_method(results, results_folder)
    print("[run_stats] uninformative row rates per method:")
    for _, row in df_zero.iterrows():
        print(
            f"  {row['method']:20s}  "
            f"{row['uninformative_rate']:.1%} uninformative  "
            f"(zero: {int(row['n_zero'])}, constant: {int(row['n_constant_nonzero'])}, "
            f"total: {int(row['n_total'])})"
        )

    # Filter to requested indices for the remaining statistics
    if indices is not None:
        missing = [k for k in indices if k not in results]
        if missing:
            print(f"[run_stats] WARNING: requested indices not found in folder: {missing}")
        results_filtered = {k: results[k] for k in indices if k in results}
        suffix = "_filtered"
    else:
        results_filtered = results
        suffix = ""

    if not results_filtered:
        print("[run_stats] No matching indices after filtering - aborting.")
        return

    print(
        f"[run_stats] running pairwise stats on: {list(results_filtered.keys())}"
    )

    # 2. Top-k percentage
    topk_percentage(results_filtered, results_folder, suffix=suffix)

    # 3. Pairwise agreement - all instances (mid-rank imputation for zeros)
    print("[run_stats] computing pairwise Spearman / Kendall (all instances) ...")
    pairwise_ranking_agreement(results_filtered, results_folder, suffix=suffix)
    pairwise_kendalls_tau(results_filtered, results_folder, suffix=suffix)

    # 4. Pairwise agreement - non-zero instances only
    print("[run_stats] computing pairwise Spearman / Kendall (non-zero only) ...")
    pairwise_ranking_agreement_nonzero(results_filtered, results_folder, suffix=suffix)
    pairwise_kendalls_tau_nonzero(results_filtered, results_folder, suffix=suffix)
    topk_jaccard_nonzero(results_filtered, results_folder, top_k=3, suffix=suffix)

    print(f"[run_stats] exp_basic stats saved to {results_folder}/")


# ---------------------------------------------------------------------------
# Experiment 5: stability across runs
def compute_stats_experiment5(foldername: str) -> None:
    """Compute stability statistics for exp_stability results."""
    run_dirs = sorted(
        d for d in os.listdir(foldername)
        if d.startswith("run_") and os.path.isdir(os.path.join(foldername, d))
    )
    if len(run_dirs) < 2:
        print(f"[run_stats] fewer than 2 run_* folders in {foldername}, skipping")
        return

    first_run_dir = os.path.join(foldername, run_dirs[0])
    index_names = sorted(
        os.path.splitext(fn)[0]
        for fn in os.listdir(first_run_dir)
        if fn.endswith(".csv")
    )
    if not index_names:
        print(f"[run_stats] no CSVs found in {first_run_dir}")
        return

    feature_names = _load_feature_names(first_run_dir)

    summary_rows = []
    for idx_name in index_names:
        runs_dict: Dict[str, np.ndarray] = {}
        for run_label in run_dirs:
            csv_path = os.path.join(foldername, run_label, f"{idx_name}.csv")
            if not os.path.exists(csv_path):
                continue
            runs_dict[run_label] = np.array(pd.read_csv(csv_path, header=None))

        if len(runs_dict) < 2:
            print(f"  [{idx_name}] fewer than 2 runs found, skipping")
            continue

        n_features = list(runs_dict.values())[0].shape[1]
        feat_names = feature_names if feature_names else [f"Feature {i}" for i in range(n_features)]

        out_dir = os.path.join(foldername, "stats", idx_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"  [{idx_name}] {len(runs_dict)} runs → {out_dir}")
        avg_jac, _ = topk_jaccard_across_runs(runs_dict, out_dir, top_k=3)
        avg_inv, _ = rank_inversion_rate(runs_dict, feat_names, out_dir)
        avg_sp, avg_kt, _ = spearman_kendall_across_runs(runs_dict, out_dir)
        summary_rows.append({
            "index": idx_name,
            "topk_jaccard_top3": avg_jac,
            "rank_consistency": 1.0 - avg_inv,
            "mean_spearman": avg_sp,
            "mean_kendall": avg_kt,
        })

    if summary_rows:
        stats_dir = os.path.join(foldername, "stats")
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(stats_dir, "stability_summary.csv"), index=False,
        )
        print(f"[run_stats] summary table saved to {stats_dir}/stability_summary.csv")

    print(f"[run_stats] exp_stability stats saved in {foldername}/stats/")


# ---------------------------------------------------------------------------
# Experiment: background-set comparison
# ---------------------------------------------------------------------------
def compute_stats_bg_sets(foldername: str) -> None:
    """Compute cross-background-config statistics for exp_bg_sets results.

    Expected layout::

        foldername/
            all/         *.csv      (reference)
            kmeans_10/   *.csv
            kmeans_40/   *.csv
            ...

    For every index the function computes pairwise Spearman, Kendall, and
    top-k Jaccard across configs, plus uninformative-row rates.

    Additionally produces three **summary tables** (saved in ``stats/``):

    - ``bg_vs_all_spearman.csv``
    - ``bg_vs_all_kendall.csv``
    - ``bg_vs_all_jaccard_top3.csv``

    Each table has indices as rows and BG configs (excl. "all") as columns,
    showing how each approximation compares to the full background.
    """
    config_dirs = sorted(
        d for d in os.listdir(foldername)
        if os.path.isdir(os.path.join(foldername, d))
        and not d.startswith("stats")
    )
    if len(config_dirs) < 2:
        print(f"[run_stats] fewer than 2 config folders in {foldername}, skipping")
        return

    first_dir = os.path.join(foldername, config_dirs[0])
    index_names = sorted(
        os.path.splitext(fn)[0]
        for fn in os.listdir(first_dir)
        if fn.endswith(".csv")
    )

    # Collectors for summary tables (index_name → {config → value})
    spearman_rows = {}
    kendall_rows = {}
    jaccard_rows = {}

    for idx_name in index_names:
        configs_dict: Dict[str, np.ndarray] = {}
        for cfg_label in config_dirs:
            csv_path = os.path.join(foldername, cfg_label, f"{idx_name}.csv")
            if not os.path.exists(csv_path):
                continue
            configs_dict[cfg_label] = np.array(pd.read_csv(csv_path, header=None))

        if len(configs_dict) < 2:
            print(f"  [{idx_name}] fewer than 2 configs found, skipping")
            continue

        out_dir = os.path.join(foldername, "stats", idx_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"  [{idx_name}] {len(configs_dict)} configs → {out_dir}")
        _, jaccard_df = topk_jaccard_across_configs(configs_dict, out_dir, top_k=3)

        configs_as_lists = {cfg: arr.tolist() for cfg, arr in configs_dict.items()}
        uninformative_row_rate_per_method(configs_as_lists, out_dir)

        spearman_matrix, _ = pairwise_ranking_agreement(
            configs_as_lists, out_dir, suffix="_configs",
        )
        kendall_matrix, _ = pairwise_kendalls_tau(
            configs_as_lists, out_dir, suffix="_configs",
        )

        # Extract "all vs X" values for the summary tables
        sorted_configs = sorted(configs_dict.keys())
        if "all" in sorted_configs:
            all_idx = sorted_configs.index("all")
            other_configs = [c for c in sorted_configs if c != "all"]

            sp_row = {}
            kt_row = {}
            for c in other_configs:
                c_idx = sorted_configs.index(c)
                sp_row[c] = float(spearman_matrix[all_idx, c_idx])
                kt_row[c] = float(kendall_matrix[all_idx, c_idx])
            spearman_rows[idx_name] = sp_row
            kendall_rows[idx_name] = kt_row

            jac_row = {}
            jac_pairs = jaccard_df[jaccard_df["config_i"] != "OVERALL"]
            for _, r in jac_pairs.iterrows():
                ci, cj = r["config_i"], r["config_j"]
                if ci == "all":
                    jac_row[cj] = float(r["mean_jaccard"])
                elif cj == "all":
                    jac_row[ci] = float(r["mean_jaccard"])
            jaccard_rows[idx_name] = jac_row

    # Write summary tables
    stats_dir = os.path.join(foldername, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    for name, rows_dict in [
        ("bg_vs_all_spearman", spearman_rows),
        ("bg_vs_all_kendall", kendall_rows),
        ("bg_vs_all_jaccard_top3", jaccard_rows),
    ]:
        if rows_dict:
            df = pd.DataFrame.from_dict(rows_dict, orient="index")
            df.index.name = "index"
            df = df[sorted(df.columns)]
            df.to_csv(os.path.join(stats_dir, f"{name}.csv"))
            print(f"[run_stats] saved {stats_dir}/{name}.csv")

            # Heatmap
            fig, ax = plt.subplots(figsize=(max(6, len(df.columns) * 0.9), max(4, len(df) * 0.45)))
            sns.heatmap(df, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                        linewidths=0.5, ax=ax)
            ax.set_ylabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
            fig.tight_layout()
            fig.savefig(os.path.join(stats_dir, f"{name}.png"), dpi=150)
            plt.close(fig)
            print(f"[run_stats] saved {stats_dir}/{name}.png")

    print(f"[run_stats] exp_bg_sets stats saved in {foldername}/stats/")


# ---------------------------------------------------------------------------
# Experiment: pivot comparison
# ---------------------------------------------------------------------------
def compute_stats_pivot(foldername: str) -> None:
    """Compare majority vs veto results for every index in exp_pivot results.

    Expected layout::

        foldername/
            majority/   *.csv
            veto/       *.csv

    Saves:
        foldername/stats/<index>/pivot_comparison_summary.csv
        foldername/stats/<index>/pivot_per_sample_correlations.csv
        foldername/stats/<index>/pivot_score_ratio.csv
        foldername/stats/<index>/uninformative_row_rate_per_method.csv
        foldername/stats/pivot_comparison_summary.csv
        foldername/stats/pivot_uninformative_summary.csv
    """
    maj_dir  = os.path.join(foldername, "majority")
    veto_dir = os.path.join(foldername, "veto")
    stats_dir = os.path.join(foldername, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    for d in (maj_dir, veto_dir):
        if not os.path.isdir(d):
            print(f"[run_stats] missing directory {d}, skipping pivot comparison")
            return

    index_names = sorted(
        os.path.splitext(fn)[0]
        for fn in os.listdir(maj_dir)
        if fn.endswith(".csv")
    )
    feature_names = _load_feature_names(maj_dir)
    summary_rows = []
    uninformative_rows = []

    for idx_name in index_names:
        maj_csv  = os.path.join(maj_dir,  f"{idx_name}.csv")
        veto_csv = os.path.join(veto_dir, f"{idx_name}.csv")
        if not (os.path.exists(maj_csv) and os.path.exists(veto_csv)):
            continue

        maj_arr  = np.array(pd.read_csv(maj_csv,  header=None))
        veto_arr = np.array(pd.read_csv(veto_csv, header=None))

        n_features = maj_arr.shape[1]
        feat_names = feature_names if feature_names else [f"Feature {i}" for i in range(n_features)]

        out_dir = os.path.join(foldername, "stats", idx_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"  [{idx_name}] comparing majority vs veto → {out_dir}")
        df_zero = uninformative_row_rate_per_method(
            {"majority": maj_arr, "veto": veto_arr},
            out_dir,
        )
        summary = compare_pivot_methods(maj_arr, veto_arr, feat_names, out_dir)
        zero_by_method = df_zero.set_index("method")
        summary_rows.append({
            "index": idx_name,
            "spearman_mean": summary["spearman_mean"],
            "kendall_mean": summary["kendall_mean"],
            "top1_flip_rate": summary["top1_flip_rate"],
            "top3_jaccard_mean": summary["topk_jaccard_mean"],
            "n_samples": maj_arr.shape[0],
        })
        uninformative_rows.append({
            "index": idx_name,
            "n_samples": maj_arr.shape[0],
            "majority_n_zero": zero_by_method.loc["majority", "n_zero"],
            "majority_n_constant_nonzero": zero_by_method.loc["majority", "n_constant_nonzero"],
            "majority_n_uninformative": zero_by_method.loc["majority", "n_uninformative"],
            "majority_uninformative_rate": zero_by_method.loc["majority", "uninformative_rate"],
            "veto_n_zero": zero_by_method.loc["veto", "n_zero"],
            "veto_n_constant_nonzero": zero_by_method.loc["veto", "n_constant_nonzero"],
            "veto_n_uninformative": zero_by_method.loc["veto", "n_uninformative"],
            "veto_uninformative_rate": zero_by_method.loc["veto", "uninformative_rate"],
        })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(stats_dir, "pivot_comparison_summary.csv"),
            index=False,
        )
    if uninformative_rows:
        pd.DataFrame(uninformative_rows).to_csv(
            os.path.join(stats_dir, "pivot_uninformative_summary.csv"),
            index=False,
        )

    print(f"[run_stats] exp_pivot stats saved in {foldername}/stats/")


# ---------------------------------------------------------------------------
# Experiment: useless feature
# ---------------------------------------------------------------------------
def compute_stats_useless(foldername: str, useless_feature_name: str) -> None:
    """Evaluate suppression of a known-useless feature.

    Parameters
    ----------
    foldername : str
        Folder with per-method CSVs (same format as exp_basic output).
    useless_feature_name : str
        Exact name of the useless feature as it appears in ``feature_names.txt``,
        e.g. ``"Noise_Uniform"`` or ``"Constant_Zero"``.
    """
    results_folder = os.path.join(foldername, "stats")
    os.makedirs(results_folder, exist_ok=True)

    results = read_results_from_folder(foldername)
    if not results:
        print(f"[run_stats] No CSV results found in {foldername}")
        return

    n_features = 0
    for v in results.values():
        if v and isinstance(v[0], (list, np.ndarray)):
            n_features = len(v[0])
            break
    feature_names = _load_feature_names(foldername, n_features)

    if useless_feature_name not in feature_names:
        print(
            f"[run_stats] '{useless_feature_name}' not found in feature names: {feature_names}"
        )
        return

    useless_idx = feature_names.index(useless_feature_name)

    # Convert list-of-lists to numpy arrays
    results_np: Dict[str, np.ndarray] = {
        k: np.array(v) for k, v in results.items()
    }

    print(f"  useless feature: '{useless_feature_name}' (index {useless_idx})")
    useless_feature_metrics(results_np, useless_idx, feature_names, results_folder)
    print(f"[run_stats] exp_useless stats saved in {results_folder}/")


# ---------------------------------------------------------------------------
# Experiment: synthetic / ground-truth
# ---------------------------------------------------------------------------
def compute_stats_synthetic(foldername: str, true_weights: List[float]) -> None:
    """Evaluate methods against known ground-truth weights.

    Parameters
    ----------
    foldername : str
        Folder with per-method CSVs (exp_synthetic output).
    true_weights : list[float]
        The true model weights in feature order, e.g. ``[10, 5, 0.5, 0.01, 0, -10]``.
    """
    results_folder = os.path.join(foldername, "stats")
    os.makedirs(results_folder, exist_ok=True)

    results = read_results_from_folder(foldername)
    if not results:
        print(f"[run_stats] No CSV results found in {foldername}")
        return

    n_features = len(true_weights)
    feature_names = _load_feature_names(foldername, n_features)

    results_np: Dict[str, np.ndarray] = {
        k: np.array(v) for k, v in results.items()
    }

    ground_truth_metrics(results_np, np.array(true_weights), feature_names, results_folder)
    uninformative_row_rate_per_method(results, results_folder)
    topk_placement_breakdown(
        results_np, feature_names, results_folder,
        exclude_methods={"bi_neg", "ssi_neg"},
    )
    print(f"[run_stats] exp_synthetic stats saved in {results_folder}/")


# ---------------------------------------------------------------------------
# Experiment: nonlinear synthetic
# ---------------------------------------------------------------------------
def compute_stats_nonlinear(foldername: str) -> None:
    """Compute stats for the nonlinear synthetic experiment.

    Combines basic pairwise-agreement statistics, useless-feature
    suppression analysis, and ground-truth comparison using the RF model's
    built-in Gini feature importances as ground truth.
    """
    compute_stats_basic(foldername)
    compute_stats_useless(foldername, "Useless_Feat")

    # --- ground-truth comparison via RF feature importances ---
    results = read_results_from_folder(foldername)
    if not results:
        return

    n_features = 7
    feature_names = _load_feature_names(foldername, n_features)

    assets_path = os.path.join(
        os.path.dirname(__file__), "training", "nonlinear_assets", "processed_data.joblib"
    )
    data_assets = joblib.load(assets_path)
    gt_importance = np.array(data_assets["feature_importances"])

    results_np = {k: np.array(v) for k, v in results.items()}
    results_folder = os.path.join(foldername, "stats")
    ground_truth_metrics(results_np, gt_importance, feature_names, results_folder)
    topk_placement_breakdown(
        results_np, feature_names, results_folder,
        exclude_methods={"bi_neg", "ssi_neg"},
    )
    print(f"[run_stats] nonlinear ground-truth comparison saved to {results_folder}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute aggregate statistics on experiment results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("folder", help="Folder containing experiment results")
    parser.add_argument(
        "--exp-basic", action="store_true",
        help="Basic mode: zero-rate report + pairwise agreement on all and non-zero-only instances",
    )
    parser.add_argument(
        "--indices", nargs="+", metavar="METHOD",
        help="For --exp-basic: restrict pairwise/average-score stats to these method names, "
             "e.g. --indices shap lime.  The uninformative-row report always covers all methods.",
    )
    parser.add_argument(
        "--exp-stability", dest="exp_stability", action="store_true",
        help="Stability mode: Jaccard + score std + CV + rank-inversion across run_* subfolders",
    )
    parser.add_argument(
        "--exp-bg", action="store_true",
        help="Background-set mode: score variance + top-k Jaccard across config subfolders",
    )
    parser.add_argument(
        "--exp-pivot", action="store_true",
        help="Pivot mode: compare majority/ vs veto/ subfolders",
    )
    parser.add_argument(
        "--exp-useless", metavar="FEATURE_NAME",
        help="Useless-feature mode: name of the known-useless feature, "
             "e.g. 'Noise_Uniform' or 'Constant_Zero'",
    )
    parser.add_argument(
        "--exp-synthetic", metavar="WEIGHTS", nargs="+", type=float,
        help="Synthetic mode: space-separated true model weights, "
             "e.g. --exp-synthetic 10 5 0.5 0.01 0 -10",
    )
    parser.add_argument(
        "--exp-nonlinear", action="store_true",
        help="Nonlinear synthetic mode: pairwise agreement + useless-feature suppression",
    )
    args = parser.parse_args()

    if args.exp_basic:
        compute_stats_basic(args.folder, indices=args.indices)
    elif args.exp_stability:
        compute_stats_experiment5(args.folder)
    elif args.exp_bg:
        compute_stats_bg_sets(args.folder)
    elif args.exp_pivot:
        compute_stats_pivot(args.folder)
    elif args.exp_useless:
        compute_stats_useless(args.folder, args.exp_useless)
    elif args.exp_synthetic:
        compute_stats_synthetic(args.folder, args.exp_synthetic)
    elif args.exp_nonlinear:
        compute_stats_nonlinear(args.folder)
    else:
        compute_and_save_stats(args.folder)
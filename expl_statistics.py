"""Statistics and ranking utilities for explanation methods.

This module provides aggregation, correlation, top-k agreement and
ground-truth comparison metrics used by ``run_stats.py`` to summarise the raw
per-sample explanation scores produced by ``explainer.py``.

Tie handling
------------
Ranking-based metrics use ``scipy.stats.rankdata`` with one of two
conventions, chosen for what each metric needs:

* ``method='dense'`` for set-membership / placement statistics
  (:func:`aggregate_and_save_rankings`, :func:`topk_percentage`,
  :func:`topk_placement_breakdown`).  Tied features share the same rank and
  the next distinct value gets the immediately following rank, so place
  numbers correspond to *distinct score groups* (place 1 = best group,
  place 2 = next group, ...).  This matches the convention used in the
  related literature.
* ``method='average'`` for Spearman / Kendall correlation functions, which
  is the statistically standard treatment for tied ranks.

Top-k set selection uses :func:`_topk_set_tied`, which expands the set to
include every feature whose score equals the k-th largest value, so ties at
the boundary do not produce arbitrary inclusion order.
"""

import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, rankdata, spearmanr


def _topk_set_tied(arr, k):
    """Return the set of feature indices in the top-k, expanding ties.

    Any feature whose score is greater than or equal to the k-th largest score
    is included.  When several features share the boundary score, the returned
    set has more than ``k`` elements; this avoids the arbitrary tie-breaking
    that ``np.argsort`` would otherwise introduce.
    """
    arr = np.asarray(arr).ravel()
    if arr.size == 0:
        return set()
    k = min(k, arr.size)
    threshold = np.partition(arr, -k)[-k]
    return set(np.where(arr >= threshold)[0].tolist())


def _argmax_set_tied(arr):
    """Return all indices tied for the highest value in ``arr``."""
    arr = np.asarray(arr).ravel()
    if arr.size == 0:
        return set()
    m = arr.max()
    return set(np.where(arr == m)[0].tolist())


def aggregate_and_save_rankings(explanations_dict, feature_names, save_folder="results"):
    """Compute per-method feature rankings and save them to CSV.

    For each method the mean absolute score per feature is computed across all
    samples and features are assigned dense ranks.  The resulting table has one
    column per method and one row per rank position.
    """
    os.makedirs(save_folder, exist_ok=True)
    csv_path = os.path.join(save_folder, "feature_rankings_per_method.csv")

    n_features = len(feature_names)
    rankings_output = {}

    for method, values in explanations_dict.items():
        abs_vals = np.abs(np.asarray(values))
        if abs_vals.ndim > 1:
            abs_vals = np.mean(abs_vals, axis=0)

        ranks = rankdata(-abs_vals, method="dense")
        order = np.argsort(ranks, kind="stable")
        rankings_output[method] = [
            f"{feature_names[i]} ({abs_vals[i]:.4f}) [rank {int(ranks[i])}]"
            for i in order
        ]

    df_ranks = pd.DataFrame(rankings_output)
    df_ranks.index = [f"Position {i + 1}" for i in range(n_features)]
    df_ranks.to_csv(csv_path, index_label="Position")

    print(f"Individual rankings saved to CSV: {csv_path}")
    return df_ranks


def topk_percentage(result, foldername, top_k=3, suffix=""):
    """Per-method, per-feature percentage of appearances in the top-k positions.

    For every position 1..``top_k`` the function counts, for each feature, the
    fraction of samples in which that feature was assigned that rank (dense
    tie-breaking).  Two CSVs are written: one over all instances and one
    restricted to instances where at least two distinct scores were observed.
    """
    all_methods_data = []
    nonzero_methods_data = []

    for method_name, score_lists in result.items():
        score_lists = list(score_lists)
        total = len(score_lists)

        if total == 0:
            n_features = 0
        else:
            n_features = int(max(np.asarray(s).ravel().shape[0] for s in score_lists))

        # counts_pos[pos, feature] = times the feature was assigned rank pos+1
        counts_pos = np.zeros((top_k, n_features), dtype=float)
        counts_pos_nonzero = np.zeros((top_k, n_features), dtype=float)
        nonzero_total = 0

        for scores in score_lists:
            arr = np.asarray(scores).ravel()
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if arr.size == 0:
                continue

            ranks = rankdata(-arr, method="dense")

            for pos in range(top_k):
                target_rank = pos + 1
                feats_at_rank = np.where(ranks == target_rank)[0]
                for f_idx in feats_at_rank:
                    if f_idx < n_features:
                        counts_pos[pos, f_idx] += 1

            informative = not np.all(arr == arr[0])
            if informative:
                nonzero_total += 1
                for pos in range(top_k):
                    target_rank = pos + 1
                    feats_at_rank = np.where(ranks == target_rank)[0]
                    for f_idx in feats_at_rank:
                        if f_idx < n_features:
                            counts_pos_nonzero[pos, f_idx] += 1

        perc_pos = (counts_pos / total) * 100 if total > 0 else np.zeros_like(counts_pos)
        perc_pos_nonzero = (
            (counts_pos_nonzero / nonzero_total) * 100
            if nonzero_total > 0 else np.zeros_like(counts_pos_nonzero)
        )

        for f_idx in range(n_features):
            row_all = {"method": method_name, "feature_idx": f_idx}
            row_nonzero = {"method": method_name, "feature_idx": f_idx}
            for pos in range(top_k):
                row_all[f"rank_{pos + 1}_percent"] = perc_pos[pos, f_idx]
                row_all[f"rank_{pos + 1}_count"] = int(counts_pos[pos, f_idx])
                row_nonzero[f"rank_{pos + 1}_percent"] = perc_pos_nonzero[pos, f_idx]
                row_nonzero[f"rank_{pos + 1}_count"] = int(counts_pos_nonzero[pos, f_idx])

            all_methods_data.append(row_all)
            nonzero_methods_data.append(row_nonzero)

    df_all = pd.DataFrame(all_methods_data)
    df_nonzero = pd.DataFrame(nonzero_methods_data)

    os.makedirs(foldername, exist_ok=True)
    df_all.to_csv(os.path.join(foldername, f"topk_perc_all{suffix}.csv"), index=False)
    df_nonzero.to_csv(os.path.join(foldername, f"topk_perc_nonzero{suffix}.csv"), index=False)
    return df_all, df_nonzero


def create_heatmap(df, filepath, cbar_label="Spearman Correlation"):
    """Render a square correlation heatmap and save it to ``filepath``."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, center=0,
        annot_kws={"size": 14, "weight": "bold"},
        cbar_kws={"label": cbar_label},
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def pairwise_ranking_agreement(results_dict, foldername, suffix=""):
    """Compute pairwise Spearman rank correlations between explanation methods.

    Per-instance ranks are produced with ``rankdata(method='average')`` so that
    ties share the same rank.  Instances where a method returns an all-zero
    score vector are assigned a uniform mid-rank, neutralising their
    contribution to the correlation.
    """
    methods = list(results_dict.keys())
    M = len(methods)
    if M == 0:
        return np.eye(0), {}

    N_instances = len(results_dict[methods[0]])
    n_features = len(results_dict[methods[0]][0]) if N_instances > 0 else 0

    ranks = {}
    for m in methods:
        ranks_list = []
        for fi in results_dict[m]:
            fi = np.asarray(fi)
            if np.all(fi == 0):
                ranks_list.append(np.full_like(fi, (n_features + 1) / 2, dtype=float))
            else:
                ranks_list.append(rankdata(-fi, method="average"))
        ranks[m] = np.array(ranks_list)

    avg_corr_matrix = np.eye(M)
    per_instance_corrs = {}

    for i, j in combinations(range(M), 2):
        method_i = methods[i]
        method_j = methods[j]
        corr_list = []
        for inst in range(N_instances):
            rho, _ = spearmanr(ranks[method_i][inst], ranks[method_j][inst])
            corr_list.append(rho)
        per_instance_corrs[(method_i, method_j)] = corr_list
        avg_corr_matrix[i, j] = avg_corr_matrix[j, i] = np.nanmean(corr_list)

    out_dir = os.path.join(foldername, f"pairwise_corr{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    df_avg_corr = pd.DataFrame(avg_corr_matrix, index=methods, columns=methods)
    df_avg_corr.to_csv(os.path.join(out_dir, "average_correlation_matrix.csv"))
    create_heatmap(df_avg_corr, os.path.join(out_dir, "average_correlation_heatmap.png"))

    for (method_i, method_j), corr_list in per_instance_corrs.items():
        df_corr = pd.DataFrame({f"{method_i}_vs_{method_j}": corr_list})
        df_corr.to_csv(
            os.path.join(out_dir, f"{method_i}_vs_{method_j}_per_instance.csv"),
            index=False,
        )

    return avg_corr_matrix, per_instance_corrs


def average_feature_scores(results_dict, foldername, suffix=""):
    """Calculate average scores per feature and method."""
    all_methods_data = []

    for method_name, score_lists in results_dict.items():
        score_lists = list(score_lists)
        total = len(score_lists)

        if total == 0:
            n_features = 0
        else:
            n_features = int(max(np.asarray(s).ravel().shape[0] for s in score_lists))

        sum_scores = np.zeros(n_features, dtype=float)
        count_nonzero = np.zeros(n_features, dtype=int)

        for scores in score_lists:
            arr = np.asarray(scores).ravel()
            for f_idx in range(n_features):
                if f_idx < arr.size:
                    sum_scores[f_idx] += arr[f_idx]
                    if arr[f_idx] != 0:
                        count_nonzero[f_idx] += 1

        avg_scores = sum_scores / total if total > 0 else np.zeros_like(sum_scores)
        avg_scores_nonzero = np.divide(
            sum_scores,
            count_nonzero,
            out=np.zeros_like(sum_scores),
            where=count_nonzero != 0
        )

        for f_idx in range(n_features):
            row = {
                "method": method_name,
                "feature_idx": f_idx,
                "average_score": avg_scores[f_idx],
                "average_score_nonzero": avg_scores_nonzero[f_idx],
                "count_nonzero": count_nonzero[f_idx]
            }
            all_methods_data.append(row)

    df_avg = pd.DataFrame(all_methods_data)
    os.makedirs(foldername, exist_ok=True)
    df_avg.to_csv(os.path.join(foldername, f"average_feature_scores{suffix}.csv"), index=False)
    return df_avg


def pairwise_kendalls_tau(results_dict, foldername, suffix=""):
    """Compute pairwise Kendall's tau between explanation methods.

    Uses the same ranking and tie-handling convention as
    :func:`pairwise_ranking_agreement`.
    """
    methods = list(results_dict.keys())
    M = len(methods)
    if M == 0:
        return np.eye(0), {}

    N_instances = len(results_dict[methods[0]])
    n_features = len(results_dict[methods[0]][0]) if N_instances > 0 else 0

    ranks = {}
    for m in methods:
        ranks_list = []
        for fi in results_dict[m]:
            fi = np.asarray(fi)
            if np.all(fi == 0):
                ranks_list.append(np.full_like(fi, (n_features + 1) / 2, dtype=float))
            else:
                ranks_list.append(rankdata(-fi, method="average"))
        ranks[m] = np.array(ranks_list)

    avg_tau_matrix = np.eye(M)
    per_instance_taus = {}

    for i, j in combinations(range(M), 2):
        method_i = methods[i]
        method_j = methods[j]
        tau_list = []
        for inst in range(N_instances):
            tau, _ = kendalltau(ranks[method_i][inst], ranks[method_j][inst])
            tau_list.append(tau)
        per_instance_taus[(method_i, method_j)] = tau_list
        avg_tau_matrix[i, j] = avg_tau_matrix[j, i] = np.nanmean(tau_list)

    out_dir = os.path.join(foldername, f"kendalls_tau{suffix}")
    os.makedirs(out_dir, exist_ok=True)

    df_avg_tau = pd.DataFrame(avg_tau_matrix, index=methods, columns=methods)
    df_avg_tau.to_csv(os.path.join(out_dir, "average_tau_matrix.csv"))
    create_heatmap(df_avg_tau, os.path.join(out_dir, "average_tau_heatmap.png"),
                   cbar_label="Kendall's tau")

    for (method_i, method_j), tau_list in per_instance_taus.items():
        df_tau = pd.DataFrame({f"{method_i}_vs_{method_j}": tau_list})
        df_tau.to_csv(
            os.path.join(out_dir, f"{method_i}_vs_{method_j}_per_instance.csv"),
            index=False,
        )

    return avg_tau_matrix, per_instance_taus


def uninformative_row_rate_per_method(results_dict, foldername):
    """Report the fraction of uninformative instances per method.

    An instance is uninformative for ranking purposes if all its feature scores
    are identical - either all zero (method abstained) or all the same non-zero
    value (constant output, carries no ranking signal).  Both are counted and
    reported separately as well as combined.

    Parameters
    ----------
    results_dict : dict[str, list]
        Mapping method-name → list of per-instance score arrays.
    foldername : str

    Returns
    -------
    pd.DataFrame  columns: method, n_total, n_zero, n_constant_nonzero,
                            n_uninformative, uninformative_rate
    """
    os.makedirs(foldername, exist_ok=True)
    rows = []
    for method, score_lists in results_dict.items():
        n_total = len(score_lists)
        n_zero = 0
        n_constant_nonzero = 0
        for s in score_lists:
            arr = np.asarray(s, dtype=float).ravel()
            if arr.size == 0:
                continue
            if np.all(arr == 0):
                n_zero += 1
            elif np.all(arr == arr[0]):
                n_constant_nonzero += 1
        n_uninformative = n_zero + n_constant_nonzero
        rows.append({
            "method":               method,
            "n_total":              n_total,
            "n_zero":               n_zero,
            "n_constant_nonzero":   n_constant_nonzero,
            "n_uninformative":      n_uninformative,
            "uninformative_rate":   n_uninformative / n_total if n_total > 0 else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("uninformative_rate", ascending=False)
    df.to_csv(os.path.join(foldername, "uninformative_row_rate_per_method.csv"), index=False)
    return df


def _is_uninformative(arr):
    """Return True if arr is all-zero or all-constant (no ranking signal)."""
    arr = np.asarray(arr, dtype=float).ravel()
    if arr.size == 0:
        return True
    return bool(np.all(arr == arr[0]))


def pairwise_ranking_agreement_nonzero(results_dict, foldername, suffix=""):
    """Pairwise Spearman correlation, skipping uninformative instances.

    An instance is skipped for a given pair (method_i, method_j) when either
    method returned an all-zero or all-constant score vector, as such rows
    carry no ranking signal.  This avoids the artificial mid-rank imputation
    used in ``pairwise_ranking_agreement`` and gives a cleaner picture of
    agreement on instances that both methods actually differentiated.

    An extra column ``n_compared`` is added to the summary so that pairs with
    very few shared informative instances can be identified.

    Parameters
    ----------
    results_dict : dict[str, list]
    foldername : str

    Returns
    -------
    avg_corr_matrix : np.ndarray  (M, M)
    per_instance_corrs : dict
    """
    methods = list(results_dict.keys())
    M = len(methods)
    if M == 0:
        return np.eye(0), {}

    N_instances = len(results_dict[methods[0]])

    def _ranks(arr):
        arr = np.asarray(arr, dtype=float)
        return rankdata(-arr, method="average")

    avg_corr_matrix = np.eye(M)
    per_instance_corrs = {}
    n_compared = {}

    for i, j in combinations(range(M), 2):
        method_i, method_j = methods[i], methods[j]
        corr_list = []
        for inst in range(N_instances):
            vi = np.asarray(results_dict[method_i][inst])
            vj = np.asarray(results_dict[method_j][inst])
            if _is_uninformative(vi) or _is_uninformative(vj):
                continue   # skip - at least one method gave no ranking signal
            rho, _ = spearmanr(_ranks(vi), _ranks(vj))
            corr_list.append(rho)

        n_compared[(method_i, method_j)] = len(corr_list)
        per_instance_corrs[(method_i, method_j)] = corr_list
        avg_corr_matrix[i, j] = avg_corr_matrix[j, i] = (
            np.nanmean(corr_list) if corr_list else np.nan
        )

    out_dir = os.path.join(foldername, f"pairwise_corr_nonzero{suffix}")
    os.makedirs(out_dir, exist_ok=True)

    df_avg = pd.DataFrame(avg_corr_matrix, index=methods, columns=methods)
    df_avg.to_csv(os.path.join(out_dir, "average_correlation_matrix.csv"))
    create_heatmap(df_avg, os.path.join(out_dir, "average_correlation_heatmap.png"))

    # also save n_compared summary
    rows = [
        {"method_i": mi, "method_j": mj, "n_compared": n, "mean_spearman": avg_corr_matrix[methods.index(mi), methods.index(mj)]}
        for (mi, mj), n in n_compared.items()
    ]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "n_compared_per_pair.csv"), index=False)

    for (method_i, method_j), corr_list in per_instance_corrs.items():
        pd.DataFrame({f"{method_i}_vs_{method_j}": corr_list}).to_csv(
            os.path.join(out_dir, f"{method_i}_vs_{method_j}_per_instance.csv"), index=False
        )

    return avg_corr_matrix, per_instance_corrs


def pairwise_kendalls_tau_nonzero(results_dict, foldername, suffix=""):
    """Pairwise Kendall's τ, skipping uninformative instances.

    Same filtering logic as ``pairwise_ranking_agreement_nonzero``: instances
    where either method returned an all-zero or all-constant vector are excluded.

    Parameters
    ----------
    results_dict : dict[str, list]
    foldername : str

    Returns
    -------
    avg_tau_matrix : np.ndarray  (M, M)
    per_instance_taus : dict
    """
    methods = list(results_dict.keys())
    M = len(methods)
    if M == 0:
        return np.eye(0), {}

    N_instances = len(results_dict[methods[0]])

    def _ranks(arr):
        arr = np.asarray(arr, dtype=float)
        return rankdata(-arr, method="average")

    avg_tau_matrix = np.eye(M)
    per_instance_taus = {}
    n_compared = {}

    for i, j in combinations(range(M), 2):
        method_i, method_j = methods[i], methods[j]
        tau_list = []
        for inst in range(N_instances):
            vi = np.asarray(results_dict[method_i][inst])
            vj = np.asarray(results_dict[method_j][inst])
            if _is_uninformative(vi) or _is_uninformative(vj):
                continue
            tau, _ = kendalltau(_ranks(vi), _ranks(vj))
            tau_list.append(tau)

        n_compared[(method_i, method_j)] = len(tau_list)
        per_instance_taus[(method_i, method_j)] = tau_list
        avg_tau_matrix[i, j] = avg_tau_matrix[j, i] = (
            np.nanmean(tau_list) if tau_list else np.nan
        )

    out_dir = os.path.join(foldername, f"kendalls_tau_nonzero{suffix}")
    os.makedirs(out_dir, exist_ok=True)

    df_avg = pd.DataFrame(avg_tau_matrix, index=methods, columns=methods)
    df_avg.to_csv(os.path.join(out_dir, "average_tau_matrix.csv"))

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_avg, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, center=0,
                annot_kws={"size": 14, "weight": "bold"},
                cbar_kws={"label": "Kendall's τ (non-zero only)"})
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "average_tau_heatmap.png"))
    plt.close()

    rows = [
        {"method_i": mi, "method_j": mj, "n_compared": n, "mean_kendall": avg_tau_matrix[methods.index(mi), methods.index(mj)]}
        for (mi, mj), n in n_compared.items()
    ]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "n_compared_per_pair.csv"), index=False)

    for (method_i, method_j), tau_list in per_instance_taus.items():
        pd.DataFrame({f"{method_i}_vs_{method_j}": tau_list}).to_csv(
            os.path.join(out_dir, f"{method_i}_vs_{method_j}_per_instance.csv"), index=False
        )

    return avg_tau_matrix, per_instance_taus


def topk_jaccard_nonzero(results_dict, foldername, top_k=3, suffix=""):
    """Top-k Jaccard between methods, skipping uninformative instances.

    For each pair of methods and each sample, the Jaccard of their top-k sets
    is only computed when *both* methods gave an informative (non-constant,
    non-zero) score vector.

    Parameters
    ----------
    results_dict : dict[str, list]
    foldername : str
    top_k : int

    Returns
    -------
    float           overall mean Jaccard (over non-zero pairs)
    pd.DataFrame    per-pair summary with n_compared
    """
    os.makedirs(foldername, exist_ok=True)

    methods = list(results_dict.keys())
    if len(methods) < 2:
        return 1.0, pd.DataFrame()

    N_instances = len(results_dict[methods[0]])

    rows = []
    all_jaccards = []

    for method_i, method_j in combinations(methods, 2):
        jacs = []
        for inst in range(N_instances):
            vi = np.asarray(results_dict[method_i][inst])
            vj = np.asarray(results_dict[method_j][inst])
            if _is_uninformative(vi) or _is_uninformative(vj):
                continue
            si = _topk_set_tied(vi, top_k)
            sj = _topk_set_tied(vj, top_k)
            u = si | sj
            jacs.append(len(si & sj) / len(u) if u else 1.0)

        mean_jac = float(np.mean(jacs)) if jacs else np.nan
        all_jaccards.extend(jacs)
        rows.append({
            "method_i":    method_i,
            "method_j":    method_j,
            "n_compared":  len(jacs),
            "mean_jaccard": mean_jac,
        })

    overall = float(np.mean(all_jaccards)) if all_jaccards else np.nan
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"method_i": "OVERALL", "method_j": "OVERALL",
                       "n_compared": len(all_jaccards), "mean_jaccard": overall}
    df.to_csv(os.path.join(foldername, f"topk_jaccard_top{top_k}_nonzero{suffix}.csv"), index=False)
    return overall, df


def topk_jaccard_across_runs(runs_dict, foldername, top_k=3):
    """Compute average pairwise Jaccard similarity of top-k feature sets across runs."""
    run_labels = list(runs_dict.keys())
    N_runs = len(run_labels)
    if N_runs < 2:
        return 1.0, pd.DataFrame()

    N_instances = len(runs_dict[run_labels[0]])

    rows = []
    all_jaccards = []
    for i, j in combinations(range(N_runs), 2):
        label_i = run_labels[i]
        label_j = run_labels[j]
        jaccards = []
        for inst in range(N_instances):
            set_i = _topk_set_tied(runs_dict[label_i][inst], top_k)
            set_j = _topk_set_tied(runs_dict[label_j][inst], top_k)
            union = set_i | set_j
            intersection = set_i & set_j
            jac = len(intersection) / len(union) if union else 1.0
            jaccards.append(jac)
        mean_jac = float(np.mean(jaccards))
        all_jaccards.extend(jaccards)
        rows.append({"run_i": label_i, "run_j": label_j, "mean_jaccard": mean_jac})

    avg_jaccard = float(np.mean(all_jaccards))
    df_summary = pd.DataFrame(rows)
    df_summary.loc[len(df_summary)] = {"run_i": "OVERALL", "run_j": "OVERALL", "mean_jaccard": avg_jaccard}

    os.makedirs(foldername, exist_ok=True)
    df_summary.to_csv(os.path.join(foldername, f"topk_jaccard_top{top_k}.csv"), index=False)

    return avg_jaccard, df_summary


def spearman_kendall_across_runs(runs_dict, foldername):
    """Mean pairwise Spearman and Kendall correlations across runs.

    For every pair of runs and every sample, compute Spearman rho and
    Kendall tau between the rank vectors of the two runs.  Return the
    overall means.

    Parameters
    ----------
    runs_dict : dict[str, array-like]
        Mapping run-label -> 2-D array (n_samples, n_features).
    foldername : str

    Returns
    -------
    mean_spearman : float
    mean_kendall  : float
    pd.DataFrame   per-run-pair summary
    """
    os.makedirs(foldername, exist_ok=True)

    run_labels = list(runs_dict.keys())
    run_arrays = {k: np.asarray(v) for k, v in runs_dict.items()}
    n_samples = next(iter(run_arrays.values())).shape[0]

    def _ranks(arr2d):
        return np.apply_along_axis(lambda row: rankdata(-row, method="average"), 1, arr2d)

    cached_ranks = {k: _ranks(v) for k, v in run_arrays.items()}

    rows = []
    all_spearman, all_kendall = [], []

    for label_i, label_j in combinations(run_labels, 2):
        r1 = cached_ranks[label_i]
        r2 = cached_ranks[label_j]
        sp_list, kt_list = [], []
        for s in range(n_samples):
            rho, _ = spearmanr(r1[s], r2[s])
            tau, _ = kendalltau(r1[s], r2[s])
            sp_list.append(rho)
            kt_list.append(tau)
        mean_sp = float(np.nanmean(sp_list))
        mean_kt = float(np.nanmean(kt_list))
        all_spearman.extend(sp_list)
        all_kendall.extend(kt_list)
        rows.append({
            "run_i": label_i, "run_j": label_j,
            "mean_spearman": mean_sp, "mean_kendall": mean_kt,
        })

    overall_sp = float(np.nanmean(all_spearman)) if all_spearman else np.nan
    overall_kt = float(np.nanmean(all_kendall)) if all_kendall else np.nan
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"run_i": "OVERALL", "run_j": "OVERALL",
                       "mean_spearman": overall_sp, "mean_kendall": overall_kt}
    df.to_csv(os.path.join(foldername, "spearman_kendall_across_runs.csv"), index=False)

    return overall_sp, overall_kt, df


def score_std_across_runs(runs_dict, feature_names, foldername):
    """Compute standard deviation of raw scores across runs, per feature."""
    os.makedirs(foldername, exist_ok=True)

    # stack to (n_runs, n_samples, n_features)
    run_arrays = np.stack([np.asarray(v) for v in runs_dict.values()], axis=0)

    # std across runs → (n_samples, n_features)
    std_per_sample = run_arrays.std(axis=0)

    # average over samples → (n_features,)
    mean_std = std_per_sample.mean(axis=0)

    df_summary = pd.DataFrame({
        "feature":  feature_names,
        "mean_std": mean_std,
    }).sort_values("mean_std", ascending=False)
    df_summary.to_csv(os.path.join(foldername, "score_std_per_feature.csv"), index=False)

    df_full = pd.DataFrame(std_per_sample, columns=feature_names)
    df_full.to_csv(os.path.join(foldername, "score_std_per_sample.csv"), index=False)

    return df_summary


# ---------------------------------------------------------------------------
# Stability metrics: coefficient of variation + rank inversion rate
# ---------------------------------------------------------------------------

def coefficient_of_variation_across_runs(runs_dict, feature_names, foldername):
    """Compute the Coefficient of Variation (CV = std / |mean|) per feature across runs.

    CV is scale-independent, making it comparable across features with very
    different score magnitudes.  A high CV signals instability.

    Parameters
    ----------
    runs_dict : dict[str, array-like]
        Mapping run-label → 2-D array of shape (n_samples, n_features).
    feature_names : list[str]
    foldername : str
        Directory where results are written.

    Returns
    -------
    pd.DataFrame  columns: feature, mean_score, std_score, cv
    """
    os.makedirs(foldername, exist_ok=True)

    # (n_runs, n_samples, n_features)
    run_arrays = np.stack([np.asarray(v) for v in runs_dict.values()], axis=0)

    # collapse samples first → (n_runs, n_features)  mean per run per feature
    run_means = run_arrays.mean(axis=1)

    # across-run statistics → (n_features,)
    grand_mean = run_means.mean(axis=0)
    grand_std  = run_means.std(axis=0)

    # CV = std / |mean|; set to NaN where mean ≈ 0 to avoid division noise
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.where(np.abs(grand_mean) > 1e-12, grand_std / np.abs(grand_mean), np.nan)

    df = pd.DataFrame({
        "feature":    feature_names,
        "mean_score": grand_mean,
        "std_score":  grand_std,
        "cv":         cv,
    }).sort_values("cv", ascending=False, na_position="last")

    df.to_csv(os.path.join(foldername, "coefficient_of_variation.csv"), index=False)
    return df


def rank_inversion_rate(runs_dict, feature_names, foldername):
    """Fraction of feature pairs whose relative ranking flips between run pairs.

    For every ordered pair of runs (r1, r2) and every sample, we count the
    fraction of feature-pairs (i, j) whose rank order is reversed:

        inversion_rate = |{(i,j): rank_i^r1 < rank_j^r1  and  rank_i^r2 > rank_j^r2}|
                         ─────────────────────────────────────────────────────────────
                                              C(n_features, 2)

    The reported overall rate is the mean over all run-pairs and all samples.

    Parameters
    ----------
    runs_dict : dict[str, array-like]
        Mapping run-label → 2-D array (n_samples, n_features).
    feature_names : list[str]
    foldername : str

    Returns
    -------
    float   overall mean inversion rate
    pd.DataFrame  per-run-pair summary
    """
    os.makedirs(foldername, exist_ok=True)

    run_labels = list(runs_dict.keys())
    run_arrays = {k: np.asarray(v) for k, v in runs_dict.items()}

    n_samples, n_features = next(iter(run_arrays.values())).shape
    n_pairs_feat = n_features * (n_features - 1) / 2  # C(n,2)

    def _ranks(arr2d):
        return np.apply_along_axis(lambda row: rankdata(-row, method="average"), 1, arr2d)

    cached_ranks = {k: _ranks(v) for k, v in run_arrays.items()}

    rows = []
    all_rates = []

    for label_i, label_j in combinations(run_labels, 2):
        r1 = cached_ranks[label_i]   # (n_samples, n_features)
        r2 = cached_ranks[label_j]

        inversion_rates = []
        for s in range(n_samples):
            inv = 0
            for fi, fj in combinations(range(n_features), 2):
                # r1[s, fi] < r1[s, fj] feature i ranked higher in run 1
                if (r1[s, fi] < r1[s, fj]) != (r2[s, fi] < r2[s, fj]):
                    inv += 1
            inversion_rates.append(inv / n_pairs_feat if n_pairs_feat > 0 else 0.0)

        mean_rate = float(np.mean(inversion_rates))
        all_rates.extend(inversion_rates)
        rows.append({
            "run_i": label_i,
            "run_j": label_j,
            "mean_inversion_rate": mean_rate,
        })

    overall = float(np.mean(all_rates))
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"run_i": "OVERALL", "run_j": "OVERALL", "mean_inversion_rate": overall}
    df.to_csv(os.path.join(foldername, "rank_inversion_rate.csv"), index=False)

    return overall, df


# ---------------------------------------------------------------------------
# Background-set comparison metrics
# ---------------------------------------------------------------------------

def score_variance_across_configs(configs_dict, feature_names, foldername):
    """Compute per-feature score variance across different background configurations.

    Parameters
    ----------
    configs_dict : dict[str, array-like]
        Mapping config-label (e.g. "kmeans_10") → 2-D array (n_samples, n_features).
    feature_names : list[str]
    foldername : str

    Returns
    -------
    pd.DataFrame  columns: feature, mean_score, var_score, std_score
    """
    os.makedirs(foldername, exist_ok=True)

    # (n_configs, n_features)  - collapse samples by mean per config
    config_means = np.stack(
        [np.asarray(v).mean(axis=0) for v in configs_dict.values()], axis=0
    )

    grand_mean = config_means.mean(axis=0)
    grand_var  = config_means.var(axis=0)
    grand_std  = config_means.std(axis=0)

    df = pd.DataFrame({
        "feature":    feature_names,
        "mean_score": grand_mean,
        "var_score":  grand_var,
        "std_score":  grand_std,
    }).sort_values("var_score", ascending=False)

    df.to_csv(os.path.join(foldername, "score_variance_across_configs.csv"), index=False)
    return df


def topk_jaccard_across_configs(configs_dict, foldername, top_k=3):
    """Pairwise Jaccard of top-k feature sets across background configurations.

    Mirrors ``topk_jaccard_across_runs`` but for different background configs
    rather than stability runs.

    Parameters
    ----------
    configs_dict : dict[str, array-like]
        Mapping config-label → 2-D array (n_samples, n_features).
    foldername : str
    top_k : int

    Returns
    -------
    float           overall mean Jaccard
    pd.DataFrame    per-pair summary
    """
    os.makedirs(foldername, exist_ok=True)

    labels = list(configs_dict.keys())
    if len(labels) < 2:
        return 1.0, pd.DataFrame()

    n_samples = next(iter(configs_dict.values())).shape[0]

    rows = []
    all_jaccards = []

    for label_i, label_j in combinations(labels, 2):
        arr_i = np.asarray(configs_dict[label_i])
        arr_j = np.asarray(configs_dict[label_j])
        jacs = []
        for s in range(n_samples):
            si = _topk_set_tied(arr_i[s], top_k)
            sj = _topk_set_tied(arr_j[s], top_k)
            u = si | sj
            jacs.append(len(si & sj) / len(u) if u else 1.0)
        mean_jac = float(np.mean(jacs))
        all_jaccards.extend(jacs)
        rows.append({"config_i": label_i, "config_j": label_j, "mean_jaccard": mean_jac})

    overall = float(np.mean(all_jaccards))
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"config_i": "OVERALL", "config_j": "OVERALL", "mean_jaccard": overall}
    df.to_csv(os.path.join(foldername, f"topk_jaccard_top{top_k}_configs.csv"), index=False)

    return overall, df


# ---------------------------------------------------------------------------
# Pivot comparison metrics
# ---------------------------------------------------------------------------

def compare_pivot_methods(majority_scores, veto_scores, feature_names, foldername,
                          top_k=3):
    """Compare majority vs veto pivot results per index.

    Computes:
    - Spearman and Kendall rank correlations between the two pivot methods.
    - Top-1 flip rate: fraction of samples where the top-ranked feature differs.
    - Top-k Jaccard: mean Jaccard similarity of the top-k feature sets.
    - Per-feature score ratio  veto / majority  (averaged over samples).

    Parameters
    ----------
    majority_scores : array-like, shape (n_samples, n_features)
    veto_scores : array-like, shape (n_samples, n_features)
    feature_names : list[str]
    foldername : str
    top_k : int

    Returns
    -------
    dict  with keys: spearman_mean, kendall_mean, top1_flip_rate,
                     topk_jaccard_mean, df_score_ratio
    """
    os.makedirs(foldername, exist_ok=True)

    maj = np.asarray(majority_scores)
    veto = np.asarray(veto_scores)
    n_samples = maj.shape[0]

    spearman_vals, kendall_vals = [], []
    top1_flips = 0
    jaccard_vals = []

    for s in range(n_samples):
        r_maj  = rankdata(-maj[s],  method="average")
        r_veto = rankdata(-veto[s], method="average")

        rho, _ = spearmanr(r_maj, r_veto)
        tau, _ = kendalltau(r_maj, r_veto)
        spearman_vals.append(rho)
        kendall_vals.append(tau)

        if _argmax_set_tied(maj[s]) != _argmax_set_tied(veto[s]):
            top1_flips += 1

        # Top-k Jaccard similarity
        k = min(top_k, maj.shape[1])
        topk_maj = _topk_set_tied(maj[s], k)
        topk_veto = _topk_set_tied(veto[s], k)
        union = topk_maj | topk_veto
        jaccard_vals.append(len(topk_maj & topk_veto) / len(union) if union else 1.0)

    spearman_mean = float(np.nanmean(spearman_vals))
    kendall_mean  = float(np.nanmean(kendall_vals))
    top1_flip_rate = top1_flips / n_samples if n_samples > 0 else 0.0
    topk_jaccard_mean = float(np.nanmean(jaccard_vals))

    # Per-sample score ratio (veto / majority); guard against zero majority scores
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(np.abs(maj) > 1e-12, veto / maj, np.nan)
    mean_ratio = np.nanmean(ratio, axis=0)

    df_ratio = pd.DataFrame({
        "feature":         feature_names,
        "mean_score_maj":  maj.mean(axis=0),
        "mean_score_veto": veto.mean(axis=0),
        "mean_score_ratio_veto_over_maj": mean_ratio,
    })

    # Save
    summary = pd.DataFrame([{
        "spearman_mean":       spearman_mean,
        "kendall_mean":        kendall_mean,
        "top1_flip_rate":      top1_flip_rate,
        f"top{top_k}_jaccard_mean": topk_jaccard_mean,
        "n_samples":           n_samples,
    }])
    summary.to_csv(os.path.join(foldername, "pivot_comparison_summary.csv"), index=False)
    df_ratio.to_csv(os.path.join(foldername, "pivot_score_ratio.csv"), index=False)

    # Per-sample Spearman/Kendall/Jaccard distributions
    pd.DataFrame({
        "spearman": spearman_vals,
        "kendall": kendall_vals,
        f"top{top_k}_jaccard": jaccard_vals,
    }).to_csv(
        os.path.join(foldername, "pivot_per_sample_correlations.csv"), index=False
    )

    return {
        "spearman_mean":      spearman_mean,
        "kendall_mean":       kendall_mean,
        "top1_flip_rate":     top1_flip_rate,
        "topk_jaccard_mean":  topk_jaccard_mean,
        "df_score_ratio":     df_ratio,
    }


# ---------------------------------------------------------------------------
# Useless feature metrics
# ---------------------------------------------------------------------------

def useless_feature_metrics(results_dict, useless_feature_idx, feature_names, foldername):
    """Evaluate how well each method suppresses a known-useless feature.

    Metrics computed per method:
    - **mean_rank**       - average rank of the useless feature (1 = most important).
                           Should be close to n_features for a good method.
    - **suppression_ratio** - mean(|score_useless|) / mean(|score_real|).
                              Closer to 0 is better.
    - **detection_rate**  - fraction of samples where the useless feature is
                            *not* in the top-k (default k=3).
    - **score_percentile** - mean percentile of |score_useless| among all
                             feature |scores| (0th percentile = lowest = best).

    Parameters
    ----------
    results_dict : dict[str, array-like]
        Mapping method-name → 2-D array (n_samples, n_features).
    useless_feature_idx : int
        Column index of the known-useless feature.
    feature_names : list[str]
    foldername : str
    top_k : int  (keyword)
        Top-k window used for detection_rate.

    Returns
    -------
    pd.DataFrame  one row per method
    """
    os.makedirs(foldername, exist_ok=True)

    top_k = 3
    rows = []

    for method, scores in results_dict.items():
        arr = np.abs(np.asarray(scores))       # (n_samples, n_features)
        n_samples, n_features = arr.shape

        useless_scores = arr[:, useless_feature_idx]          # (n_samples,)
        real_mask = np.ones(n_features, dtype=bool)
        real_mask[useless_feature_idx] = False
        real_scores = arr[:, real_mask]                       # (n_samples, n_features-1)

        # rank of useless feature (1 = highest score)
        ranks = np.apply_along_axis(
            lambda row: rankdata(-row, method="average"), 1, arr
        )                                                      # (n_samples, n_features)
        mean_rank = float(ranks[:, useless_feature_idx].mean())

        # suppression ratio
        mean_useless = float(useless_scores.mean())
        mean_real = float(real_scores.mean()) if real_scores.size > 0 else 1.0
        suppression_ratio = mean_useless / mean_real if mean_real > 1e-12 else np.nan

        # detection rate: useless feature NOT in top-k
        not_in_topk = np.array([
            useless_feature_idx not in _topk_set_tied(arr[s], top_k)
            for s in range(n_samples)
        ])
        detection_rate = float(not_in_topk.mean())

        # score percentile of useless feature
        percentiles = []
        for s in range(n_samples):
            row = arr[s]
            u_score = row[useless_feature_idx]
            pct = float(np.mean(row < u_score) * 100)
            percentiles.append(pct)
        mean_percentile = float(np.mean(percentiles))

        rows.append({
            "method":            method,
            "useless_feature":   feature_names[useless_feature_idx],
            "mean_rank":         mean_rank,
            "n_features":        n_features,
            "mean_rank_norm":    mean_rank / n_features,
            "suppression_ratio": suppression_ratio,
            "detection_rate":    detection_rate,
            "mean_score_percentile": mean_percentile,
        })

    df = pd.DataFrame(rows).sort_values("mean_rank", ascending=False)
    df.to_csv(os.path.join(foldername, "useless_feature_metrics.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# Synthetic / ground-truth metrics
# ---------------------------------------------------------------------------

def ground_truth_metrics(results_dict, true_weights, feature_names, foldername):
    """Evaluate explanation quality against known ground-truth feature weights.

    Applicable to ``exp_synthetic`` where ``TRUE_WEIGHTS`` are known.

    Metrics per method:
    - **spearman**    - Spearman correlation between method ranking and GT ranking
                        (both ranked by absolute weight / absolute score).
    - **kendall**     - Kendall's τ between the same two rankings.
    - **ndcg**        - Normalised Discounted Cumulative Gain using |GT weight|
                        as relevance and method ranking as the predicted order.
    - **mare**        - Mean Absolute Rank Error: average |predicted_rank − gt_rank|.
    - **topk_jaccard** - Jaccard similarity between the method's top-k features
                          and the ground-truth top-k (by |weight|), averaged over samples.
    - **sign_agree**  (for signed methods only) - fraction of features where
                        sign(score) == sign(true_weight), ignoring zero-weight features.

    Parameters
    ----------
    results_dict : dict[str, array-like]
        Mapping method-name → 2-D array (n_samples, n_features)  **or**
        1-D array (n_features,) for methods that return a single global ranking.
    true_weights : array-like, shape (n_features,)
        The true model weights (may be negative).
    feature_names : list[str]
    foldername : str

    Returns
    -------
    pd.DataFrame  one row per method
    """
    os.makedirs(foldername, exist_ok=True)

    true_weights = np.asarray(true_weights, dtype=float)
    n_features = len(true_weights)

    top_k = 3

    # Ground-truth ranking by absolute weight (rank 1 = most important)
    gt_abs = np.abs(true_weights)
    gt_rank = rankdata(-gt_abs, method="average")   # (n_features,)
    gt_topk = _topk_set_tied(gt_abs, top_k)

    # Ideal DCG (denominator for NDCG): relevance = |gt_weight|, sorted descending
    sorted_rel = np.sort(gt_abs)[::-1]
    idcg = float(np.sum(sorted_rel / np.log2(np.arange(2, n_features + 2))))

    def _ndcg(predicted_rank):
        """Compute DCG/IDCG given predicted ranks (1 = best)."""
        order = np.argsort(predicted_rank)           # index of feature at each position
        dcg = float(np.sum(gt_abs[order] / np.log2(np.arange(2, n_features + 2))))
        return dcg / idcg if idcg > 0 else 0.0

    rows = []

    for method, scores in results_dict.items():
        arr = np.asarray(scores)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        n_samples = arr.shape[0]

        spearman_vals, kendall_vals, ndcg_vals, mare_vals = [], [], [], []
        topk_jaccard_vals = []
        sign_agree_vals = []

        for s in range(n_samples):
            row = arr[s]
            abs_row = np.abs(row)

            if np.all(abs_row == 0):
                pred_rank = np.full(n_features, (n_features + 1) / 2.0)
            else:
                pred_rank = rankdata(-abs_row, method="average")

            rho, _ = spearmanr(pred_rank, gt_rank)
            tau, _ = kendalltau(pred_rank, gt_rank)
            ndcg_val = _ndcg(pred_rank)
            mare_val = float(np.mean(np.abs(pred_rank - gt_rank)))

            spearman_vals.append(rho)
            kendall_vals.append(tau)
            ndcg_vals.append(ndcg_val)
            mare_vals.append(mare_val)

            # top-k Jaccard vs ground truth
            pred_topk = _topk_set_tied(abs_row, top_k)
            union = gt_topk | pred_topk
            jac = len(gt_topk & pred_topk) / len(union) if union else 1.0
            topk_jaccard_vals.append(jac)

            # sign agreement (only for non-zero GT weights)
            nonzero_mask = true_weights != 0
            if nonzero_mask.any():
                agree = np.sign(row[nonzero_mask]) == np.sign(true_weights[nonzero_mask])
                sign_agree_vals.append(float(agree.mean()))

        rows.append({
            "method":          method,
            "spearman_mean":   float(np.nanmean(spearman_vals)),
            "spearman_std":    float(np.nanstd(spearman_vals)),
            "kendall_mean":    float(np.nanmean(kendall_vals)),
            "kendall_std":     float(np.nanstd(kendall_vals)),
            "ndcg_mean":       float(np.nanmean(ndcg_vals)),
            "ndcg_std":        float(np.nanstd(ndcg_vals)),
            "mare_mean":       float(np.nanmean(mare_vals)),
            "mare_std":        float(np.nanstd(mare_vals)),
            "topk_jaccard_mean": float(np.nanmean(topk_jaccard_vals)),
            "topk_jaccard_std":  float(np.nanstd(topk_jaccard_vals)),
            "sign_agree_mean": float(np.nanmean(sign_agree_vals)) if sign_agree_vals else np.nan,
        })

    df = pd.DataFrame(rows).sort_values("ndcg_mean", ascending=False)
    df.to_csv(os.path.join(foldername, "ground_truth_metrics.csv"), index=False)

    # Also save GT reference table for convenience
    df_gt = pd.DataFrame({
        "feature":      feature_names,
        "true_weight":  true_weights,
        "abs_weight":   gt_abs,
        "gt_rank":      gt_rank,
    }).sort_values("gt_rank")
    df_gt.to_csv(os.path.join(foldername, "ground_truth_reference.csv"), index=False)

    return df


# ---------------------------------------------------------------------------
# Top-k placement breakdown (place 1, place 2, place 3)
# ---------------------------------------------------------------------------

def topk_placement_breakdown(results_dict, feature_names, foldername,
                             top_k=3, exclude_methods=None):
    """Per-method breakdown of how often each feature occupies place 1, 2, ... k.

    For every method and every feature, report the percentage of samples
    where that feature is the 1st, 2nd, or 3rd most important (by |score|).

    Parameters
    ----------
    results_dict : dict[str, np.ndarray]
        {method_name: (n_samples, n_features)}
    feature_names : list[str]
    foldername : str
    top_k : int
    exclude_methods : set or None
        Method names to skip (e.g. ``{"bi_neg", "ssi_neg"}``).

    Returns
    -------
    pd.DataFrame  with columns: method, feature, place_1_pct, place_2_pct, place_3_pct
    """
    os.makedirs(foldername, exist_ok=True)
    exclude = exclude_methods or set()
    rows = []

    for method, scores in results_dict.items():
        if method in exclude:
            continue
        arr = np.abs(np.asarray(scores, dtype=float))
        n_samples, n_features = arr.shape
        ranks = np.apply_along_axis(
            lambda row: rankdata(-row, method="dense"), 1, arr
        )

        for feat_idx, feat_name in enumerate(feature_names):
            row = {"method": method, "feature": feat_name}
            for k in range(min(top_k, n_features)):
                target_rank = k + 1
                pct = float(np.mean(ranks[:, feat_idx] == target_rank) * 100)
                row[f"place_{k + 1}_pct"] = pct
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(foldername, "topk_placement_breakdown.csv"), index=False)
    return df

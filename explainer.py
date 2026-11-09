"""Core explainer for computing feature-importance indices.

Provides :class:`BackgroundSet`, which materialises the background data used
as a baseline for each method, and :class:`My_Explainer`, which dispatches the
requested explanation indices and writes their results to disk.
"""

import itertools
import math
import os

import numpy as np

from background_method import BackgroundMethod

RANDOM_SEED = 42
ALL_IDX = [
    "shap", "lime", "bi", "ssi", "bi_neg", "ssi_neg", "bi_full", "ssi_full",
    "sv", "dp", "hp", "ri", "rwi", "bww", "rwi_grid", "bww_grid"
]


class BackgroundSet:
    """Manages background data selection for explainability methods."""

    def __init__(self, data, method):
        self.data = self._calculate_background_set(background_set=data, background_method=method)
        self.method = method
        self.rng = np.random.default_rng(RANDOM_SEED)

    def get_data(self):
        """Get background data, sampling if method is 'random'."""
        if self.method.method == "random":
            idx = self.rng.choice(self.data.shape[0], size=int(self.method.param))
            return self.data[idx]
        return self.data

    def get_data_all(self):
        """Get all background data without sampling."""
        return self.data

    def _calculate_background_set(self, background_set, background_method):
        """Calculate background set based on the specified method."""
        if background_method.method == "all":
            return np.array(background_set)
        elif background_method.method == "kmeans":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=background_method.param, random_state=RANDOM_SEED, init='random')
            kmeans.fit(background_set)
            return kmeans.cluster_centers_
        elif background_method.method == "kernel_density":
            from sklearn.neighbors import KernelDensity
            kde = KernelDensity(kernel="gaussian").fit(background_set)
            return kde.sample(background_method.param, random_state=RANDOM_SEED)
        elif background_method.method == "random":
            return np.array(background_set)
        else:
            raise ValueError(f"Unknown background-set method: {background_method.method!r}")


class My_Explainer:
    """Compute feature-importance indices for a binary classifier.

    Parameters
    ----------
    predict_fn : callable
        Function mapping a 2-D array of inputs to a 1-D array of class-1
        probabilities (or a (n, 1) array; both shapes are accepted).
    background_set : array-like
        Reference data used to mask out features when forming coalitions.
    background_method : BackgroundMethod
        Strategy used to subsample / cluster ``background_set``.
    pivot_method : {"majority", "veto"}
        Rule for deciding whether a coalition is winning when more than one
        background sample is used.  ``"majority"`` requires more than half of
        the imputations to flip the prediction; ``"veto"`` requires all of
        them.
    save_to_file : bool
        If True, every index is appended to its own CSV under ``out_dir``
        immediately after computation (useful for long-running jobs).
    verbose : bool
        If True, prints progress messages and tqdm bars.
    """

    def __init__(self, predict_fn, background_set, background_method, pivot_method="majority",
                 save_to_file=True, verbose=True):
        self.predict_fn = predict_fn
        self.background_set = BackgroundSet(data=background_set, method=background_method)
        self.pivot_method = pivot_method
        self.save_to_file = save_to_file
        self.verbose = verbose
        self.result = dict()
        self.n_features = self.background_set.get_data().shape[1]

    def _log(self, msg: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(msg)

    def explain(self, data, indices, out_dir=None):
        """Compute the requested indices for every row of ``data``.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Instances to explain.
        indices : list of str
            Subset of :data:`ALL_IDX` identifying which methods to run.
        out_dir : str, optional
            If given, each index is also written to ``<out_dir>/<index>.csv``
            as it finishes.
        """
        if not isinstance(indices, list):
            raise TypeError("The 'indices' parameter must be a list.")
        for idx in indices:
            if not isinstance(idx, str):
                raise TypeError("All elements in 'indices' must be strings.")

        self.data = np.array(data)
        self.n_features = self.data.shape[1]
        for idx in indices:
            self.result[idx] = []

        if "shap" in indices:
            self.result["shap"] = self._get_shap()
            self._log("  [shap] done")
            if out_dir:
                self._save_single_method(out_dir, "shap")

        if "lime" in indices:
            self.result["lime"] = self._get_lime()
            self._log("  [lime] done")
            if out_dir:
                self._save_single_method(out_dir, "lime")

        dp_based = {"bi", "ssi", "bi_neg", "ssi_neg", "bi_full", "ssi_full", "rwi", "bww", "rwi_grid", "bww_grid"}
        if dp_based & set(indices):
            bi, ssi, bi_neg, ssi_neg, bi_full, ssi_full = self.get_banzhaf_shapley_shubik()
            for key, val in [("bi", bi), ("ssi", ssi), ("bi_neg", bi_neg),
                             ("ssi_neg", ssi_neg), ("bi_full", bi_full), ("ssi_full", ssi_full)]:
                if key in indices:
                    self.result[key] = val
                    self._log(f"  [{key}] done")
                    if out_dir:
                        self._save_single_method(out_dir, key)

            if "rwi" in indices:
                self.result["rwi"] = self.get_range_weighted_index()
                self._log("  [rwi] done")
                if out_dir:
                    self._save_single_method(out_dir, "rwi")

            if "bww" in indices:
                self.result["bww"] = self.get_banzhaf_winning_width()
                self._log("  [bww] done")
                if out_dir:
                    self._save_single_method(out_dir, "bww")

            if "rwi_grid" in indices:
                self.result["rwi_grid"] = self.get_range_weighted_index_grid()
                self._log("  [rwi_grid] done")
                if out_dir:
                    self._save_single_method(out_dir, "rwi_grid")

            if "bww_grid" in indices:
                self.result["bww_grid"] = self.get_banzhaf_winning_width_grid()
                self._log("  [bww_grid] done")
                if out_dir:
                    self._save_single_method(out_dir, "bww_grid")

        if "sv" in indices:
            self.result["sv"] = self._get_shapley_value()
            self._log("  [sv] done")
            if out_dir:
                self._save_single_method(out_dir, "sv")

        if "dp" in indices or "hp" in indices or "ri" in indices:
            if not hasattr(self, 'dp_tables') or len(self.dp_tables) != len(self.data):
                self._build_dp_tables()

        if "dp" in indices:
            self.result["dp"] = self._get_deegan_packel()
            self._log("  [dp] done")
            if out_dir:
                self._save_single_method(out_dir, "dp")

        if "hp" in indices:
            self.result["hp"] = self._get_holler_packel()
            self._log("  [hp] done")
            if out_dir:
                self._save_single_method(out_dir, "hp")

        if "ri" in indices:
            self.result["ri"] = self._get_responsibility()
            self._log("  [ri] done")
            if out_dir:
                self._save_single_method(out_dir, "ri")

        return self.result

    def get_banzhaf_winning_width(self):
        """Banzhaf Winning Width - batched binary search version.

        For each pivot edge we find the full winning interval [lo_boundary,
        hi_boundary] by searching in both directions from sample[j]. All
        ongoing searches share a single predict_fn call per binary-search step.
        """
        from tqdm import tqdm

        n_samples = self.data.shape[0]
        n_features = self.data.shape[1]

        if not hasattr(self, 'dp_tables') or len(self.dp_tables) != n_samples:
            self._build_dp_tables()

        all_bww = []

        for sample_idx in tqdm(range(n_samples), desc="  [bww]", disable=not self.verbose):
            sample = self.data[sample_idx]
            dp_winning = self.dp_tables[sample_idx]
            background_data = self.dp_backgrounds[sample_idx]

            feat_mins = np.minimum(background_data.min(axis=0), sample)
            feat_maxs = np.maximum(background_data.max(axis=0), sample)
            margin = 0.1 * (feat_maxs - feat_mins + 1e-8)

            result = self._collect_pivots_and_base_rows(sample, dp_winning, background_data, n_features)
            if result is None:
                all_bww.append(np.zeros(n_features))
                continue

            pivot_features, bg_indices, base_rows, n_piv = result

            lo_lower = feat_mins[pivot_features] - margin[pivot_features]
            hi_lower = sample[pivot_features].copy()
            lo_upper = sample[pivot_features].copy()
            hi_upper = feat_maxs[pivot_features] + margin[pivot_features]

            N_STEPS = 15
            for _ in range(N_STEPS):
                mid_lower = ((lo_lower + hi_lower) / 2).astype(np.float32)
                mid_upper = ((lo_upper + hi_upper) / 2).astype(np.float32)

                batch = np.empty((2 * n_piv, n_features), dtype=np.float32)
                batch[:n_piv] = base_rows
                batch[n_piv:] = base_rows

                pf_arr = pivot_features
                batch[np.arange(n_piv), pf_arr] = mid_lower
                batch[np.arange(n_piv, 2 * n_piv), pf_arr] = mid_upper

                preds = np.asarray(self.predict_fn(batch)).flatten()
                wins_lower = preds[:n_piv] >= 0.5
                wins_upper = preds[n_piv:] >= 0.5

                hi_lower[wins_lower] = mid_lower[wins_lower]
                lo_lower[~wins_lower] = mid_lower[~wins_lower]

                lo_upper[wins_upper] = mid_upper[wins_upper]
                hi_upper[~wins_upper] = mid_upper[~wins_upper]

            lower_boundary = (lo_lower + hi_lower) / 2
            upper_boundary = (lo_upper + hi_upper) / 2

            width_scores = np.zeros(n_features)
            for j in range(n_features):
                mask_j = pivot_features == j
                if not np.any(mask_j):
                    continue
                widths = upper_boundary[mask_j] - lower_boundary[mask_j]
                feat_range = feat_maxs[j] - feat_mins[j] + 2 * margin[j]
                if feat_range > 0:
                    width_scores[j] = widths.sum() / feat_range

            width_scores /= (1 << (n_features - 1))
            all_bww.append(width_scores)

        return np.array(all_bww)

    def _collect_pivots_and_base_rows(self, sample, dp_winning, background_data, n_features):
        """Collect all pivot edges and build base_rows vectorized.

        A pivot edge is (mask, feature_j) where adding j to mask flips 0 to 1.
        Each pivot edge is expanded over all background points.

        Returns (pivot_features, bg_indices, base_rows, n_piv) or None if no pivots.
        """
        n_bg = len(background_data)

        edges = []
        for mask in range(1 << n_features):
            for j in range(n_features):
                if mask & (1 << j):
                    continue
                if dp_winning[mask] != 0 or dp_winning[mask | (1 << j)] != 1:
                    continue
                edges.append((mask, j))

        if not edges:
            return None

        n_edges = len(edges)
        n_piv = n_edges * n_bg

        edge_features = np.array([j for (_, j) in edges], dtype=np.intp)
        templates = np.tile(background_data, (n_edges, 1, 1))

        for e_idx, (mask, j) in enumerate(edges):
            coalition = [k for k in range(n_features) if mask & (1 << k)]
            if coalition:
                for k in coalition:
                    templates[e_idx, :, k] = sample[k]

        base_rows = templates.reshape(n_piv, n_features).astype(np.float32)
        pivot_features = np.repeat(edge_features, n_bg)
        bg_indices = np.tile(np.arange(n_bg), n_edges)

        return pivot_features, bg_indices, base_rows, n_piv

    def _save_single_method(self, out_dir, method):
        """Save a single explanation method's results to a CSV file."""
        if not self.save_to_file:
            return

        os.makedirs(out_dir, exist_ok=True)
        explanations = self.result.get(method, [])
        file_path = os.path.join(out_dir, f"{method}.csv")

        with open(file_path, "w", encoding="utf-8") as f:
            for inst in explanations:
                arr = np.asarray(inst).flatten()
                f.write(",".join(map(str, arr.tolist())) + "\n")

    def get_range_weighted_index(self):
        """Range-Weighted Index - batched binary search version.

        For every pivot edge (mask, feature, bg_point) we binary-search for the
        threshold where the prediction flips. All ongoing searches share a single
        predict_fn call per binary-search step.
        """
        from tqdm import tqdm

        n_samples = self.data.shape[0]
        n_features = self.data.shape[1]

        if not hasattr(self, 'dp_tables') or len(self.dp_tables) != n_samples:
            self._build_dp_tables()

        all_rwi = []

        for sample_idx in tqdm(range(n_samples), desc="  [rwi]", disable=not self.verbose):
            sample = self.data[sample_idx]
            dp_winning = self.dp_tables[sample_idx]
            background_data = self.dp_backgrounds[sample_idx]

            result = self._collect_pivots_and_base_rows(sample, dp_winning, background_data, n_features)
            if result is None:
                all_rwi.append(np.zeros(n_features))
                continue

            pivot_features, bg_indices, base_rows, n_piv = result

            lo = background_data[bg_indices, pivot_features].astype(np.float64)
            hi = np.full(n_piv, dtype=np.float64, fill_value=0)
            for j in range(n_features):
                mask_j = pivot_features == j
                hi[mask_j] = sample[j]

            swap = lo > hi
            lo[swap], hi[swap] = hi[swap], lo[swap]

            N_STEPS = 10
            for _ in range(N_STEPS):
                mid = ((lo + hi) / 2).astype(np.float32)

                batch = base_rows.copy()
                batch[np.arange(n_piv), pivot_features] = mid

                preds = np.asarray(self.predict_fn(batch)).flatten()
                wins = preds >= 0.5

                hi[wins] = mid[wins]
                lo[~wins] = mid[~wins]

            boundary = (lo + hi) / 2
            scores = np.zeros(n_features)

            bg_means = background_data[:, :].mean(axis=0)
            for j in range(n_features):
                mask_j = pivot_features == j
                if not np.any(mask_j):
                    continue
                total_span = abs(sample[j] - bg_means[j])
                if total_span > 0:
                    widths = np.abs(sample[j] - boundary[mask_j])
                    scores[j] = widths.sum() / total_span

            scores /= (1 << (n_features - 1))
            all_rwi.append(scores)

        return np.array(all_rwi)

    def get_range_weighted_index_grid(self):
        """Range-Weighted Index - grid scan version.

        Same as get_range_weighted_index but replaces the 10-step binary search
        with a 50-point grid scan that finds ALL threshold crossings in the
        interval and picks the one nearest to x_i.  This is correct even when
        the model's 1-D slice is non-monotonic (multiple crossings).
        """
        from tqdm import tqdm

        N_GRID = 50

        n_samples = self.data.shape[0]
        n_features = self.data.shape[1]

        if not hasattr(self, 'dp_tables') or len(self.dp_tables) != n_samples:
            self._build_dp_tables()

        all_rwi = []

        for sample_idx in tqdm(range(n_samples), desc="  [rwi_grid]", disable=not self.verbose):
            sample = self.data[sample_idx]
            dp_winning = self.dp_tables[sample_idx]
            background_data = self.dp_backgrounds[sample_idx]

            result = self._collect_pivots_and_base_rows(sample, dp_winning, background_data, n_features)
            if result is None:
                all_rwi.append(np.zeros(n_features))
                continue

            pivot_features, bg_indices, base_rows, n_piv = result

            lo = background_data[bg_indices, pivot_features].astype(np.float64)
            hi = np.full(n_piv, dtype=np.float64, fill_value=0)
            for j in range(n_features):
                mask_j = pivot_features == j
                hi[mask_j] = sample[j]

            swap = lo > hi
            lo[swap], hi[swap] = hi[swap], lo[swap]

            # Build grid: shape (n_piv, N_GRID)
            t = np.linspace(0.0, 1.0, N_GRID)
            grid_vals = lo[:, None] + t[None, :] * (hi - lo)[:, None]  # (n_piv, N_GRID)

            # Evaluate all grid points in one batched call
            batch = np.repeat(base_rows, N_GRID, axis=0)  # (n_piv*N_GRID, n_features)
            pf_rep = np.repeat(pivot_features, N_GRID)
            batch[np.arange(n_piv * N_GRID), pf_rep] = grid_vals.ravel().astype(np.float32)

            preds = np.asarray(self.predict_fn(batch)).flatten().reshape(n_piv, N_GRID)
            wins = preds >= 0.5  # (n_piv, N_GRID)

            # For each pivot, find crossings and pick nearest to x_j (grid index -1 = hi = x_j side)
            boundary = (lo + hi) / 2  # fallback: midpoint
            for k in range(n_piv):
                sign_changes = np.where(np.diff(wins[k].astype(np.int8)))[0]
                if len(sign_changes) == 0:
                    continue
                # crossing k falls between grid_vals[k, sc] and grid_vals[k, sc+1]
                crossing_vals = (grid_vals[k, sign_changes] + grid_vals[k, sign_changes + 1]) / 2
                # pick the crossing nearest to hi[k] (the x_j side)
                nearest = crossing_vals[np.argmin(np.abs(crossing_vals - hi[k]))]
                boundary[k] = nearest

            scores = np.zeros(n_features)
            bg_means = background_data.mean(axis=0)
            for j in range(n_features):
                mask_j = pivot_features == j
                if not np.any(mask_j):
                    continue
                total_span = abs(sample[j] - bg_means[j])
                if total_span > 0:
                    widths = np.abs(sample[j] - boundary[mask_j])
                    scores[j] = widths.sum() / total_span

            scores /= (1 << (n_features - 1))
            all_rwi.append(scores)

        return np.array(all_rwi)

    def get_banzhaf_winning_width_grid(self):
        """Banzhaf Winning Width - grid scan version.

        Same as get_banzhaf_winning_width but replaces the two-directional
        binary search with a 100-point grid scan over the full feature range.
        For each pivot the method finds the contiguous winning interval that
        contains x_i and returns its width.  Correct when the model's 1-D
        slice is non-monotonic (multiple winning islands).
        """
        from tqdm import tqdm

        N_GRID = 100

        n_samples = self.data.shape[0]
        n_features = self.data.shape[1]

        if not hasattr(self, 'dp_tables') or len(self.dp_tables) != n_samples:
            self._build_dp_tables()

        all_bww = []

        for sample_idx in tqdm(range(n_samples), desc="  [bww_grid]", disable=not self.verbose):
            sample = self.data[sample_idx]
            dp_winning = self.dp_tables[sample_idx]
            background_data = self.dp_backgrounds[sample_idx]

            feat_mins = np.minimum(background_data.min(axis=0), sample)
            feat_maxs = np.maximum(background_data.max(axis=0), sample)
            margin = 0.1 * (feat_maxs - feat_mins + 1e-8)

            result = self._collect_pivots_and_base_rows(sample, dp_winning, background_data, n_features)
            if result is None:
                all_bww.append(np.zeros(n_features))
                continue

            pivot_features, bg_indices, base_rows, n_piv = result

            range_lo = feat_mins[pivot_features] - margin[pivot_features]
            range_hi = feat_maxs[pivot_features] + margin[pivot_features]

            # Build grid: shape (n_piv, N_GRID)
            t = np.linspace(0.0, 1.0, N_GRID)
            grid_vals = range_lo[:, None] + t[None, :] * (range_hi - range_lo)[:, None]

            # Evaluate all grid points in one batched call
            batch = np.repeat(base_rows, N_GRID, axis=0)
            pf_rep = np.repeat(pivot_features, N_GRID)
            batch[np.arange(n_piv * N_GRID), pf_rep] = grid_vals.ravel().astype(np.float32)

            preds = np.asarray(self.predict_fn(batch)).flatten().reshape(n_piv, N_GRID)
            wins = preds >= 0.5  # (n_piv, N_GRID)

            # For each pivot, find the contiguous winning run that contains x_j
            x_vals = sample[pivot_features]  # value of x for each pivot's feature
            widths = np.zeros(n_piv)
            for k in range(n_piv):
                x_k = x_vals[k]
                # find grid index closest to x_k
                x_idx = int(np.argmin(np.abs(grid_vals[k] - x_k)))
                if not wins[k, x_idx]:
                    continue  # x_k is in a losing region for this pivot
                # expand left and right while winning
                left = x_idx
                while left > 0 and wins[k, left - 1]:
                    left -= 1
                right = x_idx
                while right < N_GRID - 1 and wins[k, right + 1]:
                    right += 1
                widths[k] = grid_vals[k, right] - grid_vals[k, left]

            width_scores = np.zeros(n_features)
            for j in range(n_features):
                mask_j = pivot_features == j
                if not np.any(mask_j):
                    continue
                feat_range = feat_maxs[j] - feat_mins[j] + 2 * margin[j]
                if feat_range > 0:
                    width_scores[j] = widths[mask_j].sum() / feat_range

            width_scores /= (1 << (n_features - 1))
            all_bww.append(width_scores)

        return np.array(all_bww)

    def _build_dp_tables(self):
        """Build the DP winning table for every sample.

        Each sample gets its own fresh random background subset (stored in self.dp_backgrounds).
        Masks are batched for efficient prediction.
        """
        from tqdm import tqdm

        n_samples, n_features = self.data.shape
        total_masks = 1 << n_features
        MASKS_PER_BATCH = 256
        self.dp_tables = []
        self.dp_backgrounds = []
        self.dp_minimals = []

        for i in tqdm(range(n_samples), desc="  [dp tables]", disable=not self.verbose):
            background_data = self.background_set.get_data().astype(np.float32)
            self.dp_backgrounds.append(background_data)
            sample = self.data[i]
            n_bg = len(background_data)
            half_bg = n_bg / 2

            dp_winning = np.zeros(total_masks, dtype=np.int8)

            for batch_start in range(0, total_masks, MASKS_PER_BATCH):
                batch_end = min(batch_start + MASKS_PER_BATCH, total_masks)
                batch_size = batch_end - batch_start

                mega_batch = np.tile(background_data, (batch_size, 1))

                for k, mask in enumerate(range(batch_start, batch_end)):
                    indices = [j for j in range(n_features) if (mask & (1 << j))]
                    if indices:
                        row_start = k * n_bg
                        mega_batch[row_start:row_start + n_bg, indices] = sample[indices]

                all_preds = np.asarray(self.predict_fn(mega_batch)).flatten()

                for k, mask in enumerate(range(batch_start, batch_end)):
                    preds = all_preds[k * n_bg:(k + 1) * n_bg]
                    num_winning = (preds >= 0.5).sum()

                    if self.pivot_method == "majority":
                        dp_winning[mask] = 1 if num_winning > half_bg else 0
                    elif self.pivot_method == "veto":
                        dp_winning[mask] = 1 if num_winning == n_bg else 0

            self.dp_tables.append(dp_winning)
            self.dp_minimals.append(self._find_minimal_winning(dp_winning, n_features))

    @staticmethod
    def _find_minimal_winning(dp_winning, n):
        """Find all minimal winning coalitions from a DP table.

        A coalition is minimal winning if it wins and no proper subset wins.
        Uses superset pruning: once a minimal is found, all its supersets are
        skipped, which dramatically reduces work for large n.

        Returns a list of bitmasks.
        """
        total = 1 << n
        not_minimal = np.zeros(total, dtype=bool)
        minimal = []

        for mask in range(1, total):
            if dp_winning[mask] != 1:
                continue
            if not_minimal[mask]:
                continue
            is_min = True
            sub = (mask - 1) & mask
            while sub > 0:
                if dp_winning[sub] == 1:
                    is_min = False
                    break
                sub = (sub - 1) & mask
            if is_min:
                minimal.append(mask)
                complement = ((1 << n) - 1) ^ mask
                sup = complement
                while sup > 0:
                    not_minimal[mask | sup] = True
                    sup = (sup - 1) & complement
        return minimal

    def get_banzhaf_shapley_shubik(self):
        """Compute Banzhaf and Shapley-Shubik indices for all samples."""
        n_samples, n_features = self.data.shape

        all_banzhaf = []
        all_shapley = []
        all_banzhaf_neg = []
        all_shapley_neg = []

        ss_weights = np.zeros(n_features)
        for s in range(n_features):
            ss_weights[s] = (math.factorial(s) * math.factorial(n_features - s - 1)) / math.factorial(n_features)

        if not hasattr(self, 'dp_tables') or len(self.dp_tables) != n_samples:
            self._build_dp_tables()

        for i in range(n_samples):
            dp_winning = self.dp_tables[i]

            sample_banzhaf = np.zeros(n_features)
            sample_shapley = np.zeros(n_features)
            sample_banzhaf_neg = np.zeros(n_features)
            sample_shapley_neg = np.zeros(n_features)

            for mask in range(1 << n_features):
                s_size = bin(mask).count('1')

                for j in range(n_features):
                    if not (mask & (1 << j)):
                        if dp_winning[mask] == 0 and dp_winning[mask | (1 << j)] == 1:
                            sample_banzhaf[j] += 1
                            sample_shapley[j] += ss_weights[s_size]
                        elif dp_winning[mask] == 1 and dp_winning[mask | (1 << j)] == 0:
                            sample_banzhaf_neg[j] += -1
                            sample_shapley_neg[j] += -ss_weights[s_size]

            total_b = np.sum(sample_banzhaf)
            all_banzhaf.append(sample_banzhaf / total_b if total_b > 0 else sample_banzhaf)
            all_shapley.append(sample_shapley)

            total_b_neg = np.sum(sample_banzhaf_neg)
            all_banzhaf_neg.append(sample_banzhaf_neg / total_b_neg if total_b_neg > 0 else sample_banzhaf_neg)
            all_shapley_neg.append(sample_shapley_neg)

        return (
            np.array(all_banzhaf), np.array(all_shapley),
            np.array(all_banzhaf_neg), np.array(all_shapley_neg),
            np.array(all_banzhaf) + np.array(all_banzhaf_neg),
            np.array(all_shapley) + np.array(all_shapley_neg)
        )

    def _get_shapley_value(self):
        """Exact Shapley value via permutation enumeration, averaged over background set."""
        from tqdm import tqdm

        n_features = self.data.shape[1]
        feature_indices = list(range(n_features))
        all_shapley = []

        perms = list(itertools.permutations(feature_indices))
        total_permutations = len(perms)

        for sample in tqdm(self.data, desc="  [sv]", disable=not self.verbose):
            background_data = self.background_set.get_data().astype(np.float32)
            n_baselines = len(background_data)

            marginal_sums = np.zeros(n_features)

            for perm in perms:
                batch = background_data.copy()
                preds_before = np.asarray(self.predict_fn(batch)).flatten()

                for f_idx in perm:
                    batch[:, f_idx] = sample[f_idx]
                    preds_after = np.asarray(self.predict_fn(batch)).flatten()
                    marginal_sums[f_idx] += (preds_after - preds_before).sum()
                    preds_before = preds_after

            all_shapley.append(marginal_sums / (total_permutations * n_baselines))

        return all_shapley

    def _get_deegan_packel(self):
        """Compute Deegan-Packel index for all samples."""
        n = self.n_features
        all_dp = []
        for i, _ in enumerate(self.data):
            dp_counts = np.zeros(n)
            for mask in self.dp_minimals[i]:
                members = [j for j in range(n) if mask & (1 << j)]
                size = len(members)
                for j in members:
                    dp_counts[j] += 1.0 / size
            total = dp_counts.sum()
            all_dp.append(dp_counts / total if total > 0 else dp_counts)
        return all_dp

    def _get_responsibility(self):
        """Compute Responsibility index for all samples."""
        n = self.n_features
        all_ri = []
        for i, _ in enumerate(self.data):
            scores = np.zeros(n)
            for j in range(n):
                min_size = None
                for mask in self.dp_minimals[i]:
                    if mask & (1 << j):
                        size = bin(mask).count('1')
                        if min_size is None or size < min_size:
                            min_size = size
                if min_size is not None:
                    scores[j] = 1.0 / min_size
            all_ri.append(scores)
        return all_ri

    def _get_holler_packel(self):
        """Compute Holler-Packel index for all samples."""
        n = self.n_features
        all_hp = []
        for i, _ in enumerate(self.data):
            hp_counts = np.zeros(n)
            for mask in self.dp_minimals[i]:
                for j in range(n):
                    if mask & (1 << j):
                        hp_counts[j] += 1
            all_hp.append(hp_counts)
        return all_hp

    def _get_shap(self):
        """Compute SHAP values using KernelExplainer."""
        import shap
        from tqdm import tqdm

        np.random.seed(RANDOM_SEED)

        if self.background_set.method.method == "random":
            shap_results = []
            for sample in tqdm(self.data, desc="  [shap]", disable=not self.verbose):
                background_np = np.array(self.background_set.get_data())
                explainer = shap.KernelExplainer(self.predict_fn, background_np)
                shap_val = explainer.shap_values(sample.reshape(1, -1), nsamples="auto", silent=True)
                shap_results.append(np.array(shap_val).flatten())
            return np.array(shap_results)
        else:
            background_np = np.array(self.background_set.get_data_all())
            explainer = shap.KernelExplainer(self.predict_fn, background_np)
            shap_vals = explainer.shap_values(self.data, nsamples="auto", silent=True)
            return np.array(shap_vals)

    def _get_lime(self):
        """Compute LIME explanations."""
        from lime.lime_tabular import LimeTabularExplainer
        from tqdm import tqdm

        if self.background_set.method.method == "random":
            lime_results = []
            for sample in tqdm(self.data, desc="  [lime]", disable=not self.verbose):
                background_np = np.array(self.background_set.get_data())
                n_features = background_np.shape[1]
                feature_names = [f"x{i}" for i in range(n_features)]

                explainer = LimeTabularExplainer(
                    background_np,
                    mode="classification",
                    feature_names=feature_names,
                    discretize_continuous=False,
                    random_state=RANDOM_SEED
                )

                def lime_predict(x):
                    probs = np.asarray(self.predict_fn(x)).flatten()
                    return np.column_stack([1 - probs, probs])

                exp = explainer.explain_instance(sample, lime_predict, num_features=n_features)
                weights = np.zeros(n_features)
                for idx, w in exp.as_map()[1]:
                    weights[idx] = w
                lime_results.append(weights)
            return np.array(lime_results)
        else:
            background_np = np.array(self.background_set.get_data_all())
            n_features = background_np.shape[1]
            feature_names = [f"x{i}" for i in range(n_features)]

            explainer = LimeTabularExplainer(
                background_np,
                mode="classification",
                feature_names=feature_names,
                discretize_continuous=False,
                random_state=RANDOM_SEED
            )

            def lime_predict(x):
                probs = np.asarray(self.predict_fn(x)).flatten()
                return np.column_stack([1 - probs, probs])

            lime_results = []
            for sample in tqdm(self.data, desc="  [lime]", disable=not self.verbose):
                exp = explainer.explain_instance(sample, lime_predict, num_features=n_features)
                weights = np.zeros(n_features)
                for idx, w in exp.as_map()[1]:
                    weights[idx] = w
                lime_results.append(weights)
            return np.array(lime_results)
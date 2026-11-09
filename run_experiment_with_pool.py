"""Run one or more explainability experiments in parallel.

Each *experiment* loads a user workflow (data + model + predict function),
constructs a :class:`~explainer.My_Explainer`, computes the requested indices
on the positive test samples, and writes one CSV per index to the output
folder.

The module exposes both a :func:`main` CLI and the lower-level
:func:`run_experiment_with_pool` function used by the scripts under
``experiments/``.  Heavy dependencies (``torch``, ``shap``, ``lime``) are
imported lazily inside :func:`run_experiment` so the parent process that
manages the worker pool does not pay the cost of loading heavy dependencies.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

from background_method import BackgroundMethod

# Duplicated from explainer.ALL_IDX so the parent process never imports the
# heavy explainer module (torch / shap / lime).
ALL_IDX = [
    "shap", "lime", "bi", "ssi", "bi_neg", "ssi_neg", "bi_full", "ssi_full",
    "sv", "dp", "hp", "ri", "rwi", "bww",
]


def parse_background(s: str) -> BackgroundMethod:
    """Parse 'method:param' string into a BackgroundMethod."""
    if ":" in s:
        method, param = s.split(":", 1)
        return BackgroundMethod(method, int(param))
    return BackgroundMethod(s, None)


def run_experiment(
    workflow_name: str,
    output_folder: str,
    background: BackgroundMethod,
    pivot_method: str,
    indices: list,
    max_samples: int = None,
    background_n_samples: int = None,
    background_seed: int = None,
    force: bool = False,
    save_to_file: bool = True,
    verbose: bool = True,
):
    """Run a single experiment.

    Parameters
    ----------
    background_n_samples : int, optional
        If set, draw this many rows from ``X_background`` once (using
        ``background_seed`` for reproducibility) and pass the resulting
        fixed subset to the explainer with method ``all``.  This ensures
        every sample in the run sees the *same* background set, so
        across-run variation reflects background-set sensitivity rather
        than per-sample randomness.
    background_seed : int, optional
        RNG seed used when ``background_n_samples`` is set.  Defaults to
        the run index if not provided.
    """

    def log(msg):
        if verbose:
            print(msg, flush=True)

    t0 = time.time()
    log(f"[experiment] workflow    : {workflow_name}")
    log(f"[experiment] output      : {output_folder}")
    bg_param = f"{background.method}:{background.param}" if background.param else background.method
    log(f"[experiment] background  : {bg_param}")
    log(f"[experiment] pivot method: {pivot_method}")
    log(f"[experiment] indices     : {indices}")

    # Heavy imports live here so the parent (parallel-launcher) process
    # does not pay the cost of loading torch / shap / lime.
    from explainer import My_Explainer
    from helpers import load_workflow, save_feature_names, get_positive_samples

    workflow = load_workflow(workflow_name)
    workflow_data = workflow["load_everything"]()
    X_test, X_background = workflow["data"]()
    predict_fn = lambda x: workflow["predict"](workflow_data, x)

    if background_n_samples is not None:
        import numpy as np
        rng = np.random.default_rng(background_seed)
        n = min(background_n_samples, len(X_background))
        idx = rng.choice(len(X_background), size=n, replace=True)
        X_background = X_background[idx]
        background = BackgroundMethod("all", None)
        log(f"[experiment] pre-sampled : {n} background rows (seed={background_seed})")

    X_test_positive = get_positive_samples(X_test, predict_fn)
    if max_samples is not None:
        X_test_positive = X_test_positive[:max_samples]
    log(f"[experiment] explaining {len(X_test_positive)} positive samples.")

    if os.path.exists(output_folder):
        if force:
            import shutil
            shutil.rmtree(output_folder)
        else:
            raise FileExistsError(
                f"Output folder '{output_folder}' already exists. Use --force to overwrite."
            )
    os.makedirs(output_folder)
    save_feature_names(workflow_data, output_folder, X_test.shape[1])

    expl = My_Explainer(predict_fn, X_background, background, pivot_method, save_to_file)
    expl.explain(X_test_positive, indices, out_dir=output_folder)

    log(f"[experiment] done in {time.time() - t0:.1f}s → {output_folder}/")
    return expl.result


def _worker_wrapper(config: dict):
    """Wrapper function for multiprocessing.Pool."""
    try:
        result = run_experiment(**config)
        return (True, config.get("output_folder", "?"), result)
    except Exception as exc:
        return (False, config.get("output_folder", "?"), str(exc))


def run_experiment_with_pool(configs: list, max_workers: int = None):
    """Run multiple experiments in parallel using multiprocessing.Pool."""
    if not configs:
        print("[pool] No configs to run")
        return []

    if max_workers is None:
        max_workers = min(len(configs), os.cpu_count() or 1)

    print(f"[pool] {len(configs)} jobs, max {max_workers} workers", flush=True)
    t0 = time.time()

    prepared_configs = []
    for cfg in configs:
        config = dict(cfg)

        if isinstance(config.get("background"), str):
            config["background"] = parse_background(config["background"])

        config.setdefault("pivot_method", "majority")
        config.setdefault("indices", ALL_IDX)
        config.setdefault("max_samples", None)
        config.setdefault("background_n_samples", None)
        config.setdefault("background_seed", None)
        config.setdefault("force", False)
        config.setdefault("save_to_file", True)
        config.setdefault("verbose", True)

        prepared_configs.append(config)

    results = []
    if max_workers == 1:
        # Run sequentially in the main process so that sklearn's internal Loky
        # parallelism (n_jobs=-1) works without triggering the
        # "Loky-backed parallel loops cannot be called in a multiprocessing"
        # warning that occurs when joblib tries to spawn workers inside a Pool.
        results = [_worker_wrapper(cfg) for cfg in prepared_configs]
    else:
        with Pool(processes=max_workers) as pool:
            results = pool.map(_worker_wrapper, prepared_configs)

    print("\n[pool] Results:", flush=True)
    for success, label, result_or_error in results:
        if success:
            print(f"  ✓ {label}", flush=True)
        else:
            print(f"  ✗ {label} - {result_or_error}", flush=True)

    print(f"[pool] all done in {time.time() - t0:.1f}s", flush=True)

    return results


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Run explainability experiments using multiprocessing.Pool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-file", type=str, default=None,
                        help="JSON file with list of experiment configs")
    parser.add_argument("--workflow", required=False,
                        help="Path to user script")
    parser.add_argument("--output", required=False,
                        help="Output folder for results")
    parser.add_argument("--background", default="kmeans:10",
                        help="Background method and param")
    parser.add_argument("--pivot", default="majority", choices=["majority", "veto"],
                        help="Pivot method for Banzhaf/SSI")
    parser.add_argument("--indices", nargs="+", default=ALL_IDX,
                        help="Explanation indices to compute")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of positive samples")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite output folder if it exists")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Number of parallel workers")
    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, 'r') as f:
            configs = json.load(f)

        for cfg in configs:
            if "background" in cfg and isinstance(cfg["background"], str):
                cfg["background"] = parse_background(cfg["background"])

        run_experiment_with_pool(configs, max_workers=args.max_workers)
    else:
        if not args.workflow or not args.output:
            parser.error("Either --config-file or both --workflow and --output are required")

        config = {
            "workflow_name": args.workflow,
            "output_folder": args.output,
            "background": parse_background(args.background),
            "pivot_method": args.pivot,
            "indices": args.indices,
            "max_samples": args.max_samples,
            "force": args.force,
        }

        run_experiment_with_pool([config], max_workers=args.max_workers)


if __name__ == "__main__":
    main()


def _test_worker(config: dict):
    """Test worker function for debugging pool functionality."""
    import os
    pid = os.getpid()
    workflow = config.get("workflow_name", "N/A")
    output = config.get("output_folder", "N/A")
    bg = config.get("background", "N/A")

    print(f"[PID {pid}] workflow={workflow}, output={output}, background={bg}", flush=True)
    time.sleep(1)

    return (True, output, f"completed by PID {pid}")


def test_pool():
    """Test the pool functionality without running actual experiments."""
    print("=" * 60)
    print("POOL TEST - just printing parameters from each worker")
    print("=" * 60)

    test_configs = [
        {
            "workflow_name": "user_scripts/breast_cancer_user_script.py",
            "output_folder": "results/test_pool_1",
            "background": "kmeans:40",
            "pivot_method": "majority",
        },
        {
            "workflow_name": "user_scripts/german_credit_user_script.py",
            "output_folder": "results/test_pool_2",
            "background": "random:20",
            "pivot_method": "veto",
        },
        {
            "workflow_name": "user_scripts/linear_user_script.py",
            "output_folder": "results/test_pool_3",
            "background": "kernel_density:10",
            "pivot_method": "majority",
        },
        {
            "workflow_name": "user_scripts/breast_cancer_user_script.py",
            "output_folder": "results/test_pool_4",
            "background": "kmeans:10",
            "pivot_method": "majority",
        },
    ]

    max_workers = 2
    print(f"\n[test] Running {len(test_configs)} test jobs with {max_workers} workers\n")
    t0 = time.time()

    with Pool(processes=max_workers) as pool:
        results = pool.map(_test_worker, test_configs)

    print("\n[test] Results:")
    for success, label, result in results:
        status = "✓" if success else "✗"
        print(f"  {status} {label}: {result}")

    print(f"\n[test] Test completed in {time.time() - t0:.1f}s")
    print("=" * 60)
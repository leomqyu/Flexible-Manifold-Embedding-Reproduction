"""Evaluate the FME models on the paper's semi-supervised and FME/U protocols."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from fme import FlexibleManifoldEmbedding, UnsupervisedFME, knn_laplacian
from fme.baselines import LinearLapRLS, LocalityPreservingProjection, graph_propagation
from fme.data import BENCHMARKS, load_dataset, pca_preprocess, semi_split, unsupervised_split

ROOT = Path(__file__).resolve().parent


def values(s):
    out = tuple(float(v) for v in s.split(","))
    if not out or any(not np.isfinite(x) or x <= 0 for x in out):
        raise argparse.ArgumentTypeError("parameter values must be positive floats")
    return out


def dimensions(s):
    result = tuple(int(v) for v in s.split(","))
    if not result or any(d < 1 for d in result):
        raise argparse.ArgumentTypeError("dimensions must be positive integers")
    return result


def read_hash():
    paths = [ROOT / "run_experiments.py", ROOT / "requirements.txt",
             *sorted((ROOT / "fme").glob("*.py"))]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def load_record_config(args, X, y):
    data_file = Path(args.data_dir) / f"{args.dataset}.npz"
    data_hash = hashlib.sha256(data_file.read_bytes()).hexdigest() if data_file.exists() else None
    return dict(dataset=args.dataset, mode=args.mode, per_class=list(args.per_class),
                dims=list(args.dims), mu=list(args.mu), gamma=list(args.gamma),
                selection=args.selection, n_neighbors=args.neighbors, bandwidth=args.bandwidth,
                pca_variance=args.pca_variance, seed=args.seed, split_start=args.split_start,
                splits=args.splits, threads=args.threads, dataset_sha256=data_hash,
                n_samples=len(X), n_features=X.shape[1], n_classes=len(np.unique(y)),
                source_sha256=read_hash(), python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)


def atomic_json(file, value):
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix(file.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(file)


def evaluate_projection(Z_train, Z_test, y_train, y_test, labeled_indices):
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(Z_train[labeled_indices], y_train[labeled_indices])
    mask = np.ones(len(y_train), dtype=bool)
    mask[labeled_indices] = False
    metrics = {}
    if np.any(mask):
        metrics["unlabeled"] = float(clf.score(Z_train[mask], y_train[mask]))
    metrics["test"] = float(clf.score(Z_test, y_test))
    return metrics


def select(rows, selection):
    """Grid selection for reported results.

    `paper_oracle` selects the maximum OUTER accuracy separately for each
    metric: this resembles the published best-of-grid protocol but is
    optimistically biased. It MUST NOT be described as held-out tuning.
    `fixed` uses the first hyperparameter setting only.
    """
    if selection == "fixed":
        return [x for x in rows if x["candidate"] == 0]
    best = {}
    for r in rows:
        key = (r["method"], r["phase"], r["dimension"])
        if key not in best or r["accuracy"] > best[key]["accuracy"]:
            best[key] = r
    return list(best.values())


def _rows(method, metrics, split, seed, regime, dim, mu=None, gamma=None,
          candidate=0, status="ok"):
    return [dict(method=method, phase=phase, split=split, seed=seed,
                 regime=regime, dimension=dim, mu=mu, gamma=gamma,
                 candidate=candidate, accuracy=float(acc), status=status)
            for phase, acc in metrics.items()]


def one_semi(X, y, args, split, regime):
    seed = args.seed + split
    train, test, labeled_full = semi_split(y, seed, regime)
    Xt, Xv, pca = pca_preprocess(X[train], X[test], args.pca_variance)
    yt, yv = y[train], y[test]
    # ``train`` is shuffled; map absolute labeled indices to train-row offsets.
    positions = np.flatnonzero(np.isin(train, labeled_full))
    yu = np.full(len(train), -1, dtype=int)
    yu[positions] = yt[positions]
    L, _, scale = knn_laplacian(Xt, args.neighbors, args.bandwidth)
    rows = []
    candidate = 0
    for mu in args.mu:
        for gamma in args.gamma:
            model = FlexibleManifoldEmbedding(mu=mu, gamma=gamma, n_neighbors=args.neighbors)
            model.fit(Xt, yu, laplacian=L)
            metrics = evaluate_projection(model.transform(Xt), model.transform(Xv),
                                          yt, yv, positions)
            rows.extend(_rows("FME", metrics, split, seed, regime,
                              len(np.unique(y)), mu, gamma, candidate))
            candidate += 1
    for c, mu in enumerate(args.mu):
        baseline = LinearLapRLS(ridge=mu).fit(Xt, yu, L)
        metrics = evaluate_projection(baseline.transform(Xt), baseline.transform(Xv),
                                      yt, yv, positions)
        rows.extend(_rows("LapRLS/L", metrics, split, seed, regime,
                          len(np.unique(y)), mu, None, c))
    for method in ("lgc", "gfhf"):
        prediction, scores = graph_propagation(yu, L, variant=method)
        mask = yu == -1
        rows.extend(_rows(method.upper(), {"unlabeled":np.mean(prediction[mask] == yt[mask])},
                          split, seed, regime, len(np.unique(y))))
    return select(rows, args.selection)


def one_unsup(X, y, args, split):
    seed = args.seed + split
    train, test = unsupervised_split(y, seed, per_class=5)
    Xt, Xv, pca = pca_preprocess(X[train], X[test], args.pca_variance)
    yt, yv = y[train], y[test]
    L, _, _ = knn_laplacian(Xt, args.neighbors, args.bandwidth)
    rows = []
    for dim in args.dims:
        if dim >= min(len(Xt)-1, Xt.shape[1]+1):
            continue
        candidate = 0
        for mu in args.mu:
            for gamma in args.gamma:
                model = UnsupervisedFME(n_components=dim, mu=mu, gamma=gamma,
                                        n_neighbors=args.neighbors, random_state=seed)
                model.fit(Xt, laplacian=L)
                metrics = evaluate_projection(model.transform(Xt), model.transform(Xv),
                                              yt, yv, np.arange(len(yt)))
                rows.extend(_rows("FME/U", metrics, split, seed, 5, dim,
                                  mu, gamma, candidate))
                candidate += 1
        pca_model = PCA(n_components=dim, svd_solver="full").fit(Xt)
        metrics = evaluate_projection(pca_model.transform(Xt), pca_model.transform(Xv),
                                      yt, yv, np.arange(len(yt)))
        rows.extend(_rows("PCA", metrics, split, seed, 5, dim))
        for sr, method in ((False, "LPP"), (True, "LPP-SR")):
            baseline = LocalityPreservingProjection(n_components=dim,
                        spectral_regression=sr).fit(Xt, L)
            metrics = evaluate_projection(baseline.transform(Xt), baseline.transform(Xv),
                                          yt, yv, np.arange(len(yt)))
            rows.extend(_rows(method, metrics, split, seed, 5, dim))
    return select(rows, args.selection)


def write_results(output, records, split_count):
    import pandas as pd
    table = pd.DataFrame.from_records(records)
    if table.empty:
        return
    cols = ["split", "seed", "regime", "method", "phase", "dimension", "mu",
            "gamma", "candidate", "accuracy", "status"]
    table = table[cols].sort_values(["split", "regime", "method", "dimension", "phase"])
    table.to_csv(output / "runs.csv", index=False)
    groups = ["regime", "method", "phase", "dimension"]
    summary = table.groupby(groups,dropna=False)["accuracy"].agg(
        n="count", mean="mean", std="std").reset_index()
    summary["mean_pct"] = summary["mean"] * 100
    summary["std_pct"] = summary["std"] * 100
    summary.to_csv(output / "summary.csv", index=False)
    lines = ["# Experiment summary", "", f"Completed outer splits: {len(set(table['split']))}/{split_count}.",
             "", "Accuracy is in percent. Standard deviation uses ddof=1; N/A for one split.", "",
             "| Regime | Method | Phase | Dim | Splits | Mean ± sample SD |",
             "|---:|---|---|---:|---:|---:|"]
    for item in summary.itertuples():
        sd = f"{item.std_pct:.2f}" if np.isfinite(item.std_pct) else "N/A"
        lines.append(f"| {item.regime} | {item.method} | {item.phase} | {item.dimension} | {item.n} | {item.mean_pct:.2f} ± {sd} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="digits")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--mode", choices=("semi", "unsup"), default="semi")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--split-start", type=int, default=0)
    parser.add_argument("--per-class", type=dimensions, default=None)
    parser.add_argument("--dims", type=dimensions, default=(2, 4, 8, 16, 32))
    parser.add_argument("--mu", type=values, default=(1.,))
    parser.add_argument("--gamma", type=values, default=(1.,))
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--bandwidth", type=float, default=None)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--selection", choices=("fixed", "paper_oracle"), default="fixed")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Development smoke test only; do not use for paper comparison")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.bandwidth is None:
        args.bandwidth = "median"
    if min(args.splits,args.neighbors,args.threads) < 1 or args.split_start < 0:
        parser.error("splits, neighbors, threads must be positive and split-start nonnegative")
    if args.seed < 0 or args.seed + args.split_start + args.splits >= 2**32:
        parser.error("seed sequence must fit within uint32")
    X, y = load_dataset(args.dataset,args.data_dir)
    if args.max_samples:
        _, subset = train_test_split(np.arange(len(y)), test_size=args.max_samples,
                                     stratify=y, random_state=args.seed)
        X, y = X[subset], y[subset]
        print("WARNING: development subsample; results are not a paper reproduction",flush=True)
    if args.per_class is None:
        args.per_class = BENCHMARKS.get(args.dataset,{}).get("semi_labels",(1, 2, 3))
    if args.mode == "unsup" and args.dataset == "20news":
        parser.error("The paper's FME/U experiments concern face datasets only")
    if args.output is None:
        args.output = ROOT / "outputs" / args.dataset / args.mode
    config = load_record_config(args,X,y)
    progress = args.output / "progress.json"
    if args.resume:
        if not progress.exists():
            parser.error("No progress.json exists for --resume")
        saved = json.loads(progress.read_text())
        if saved["config"] != config:
            parser.error("Resume requires identical configuration, versions and source files")
        records = saved["records"]
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        records = []
        atomic_json(progress,{"config":config,"records":records})
    completed = {(r["split"],r["regime"]) for r in records}
    with threadpool_limits(limits=args.threads):
        for i in range(args.split_start,args.split_start+args.splits):
            regimes = args.per_class if args.mode=="semi" else (5,)
            for regime in regimes:
                if (i,regime) in completed:
                    continue
                print(f"Split {i}, labels/class {regime}: {args.dataset} {args.mode}",flush=True)
                result = one_semi(X,y,args,i,regime) if args.mode=="semi" else one_unsup(X,y,args,i)
                if not result:
                    raise ValueError("No experiment results; check embedding dimensions")
                records.extend(result)
                atomic_json(progress,{"config":config,"records":records})
                write_results(args.output,records,args.splits)
    print((args.output / "summary.md").read_text(),flush=True)


if __name__ == "__main__":
    main()

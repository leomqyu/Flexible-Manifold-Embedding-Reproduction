"""Data loading for paper benchmarks and locally supplied arrays."""

from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

BENCHMARKS = {
    "umist": {"classes": 20, "semi_labels": (1, 2, 3), "unsup_per_class": 5},
    "cmu_pie": {"classes": 68, "semi_labels": (1, 2, 3), "unsup_per_class": 5},
    "yale_b": {"classes": 38, "semi_labels": (1, 2, 3), "unsup_per_class": 5},
    "coil20": {"classes": 20, "semi_labels": (1, 2, 3), "unsup_per_class": 5},
    "20news": {"classes": 4, "semi_labels": (10, 20, 30), "unsup_per_class": None},
}


def load_dataset(dataset, data_dir="data"):
    """Read data/<dataset>.npz with X (n,d) and integer class labels y (n,).

    A small built-in digits dataset is provided solely for numerical smoke
    tests, never as a substitute for any of the paper's benchmarks.
    """
    if dataset == "digits":
        from sklearn.datasets import load_digits
        bunch = load_digits()
        return np.asarray(bunch.data, dtype=float), np.asarray(bunch.target, dtype=int)
    path = Path(data_dir) / f"{dataset}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}. See README for format/source restrictions.")
    with np.load(path, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["y"])
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("Expected X=(n,d) and y=(n,)")
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite dataset entries")
    classes, y = np.unique(y, return_inverse=True)
    return X, y.astype(int)


def semi_split(y, seed, per_class):
    """Paper split: 50% outer train/test and exactly `per_class` labeled per class."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = len(y) // 2
    train, test = idx[:cut], idx[cut:]
    y = np.asarray(y)
    chosen = []
    for label in np.unique(y):
        candidates = train[y[train] == label]
        if len(candidates) < per_class:
            raise ValueError(f"Insufficient training observations of class {label}")
        chosen.extend(rng.choice(candidates, per_class, replace=False).tolist())
    chosen = np.asarray(chosen, dtype=int)
    return train, test, chosen


def unsupervised_split(y, seed, per_class=5):
    """Paper FME/U protocol: five training images per class, remaining as test."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    chosen = []
    for label in np.unique(y):
        candidates = np.flatnonzero(y == label)
        if len(candidates) <= per_class:
            raise ValueError(f"Insufficient observations of class {label}")
        chosen.extend(rng.choice(candidates, per_class, replace=False).tolist())
    train = np.asarray(chosen, dtype=int)
    mask = np.ones(len(y), dtype=bool)
    mask[train] = False
    return train, np.flatnonzero(mask)


def pca_preprocess(X_train, X_test, variance=0.95):
    """Fit PCA exclusively on the outer training set (no test leakage)."""
    n, d = X_train.shape
    if variance is None:
        return np.asarray(X_train, float), np.asarray(X_test, float), None
    if not 0 < variance <= 1:
        raise ValueError("variance must be in (0,1]")
    max_dim = min(n, d)
    if variance == 1:
        components = max_dim
    else:
        components = variance
    model = PCA(n_components=components, svd_solver="full")
    train = model.fit_transform(X_train)
    test = model.transform(X_test)
    return train, test, model

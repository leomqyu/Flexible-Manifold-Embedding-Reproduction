"""Reference baselines for the experiments in Nie et al. (2010)."""

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, spsolve

from .model import _RidgeResidual


def _one_hot(y):
    y = np.asarray(y)
    labeled = y != -1
    classes = np.unique(y[labeled])
    if len(classes) < 2:
        raise ValueError("at least two labeled classes are required")
    Y = (y[:, None] == classes[None, :]).astype(float)
    return classes, labeled, Y


def graph_propagation(y, L, variant="gfhf", alpha=0.99):
    """Transductive predictions; does not provide a test-sample transform."""
    classes, labeled, Y = _one_hot(y)
    n = len(y)
    L = sparse.csr_matrix(L)
    if L.shape != (n, n):
        raise ValueError("L and y dimension mismatch")
    if variant == "gfhf":
        unlabel = ~labeled
        F = Y.copy()
        if np.any(unlabel):
            A = L[unlabel][:, unlabel].tocsc()
            B = -(L[unlabel][:, labeled] @ Y[labeled])
            # A component without labels is genuinely unidentifiable.
            # A tiny ridge supplies a deterministic finite solution.
            A = A + sparse.eye(A.shape[0], format="csc") * 1e-10
            F[unlabel] = spsolve(A, B)
    elif variant == "lgc":
        if not 0 < alpha < 1:
            raise ValueError("LGC alpha must lie in (0,1)")
        D = np.asarray(L.diagonal()).ravel()
        invsqrt = np.zeros_like(D)
        invsqrt[D > 0] = 1.0 / np.sqrt(D[D > 0])
        B = sparse.diags(invsqrt)
        normL = sparse.eye(n, format="csr") - B @ (sparse.diags(D) - L) @ B
        A = (1 - alpha) * sparse.eye(n, format="csr") + alpha * normL
        F = spsolve(A.tocsc(), (1 - alpha) * Y)
    else:
        raise ValueError("variant must be 'lgc' or 'gfhf'")
    return classes[np.argmax(F, axis=1)], F


class LinearLapRLS:
    """Hard-linear Laplacian-regularized least-squares control."""

    def __init__(self, ridge=1.0):
        if ridge <= 0:
            raise ValueError("ridge must be >0")
        self.ridge = ridge

    def fit(self, X, y, L):
        X = np.asarray(X, dtype=float)
        classes, labeled, Y = _one_hot(y)
        Z = np.column_stack((X, np.ones(len(X))))
        if sparse.issparse(L):
            A = Z.T @ (L @ Z)
        else:
            A = Z.T @ L @ Z
        A += Z[labeled].T @ Z[labeled]
        A[:X.shape[1], :X.shape[1]] += self.ridge * np.eye(X.shape[1])
        rhs = Z[labeled].T @ Y[labeled]
        sol = np.linalg.solve(A, rhs)
        self.coef_, self.intercept_ = sol[:-1], sol[-1]
        self.classes_ = classes
        return self

    def transform(self, X):
        return np.asarray(X) @ self.coef_ + self.intercept_


class LocalityPreservingProjection:
    """Generalized graph eigenproblem LPP with optional spectral regression."""

    def __init__(self, n_components=2, spectral_regression=False, ridge=1e-3):
        self.n_components = int(n_components)
        self.spectral_regression = spectral_regression
        self.ridge = ridge

    def fit(self, X, L):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if not 1 <= self.n_components < min(n, d + 1):
            raise ValueError("invalid number of embedding components")
        L = sparse.csr_matrix(L)
        if L.shape != (n, n):
            raise ValueError("L shape mismatch")
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        deg = np.asarray(L.diagonal()).ravel()
        if np.any(deg <= 0):
            raise ValueError("LPP requires positive graph degrees")
        k = self.n_components
        if self.spectral_regression:
            # The trivial generalized eigenvector is constant. Skip it.
            if n <= k + 2:
                raise ValueError("LPP-SR needs more samples than embedding dimensions")
            vals, vecs = eigsh(L, M=sparse.diags(deg), k=k + 1, sigma=-1e-7)
            inds = np.argsort(vals)[1:k + 1]
            scores = vecs[:, inds]
            ridge = _RidgeResidual(X, gamma=1.0 / self.ridge)
            self.coef_ = ridge.coef(scores)
        else:
            A = Xc.T @ (L @ Xc)
            B = Xc.T @ (deg[:, None] * Xc)
            eps = self.ridge * max(np.trace(B) / d, 1e-12)
            vals, vecs = eigh((A + A.T) * 0.5,
                              (B + B.T) * 0.5 + eps * np.eye(d),
                              subset_by_index=(0, k - 1))
            self.coef_ = vecs
        return self

    def transform(self, X):
        return (np.asarray(X) - self.mean_) @ self.coef_

"""Flexible manifold embedding (Nie et al., IEEE TIP 2010, §§III–IV).

Row-major convention: X is (n_samples, n_features), F is (n_samples,
embedding_dim). The published regularizer is mu*(||W||_F^2 +
gamma*||F-XW-1b^T||_F^2); the names mu and gamma are not swapped.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, cg, eigsh
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from .graph import knn_laplacian


class _RidgeResidual:
    """Eliminate W,b without forming an n-by-n projection matrix."""

    def __init__(self, X, gamma):
        self.mean = X.mean(axis=0)
        self.centered = X - self.mean
        self.n, self.d = X.shape
        self.gamma = gamma
        self.primal = self.d <= self.n
        if self.primal:
            gram = self.centered.T @ self.centered
            system = np.eye(self.d) + gamma * gram
        else:
            gram = self.centered @ self.centered.T
            system = np.eye(self.n) + gamma * gram
        self.factor = cho_factor(system, overwrite_a=True, check_finite=False)

    def coef(self, targets):
        target = targets - targets.mean(axis=0)
        if self.primal:
            return self.gamma * cho_solve(self.factor, self.centered.T @ target, check_finite=False)
        return self.gamma * (self.centered.T @ cho_solve(self.factor, target, check_finite=False))

    def apply(self, targets):
        """(H - gamma*HX*(I+gamma*X^THX)^-1*X^TH) targets."""
        targets = np.asarray(targets)
        v = targets - targets.mean(axis=0)
        if self.primal:
            out = v - self.gamma * self.centered @ cho_solve(
                self.factor, self.centered.T @ v, check_finite=False)
        else:
            out = cho_solve(self.factor, v, check_finite=False)
        # Keep numerically centered when multiplying many times in eigsh/cg.
        return out - out.mean(axis=0)


def _validated(X, mu, gamma):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError("X must be finite and have shape (n>=3, d>=1)")
    if not np.isfinite(mu) or not np.isfinite(gamma) or mu <= 0 or gamma <= 0:
        raise ValueError("mu and gamma must be positive and finite")
    return X


def _graph(X, supplied, n_neighbors, bandwidth):
    if supplied is None:
        L, _, _ = knn_laplacian(X, n_neighbors, bandwidth)
    else:
        L = supplied
        if L.shape != (len(X), len(X)):
            raise ValueError("the supplied Laplacian has the wrong shape")
        if not issparse(L):
            L = np.asarray(L, dtype=np.float64)
        if np.max(np.abs(np.asarray(L.sum(axis=1)).ravel())) > 1e-7:
            raise ValueError("Laplacian rows must sum to zero")
    return L


class FlexibleManifoldEmbedding(BaseEstimator, ClassifierMixin):
    """Semi-supervised FME; `y=-1` means an unlabeled graph node.

    Attributes
    ----------
    scores_ : ndarray, (n, n_classes)
        Optimized transductive F (not the same as inductive projections).
    coef_ : ndarray, (d, n_classes)
        Learned out-of-sample projection W.
    intercept_ : ndarray, (n_classes,)
        Out-of-sample intercept b.
    """

    def __init__(self, mu=1.0, gamma=1.0, n_neighbors=5,
                 bandwidth="median", tol=1e-9, max_iter=5000,
                 dense_threshold=400):
        self.mu = mu
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.bandwidth = bandwidth
        self.tol = tol
        self.max_iter = max_iter
        self.dense_threshold = dense_threshold

    def fit(self, X, y, *, laplacian=None):
        X = _validated(X, self.mu, self.gamma)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must have one entry per row of X")
        labeled = y != -1
        classes = np.unique(y[labeled])
        if len(classes) < 2:
            raise ValueError("at least two classes must have labels")
        u = labeled.astype(float)
        Y = np.zeros((len(X), len(classes)), dtype=float)
        for i, label in enumerate(classes):
            Y[labeled & (y == label), i] = 1.0
        L = _graph(X, laplacian, self.n_neighbors, self.bandwidth)
        ridge = _RidgeResidual(X, self.gamma)
        n = len(X)
        a = self.mu * self.gamma

        def matvec(z):
            z = np.asarray(z, dtype=float).reshape(n, -1)
            return (u[:, None] * z + L @ z + a * ridge.apply(z)).reshape(-1)

        if n <= self.dense_threshold:
            eye = np.eye(n)
            matrix = np.column_stack([matvec(eye[:, i]) for i in range(n)])
            # Small-system direct solution also serves as an oracle for tests.
            F = np.linalg.solve((matrix + matrix.T) * 0.5, u[:, None] * Y)
        else:
            operator = LinearOperator((n, n), matvec=lambda v:matvec(v), dtype=float)
            solutions = []
            for column in range(len(classes)):
                v, info = cg(operator, u * Y[:, column], rtol=self.tol,
                             atol=0.0, maxiter=self.max_iter)
                if info != 0:
                    raise RuntimeError(f"FME conjugate gradient did not converge (info={info})")
                solutions.append(v)
            F = np.column_stack(solutions)
        W = ridge.coef(F)
        self.coef_ = W
        self.intercept_ = F.mean(axis=0) - ridge.mean @ W
        self.scores_ = F
        self.classes_ = classes
        self.laplacian_ = L
        self.residual_ = F - (X @ W + self.intercept_)
        self.objective_ = self.objective(X, Y, u, L, F, W, self.intercept_)
        self.train_projection_ = self.transform(X)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def decision_function(self, X):
        return self.transform(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def predict_transductive(self):
        return self.classes_[np.argmax(self.scores_, axis=1)]

    def objective(self, X, Y, u, L, F, W, b):
        diff = F - Y
        residual = F - np.asarray(X) @ W - b
        return float(np.sum(u[:, None] * diff**2) + np.sum(F * (L @ F))
                     + self.mu * (np.sum(W**2) + self.gamma * np.sum(residual**2)))


class UnsupervisedFME(BaseEstimator, TransformerMixin):
    """FME/U: smallest nonconstant eigenvectors of L + mu*gamma*P.

    Embeddings are centered and have orthonormal columns. The constant
    eigenvector is excluded even when its eigenspace is numerically degenerate.
    """

    def __init__(self, n_components=2, mu=1.0, gamma=1.0,
                 n_neighbors=5, bandwidth="median", tol=1e-8,
                 max_iter=10000, dense_threshold=400, random_state=0):
        self.n_components = n_components
        self.mu = mu
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.bandwidth = bandwidth
        self.tol = tol
        self.max_iter = max_iter
        self.dense_threshold = dense_threshold
        self.random_state = random_state

    def fit(self, X, y=None, *, laplacian=None):
        X = _validated(X, self.mu, self.gamma)
        n = len(X)
        k = int(self.n_components)
        if not 1 <= k < n - 1:
            raise ValueError("n_components must lie in [1, n_samples - 2]")
        L = _graph(X, laplacian, self.n_neighbors, self.bandwidth)
        ridge = _RidgeResidual(X, self.gamma)
        alpha = self.mu * self.gamma

        def base(v):
            v = np.asarray(v).reshape(n, -1)
            return L @ v + alpha * ridge.apply(v)

        if n <= self.dense_threshold:
            # Work in an orthonormal basis of the mean-zero subspace.
            # This avoids inadvertently selecting the trivial constant mode.
            from scipy.linalg import null_space
            Q = null_space(np.ones((1, n)))
            op = base(Q)
            A = (Q.T @ op + op.T @ Q) * 0.5
            values, vecs = eigh(A, subset_by_index=(0, k - 1))
            F = Q @ vecs
        else:
            # Add a constant-only penalty to move the trivial zero mode away
            # from the eigenpairs sought; the centered eigenvectors are unchanged.
            degree = np.asarray(L.diagonal()).ravel()
            shift = 100 * (np.max(degree) + alpha + 1)
            def mv(z):
                z = z.reshape(n, 1)
                return (base(z) + shift * np.mean(z)).reshape(-1)
            operator = LinearOperator((n, n), matvec=mv, dtype=float)
            values, F = eigsh(operator, k=k, which="SA", tol=self.tol,
                              maxiter=self.max_iter)
            order = np.argsort(values)
            values, F = values[order], F[:, order]
            F -= F.mean(axis=0)
            F, _ = np.linalg.qr(F)
        self.embedding_ = F
        self.eigenvalues_ = values
        self.coef_ = ridge.coef(F)
        self.intercept_ = F.mean(axis=0) - ridge.mean @ self.coef_
        self.train_projection_ = self.transform(X)
        self.laplacian_ = L
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

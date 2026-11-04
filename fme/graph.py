"""Sparse neighborhood graph construction for manifold regularization."""

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def knn_laplacian(X, n_neighbors=5, bandwidth="median", symmetric="union"):
    """Build symmetric heat-kernel affinities and an unnormalized Laplacian.

    The paper uses a Gaussian-weighted nearest-neighbor graph. The number of
    neighbors and bandwidth rule remain explicit to make graph conventions
    testable and to avoid silently conflating different literature variants.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or len(X) < 3 or not np.isfinite(X).all():
        raise ValueError("X must be a finite 2-D array containing >=3 samples")
    n = len(X)
    k = int(n_neighbors)
    if not 1 <= k < n:
        raise ValueError("n_neighbors must lie in [1, n_samples - 1]")
    if symmetric not in ("union", "mutual"):
        raise ValueError("symmetric must be union or mutual")
    neigh = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    distances, indices = neigh.kneighbors(X)
    # Remove the query sample by index instead of assuming it occupies index 0
    # when two or more observations are identical.
    rows = []
    columns = []
    square_distances = []
    for i in range(n):
        valid = indices[i] != i
        ids = indices[i, valid][:k]
        ds = distances[i, valid][:k]
        if len(ids) != k:
            raise ValueError("insufficient nonself neighbors")
        rows.extend([i] * k)
        columns.extend(ids.tolist())
        square_distances.extend((ds * ds).tolist())
    square_distances = np.asarray(square_distances, dtype=np.float64)
    positive = square_distances[square_distances > 0]
    if isinstance(bandwidth, str):
        if bandwidth != "median":
            raise ValueError("bandwidth must be 'median' or a positive scalar")
        scale = float(np.median(positive)) if len(positive) else 1.0
    else:
        scale = float(bandwidth)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("bandwidth must be positive and finite")
    weight = np.exp(-square_distances / scale)
    S = sparse.csr_matrix((weight, (rows, columns)), shape=(n, n))
    if symmetric == "union":
        S = S.maximum(S.T)
    else:
        S = S.minimum(S.T)
    S.setdiag(0)
    S.eliminate_zeros()
    L = sparse.diags(np.asarray(S.sum(axis=1)).ravel()) - S
    return L.tocsr(), S.tocsr(), scale

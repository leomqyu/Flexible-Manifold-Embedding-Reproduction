"""Flexible manifold embedding for semi-supervised and unsupervised learning."""

from .graph import knn_laplacian
from .model import FlexibleManifoldEmbedding, UnsupervisedFME

__all__ = ["knn_laplacian", "FlexibleManifoldEmbedding", "UnsupervisedFME"]

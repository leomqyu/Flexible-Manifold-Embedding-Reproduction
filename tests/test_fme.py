"""CPU-only correctness tests for the exact FME quadratic objectives."""

import numpy as np
import pytest
from scipy.linalg import null_space
from sklearn.datasets import make_blobs

from fme import FlexibleManifoldEmbedding, UnsupervisedFME, knn_laplacian
from fme.baselines import LinearLapRLS, LocalityPreservingProjection, graph_propagation
from fme.data import semi_split, unsupervised_split, pca_preprocess
from fme.model import _RidgeResidual


def synthetic(n=32, d=5, seed=12):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n,d))
    y = np.repeat([0,1], n//2)
    return X, y


def test_graph_properties_duplicate_rows():
    X,_ = synthetic()
    X[1] = X[0]
    L,S,_ = knn_laplacian(X,5)
    np.testing.assert_allclose((S-S.T).toarray(),0,atol=1e-12)
    np.testing.assert_allclose(np.asarray(L.sum(axis=1)),0,atol=1e-12)
    assert np.all(S.diagonal() == 0)
    np.testing.assert_array_less(-1e-9,np.linalg.eigvalsh(L.toarray()))


def test_fme_recovered_stationary_point():
    X,y = synthetic(n=28,d=4)
    y[[1,4,6,14,18,23]]=-1
    model=FlexibleManifoldEmbedding(mu=.18,gamma=3.0).fit(X,y)
    F,W,b=model.scores_,model.coef_,model.intercept_
    L=model.laplacian_.toarray()
    Y=(y[:,None]==model.classes_[None,:]).astype(float)
    u=(y!=-1).astype(float)
    residual=F-X@W-b
    grad_F = 2*(u[:,None]*(F-Y)+L@F+model.mu*model.gamma*residual)
    grad_W = 2*model.mu*(W-model.gamma*X.T@residual)
    grad_b = np.sum(residual,axis=0)
    np.testing.assert_allclose(grad_F,0,atol=3e-8)
    np.testing.assert_allclose(grad_W,0,atol=3e-8)
    np.testing.assert_allclose(grad_b,0,atol=3e-8)
    assert np.isfinite(model.objective_)


def test_ridge_elimination_and_psd():
    X,_=synthetic(23,6)
    ridge=_RidgeResidual(X,2.5)
    rng=np.random.default_rng(1)
    v=rng.standard_normal((len(X),5))
    p=ridge.apply(v)
    np.testing.assert_allclose(p.mean(axis=0),0,atol=1e-12)
    assert np.sum(v*p)>-1e-8
    W=ridge.coef(v)
    residual=v-(X@W + (v.mean(axis=0)-X.mean(axis=0)@W))
    np.testing.assert_allclose(ridge.gamma*np.sum(residual**2)+np.sum(W**2),
                              ridge.gamma*np.sum(v*p),rtol=1e-10,atol=1e-10)


def test_unlabeled_scores_and_inductive_predictions_distinct():
    X,y=synthetic()
    y[[1,2,3,17,18,19]]=-1
    m=FlexibleManifoldEmbedding(mu=.2,gamma=.3).fit(X,y)
    assert m.transform(X).shape==m.scores_.shape
    assert np.linalg.norm(m.scores_-m.transform(X)) > 1e-4
    assert m.predict(X).shape == (len(X),)


def test_unsupervised_centering_orthogonality_eigensystem():
    X,_=synthetic(n=34,d=6)
    m=UnsupervisedFME(n_components=3,mu=2,gamma=.6).fit(X)
    F=m.embedding_
    np.testing.assert_allclose(F.mean(axis=0),0,atol=1e-10)
    np.testing.assert_allclose(F.T@F,np.eye(3),atol=1e-10)
    Q=m.laplacian_.toarray()
    ridge=_RidgeResidual(X,.6)
    for j in range(F.shape[1]):
        af=Q@F[:,j]+2*.6*ridge.apply(F[:,j:j+1]).ravel()
        np.testing.assert_allclose(af,m.eigenvalues_[j]*F[:,j],atol=1e-8)
    assert m.transform(X[:2]).shape==(2,3)


def test_unsup_primal_dual_equivalence():
    rng=np.random.default_rng(23)
    X1=rng.standard_normal((19,5))
    X2=rng.standard_normal((19,37))
    for X in (X1,X2):
        ridge=_RidgeResidual(X,1.7)
        target=rng.standard_normal((len(X),3))
        W=ridge.coef(target)
        grad=W+1.7*X.T@(X@W+(target.mean(axis=0)-X.mean(axis=0)@W)-target)
        np.testing.assert_allclose(grad,0,atol=1e-9)


def test_baselines_finite():
    X,y=synthetic(n=40,d=8)
    L,_,_=knn_laplacian(X,5)
    sparse_y=np.full_like(y,-1)
    sparse_y[[0,2,22,25]]=y[[0,2,22,25]]
    assert LinearLapRLS(ridge=.3).fit(X,sparse_y,L).transform(X[:2]).shape == (2,2)
    for variant in ("lgc","gfhf"):
        pred,scores=graph_propagation(sparse_y,L,variant)
        assert np.isfinite(scores).all()
        assert len(pred)==len(y)
    for sr in (True,False):
        model=LocalityPreservingProjection(2,spectral_regression=sr).fit(X,L)
        assert np.isfinite(model.transform(X)).all()


def test_split_and_preprocessing_no_test_leakage():
    X,y=synthetic(100,10)
    a,b,l=semi_split(y,99,2)
    assert not set(a).intersection(b)
    assert set(l).issubset(a)
    assert np.bincount(y[l]).tolist()==[2,2]
    train,test=unsupervised_split(y,99,5)
    assert np.bincount(y[train]).tolist()==[5,5]
    A,B,pca=pca_preprocess(X[a],X[b],.95)
    np.testing.assert_allclose(A,pca.transform(X[a]))
    np.testing.assert_allclose(B,pca.transform(X[b]))
    X[b] += 10000
    np.testing.assert_allclose(A,pca.transform(X[a]))


def test_iterative_semi_matches_dense():
    X,y=synthetic(n=42,d=7,seed=21)
    for i in [2,3,4,5,24,25,26,27]:
        y[i]=-1
    exact=FlexibleManifoldEmbedding(mu=.6,gamma=.2,dense_threshold=100).fit(X,y)
    cg_model=FlexibleManifoldEmbedding(mu=.6,gamma=.2,dense_threshold=10,tol=1e-11).fit(X,y)
    np.testing.assert_allclose(cg_model.scores_,exact.scores_,atol=2e-7,rtol=1e-7)
    np.testing.assert_allclose(cg_model.coef_,exact.coef_,atol=2e-7,rtol=1e-7)


def test_iterative_unsup_matches_dense_eigenspace():
    X,_=synthetic(n=42,d=7,seed=21)
    m1=UnsupervisedFME(n_components=3,mu=.3,gamma=2.0,dense_threshold=100).fit(X)
    m2=UnsupervisedFME(n_components=3,mu=.3,gamma=2.0,dense_threshold=10).fit(X)
    np.testing.assert_allclose(m2.eigenvalues_,m1.eigenvalues_,atol=1e-7,rtol=1e-7)
    # Eigenvectors are identifiable only up to sign/orthogonal rotation.
    np.testing.assert_allclose(m2.embedding_@m2.embedding_.T,
                              m1.embedding_@m1.embedding_.T,atol=1e-5,rtol=1e-5)

# Flexible Manifold Embedding — Reproduction

A NumPy/SciPy implementation of the semi-supervised **FME** and unsupervised **FME/U** algorithms of Nie, Xu, Tsang, and Zhang, *Flexible Manifold Embedding: A Framework for Semi-Supervised and Unsupervised Dimension Reduction*, IEEE Transactions on Image Processing **19**(7), 1921–1932 (2010). [DOI](https://doi.org/10.1109/TIP.2010.2044958).

The repository distinguishes **numerical verification**, **small development experiments**, and **paper-benchmark reproduction**. No manuscript benchmark accuracy is claimed until the corresponding original dataset, preprocessing, graph construction, and parameter protocol have been verified.

## Structure

```text
fme/
  model.py           FME and FME/U solvers; out-of-sample mappings
  graph.py           symmetric nearest-neighbor Gaussian graph
  baselines.py       LGC, GFHF, LapRLS/L, LPP, LPP-SR
  data.py            benchmark data format, split logic, train-only PCA
run_experiments.py   split evaluation, parameter sweeps, resume, summaries
scripts/
  prepare_images.py  class-folder image conversion
  prepare_20news.py  optional approximate 20NEWS preprocessing
requirements.txt
tests/               numerical and protocol regression tests
outputs/             generated at runtime; not tracked
```

## Installation

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

All algorithms work on CPU; a GPU is not required. Full image/text benchmarks require obtaining the original data and can be computationally demanding.

## Mathematical model

Let `X` have shape `(n_samples, n_features)`; `F` is the vector of train-graph scores/coordinates, `Y` is the class indicator matrix for labeled observations, `U` is the diagonal label indicator, `L` is the graph Laplacian, and `h(X) = XW + 1 b^T`.

For semi-supervised learning, this implementation minimizes the paper's joint objective:

\[
\operatorname{tr}[(F-Y)^T U(F-Y)] + \operatorname{tr}(F^T L F)
+ \mu\left(\|W\|_F^2+\gamma\|F-XW-\mathbf{1}b^T\|_F^2\right).
\]

The **paper's** `mu` multiplies both regularizers; its `gamma` controls the relative residual penalty. Eliminating the ridge regression variables gives the centered residual operator

\[
H=I-\tfrac{11^T}{n},\quad
P=H-\gamma HX (I+\gamma X^THX)^{-1} X^T H,\quad
(U+L+\mu\gamma P)F=UY.
\]

`FlexibleManifoldEmbedding.scores_` contains the optimized **transductive** `F`. The inductive embedding is **not** `F`: use `transform(X)` to obtain `h(X)` from learned `coef_` and `intercept_`.

For FME/U, the method minimizes `tr[F^T(L+mu*gamma*P)F]` with centered, unit-orthogonal embedding columns and then regresses the resulting coordinates onto `X`. The lowest *nonconstant* eigenvectors solve the constrained problem. Dense direct and iterative paths are implemented without forming `P` explicitly.

## Paper experiment protocols

| Experiment | Datasets | Split | Labeled training points | Report |
|---|---|---|---|---|
| Semi-supervised FME | UMIST, CMU PIE (C27), Yale-B, COIL-20, 20NEWS (4 rec classes) | 50% train / 50% test | 1, 2, 3 per class for images; 10, 20, 30 for text | Graph-unlabeled accuracy and unseen-test accuracy separately |
| Unsupervised FME/U | UMIST, CMU PIE, Yale-B | 5 images per class train; remainder test | Labels used only for 1-NN evaluation | Unseen-test accuracy vs dimension |

The paper applies PCA (95% retained energy) before the graph/dimension-reduction methods. This implementation **fits PCA only on the training split**, avoiding test-information leakage. For the semi-supervised methods, the nearest-neighbor classifier in the learned embedding is trained on labeled training points; graph-only LGC/GFHF have **no** out-of-sample test score. The code also supports the paper's best-of-grid reporting as `--selection paper_oracle`, which explicitly selects the best result *using outer evaluation labels*. This practice is optimistic and must not be interpreted as a proper validation protocol. The default `fixed` selection avoids this test-label leakage.

**Parameter and graph caveats.** The paper states that Gaussian affinities are computed on a nearest-neighbor graph with its bandwidth selected as in an earlier reference. The precise implementation of that bandwidth and the parameter grid are not fully recoverable from the public text available to this implementation. Accordingly, the graph neighbor count (`--neighbors`, default 5), Gaussian bandwidth (`--bandwidth`, default median nonzero squared neighbor distance), and parameter lists are explicit configuration choices **rather than claimed exact original settings**. They must be audited before comparing numerical accuracy with Table I or Table II. The source image releases and normalization conventions likewise need to be verified dataset by dataset.

### Data format

Place arrays at `data/<dataset>.npz` using keys `X` (finite numeric array of shape `n × d`) and `y` (integer or string class labels, length `n`). Names used by the script: `umist`, `cmu_pie`, `yale_b`, `coil20`, `20news`. All label identifiers are remapped internally to `0..C-1`.

For class-folder image archives, first prepare `data/coil20.npz`, for example:

```bash
python scripts/prepare_images.py /path/to/coil20-class-folders data/coil20.npz --width 32 --height 32
```

UMIST uses 28×23 grayscale images in the paper; CMU PIE C27, Yale-B, and COIL-20 use 32×32 grayscale. Folder/label conventions must be checked against the chosen original release. Face datasets are not downloaded or redistributed here due to source availability and licensing. `scripts/prepare_20news.py` downloads a scikit-learn version and therefore produces an **approximation**, not the exact paper's 20-news-18828 version with its original 8,014-dimensional TF-IDF preprocessing. Do not label that output an exact reproduction.

### Running

Quick CPU verification on scikit-learn digits (not one of the paper datasets):

```bash
python run_experiments.py --dataset digits --max-samples 180 --mode semi --splits 1 --per-class 1 --mu 0.1 --gamma 1 --output outputs/digits-check
python run_experiments.py --dataset digits --max-samples 160 --mode unsup --splits 1 --dims 2,4,8 --mu 0.1 --gamma 1 --output outputs/digits-unsup-check
```

After supplying the original UMIST arrays:

```bash
python run_experiments.py --dataset umist --mode semi --splits 20 --per-class 1,2,3 --mu 1e-6,1e-3,1,1e3,1e6 --gamma 1e-6,1e-3,1,1e3,1e6 --selection paper_oracle --output outputs/umist-semi-paper
python run_experiments.py --dataset umist --mode unsup --splits 20 --dims 2,4,8,16 --mu 1e-6,1e-3,1,1e3,1e6 --gamma 1e-6,1e-3,1,1e3,1e6 --selection paper_oracle --output outputs/umist-unsup-paper
```

These commands are a **parameter sweep template**, not a verified transcription of the original grid; confirm the grid, final embedding dimension, and bandwidth before declaring exact numerical replication. Use `--threads 1` to avoid oversubscribing BLAS when launching multiple processes. Different workers can use distinct `--split-start` ranges and `--output` directories. Use `--resume` with identical arguments to resume one interrupted directory; a configuration/source hash guards against accidental mixing.

Run outputs (`outputs/<run>/`) contain `runs.csv`, `summary.csv`, `summary.md`, and `progress.json`. Results are created from actual runs only; none are included as published paper numbers. Use independent output directories for distinct configurations.

## Comparison scope and verification

Implemented baselines: graph-only **LGC**, **GFHF**; hard-linear **LapRLS/L**; **PCA**, **LPP**, and **LPP-SR**. The original paper also compares **MFA**, **TCA**, and **SDA** in Table I; these are **not yet implemented** here. LPP and spectral-regression defaults have explicit regularization choices that may not match the authors' original baselines. No claims of reproduction of all methods or exact published accuracy are made.

The CPU tests verify graph symmetry and PSD properties, closed-form FME first-order conditions, ridge elimination in both primal/dual regimes, centered FME/U eigenvectors, baseline finiteness, and split/preprocessing boundaries. Final manuscript-benchmark replication requires the precise original datasets, paper-specific graph bandwidth, parameter grids, and full repeated experiments.

### Plotting and parallel split aggregation

The result plotter uses measured `summary.csv` values and displays sample-standard-deviation error bars, when multiple splits exist:

```bash
python scripts/plot_results.py outputs/umist-unsup-paper/summary.csv --phase test --output outputs/umist-unsup-paper/accuracy_vs_dimension.png
```

Parallel workers should use non-overlapping `--split-start` ranges and distinct output folders. After all workers finish, combine their records (configuration and overlap are verified):

```bash
python scripts/merge_runs.py outputs/worker0 outputs/worker1 --output outputs/merged
```

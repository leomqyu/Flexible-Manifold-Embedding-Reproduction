"""Prepare an approximate 20NEWS dataset from scikit-learn's mirror.

This is NOT identical to the 20-news-18828 release in Nie et al. (2010);
results must not be presented as faithful Table I reproduction.
"""

import argparse
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

CATEGORIES=("rec.autos","rec.motorcycles","rec.sport.baseball","rec.sport.hockey")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("data/20news.npz"))
    args=parser.parse_args()
    bunch=fetch_20newsgroups(subset="all",categories=list(CATEGORIES),
                             remove=(),shuffle=False)
    vectorizer=TfidfVectorizer(max_features=8014,sublinear_tf=False,norm="l2")
    X=vectorizer.fit_transform(bunch.data)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,X=X.toarray().astype(np.float32),
                        y=bunch.target.astype(np.int64))
    print("Created an approximation, not the original 20-news-18828 protocol:",args.output)
    print("Shape:",X.shape)

if __name__=="__main__":main()

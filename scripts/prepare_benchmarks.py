"""Convert the locally obtained benchmark files into data/<dataset>.npz.

Expected raw files (not redistributed with this repository):
  umist_cropped.mat  UMIST Cropped, 20 subjects, 575 images, 112x92.
  CMUPIEData.mat     CMU PIE C27, 68 subjects, 42 images each, 32x32.
  YaleB_32x32.mat    Extended Yale B, 38 subjects, 2414 images, 32x32.
  COIL20.mat         COIL-20, 20 objects, 1440 images, 32x32.
  20news-18828.tar.gz  The original 20 Newsgroups 18828 archive.

The UMIST images are resized to the paper's 28x23 pixels. The text features
are a scikit-learn TF-IDF approximation with 8014 features; see the README for
the exact provenance and for the differences from the paper's preprocessing.
"""

import argparse
import tarfile
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

CATEGORIES = ("rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey")


def save(path, X, y):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=np.asarray(X, dtype=np.float32),
                        y=np.asarray(y, dtype=np.int64))
    print(f"{path.name}: X={X.shape}, classes={len(np.unique(y))}, "
          f"counts={np.bincount(np.asarray(y, dtype=int)).min()}.."
          f"{np.bincount(np.asarray(y, dtype=int)).max()}")


def umist(raw, out):
    data = sio.loadmat(raw / "umist_cropped.mat", squeeze_me=True)
    X, y = [], []
    for label, images in enumerate(data["facedat"]):
        images = np.asarray(images)
        if images.ndim == 2:
            images = images[..., None]
        for index in range(images.shape[2]):
            image = Image.fromarray(images[:, :, index].astype(np.uint8), mode="L")
            image = image.resize((23, 28), Image.Resampling.BILINEAR)
            X.append(np.asarray(image, dtype=np.float32).ravel())
            y.append(label)
    save(out / "umist.npz", np.vstack(X), y)


def pie(raw, out):
    data = sio.loadmat(raw / "CMUPIEData.mat", squeeze_me=True,
                       struct_as_record=False)["CMUPIEData"]
    X = np.stack([np.asarray(item.pixels).ravel() for item in data])
    y = np.array([int(item.label) - 1 for item in data])
    save(out / "cmu_pie.npz", X, y)


def yale(raw, out):
    data = sio.loadmat(raw / "YaleB_32x32.mat")
    X = np.asarray(data["fea"])
    y = np.asarray(data["gnd"]).ravel().astype(int) - 1
    save(out / "yale_b.npz", X, y)


def coil(raw, out):
    data = sio.loadmat(raw / "COIL20.mat")
    X = np.asarray(data["fea"])
    y = np.asarray(data["gnd"]).ravel().astype(int) - 1
    save(out / "coil20.npz", X, y)


def news(archive, out):
    texts = {category: [] for category in CATEGORIES}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) == 3 and parts[1] in texts:
                stream = handle.extractfile(member)
                if stream is not None:
                    texts[parts[1]].append(stream.read().decode("latin-1", errors="ignore"))
    documents, labels = [], []
    for label, category in enumerate(CATEGORIES):
        documents.extend(texts[category])
        labels.extend([label] * len(texts[category]))
    vectorizer = TfidfVectorizer(max_features=8014, sublinear_tf=False,
                                 norm="l2", stop_words=None)
    X = vectorizer.fit_transform(documents).toarray()
    save(out / "20news.npz", X, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--news-archive", type=Path,
                        default=Path("20news-18828.tar.gz"))
    parser.add_argument("--skip-news", action="store_true")
    args = parser.parse_args()
    umist(args.raw_dir, args.output_dir)
    pie(args.raw_dir, args.output_dir)
    yale(args.raw_dir, args.output_dir)
    coil(args.raw_dir, args.output_dir)
    if not args.skip_news:
        news(args.news_archive, args.output_dir)


if __name__ == "__main__":
    main()

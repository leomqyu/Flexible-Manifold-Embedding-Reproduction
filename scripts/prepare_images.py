"""Convert class-folder images into a numpy dataset without changing labels.

Input tree: root/class_name/image.ext. Each image becomes one flattened
8-bit grayscale vector, resized only when --width/--height are provided.
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

EXTS = {".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def convert(root, output, width=None, height=None):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    if (width is None) != (height is None):
        raise ValueError("width and height must be provided together")
    X, y, names = [], [], []
    for i, folder in enumerate(sorted(p for p in root.iterdir() if p.is_dir())):
        images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)
        if not images:
            continue
        for file in images:
            with Image.open(file) as image:
                image = image.convert("L")
                if width is not None:
                    image = image.resize((width,height),Image.Resampling.BILINEAR)
                X.append(np.asarray(image,dtype=np.float64).ravel())
            y.append(i)
            names.append(str(file.relative_to(root)))
    if not X:
        raise ValueError("No images in class directories")
    dim = len(X[0])
    if any(len(v) != dim for v in X):
        raise ValueError("Inconsistent image shapes; supply --width and --height")
    out = Path(output)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,X=np.vstack(X),y=np.asarray(y),filenames=np.asarray(names))
    print(f"Saved {len(X)} images with {dim} features: {out}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--width",type=int)
    p.add_argument("--height",type=int)
    a=p.parse_args()
    convert(a.root,a.output,a.width,a.height)

if __name__=="__main__":main()

"""Read the stock TH12 0.02a display-FPS glyphs in a 640x480 color capture."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_display_fps(path, templates, labels):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape != (480, 640):
        raise ValueError(f"Expected a native 640x480 client image at {path}")
    band = gray[469:479, 586:620].astype(np.float32)
    contrast = band - cv2.GaussianBlur(band, (5, 5), 1.2)
    glyphs = np.stack([contrast[:, x:x + 8].reshape(-1) for x in (2, 9, 22)])
    glyphs -= glyphs.mean(axis=1, keepdims=True)
    glyphs /= np.maximum(np.linalg.norm(glyphs, axis=1, keepdims=True), 0.001)
    distances = np.square(glyphs[:, None] - templates[None]).sum(axis=2)
    per_digit = np.stack([distances[:, labels == digit].min(axis=1) for digit in range(10)], axis=1)
    order = np.argsort(per_digit, axis=1)
    margins = per_digit[np.arange(3), order[:, 1]] - per_digit[np.arange(3), order[:, 0]]
    fps = round(float(order[:, 0] @ [10, 1, 0.1]), 1)
    return dict(display_fps=fps, min_digit_margin=float(margins.min()),
                max_digit_distance=float(per_digit.min(axis=1).max()),
                readable=bool(margins.min() >= 0.10 and per_digit.min(axis=1).max() < 0.24 and 15 <= fps <= 65))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("images", nargs="+", type=Path)
    args.add_argument("--templates", type=Path, default=Path(__file__).with_name("hud-fps-glyphs.npz"))
    parsed = args.parse_args()
    with np.load(parsed.templates, allow_pickle=False) as data:
        for image in parsed.images:
            print(json.dumps(dict(image=str(image), **read_display_fps(image, data["templates"], data["digits"]))))

"""Compare original stock UI photographs with calibrated observations of trial 0.01a."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

REGIONS = {
    "title": ((420,330,590,450), .68),
    "normal": ((260,185,380,246), .81),
    "hard": ((253,186,387,242), .78),
    "lunatic": ((253,186,387,242), .78),
    "character": ((10,100,220,350), .82),
    "wolf": ((330,62,491,145), .75),
    "game_over": ((43,187,118,222), .72),
    "score_ranking": ((155,61,295,79), .78),
}

def comparison(path, key):
    region, threshold = REGIONS[key]
    with Image.open(path) as image:
        observed = np.array(image.convert("L").crop(region))
    with np.load(Path(__file__).with_name("original-ui-references.npz")) as refs:
        original = refs[key]
    if key in ("game_over", "score_ranking"):
        cutoff = 225 if key == "game_over" else 220
        shape = original > cutoff
        score = np.count_nonzero((observed > cutoff) & shape) / np.count_nonzero(shape)
    else:
        score = np.corrcoef(observed.ravel(), original.ravel())[0,1]
    return float(score), bool(np.isfinite(score) and score >= threshold)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path)
    p.add_argument("key", choices=REGIONS)
    args = p.parse_args()
    score, matches = comparison(args.image, args.key)
    print(score)
    raise SystemExit(0 if matches else 1)

"""Read the original TH13 Trial encounter countdown and native score-ranking screen.

Inputs are the 640x480 client region. The policy never sees these template outputs.
Prepare the detector from the two previous credited calibration lanes and the observed
ranking reference, then call --image with screenshots from independent credits.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


CALIBRATION = [
    ('progress-01-a', 36, [29,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,14,13,12,11,10,9,9]),
    ('progress-02-a', 37, [29,28,27,26,25,24,23,23,21,21,19,18,18,17,16,15,14,13,13,11,11,10,9,8,8,6,5,5,2,1,0]),
]


def digits(image):
    pixels = np.asarray(image)
    region = pixels[15:31].max(axis=2).astype(np.float64)
    patches = []
    for x in (201, 213):
        patch = region[:, x:x+13]
        centered = patch - patch.mean()
        patches.append(centered.ravel() / max(np.linalg.norm(centered), 1e-8))
    return patches


def ranking_mask(image):
    pixels = np.asarray(image)
    header = pixels[61:78, 154:298].min(axis=2) > 155
    entry = pixels[318:344, 155:286].min(axis=2) > 130
    return np.r_[header.ravel(), entry.ravel()]


def build(reference_root, output):
    numbers, vectors = [], []
    for folder, first, readings in CALIBRATION:
        for index, value in enumerate(readings, first):
            frame = Image.open(reference_root / folder / f'frame-{index:03}.jpg').convert('RGB')
            numbers.extend(divmod(value, 10))
            vectors.extend(digits(frame))
    rank = Image.open(reference_root / 'progress-02-b/frame-070.jpg').convert('RGB')
    np.savez(output, digits=numbers, vectors=np.stack(vectors), ranking=ranking_mask(rank))


def read(image, templates):
    if image.size != (640, 480):
        raise ValueError('Expected a 640x480 client crop')
    correlation = np.stack(digits(image)) @ templates['vectors'].T
    best = correlation.argmax(axis=1)
    score = correlation.max(axis=1).min()
    tens, ones = templates['digits'][best]
    integer = int(10*tens + ones)
    mask = ranking_mask(image)
    reference = templates['ranking']
    overlap = float(2*(mask & reference).sum() / max(1, mask.sum() + reference.sum()))
    return dict(countdown_s=integer if integer <= 30 and score > .66 else None,
                digit_confidence=round(float(score), 3), ranking=overlap > .79,
                ranking_confidence=round(overlap, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--templates', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--build-from', type=Path)
    group.add_argument('--image', type=Path)
    args = parser.parse_args()
    if args.build_from:
        build(args.build_from, args.templates)
    else:
        image = Image.open(args.image).convert('RGB')
        with np.load(args.templates) as templates:
            print(json.dumps(read(image, templates)))


if __name__ == '__main__':
    main()

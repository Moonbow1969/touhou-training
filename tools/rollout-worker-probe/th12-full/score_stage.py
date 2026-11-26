"""Verify a TH12 native highlighted score-row stage against an audited same-font reference."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def yellow_text(path):
    with Image.open(path) as image:
        if image.size != (640, 480):
            raise ValueError('Expected the native 640x480 game window.')
        red, green, blue = np.asarray(image.convert('RGB'), dtype=np.int16).transpose(2, 0, 1)
    return (np.minimum(red, green) > 165) & (blue < 120) & (abs(red - green) < 80)


def dice(left, right):
    return 2 * int((left & right).sum()) / max(1, int(left.sum() + right.sum()))


def read_rank_stage(screen, reference, reference_y, reference_stage):
    observed, known = yellow_text(screen), yellow_text(reference)
    row = slice(reference_y - 6, reference_y + 7)
    word = known[row, 336:387]
    number = known[row, 391:405]
    if word.sum() < 10 or number.sum() < 3:
        raise ValueError('The reference needs the center of the visibly highlighted Stage row.')
    word_score, y = max((dice(word, observed[y-6:y+7, 336:387]), y) for y in range(90, 275))
    number_score = dice(number, observed[y-6:y+7, 391:405])
    return {'stage': reference_stage if min(word_score, number_score) >= .98 else None,
            'row_y': y if word_score >= .98 else None,
            'word_dice': word_score, 'digit_dice': number_score}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('screen', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--reference-y', type=int, required=True)
    parser.add_argument('--reference-stage', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(read_rank_stage(args.screen, args.reference, args.reference_y, args.reference_stage)))


if __name__ == '__main__':
    main()

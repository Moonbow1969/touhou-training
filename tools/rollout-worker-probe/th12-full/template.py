"""Build TH12's menu and resource measurement patches from a local game installation."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


RECTS = {
    'ranking-full': (121, 44, 139, 22),
    'pause': (12, 205, 72, 29),
    'gameover': (12, 205, 72, 29),
    'retry': (34, 345, 137, 25),
    'confirm': (203, 249, 150, 36),
    'hud': (460, 5, 72, 19),
    'title': (32, 374, 99, 20),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=RECTS)
    parser.add_argument('screenshot', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with Image.open(args.screenshot) as image:
        origin = {(640, 480): (32, 16), (1024, 768): (35, 38)}.get(image.size)
        if origin is None:
            parser.error('Expected a native 640x480 window or a calibrated 1024x768 desktop.')
        x, y, w, h = RECTS[args.kind]
        x += origin[0]
        y += origin[1]
        rgb = np.asarray(image.convert('RGB').crop((x, y, x + w, y + h)), dtype=np.uint32)
    gray = (77 * rgb[:, :, 0] + 150 * rgb[:, :, 1] + 29 * rgb[:, :, 2]) >> 8
    args.output.mkdir(parents=True, exist_ok=True)
    gray.astype(np.uint8).tofile(args.output / (args.kind + '.gray'))


if __name__ == '__main__':
    main()

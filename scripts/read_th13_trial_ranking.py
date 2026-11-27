"""Read the highlighted original TH13 trial Stage 1/2 ranking using calibrated glyphs."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


TRIAL_SHA256 = "09818929cceea76ff4c65630e649bbccdb85d92b2e28bb4dfe6526749e6f4fb6"
DEFAULT_CALIBRATION = Path(__file__).resolve().parents[1] / "tools/full_action/th13-trial-ranking-glyphs.json"


def glyphs(image, row):
    x, y = 3, 22 + 18 * row
    pixels = np.asarray(image.crop((x + 332, y + 95, x + 405, y + 110)).convert("RGB"), dtype=np.int16)
    red, green, blue = pixels.transpose(2, 0, 1)
    mask = (red > 100) & (green > 95) & (red + green - 2 * blue > 80)
    return mask[:, :56], mask[:, 59:]


def load_templates(path=DEFAULT_CALIBRATION):
    data = json.loads(path.read_bytes())
    if data["game_sha256"] != TRIAL_SHA256:
        raise ValueError("Ranking glyphs belong to a different game build")
    templates = []
    for glyph in data["templates"]:
        shape = tuple(glyph["shape"])
        packed = np.frombuffer(base64.b64decode(glyph["bits"]), dtype=np.uint8)
        templates.append(np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape).astype(bool))
    return templates


def f1(left, right):
    return float(2 * (left & right).sum() / max(1, left.sum() + right.sum()))


def classify(image, templates):
    if image.size != (1024, 768):
        return dict(readout=None, reason="unexpected dimensions")
    word1, digit1, word2, digit2 = templates
    matches = []
    for i in range(10):
        word, digit = glyphs(image, i)
        matches.append((max(f1(word, word1), f1(word, word2)), f1(digit, digit1), f1(digit, digit2), i))
    matches.sort(reverse=True)
    word, s1, s2, i = matches[0]
    margin = word - matches[1][0]
    accepted = word >= .85 and margin >= .35 and max(s1, s2) >= .85 and abs(s1 - s2) >= .20
    return dict(readout=(1 if s1 > s2 else 2) if accepted else None,
                ranking_row=i+1, word_f1=word, digit_1_f1=s1, digit_2_f1=s2, rank_row_margin=margin)


def scan(directory, calibration=DEFAULT_CALIBRATION):
    templates = load_templates(calibration)
    rows = []
    for receipt in sorted(directory.glob("*/receipt.json")):
        log = json.loads(receipt.read_bytes())
        if log.get("game_sha256") != TRIAL_SHA256:
            continue
        photo = receipt.parent / "ranking.png"
        native = log.get("stage2", log.get("stage2_reached"))
        if photo.is_file():
            with Image.open(photo) as image:
                result = classify(image, templates)
            digest = hashlib.sha256(photo.read_bytes()).hexdigest()
        else:
            result, digest = dict(readout=None, reason="ranking.png missing"), None
        label = result["readout"]
        rows.append(dict(path=photo.relative_to(directory).as_posix(), sha256=digest, native_stage2=native,
                         matches_native=(label == 2) == native if label is not None and native is not None else None,
                         **result))
    return dict(game_sha256=TRIAL_SHA256, calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
                credits=len(rows), rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    args = parser.parse_args()
    args.out.write_text(json.dumps(scan(args.native_dir, args.calibration), indent=2) + "\n", encoding="utf-8")

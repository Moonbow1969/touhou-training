"""Independently bind optional full-resolution motion patches to original replay camera ticks."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from replays.audit_th08_trial002_camera import digest, verify


PATCHES = np.dtype([("tick", "<u4"), ("rgb", "u1", (2, 64, 64, 3))])


def bounds(x, length):
    low = max(0, ((x + 3) // 4) * 4)
    high = min(length, ((x + 64) // 4) * 4)
    return low, high


def aligned_crop(data, left, top):
    x0, x1 = bounds(left, 384)
    y0, y1 = bounds(top, 448)
    patch = Image.fromarray(data[y0-top:y1-top, x0-left:x1-left]).convert("L")
    small = patch.resize(((x1-x0)//4, (y1-y0)//4), Image.Resampling.BOX)
    return np.asarray(small), np.s_[y0//4:y1//4, x0//4:x1//4]


def full_reference(path, left, top):
    pixels = np.fromfile(path, dtype=np.uint8).reshape(480, 640, 4)[16:464, 32:416, :3][:, :, ::-1]
    return np.asarray(Image.fromarray(pixels).crop((left, top, left+64, top+64)))


def validate(raw, replay, stage):
    frames, words, valid, positions, evidence = verify(raw, replay, stage)
    path = raw / "motion-patches.bin"
    patches = np.fromfile(path, dtype=PATCHES)
    if not 0 < len(patches) <= 4096 or patches.nbytes != path.stat().st_size:
        raise ValueError("Invalid original motion record size")
    ticks = frames["state"][:, 0].astype(np.int64)
    target_ticks = patches["tick"].astype(np.int64)
    idx = np.searchsorted(ticks, target_ticks)
    if (np.any(np.diff(target_ticks) <= 0) or np.any(idx < 1) or np.any(idx >= len(ticks)) or
            not np.array_equal(ticks[idx], target_ticks) or not np.array_equal(ticks[idx-1], target_ticks-4)):
        raise ValueError("Original patch pair is not at its claimed four-tick camera boundary")
    refs = {int(filename.stem.split("-")[1]): filename for filename in raw.glob("rgb-*.bgra")}
    checked = 0
    for placement, pair in zip(idx, patches["rgb"]):
        x, y = (positions["xy"][placement] + .5).astype(int)
        if not 0 <= x <= 384 or not 0 <= y <= 448:
            raise ValueError("Native patch coordinates lie outside the playfield")
        left, top = x-32, y-32
        for offset, rgb in enumerate(pair):
            frame = placement - 1 + offset
            reconstructed, roi = aligned_crop(rgb, left, top)
            np.testing.assert_array_equal(reconstructed, frames["obs"][frame][roi])
            tick = ticks[frame]
            if tick in refs:
                np.testing.assert_array_equal(rgb, full_reference(refs[tick], left, top))
                checked += 1
    if not checked:
        raise ValueError("No overlapping raw RGB witnesses")
    history = idx[:, None]+np.arange(-4, 1)
    four = idx >= 4
    four[four] &= (np.diff(ticks[history[four]], axis=1) == 4).all(1)
    usable = valid[idx] & four
    activity = np.abs(patches["rgb"][:,1].astype(np.int16)-patches["rgb"][:,0]).mean(axis=(1,2,3))
    return dict(stage=stage, replay_sha256=digest(replay), patch_sha256=digest(path),
                original_camera_sha256=evidence["raw_camera_sha256"], positions_sha256=evidence["positions_sha256"],
                matched_rows=len(patches), matched_older_and_current_luma_tiles=2*len(patches),
                independent_rgb_witnesses=checked, usable_causal_four_state_rows=int(usable.sum()),
                first_target_tick=int(target_ticks[0]), last_target_tick=int(target_ticks[-1]),
                mean_color_difference=float(activity.mean()), p75_color_difference=float(np.quantile(activity,.75)),
                original_convention="Two same-coordinate native RGB crops, exactly four original ticks apart; not projectile annotations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.raw, args.replay, args.stage)
    with args.out.open("x", encoding="utf-8") as writer:
        writer.write(json.dumps(result, indent=2)+"\n")

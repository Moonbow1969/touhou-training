"""Capture a selected native TH10 or TH12 retail replay.

Run this module from the repository root while the intended original Stage
selector is in the foreground. The manifest entry supplies the source SHA,
player group and split. Every accepted frame has matching float and fixed
player coordinates as well as matching original/decoded input.
"""

import argparse
import json
from pathlib import Path
from time import sleep

import keyboard
import numpy as np
import win32gui
import win32process

from .collect_replay_pixels import Sample, capture_frames, digest, pair
from .inspect_rpy import action_streams
from .retail_gameplay_reader import LAYOUTS, StockReplayReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", type=int, choices=sorted(LAYOUTS), required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = next(
        row
        for row in json.loads(args.manifest.read_bytes())
        if row["name"] == args.replay.name
    )
    if digest(args.replay) != source["sha256"] or source.get("split_conflict"):
        raise ValueError("Verify source identity and split before collecting.")
    role = {"train": "train", "held": "held", "validation": "held", "test": "held"}[
        source["split"]
    ]
    path = args.out / role / f"{args.replay.stem}-s{args.stage}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix(".json")
    if receipt.is_file():
        previous = json.loads(receipt.read_bytes())
        if previous["replay_sha256"] == source["sha256"] and previous[
            "archive_sha256"
        ] == digest(path):
            print(f"Already qualified {path}")
            return
        raise ValueError("Existing capture identity differs.")
    hwnd = win32gui.GetForegroundWindow()
    if win32process.GetWindowThreadProcessId(hwnd)[
        1
    ] != args.pid or win32gui.GetClientRect(hwnd) != (0, 0, 640, 480):
        raise RuntimeError(
            "Select the original 640×480 replay-stage window before starting."
        )
    words = next(
        np.frombuffer(data, "<u2").reshape(-1, 3)[:, 0]
        for stage, data in action_streams(args.replay, args.title)
        if stage == args.stage
    )
    reader = StockReplayReader(args.title, args.pid)
    layout = LAYOUTS[args.title]

    def observe():
        # A scene change can invalidate a pointer between the two reads.
        try:
            state = reader.snapshot()
        except OSError:
            return None
        if state is None or state.tick < 0 or state.tick > 100_000:
            return None
        x, y = state.xy
        valid = (
            -193 <= x <= 193
            and -32 <= y <= 468
            and max(abs(a - b) for a, b in zip(state.xy, state.fixed_xy)) < 2.1
        )
        return state.stage, Sample(
            state.tick, state.word, state.xy, valid, -1, state.fixed_xy
        )

    try:
        keyboard.press("z")
        try:
            sleep(0.08)
        finally:
            keyboard.release("z")
        samples, images, racing = capture_frames(hwnd, args.stage, words, observe)
    finally:
        reader.close()
    data, metrics = pair(
        samples,
        images,
        words,
        control_bits=0xF3 | layout.focus_bit,
        focus_bit=layout.focus_bit,
    )
    with path.with_suffix(".npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, **data)
    path.with_suffix(".npz.tmp").replace(path)
    report = {
        **source,
        **metrics,
        "title": f"TH{args.title}",
        "stage": args.stage,
        "split": role,
        "original_split": source["split"],
        "replay_sha256": source["sha256"],
        "archive_sha256": digest(path),
        "original_software_exe_sha256": layout.executable_sha256,
        "game_version": "1.00a retail Steam"
        if args.title == 10
        else "1.00b retail Steam",
        "focus_bit": layout.focus_bit,
        "discarded_racing_reads": racing,
        "rank": None,
        "control_filter": "Stage and player present; native float/fixed coordinate agreement under 2.1 native pixels. Dialogue and respawn are not explicitly detected.",
        "action_convention": "first future tick for movement, focus and fire; any Bomb in four future ticks",
    }
    receipt.with_suffix(".json.tmp").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    receipt.with_suffix(".json.tmp").replace(receipt)
    print(f"Qualified {path} with {metrics['frames']} aligned decisions.")


if __name__ == "__main__":
    main()

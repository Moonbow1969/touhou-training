"""Sample a stock TH13 retail replay with a causal camera/action pair every four ticks.

Start from the original game's selected replay-stage menu at 640x480 windowed.
Example: python -m replays.collect_replay_pixels --pid PID --manifest corpus.json
         --replay path/to/replay.rpy --stage 1 --out human
"""

import argparse
from dataclasses import dataclass
from hashlib import file_digest
import json
from pathlib import Path
from struct import unpack
from time import monotonic, sleep

import numpy as np
from PIL import Image, ImageGrab
import psutil
import win32gui
import win32process

from environment.structured import StateReader
from .collect_stage import CONTROL_BITS, ReplayProcess, has_control
from .inspect_rpy import action_streams

TH13_RETAIL_SHA = "1a6e3eb3b65238a739d7250606e9cb233d16b505c3340a11c4584b583388930d"
PLAYFIELD = (32, 16, 416, 464)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class Sample:
    tick: int
    word: int
    xy: tuple
    controllable: bool
    difficulty: int


def original_window(pid):
    handles = []

    def visit(handle, _):
        if (
            win32gui.IsWindowVisible(handle)
            and win32process.GetWindowThreadProcessId(handle)[1] == pid
        ):
            if "Ten Desires. ver 1.00c" in win32gui.GetWindowText(handle):
                handles.append(handle)

    win32gui.EnumWindows(visit, None)
    if len(handles) != 1 or win32gui.GetClientRect(handles[0]) != (0, 0, 640, 480):
        raise RuntimeError("Expected one TH13 1.00c window with a 640×480 client.")
    return handles[0]


def read_state(game, reader):
    offsets = game.offsets
    if game.read_game_val("game_screen") != 7 or game.read_game_val("pause_state") != 2:
        return None
    thread = reader.read_uint(offsets.game_thread.game_thread_pointer, relative=True)
    player = reader.read_uint(offsets.player.player_pointer, relative=True)
    if not thread or not player:
        return None
    stage = reader.read_uint(offsets.environment.stage, relative=True)
    state = reader.read_uint(player + offsets.player.zPlayer_state)
    return stage, Sample(
        reader.read_uint(thread + offsets.game_thread.zGameThread_stage_timer),
        reader.read_uint(offsets.statics.input, relative=True),
        unpack(
            "<ff",
            game._read_game_memory(player + offsets.player.zPlayer_pos, 8, rel=False),
        ),
        has_control(reader, thread, state),
        reader.read_uint(offsets.environment.difficulty, relative=True),
    )


def capture(game, stage, words):
    hwnd = original_window(game.pid)
    reader = StateReader(game, game.offsets, 96)
    game.start_selected_replay()
    samples, images = [], []
    next_tick = 0
    deadline = monotonic() + 12
    racing = 0
    while monotonic() < deadline:
        previous = read_state(game, reader)
        if previous is None:
            sleep(0.004)
            continue
        current_stage, state = previous
        if current_stage != stage or state.tick > len(words) + 100:
            if samples:
                break
            sleep(0.005)
            continue
        if samples and state.tick < samples[-1].tick - 100:
            break
        if state.tick < next_tick:
            sleep(0.001)
            continue
        screen = ImageGrab.grab(window=hwnd)
        check = read_state(game, reader)
        if (
            check is None
            or check[0] != stage
            or (check[1].tick, check[1].word) != (state.tick, state.word)
        ):
            racing += 1
            continue
        if screen.size != (640, 480):
            raise RuntimeError("Game client changed dimensions during capture.")
        images.append(
            np.asarray(
                screen.crop(PLAYFIELD)
                .convert("L")
                .resize((96, 112), Image.Resampling.BOX)
            )
        )
        samples.append(state)
        next_tick = (state.tick // 4 + 1) * 4
        deadline = monotonic() + 12
        if len(samples) % 250 == 0:
            print(
                f"Stage {stage} tick {state.tick}, captured {len(samples)} stable images.",
                flush=True,
            )
        if state.tick >= len(words) - 8:
            break
    if len(samples) < 100:
        raise RuntimeError("Original stage did not produce a usable capture.")
    return samples, images, racing


def movement(words):
    right = ((words & 128) != 0).astype(np.int64)
    left = ((words & 64) != 0).astype(np.int64)
    down = ((words & 32) != 0).astype(np.int64)
    up = ((words & 16) != 0).astype(np.int64)
    return 4 + right - left + 3 * (down - up)


def pair(samples, images, words):
    ticks = np.asarray([sample.tick for sample in samples])
    native = np.asarray([sample.word for sample in samples], dtype=np.uint16)
    playable = np.asarray([sample.controllable for sample in samples], dtype=bool)
    scores = []
    for offset in range(-60, 61):
        index = ticks + offset
        available = playable & (index >= 8) & (index + 4 < len(words))
        replay = words[index[available]] & CONTROL_BITS
        current = native[available] & CONTROL_BITS
        moving = ((replay | current) & 0xF0) != 0
        if moving.sum() > 20:
            scores.append(
                (
                    float(np.mean(replay[moving] == current[moving])),
                    int(moving.sum()),
                    offset,
                )
            )
    scores.sort(reverse=True)
    if not scores or scores[0][0] < 0.97:
        raise RuntimeError(
            f"Original native movement/input identity failed: {scores[:5]}"
        )
    agreement, _, offset = scores[0]
    index = ticks + offset
    keep = playable & (index >= 8) & (index + 4 < len(words))
    keep[keep] &= (native[keep] & CONTROL_BITS) == (words[index[keep]] & CONTROL_BITS)
    index = index[keep]
    future = words[index[:, None] + np.arange(1, 5)]
    history = words[index[:, None] + np.arange(-7, 1)]
    first = future[:, 0]
    action = np.column_stack(
        (
            movement(first),
            (first & 8) != 0,
            (first & 1) != 0,
            np.any((future & 2) != 0, axis=1),
        )
    ).astype(np.int64)
    pixels = np.stack(images)[keep]
    coverage = float((ticks[-1] - ticks[0]) / len(words))
    if coverage < 0.9:
        raise RuntimeError(f"Incomplete stage span: {coverage:.1%}")
    # Process sequentially to avoid an extra two-byte copy of an entire long stage.
    change = np.mean(
        [np.abs(b.astype(np.int16) - a).mean() for a, b in zip(pixels, pixels[1:])]
    )
    if change < 0.3:
        raise RuntimeError("Pixel stream is stale.")
    data = dict(
        obs=pixels,
        action=action,
        native_frame=ticks[keep],
        native_input=native[keep],
        input_chunk=future,
        past_input_chunk=history,
        xy=np.asarray([s.xy for s in samples], dtype=np.float32)[keep],
        stable=np.ones(keep.sum(), dtype=bool),
    )
    metrics = dict(
        frames=len(index),
        coverage=coverage,
        raw_stable_frames=len(samples),
        rank=sorted({s.difficulty for s in samples}),
        replay_stage_ticks=len(words),
        range_native=[int(ticks[0]), int(ticks[-1])],
        alignment_offset=offset,
        moving_input_agreement=agreement,
        alignment_candidates=scores[:5],
        moving_rows=int((action[:, 0] != 4).sum()),
        bomb_rows=int(action[:, 3].sum()),
        native_mean_pixel_change=float(change),
    )
    return data, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    row = next(
        r
        for r in json.loads(args.manifest.read_bytes())
        if r["name"] == args.replay.name
    )
    split = {"train": "train", "validation": "held", "held": "held", "test": "held"}[
        row["split"]
    ]
    if (
        digest(args.replay) != row["sha256"]
        or digest(Path(psutil.Process(args.pid).exe())) != TH13_RETAIL_SHA
    ):
        raise ValueError(
            "The source replay or original game executable fingerprint differs."
        )
    folder = args.out / split
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{args.replay.stem}-s{args.stage}"
    archive = folder / f"{stem}.npz"
    receipt = folder / f"{stem}.json"
    if receipt.exists():
        prior = json.loads(receipt.read_bytes())
        if (
            prior["replay_sha256"] == row["sha256"]
            and archive.is_file()
            and prior["archive_sha256"] == digest(archive)
        ):
            print(f"Existing verified capture {archive}")
            return
        raise RuntimeError("Existing capture fingerprint differs.")
    words = next(
        np.frombuffer(data, "<u2").reshape(-1, 3)[:, 0]
        for stage, data in action_streams(args.replay, 13)
        if stage == args.stage
    )
    game = ReplayProcess(13, args.pid)
    try:
        samples, images, racing = capture(game, args.stage, words)
    finally:
        game.close()
    data, metrics = pair(samples, images, words)
    with archive.with_suffix(".npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, **data)
    archive.with_suffix(".npz.tmp").replace(archive)
    report = {
        **row,
        **metrics,
        "title": "TH13",
        "stage": args.stage,
        "split": split,
        "original_split": row["split"],
        "replay_sha256": row["sha256"],
        "archive_sha256": digest(archive),
        "game_version": "1.00c retail Steam",
        "original_software_exe_sha256": TH13_RETAIL_SHA,
        "discarded_racing_reads": racing,
        "action_convention": "first future tick move/focus/shoot; Bomb any of next four native ticks",
    }
    receipt.with_suffix(".json.tmp").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    receipt.with_suffix(".json.tmp").replace(receipt)
    print(
        f"Qualified {archive}: {len(data['obs'])} pairs; alignment {metrics['moving_input_agreement']:.5f}"
    )


if __name__ == "__main__":
    main()

"""Measure original TH14 stepping with a fixed action trace.

Select Fast input latency in custom.exe. The probe reads the active limiter
without changing it. On compatible thprac installs, set the candidate rate
under Advanced Options (F12) before running the probe.
"""

import argparse
from collections import Counter
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import struct
from time import perf_counter, process_time, sleep

import numpy as np

from scripts.stage_policy import NormalStageOne
from scripts.train_full_policy import retry_story_game_over


TRACE = (
    (4, 0, 1, 0),
    (3, 1, 1, 0),
    (4, 0, 1, 0),
    (5, 1, 1, 0),
)


def align_start(env, target_frame):
    frame = env.stage_frame()
    if target_frame is None or target_frame == frame:
        return
    if frame > target_frame:
        raise RuntimeError(f"Retry began at frame {frame}, past requested frame {target_frame}.")
    env.release()
    env.game.require_game_focus()
    env.game.resume_game_process()
    deadline = perf_counter() + 2
    try:
        while env.stage_frame() < target_frame:
            if perf_counter() >= deadline:
                raise TimeoutError("The stage clock did not reach the requested starting frame.")
    finally:
        env.freeze()
    actual = env.stage_frame()
    if actual != target_frame:
        raise RuntimeError(f"Frame alignment overshot from {target_frame} to {actual}.")
    env.snapshot = env.reader.snapshot()
    env.first_score = env.snapshot.score
    env.last_frame = actual


def measure(env, count, target_frame):
    reset_started = perf_counter()
    reset_cpu_started = process_time()
    env.reset()
    reset_seconds = perf_counter() - reset_started
    reset_cpu_seconds = process_time() - reset_cpu_started
    align_start(env, target_frame)
    game = env.game
    initial_global = game.read_game_val("global_timer")
    initial_stage = env.stage_frame()
    global_cursor, stage_cursor = initial_global, initial_stage
    transitions = []
    started = perf_counter()
    cpu_started = process_time()
    for step in range(count):
        action = TRACE[(step // 8) % len(TRACE)]
        obs, _, done, truncated, info = env.step(np.asarray(action, dtype=np.int64))
        global_tick = game.read_game_val("global_timer")
        stage_tick = env.stage_frame()
        nearest_projectiles = np.round(obs["bullets"][:min(3, env.snapshot.bullets)], 5).tolist()
        transitions.append({
            "index": step,
            "action": action,
            "global_delta": global_tick - global_cursor,
            "stage_delta": stage_tick - stage_cursor,
            "stage_frame": stage_tick,
            "player": [round(float(obs["context"][0] * 192), 4),
                       round(float(obs["context"][1] * 448), 4)],
            "lives": env.snapshot.lives,
            "bullets": env.snapshot.bullets,
            "nearest_projectiles": nearest_projectiles,
            "native_stage_score": info["score"],
        })
        global_cursor, stage_cursor = global_tick, stage_tick
        if done or truncated:
            break
    seconds = perf_counter() - started
    cpu_seconds = process_time() - cpu_started
    transitions_bytes = json.dumps(transitions, sort_keys=True).encode()
    return {
        "decisions": len(transitions),
        "stepping_wall_s": seconds,
        "stepping_controller_cpu_s": cpu_seconds,
        "reset_wall_s": reset_seconds,
        "reset_controller_cpu_s": reset_cpu_seconds,
        "initial_stage_frame": initial_stage,
        "global_ticks": global_cursor - initial_global,
        "stage_ticks": stage_cursor - initial_stage,
        "global_ticks_per_second": (global_cursor - initial_global) / seconds,
        "stage_ticks_per_second": (stage_cursor - initial_stage) / seconds,
        "global_step_histogram": dict(Counter(row["global_delta"] for row in transitions)),
        "stage_step_histogram": dict(Counter(row["stage_delta"] for row in transitions)),
        "both_exact_four_steps": sum(row["global_delta"] == row["stage_delta"] == 4 for row in transitions),
        "outcome": info["outcome"],
        "trace_sha256": hashlib.sha256(transitions_bytes).hexdigest(),
        "transitions": transitions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parakit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--start-frame", type=int, help="Advance without input to this exact stage tick after Retry.")
    parser.add_argument("--poll-ms", type=float, nargs="+", default=[0.0], metavar="MS",
                        help="Rotate native clock polling intervals across repeats. Zero is the stock busy poll.")
    parser.add_argument("--retry-game-over", action="store_true",
                        help="Resume the verified Normal Reimu A Stage 1 pause or Retry menu.")
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1 or args.out.exists():
        parser.error("Use positive steps and repeats and a new output path.")
    if args.start_frame is not None and args.start_frame < 1:
        parser.error("Use a positive starting frame.")
    if (len(set(args.poll_ms)) != len(args.poll_ms)
            or any(not math.isfinite(value) or not 0 <= value <= 4 for value in args.poll_ms)):
        parser.error("Use distinct clock poll intervals between 0 and 4 ms.")

    from environment import interface as game

    if args.retry_game_over:
        retry_story_game_over(args.parakit)
    deadline = perf_counter() + 45
    while game.read_game_val("game_state") != 2 and perf_counter() < deadline:
        sleep(.05)
    mode = int.from_bytes(game._read_game_memory(0xD9159, 1), "little")
    if mode != 3:
        raise RuntimeError("This probe requires the game's Fast input-latency limiter mode.")
    # The limiter's double pointer is the instruction operand used by thprac for this build.
    pointer = struct.unpack("<I", game._read_game_memory(0x6A792, 4))[0]
    period = struct.unpack("<d", game._read_game_memory(pointer, 8, rel=False))[0]
    if not .001 < period < .1:
        raise RuntimeError(f"Unexpected Fast limiter period: {period}.")
    result = {
        "game": "TH14 original 1.00b",
        "stage": "Normal Reimu A Stage 1",
        "label": args.label,
        "limiter_hz": 1 / period,
        "requested_decisions": args.steps,
        "requested_start_frame": args.start_frame,
        "clock_poll_ms": args.poll_ms,
        "episodes": [],
    }
    env = NormalStageOne(args.parakit, limit=args.steps, max_misses=3)
    original_sleep = game._sleep
    try:
        for index in range(args.repeats):
            pivot = index % len(args.poll_ms)
            for interval in args.poll_ms[pivot:] + args.poll_ms[:pivot]:
                game._sleep = partial(original_sleep, poll_interval=interval / 1000) if interval else original_sleep
                row = measure(env, args.steps, args.start_frame)
                row.update(repeat=index + 1, poll_ms=interval)
                result["episodes"].append(row)
                print(json.dumps({key: value for key, value in row.items() if key != "transitions"}), flush=True)
    finally:
        game._sleep = original_sleep
        env.close()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

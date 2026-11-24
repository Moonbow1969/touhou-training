"""Measure stock TH14 replay playback speed and four-tick capture fidelity."""

import argparse
import json
import math
from collections import Counter
from hashlib import file_digest
from pathlib import Path
from time import perf_counter, process_time, sleep

import keyboard
import numpy as np

from environment.structured import StateReader
from replays.collect_stage import (
    CONTROL_BITS,
    ReplayProcess,
    has_control,
    wait_for_stage,
)
from replays.inspect_rpy import action_streams


PIXEL_SCALE = np.asarray([384, 448])


def fingerprint(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_frame(game, reader):
    thread = reader.offsets.game_thread
    pointer = reader.read_uint(thread.game_thread_pointer, relative=True)
    frame = reader.read_uint(pointer + thread.zGameThread_stage_timer)
    stage = reader.read_uint(reader.offsets.environment.stage, relative=True)
    if stage != 1 or game.read_game_val("pause_state") != 2:
        raise RuntimeError("The probe left the active Stage 1 replay.")
    return game.clock(reader), frame, pointer


def align_start(game, reader, reference):
    remainder = Counter(frame % 4 for frame in reference).most_common(1)[0][0]
    deadline = perf_counter() + 10
    while perf_counter() < deadline:
        _, frame, _ = read_frame(game, reader)
        if frame >= 300 and frame % 4 == remainder:
            game.freeze()
            _, frozen_frame, _ = read_frame(game, reader)
            if frozen_frame % 4 == remainder:
                return
            game.resume()
        sleep(.0005)
    raise TimeoutError("Could not align the replay clock to the stored four-tick reference.")


def sample_window(game, reader, labels, reference, offset, steps, fast_key, poll_seconds):
    start_tick, start_frame, _ = read_frame(game, reader)
    tick, frame = start_tick, start_frame
    global_deltas, stage_deltas = Counter(), Counter()
    counts = Counter()
    nearest_projectile_errors = []
    started = perf_counter()
    cpu_started = process_time()
    if fast_key:
        game.require_foreground()
    try:
        if fast_key:
            keyboard.press(fast_key)
        for _ in range(steps):
            game.require_foreground()
            game.resume()
            deadline = perf_counter() + 2
            while game.clock(reader) < tick + 4:
                if perf_counter() >= deadline:
                    raise TimeoutError("The replay clock stopped advancing.")
                if poll_seconds:
                    sleep(poll_seconds)
            game.freeze()
            next_tick, next_frame, pointer = read_frame(game, reader)
            global_delta, stage_delta = next_tick - tick, next_frame - frame
            global_deltas[global_delta] += 1
            stage_deltas[stage_delta] += 1
            counts["exact_four_global_and_stage"] += global_delta == stage_delta == 4
            tick, frame = next_tick, next_frame
            snapshot = reader.snapshot()
            if 0 <= frame + offset < len(labels) and has_control(reader, pointer, snapshot.player_state):
                live = reader.read_uint(reader.offsets.statics.input, relative=True) & CONTROL_BITS
                recorded = labels[frame + offset] & CONTROL_BITS
                counts["controllable_label_pairs"] += 1
                counts["aligned_label_pairs"] += live == recorded
                if (live | recorded) & 0xF0:
                    counts["moving_label_pairs"] += 1
                    counts["aligned_moving_label_pairs"] += live == recorded
            if frame in reference:
                context, bullets, bullet_count = reference[frame]
                live_bullets = snapshot.observation["bullets"]
                counts["matching_reference_frames"] += 1
                counts["matching_player_positions"] += np.allclose(
                    snapshot.observation["context"][:2], context, rtol=0, atol=1e-6,
                )
                counts["matching_projectile_counts"] += snapshot.bullets == bullet_count
                counts["matching_projectile_tensors"] += np.allclose(live_bullets, bullets, rtol=0, atol=1e-6)
                live_positions = live_bullets[live_bullets[:, 6] != 0, :2]
                ref_positions = bullets[bullets[:, 6] != 0, :2]
                if len(live_positions) and len(ref_positions):
                    live_nearest = np.linalg.norm(live_positions * PIXEL_SCALE, axis=1).min()
                    ref_nearest = np.linalg.norm(ref_positions * PIXEL_SCALE, axis=1).min()
                    nearest_projectile_errors.append(abs(live_nearest - ref_nearest))
    finally:
        if fast_key:
            keyboard.release(fast_key)
    seconds = perf_counter() - started
    return {
        "fast_forward_key": fast_key, "decisions": steps,
        "first_stage_frame": start_frame, "last_stage_frame": frame, "native_stage_ticks": frame - start_frame,
        "seconds": seconds, "native_stage_ticks_per_second": (frame - start_frame) / seconds,
        "controller_cpu_seconds": process_time() - cpu_started,
        "global_ticks_per_second": (tick - start_tick) / seconds,
        "global_delta_histogram": dict(global_deltas), "stage_delta_histogram": dict(stage_deltas),
        "nearest_projectile_error_support": len(nearest_projectile_errors),
        "nearest_projectile_error_p95_px": (
            float(np.percentile(nearest_projectile_errors, 95)) if nearest_projectile_errors else None
        ),
        **{name: int(count) for name, count in counts.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="Accepted Stage 1 observations from the same replay.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--key", choices=("left ctrl", "z"), default="left ctrl")
    parser.add_argument("--poll-ms", type=float, default=.5)
    args = parser.parse_args()
    if args.steps < 1 or args.out.exists() or not args.out.parent.is_dir():
        parser.error("Use a positive decision count and a fresh output path in an existing directory.")
    if not math.isfinite(args.poll_ms) or not 0 <= args.poll_ms <= 1:
        parser.error("Use a poll interval between 0 and 1 ms.")
    metadata = json.loads(args.reference.with_suffix(".json").read_text(encoding="utf-8"))
    replay_sha = fingerprint(args.replay)
    if (metadata["sha256"] != replay_sha or metadata["title"] != 14 or metadata["stage"] != 1
            or metadata["stride"] != 4 or metadata.get("failure") or metadata["moving_alignment"] < .95
            or metadata["coverage"] < .9):
        parser.error("The reference must be an accepted four-tick TH14 Stage 1 capture of this replay.")
    with np.load(args.reference, allow_pickle=False) as arrays:
        if (arrays["expert_action_chunk"].shape != (metadata["pairs"], 4)
                or len(arrays["stage_frame"]) != metadata["pairs"]):
            parser.error("The reference arrays do not match their accepted pair count.")
        positions = arrays["context"][:, :2]
        states = zip(positions, arrays["bullets"], arrays["bullet_count"], strict=True)
        reference = dict(zip(arrays["stage_frame"].tolist(), states, strict=True))
    if not reference:
        parser.error("The stage reference contains no accepted observations.")
    _, body = next((stage, data) for stage, data in action_streams(args.replay, 14) if stage == 1)
    labels = np.frombuffer(body, dtype="<u2").reshape(-1, 3)[:, 0]
    game = ReplayProcess(14, args.pid)
    report = {
        "title": "TH14", "stage": 1, "replay_sha256": replay_sha,
        "reference_arrays_sha256": fingerprint(args.reference), "blocks": [], "status": "started",
        "poll_ms": args.poll_ms,
        "window_design": "ABBA over consecutive slices of one native replay, with same-frame archived parity where sampled.",
    }
    try:
        game.require_stock_th14()
        reader = StateReader(game, game.offsets, 96)
        game.start_selected_replay()
        wait_for_stage(game, reader, 1, 18)
        game.require_th14_playback()
        align_start(game, reader, reference)
        for key in (None, args.key, args.key, None):
            row = sample_window(game, reader, labels, reference, metadata["offset"], args.steps, key, args.poll_ms / 1000)
            report["blocks"].append(row)
            print(json.dumps(row), flush=True)
        report["status"] = "complete"
    finally:
        try:
            game.close()
        except BaseException:
            report["cleanup"] = "failed"
            raise
        finally:
            with args.out.open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, allow_nan=False)
                output.write("\n")


if __name__ == "__main__":
    main()

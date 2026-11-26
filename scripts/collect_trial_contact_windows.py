"""Collect randomized held 12-tick injury windows from the official TH14 0.01b trial."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from shutil import disk_usage
from time import perf_counter

from scripts.benchmark_trial_workers import TrialWorker

TRIAL_SHA256 = "6c62bff1e6be644086c5811c36eb938818f14012f94ad5fce784ca7889c66d48"
SHOOT = 32
FIRST_FRAME = 105
ACTIVE = 1
HIT = 4
INJURY_CONFIRM_STEPS = 6
GIB = 1024**3


def action_mask(move, focus):
    dy, dx = divmod(move, 3)
    return (1, 0, 2)[dx] | (4, 0, 8)[dy] | (focus << 4) | SHOOT


def check_state(state):
    if state["contact_schema"] != 1 or state["mode"] != 2 or state["stage"] != 1:
        raise RuntimeError("The worker left the observed trial Stage 1 state.")
    if (state["player_state"] not in range(5) or not 0 <= state["bombs"] <= 8
            or state["bullet_count"] < 0 or len(state["bullets"]) != min(96, state["bullet_count"])):
        raise RuntimeError("The trial's contact state violates its readout contract.")


def in_range(state, radius):
    return any(active == 1 and r > 0 and x*x + y*y <= radius*radius for x, y, vx, vy, r, active in state["bullets"])


def capture(name, worker, args, seed):
    if worker["status"] != "ready" or worker["executable_sha256"] != TRIAL_SHA256:
        raise ValueError("Expected a hash-qualified deployed official trial worker.")
    rng = Random(seed)
    counters, deltas, stage_deltas, active_states = Counter(), Counter(), Counter(), Counter()
    step_count = 0
    written_bytes = 0
    byte_limit = getattr(args, "max_uncompressed_bytes", None)
    started = perf_counter()
    path = args.out / f"{name}.jsonl.gz"
    earliest_hit = None

    def step(pipe, state, mask):
        nonlocal step_count
        if step_count >= args.steps:
            raise StopIteration
        following = pipe.request(f"S {mask}")
        step_count += 1
        check_state(following)
        if following["held"] != mask:
            raise RuntimeError("Unexpected key mask after a trial decision.")
        delta = (following["ticks"] - state["ticks"]) % (2**32)
        stage_delta = following["frame"] - state["frame"]
        deltas[delta] += 1
        stage_deltas[stage_delta] += 1
        active_states[str(following["player_state"])] += 1
        exact = delta == stage_delta == following["delta"] == 4
        if not exact:
            counters["clock_mismatch"] += 1
        return following, exact

    with TrialWorker(worker["command"]) as pipe, gzip.open(path, "xt", encoding="utf-8") as trace:
        state = pipe.read(timeout=90)
        startup_seconds = perf_counter() - started
        try:
            for episode in range(args.episodes):
                if episode:
                    previous = state["frame"]
                    state = pipe.request("R")
                    if state["frame"] >= previous:
                        raise RuntimeError("Trial Retry did not rewind the observed native stage clock.")
                check_state(state)
                if (not 0 <= state["frame"] < 90 or state["delta"] != 0 or state["lives"] != 2
                        or state["player_state"] == HIT or state["held"] != 0):
                    raise RuntimeError(f"Unqualified trial opening: frame {state['frame']}, "
                                       f"lives {state['lives']}, player state {state['player_state']}.")
                counters["qualified_openings"] += 1
                while state["frame"] < args.frame_limit:
                    if state["player_state"] == HIT or state["lives"] != 2:
                        counters["unscored_hit_or_stock_change"] += 1
                        if state["frame"] < FIRST_FRAME:
                            counters["startup_hits_or_stock_changes"] += 1
                        break
                    eligible = (state["frame"] >= FIRST_FRAME and state["player_state"] == ACTIVE
                                and in_range(state, args.near_radius))
                    if not eligible:
                        state, _ = step(pipe, state, SHOOT)
                        counters["between_window_steps"] += 1
                        continue
                    sample = state
                    assigned = rng.randrange(18)
                    focus, move = divmod(assigned, 9)
                    mask = action_mask(move, focus)
                    counters["randomized_windows"] += 1
                    transitions = []
                    exact = True
                    hit_after = None
                    for turn in range(3):
                        state, good = step(pipe, state, mask)
                        exact &= good
                        transitions.append([state[k] for k in ("frame", "ticks", "player_state", "lives", "bombs", "player")])
                        if state["player_state"] == HIT and hit_after is None:
                            hit_after = turn + 1
                    if hit_after is None and state["lives"] != sample["lives"]:
                        counters["unobserved_hit_onset"] += 1
                        break
                    if hit_after is not None:
                        for _ in range(INJURY_CONFIRM_STEPS):
                            if state["lives"] < sample["lives"]:
                                break
                            state, good = step(pipe, state, mask)
                            exact &= good
                        if state["lives"] != sample["lives"] - 1:
                            counters["unconfirmed_hit"] += 1
                            break
                        earliest_hit = sample["frame"] if earliest_hit is None else min(earliest_hit, sample["frame"])
                    if not exact:
                        counters["clock_rejected_windows"] += 1
                    else:
                        actor_before = dict(player=sample["player"], prior_focus=bool(sample["held"] & 16),
                                            bullets=sample["bullets"], bullet_count=sample["bullet_count"])
                        line = json.dumps(dict(episode=episode, frame=sample["frame"], tick=sample["ticks"],
                                               lives=sample["lives"], bombs=sample["bombs"], start=actor_before,
                                               held_mask=mask, move=move, focus=focus, hit_after_step=hit_after,
                                               transitions=transitions, confirmed_lives=state["lives"])) + "\n"
                        if byte_limit is not None and written_bytes + len(line) > byte_limit:
                            counters["byte_limit_reached"] += 1
                            raise StopIteration
                        trace.write(line)
                        written_bytes += len(line)
                        counters["accepted_windows"] += 1
                        counters["injuries"] += hit_after is not None
                        if counters["accepted_windows"] % 32 == 0:
                            trace.flush()
                            print(json.dumps(dict(worker=name, **counters)), flush=True)
                    if counters["accepted_windows"] >= args.windows:
                        raise StopIteration
                    if hit_after is not None:
                        break
        except StopIteration:
            pass
    return dict(worker=name, seed=seed, counts=dict(counters), native_tick_deltas=dict(deltas),
                native_stage_deltas=dict(stage_deltas), sampled_player_states=dict(active_states),
                minimum_injury_window_frame=earliest_hit, wall_seconds=perf_counter()-started, startup_seconds=startup_seconds,
                sampled_steps=step_count, file=path.name, shard_bytes=path.stat().st_size,
                shard_sha256=sha256(path.read_bytes()).hexdigest(),
                launch={k:worker[k] for k in ("executable_sha256", "worker_source_sha256", "readout_sha256", "image_id")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers-config", type=Path, required=True)
    parser.add_argument("--workers", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2600)
    parser.add_argument("--episodes", type=int, default=18)
    parser.add_argument("--frame-limit", type=int, default=1800)
    parser.add_argument("--near-radius", type=float, default=112.)
    parser.add_argument("--seed", type=int, default=120247)
    args = parser.parse_args()
    if min(args.steps, args.windows, args.episodes) <= 0 or args.frame_limit <= FIRST_FRAME or not 0 < args.near_radius <= 192:
        parser.error("Use positive bounds and a valid trial projectile neighborhood.")
    if len(set(args.workers)) != len(args.workers) or any(not name.isalpha() for name in args.workers):
        parser.error("Use distinct alphabetical names for the worker lanes.")
    args.out.mkdir(parents=True, exist_ok=False)
    if disk_usage(args.out).free < 8*GIB:
        raise OSError("The local transition volume has less than 8 GiB free.")
    config = json.loads(args.workers_config.read_text())
    started = perf_counter()
    status_path = args.out / "collection.json"
    report = dict(status="running", setting="Official TH14 trial 0.01b, Normal Reimu A Stage 1",
                  sampling=dict(windows_per_worker=args.windows, step_limit_per_worker=args.steps,
                                episode_limit_per_worker=args.episodes, frame_limit=args.frame_limit,
                                eligible_start_frame=FIRST_FRAME, initial_lives=2, near_projectile_radius=args.near_radius,
                                move_focus="uniform independent choice among 18, held 3 × 4 native ticks",
                                focus_context="last verified held Shift command",
                                shoot=1, bomb=0, first_confirmed_native_hit_per_credit=True,
                                injury="Native player state enters 4 within the 12-tick hold, then native lives decrement.",
                                rng_controlled="Action sampler seeds only"),
                  requested_workers=args.workers, workers={}, errors={},
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    status_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with ThreadPoolExecutor(max_workers=len(args.workers)) as pool:
        jobs = {pool.submit(capture, name, config[name], args, args.seed + 1009 * index): name
                for index, name in enumerate(args.workers)}
        for future in as_completed(jobs):
            name = jobs[future]
            try:
                row = future.result()
            except Exception as exc:
                report["errors"][name] = f"{type(exc).__name__}: {exc}"
            else:
                report["workers"][name] = row
                print(json.dumps(row), flush=True)
    report["status"] = "partial" if report["errors"] else "complete"
    report["wall_seconds"] = perf_counter() - started
    status_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["workers"]:
        raise RuntimeError(json.dumps(report["errors"]))


if __name__ == "__main__":
    main()

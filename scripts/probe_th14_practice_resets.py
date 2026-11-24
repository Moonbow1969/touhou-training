"""Qualify repeated Retry from an already selected TH14 Stage 1 Practice section."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.checkpoint_storage import digest


def sample(env):
    game = env.game
    left_right, height = env.snapshot.observation["context"][:2]
    if not np.isfinite((left_right, height)).all():
        raise RuntimeError("The player position is unreadable.")
    return {
        "stage": env.current_stage(),
        "chapter": env.reader.read_uint(env.reader.offsets.statics.stage_chapter, relative=True),
        "screen": env.current_screen(),
        "state": game.read_game_val("game_state"),
        "global_tick": game.read_game_val("global_timer"),
        "stage_frame": env.stage_frame(),
        "player": [float(left_right) * 192, float(height) * 448],
        "player_state": env.snapshot.player_state,
        "resources": {key: game.read_game_val(key) for key in
                      ("lives", "bombs", "power", "life_fragments", "bomb_fragments")},
        "spell_id": env.snapshot.spell_id,
        "boss_max_hp": env.snapshot.boss.max_hp if env.snapshot.boss else None,
        "bullets": env.snapshot.bullets,
    }


def require_active(state):
    if ((state["stage"], state["screen"], state["state"]) != (1, 7, 2)
            or state["global_tick"] is None or state["stage_frame"] < 1):
        raise RuntimeError("Expected readable active TH14 Stage 1 gameplay.")


def require_opening(state, chapter, expected_resources):
    require_active(state)
    if state["chapter"] != chapter:
        raise RuntimeError(f"Expected raw practice chapter {chapter}, observed {state['chapter']}.")
    for key, expected in expected_resources.items():
        if state["resources"][key] != expected:
            raise RuntimeError(f"Expected opening {key}={expected}, observed {state['resources'][key]}.")
    if any(value is None for value in state["resources"].values()):
        raise RuntimeError("The opening resource inventory is unreadable.")


def probe(env, report, chapter, expected_resources, resets, steps, start_frame=None):
    from scripts.benchmark_native_clock import align_start

    env.reset()  # The prepared first reset adopts the current position without pressing Retry.
    state = report["prepared_opening"] = sample(env)
    require_opening(state, chapter, expected_resources)
    reference = None
    for number in range(1, resets + 1):
        row = {"reset": number, "transitions": [], "complete": False}
        report["resets"].append(row)
        end_frame = state["stage_frame"]
        started = perf_counter()
        env.reset()
        state = row["native_opening"] = sample(env)
        require_active(state)
        if state["stage_frame"] >= end_frame:
            raise RuntimeError("The native stage frame did not rewind on Retry.")
        if start_frame is not None:
            align_start(env, start_frame)
            state = sample(env)
        row["opening"] = state
        row["seconds_to_opening"] = perf_counter() - started
        require_opening(state, chapter, expected_resources)
        signature = (state["resources"], state["spell_id"], state["boss_max_hp"])
        if reference is None:
            reference = signature
        elif signature != reference:
            raise RuntimeError("Retry changed the expected opening inventory or boss phase.")

        responses = {"left": False, "right": False}
        for index in range(steps):
            left = (index // 4) % 2 == 0
            direction = "left" if left else "right"
            action = np.array([3 if left else 5, 1, 1, 0], dtype=np.int64)
            before = state
            _, _, terminated, truncated, info = env.step(action)
            state = sample(env)
            stage_delta = state["stage_frame"] - before["stage_frame"]
            global_delta = (None if state["global_tick"] is None
                            else state["global_tick"] - before["global_tick"])
            x = before["player"][0]
            dx = state["player"][0] - x
            eligible = (-173 < x < 173 and before["player_state"] == state["player_state"] == 1
                        and before["resources"]["lives"] == state["resources"]["lives"])
            response = ((-dx if left else dx) > .05) if eligible else None
            responses[direction] |= response is True
            row["transitions"].append({
                "direction": direction,
                "global_delta": global_delta, "stage_delta": stage_delta,
                "chapter": state["chapter"], "player_x": state["player"][0],
                "bullets": state["bullets"], "response": response,
            })
            require_active(state)
            if (stage_delta <= 0 or global_delta <= 0 or state["chapter"] != chapter
                    or terminated or truncated):
                raise RuntimeError(f"The fixed trace left the expected active section at step {index + 1}. "
                                   f"Native outcome was {info['outcome']!r}.")
            if (state["player_state"] == 4 or state["resources"]["lives"] != expected_resources["lives"]
                    or info["bombs_used"]):
                raise RuntimeError("The fixed trace entered a hit state, lost a life, or consumed a Bomb.")
        if not all(responses.values()):
            raise RuntimeError("Retry did not produce an observed movement in both directions.")
        row["complete"] = True


def summarize(rows):
    left_right = {direction: {"eligible": 0, "observed": 0} for direction in ("left", "right")}
    global_deltas, stage_deltas = Counter(), Counter()
    exact = steps = 0
    for row in rows:
        for step in row["transitions"]:
            steps += 1
            if step["global_delta"] is not None:
                global_deltas[step["global_delta"]] += 1
            stage_deltas[step["stage_delta"]] += 1
            exact += step["global_delta"] == step["stage_delta"] == 4
            if step["response"] is not None:
                direction = left_right[step["direction"]]
                direction["eligible"] += 1
                direction["observed"] += step["response"]
    return {
        "completed_resets": sum(row["complete"] for row in rows),
        "decisions": steps, "both_exact_four": exact,
        "unreadable_global_ticks": steps - global_deltas.total(),
        "global_tick_deltas": dict(sorted(global_deltas.items())),
        "stage_tick_deltas": dict(sorted(stage_deltas.items())),
        "horizontal": left_right,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True, help="PID of the prepared foreground Steam TH14 instance.")
    parser.add_argument("--chapter", type=int, required=True,
                        help="Expected raw native stage_chapter field after the selected section appears.")
    for resource in ("lives", "bombs", "power"):
        parser.add_argument(f"--expect-{resource}", type=int, required=True)
    parser.add_argument("--resets", type=int, default=20)
    parser.add_argument("--steps", type=int, default=32, help="Decisions to sample per Retry, at least eight.")
    parser.add_argument("--start-frame", type=int,
                        help="Align each Retry to this early native stage frame before checks (2 s alignment timeout).")
    parser.add_argument("--parakit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.pid < 1 or args.chapter < 0 or args.resets < 1 or args.steps < 8
            or args.start_frame is not None and args.start_frame < 1):
        parser.error("Use positive PID, reset and start-frame values, a nonnegative chapter, and at least eight steps.")
    expected = {key: getattr(args, f"expect_{key}") for key in ("lives", "bombs", "power")}
    if any(not 0 <= expected[key] <= cap for key, cap in (("lives", 8), ("bombs", 8), ("power", 400))):
        parser.error("Expected resources must be in TH14's native inventory ranges.")
    if not (args.parakit / "offsets.py").is_file() or args.out.exists():
        parser.error("Supply ParaKit offsets and a new output path.")

    import psutil

    from replays.collect_stage import TH14_STEAM_100B
    from scripts.stage_policy import NormalStageOne
    from scripts.train_full_policy import publish_json

    executable = Path(psutil.Process(args.pid).exe())
    sha = digest(executable)
    if sha != TH14_STEAM_100B:
        parser.error("That PID does not use the verified Steam TH14 1.00b binary.")
    from environment import interface as game

    if game._game_pid != args.pid:
        parser.error("The foreground driver found a different TH14 instance.")
    game.require_game_focus()
    report = {
        "task": "Prepared TH14 Steam 1.00b Normal Reimu A Stage 1 Practice",
        "requested_chapter": args.chapter, "expected_resources": expected,
        "requested_resets": args.resets, "steps_per_reset": args.steps,
        "requested_start_frame": args.start_frame,
        "clock_poll_ms": 0., "minimum_joint_four_tick_rate": .99, "minimum_per_direction_response_rate": .98,
        "executable_sha256": sha, "parakit_offsets_sha256": digest(args.parakit / "offsets.py"),
        "runner_sha256": digest(Path(__file__)), "resets": [],
        "status": "started",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream)
    env = None
    try:
        env = NormalStageOne(args.parakit, prepared=True, stage_practice=True, limit=args.steps + 1)
        probe(env, report, args.chapter, expected, args.resets, args.steps, args.start_frame)
        report["status"] = "measured"
    except (RuntimeError, OSError) as error:
        report.update(status="failed", error=str(error))
    finally:
        counts = report["counts"] = summarize(report["resets"])
        if report["status"] == "measured":
            passed = (
                counts["completed_resets"] == args.resets
                and counts["both_exact_four"] * 100 >= counts["decisions"] * 99
                and all(side["eligible"] > 0 and side["observed"] * 100 >= side["eligible"] * 98
                        for side in counts["horizontal"].values())
            )
            report["status"] = "passed" if passed else "failed"
        try:
            if env is not None:
                env.close()
        finally:
            publish_json(args.out, report)
    print(json.dumps({"status": report["status"], **counts, "error": report.get("error")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

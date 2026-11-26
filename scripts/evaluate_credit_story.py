"""Evaluate frozen TH14 Normal story policies on a continuous zero-Continue credit."""

import argparse
from collections import Counter
from contextlib import contextmanager
import ctypes
from importlib import import_module
import json
import os
from pathlib import Path
import re
from statistics import median
from time import perf_counter

import numpy as np
import torch

from scripts import evaluate_full_credit, native_story_restart, stage_policy
from scripts.audit_th14_clear_objective import NativeState, classify
from scripts.checkpoint_storage import digest
from scripts.evaluate_full_credit import continues, resources, verify_opening
from scripts.evaluate_sft_native import load_arms, output_lock, restore_report, trial_schedule
from scripts.evaluate_stage_policies import policy_observation
from scripts.native_story_restart import FreshCreditRequired, GameOverMenu
from scripts.pretrain_policy import ACTION_NAMES, ACTION_WIDTHS
from scripts.train_full_policy import publish_json


TARGETS = {"stage2": 1, "stage3": 2, "ending": 6}
LIMITS = {"stage2": 3300, "stage3": 8500, "ending": 45000}


def last_stage(target):
    return TARGETS[target]


def score_terminal(target, state, seen_stages, truncated):
    if state.continues:
        raise ValueError("A Continue cannot count toward a one-credit result.")
    if target == "ending":
        outcome = classify("story", 1, state, training_limit_reached=truncated)
        if outcome == "clear" and tuple(seen_stages) != tuple(range(1, 7)):
            raise ValueError("The ending did not follow all six observed native story stages.")
        return "one_credit_clear" if outcome == "clear" else outcome
    expected_stage = last_stage(target) + 1
    if state.stage == expected_stage:
        outcome = classify("stage", last_stage(target), state, training_limit_reached=truncated)
        if outcome == "clear" and tuple(seen_stages) == tuple(range(1, expected_stage + 1)):
            return "target_reached"
        raise ValueError("The target stage did not follow every preceding native stage.")
    outcome = classify("story", 1, state, training_limit_reached=truncated)
    if outcome == "clear":
        raise ValueError("The credit entered an ending outside the selected stage target.")
    return outcome


def native_marker(env, pause, *, raw_frame=None, inventory=None):
    return dict(stage=env.current_stage(), screen=env.current_screen(), pause=pause,
                frame=env.stage_frame() if raw_frame is None else raw_frame,
                resources=resources(env) if inventory is None else inventory)


@contextmanager
def bound_policy(policy, env):
    """Bind a learned native-input adapter for one credit, then detach it."""
    if not hasattr(policy, "environment"):
        yield policy
        return
    previous = policy.environment
    policy.environment = env
    try:
        yield policy
    finally:
        policy.environment = previous


def stage_row(marker):
    return dict(stage=marker["stage"], entry=marker, first_playable=None, exit=None,
                decisions=0, lives_lost=0, bombs_consumed=0, action_counts={
                    name: [0] * width for name, width in zip(ACTION_NAMES, ACTION_WIDTHS)
                })


def story_credit(env, arm, trial, args, reset_callback, progress, *, transition_recorder=None):
    env.motion_history = arm.history
    observation, reset_info = reset_callback(env, trial)
    inventory = verify_opening(env, args.start_frame)
    torch.manual_seed(trial["policy_seed"])
    reached = {1: stage_row(native_marker(env, 2, inventory=inventory))}
    reached[1]["first_playable"] = reached[1]["entry"]
    deaths = bombs = 0
    started = perf_counter()
    with bound_policy(arm.policy, env) as policy:
        while True:
            stage_before = env.snapshot.stage
            before_frame = env.last_frame
            player_before = env.snapshot.player_state
            previous_x_key = "x" in env.held
            previous_stock = env.game.read_game_val("bombs")
            stage_log = reached[stage_before]
            actor_observation = policy_observation(policy, env, observation, uses_history=arm.history)
            if transition_recorder is not None:
                transition_before = transition_recorder.before_step(env, actor_observation)
            action, _ = policy.predict(actor_observation, deterministic=False)
            if not env.action_space.contains(action):
                raise RuntimeError("The frozen policy produced an action outside the native control space.")
            stage_log["decisions"] += 1
            for name, value in zip(ACTION_NAMES, action):
                stage_log["action_counts"][name][int(value)] += 1
            observation, _, terminated, truncated, info = env.step(action)
            stage, pause = info["native_stage"], info["native_pause_state"]
            continued = continues(env)
            if transition_recorder is not None:
                transition_recorder.after_step(env, transition_before, action, info, terminated, truncated,
                                               info["misses"] - deaths, info["bombs_used"] - bombs)
            stage_log["lives_lost"] += info["misses"] - deaths
            stage_log["bombs_consumed"] += info["bombs_used"] - bombs
            death_or_bomb = info["misses"] != deaths or info["bombs_used"] != bombs
            if death_or_bomb or player_before == 4:
                stage_log.setdefault("resource_events", []).append(dict(
                    stage_frame=before_frame, player_state=player_before, previous_x_key=previous_x_key,
                    bomb_command=int(action[3]), stock_before=previous_stock,
                    stock_after=env.game.read_game_val("bombs"), deaths=info["misses"] - deaths,
                    bombs_used=info["bombs_used"] - bombs, native_stage_after=stage,
                    native_state_after=pause))
            deaths, bombs = info["misses"], info["bombs_used"]
            if continued:
                env.release()
                progress(dict(info, outcome="continued_credit_invalid", native_continues=continued,
                              stage_records=list(reached.values()), stopped_on_continue=True))
                raise RuntimeError("Native Continue advanced. This attempt is excluded from story scoring.")
            new_stage = stage != stage_before
            marker = None
            if new_stage:
                if stage != stage_before + 1 or stage > 6 or stage in reached:
                    raise RuntimeError(f"Nonconsecutive native story stage transition {stage_before} -> {stage}.")
                marker = native_marker(env, pause)
                stage_log["exit"] = dict(marker, last_controlled_frame=before_frame, reason="native_stage_advance")
                reached[stage] = stage_row(marker)
                if pause == 2:
                    reached[stage]["first_playable"] = marker
            elif reached[stage]["first_playable"] is None and pause == 2:
                reached[stage]["first_playable"] = native_marker(env, pause)
            if not (new_stage or death_or_bomb or terminated or truncated or info["steps"] % 256 == 0):
                continue
            row = dict(info, target=args.target, decision_count=info["steps"], deaths=deaths,
                       actual_bombs_consumed=bombs, native_continues=continued,
                       stage_records=list(reached.values()), start_frame=args.start_frame,
                       start_resources=inventory, reset_info=reset_info, seconds=perf_counter() - started)
            if terminated or truncated:
                env.release()
                terminal = marker if marker is not None else native_marker(env, pause)
                state = NativeState(stage, terminal["screen"], pause, terminal["resources"]["lives"], continued)
                row.update(native_terminal=terminal, stopped_before_continue=pause == 1 and state.lives < 0)
                progress(row)
                outcome = score_terminal(args.target, state, reached, truncated)
                if outcome == "ongoing":
                    raise RuntimeError("The adapter ended a live credit before its objective.")
                if not new_stage:
                    reached[stage]["exit"] = dict(terminal, last_controlled_frame=before_frame, reason=outcome)
                row.update(outcome=outcome, one_credit_clear=outcome == "one_credit_clear",
                           target_reached=outcome in ("target_reached", "one_credit_clear"))
                if transition_recorder is not None:
                    final_observation = None
                    if outcome == "time_limit" and pause == 2 and terminal["screen"] == 7:
                        final_observation = policy_observation(policy, env, observation, uses_history=arm.history)
                    transition_recorder.finish(outcome, final_observation)
                return row
            progress(row)


def summarize(completed, arms):
    summary = {}
    for name in arms:
        rows = [row["result"] for row in completed if row["trial"]["arm"] == name]
        stages = {}
        for stage in range(1, 7):
            entrants = [step["entry"]["resources"] for row in rows for step in row["stage_records"] if step["stage"] == stage]
            if entrants:
                stages[stage] = dict(entries=len(entrants), median_lives=median(row["lives"] for row in entrants),
                                     median_bombs=median(row["bombs"] for row in entrants))
        summary[name] = dict(credits=len(rows), outcomes=dict(Counter(row["outcome"] for row in rows)),
                             target_reaches=sum(row["target_reached"] for row in rows),
                             one_credit_clears=sum(row["one_credit_clear"] for row in rows),
                             stage_entries=stages)
    return summary


@contextmanager
def native_session(args, arms, menu, restart_hook):
    from environment import interface as game
    from environment.structured import StateReader, load_parakit_offsets

    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState
    continuous = 0x80000000
    keep_awake(continuous | 0x00000001 | 0x00000002)
    stage = None
    try:
        reader = StateReader(game, load_parakit_offsets(args.parakit), 96)
        preparation = native_story_restart.prepare(game, reader, menu, restart_hook)
        longest = max([12] + [ticks for arm in arms.values() for ticks in (arm.policy.motion_spans or [])])
        stage = stage_policy.NormalStageOne(
            args.parakit, limit=args.limit, max_misses=float("inf"), prepared=True,
            last_stage=last_stage(args.target), motion_history=any(arm.history for arm in arms.values()),
            history_frames=longest, focus_wait=args.focus_wait,
        )
        yield stage, preparation
    finally:
        try:
            if stage is not None:
                stage.release()
                stage.close()
        finally:
            keep_awake(continuous)


def evaluate(args, arms, specification, path, menu, restart_hook, stage_factory=native_session, reset_callback=None,
             *, transition_archive=None):
    report = restore_report(path, specification) if args.resume else dict(
        specification=specification, completed=[], restarts=[], interrupted_attempts=[], pending=None,
    )
    if report.get("in_progress") is not None:
        report["interrupted_attempts"].append(report["in_progress"])
    remaining = specification["schedule"][len(report["completed"]):]
    if transition_archive is not None:
        transition_archive.recover(report["completed"], remaining)
        matched = len(report["completed"]) // len(arms)
        report.update(matched_blocks=matched, matched_summary=summarize(report["completed"][:matched * len(arms)], arms))
    report.update(status="evaluating" if remaining else "complete", error=None, in_progress=None,
                  waiting_for_fresh_credit=None, pending=None)
    publish_json(path, report)
    if not remaining:
        return report

    if reset_callback is None:
        def reset_callback(env, trial):
            return native_story_restart.reset(env, trial, start_frame=args.start_frame,
                                              menu=menu, restart_hook=restart_hook)

    try:
        with stage_factory(args, arms, menu, restart_hook) as (stage, preparation):
            report.setdefault("session_preparations", []).append(preparation)
            for trial in remaining:
                report["pending"] = trial
                publish_json(path, report)
                print(json.dumps(dict(status="trial", **trial)), flush=True)

                def progress(result):
                    report["in_progress"] = dict(trial=trial, result=result)
                    publish_json(path, report)

                if transition_archive is None:
                    result = story_credit(stage, arms[trial["arm"]], trial, args, reset_callback, progress)
                else:
                    recorder = transition_archive.start(stage)
                    result = story_credit(stage, arms[trial["arm"]], trial, args, reset_callback, progress,
                                          transition_recorder=recorder)
                    result["transition_shard"] = transition_archive.publish(trial, result, recorder)
                report["completed"].append(dict(trial=trial, result=result))
                matched = len(report["completed"]) // len(arms)
                report.update(pending=None, in_progress=None, matched_blocks=matched,
                              matched_summary=summarize(report["completed"][:matched * len(arms)], arms))
                publish_json(path, report)
                print(json.dumps(dict(status="completed", **trial, outcome=result["outcome"],
                                      decision_count=result["decision_count"], deaths=result["deaths"],
                                      terminal=result["native_terminal"])), flush=True)
    except FreshCreditRequired as error:
        report.update(status="waiting_for_fresh_credit", pending=None,
                      waiting_for_fresh_credit=dict(native_state=error.state, detail=str(error)))
        publish_json(path, report)
        return report
    except BaseException as error:
        report.update(status="interrupted", error=dict(type=type(error).__name__, detail=str(error)))
        publish_json(path, report)
        raise
    report["status"] = "complete"
    publish_json(path, report)
    return report


def selections(args):
    arms = load_arms(args.policy, args.ppo_policy, args.ppo_template)
    if args.crossover:
        from scripts.evaluate_bomb_crossover import SPLICE, crossover_arms

        for name in args.crossover:
            if name in arms or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name):
                raise ValueError("Crossover arm names must be unique.")
            arms[name] = crossover_arms(args)[SPLICE]
    if args.timing_policy:
        from scripts.train_native_bomb_timing import load_native_bomb_arm

        for entry in args.timing_policy:
            name, anchor, path = entry.split("=", 2)
            if name in arms or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name):
                raise ValueError("Timing arm names must be unique.")
            candidate = load_native_bomb_arm(arms[anchor], Path(path))
            candidate.provenance.update(training_stage=1, deployment_stages=list(range(1, last_stage(args.target) + 1)),
                                        later_stages_are_transfer=True, anchor_arm=anchor)
            arms[name] = candidate
    if getattr(args, "run_arm", None):
        missing = set(args.run_arm) - arms.keys()
        if missing or len(set(args.run_arm)) != len(args.run_arm):
            raise ValueError("Each requested evaluation arm must be loaded and named once.")
        return {name: arms[name] for name in args.run_arm}
    return arms


def hook(path):
    module, name = path.split(":", 1)
    function = getattr(import_module(module), name)
    if not callable(function):
        raise ValueError("A restart hook must name a callable module:function.")
    return function


def run_specification(args, arms):
    sources = (evaluate_full_credit, native_story_restart, stage_policy)
    fingerprints = {module.__name__: digest(Path(module.__file__)) for module in sources}
    names = ["full_credit_reset"]
    if args.crossover:
        names.append("evaluate_bomb_crossover")
    if args.timing_policy:
        names.append("train_native_bomb_timing")
    for name in names:
        module = import_module(f"scripts.{name}")
        fingerprints[module.__name__] = digest(Path(module.__file__))
    external = import_module(args.restart_hook.partition(":")[0]) if args.restart_hook else None
    return json.loads(json.dumps({
        "schema": 1,
        "task": dict(title="TH14", version="1.00b", difficulty="Normal", shot="Reimu A", stock_story=True,
                     target=args.target, start_frame=args.start_frame, decision_limit=args.limit,
                     frames_per_action=4, all_native_lives=True, native_rng_seed_controlled=False),
        "controller": dict(frozen=True, geometry=False, aim_correction=False, automatic_bomb=False,
                           allows_continue=False, policy_unchanged_between_stages=True),
        "restart": dict(game_over_default_sha256=digest(args.game_over_default_reference)
                        if args.game_over_default_reference else None,
                        game_over_retry_sha256=digest(args.game_over_retry_reference)
                        if args.game_over_retry_reference else None,
                        menu_crop=args.menu_crop, menu_max_error=args.menu_max_error, external_hook=args.restart_hook,
                        external_hook_source_sha256=digest(Path(external.__file__)) if external else None),
        "runtime": dict(runner_sha256=digest(Path(__file__)), source_sha256=fingerprints,
                        parakit_offsets_sha256=digest(args.parakit / "offsets.py"),
                        torch=str(torch.__version__), stable_baselines3=evaluate_full_credit.evaluate_sft_native.sb3_version),
        "policies": {name: arm.provenance for name, arm in arms.items()},
        "schedule": trial_schedule(arms, args.seeds),
    }))


def main():
    from scripts.evaluate_bomb_crossover import PPO211, PPO397, TEMPLATE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", default=[], metavar="NAME=CHECKPOINT")
    parser.add_argument("--ppo-policy", action="append", default=[], metavar="NAME=ARCHIVE")
    parser.add_argument("--ppo-template", type=Path, default=TEMPLATE)
    parser.add_argument("--crossover", action="append", default=[], metavar="ARM")
    parser.add_argument("--ppo211", type=Path, default=PPO211, help="Bomb source for the published crossover.")
    parser.add_argument("--ppo397", type=Path, default=PPO397, help="Control source for the published crossover.")
    parser.add_argument("--timing-policy", action="append", default=[], metavar="NAME=ANCHOR_ARM=MODEL_JSON")
    parser.add_argument("--target", choices=tuple(TARGETS), default="ending")
    parser.add_argument("--run-arm", action="append", help="Run this arm while keeping its source models loaded.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1601, 1602, 1603, 1604])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parakit", type=Path, default=Path(__file__).resolve().parents[2] / "parakit")
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--focus-wait", type=int, default=30)
    parser.add_argument("--game-over-default-reference", type=Path)
    parser.add_argument("--game-over-retry-reference", type=Path)
    parser.add_argument("--menu-crop", action="append", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="Repeat for the first and last menu row crops in runtime screenshot coordinates.")
    parser.add_argument("--menu-max-error", type=float, default=.035)
    parser.add_argument("--restart-hook", metavar="MODULE:FUNCTION",
                        help="Optional operator-qualified same-process restart for an unsupported screen.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("The stock TH14 1.00b native adapter requires Windows.")
    if args.limit is None:
        args.limit = LIMITS[args.target]
    if min(args.start_frame, args.limit) < 1 or args.focus_wait < 0:
        parser.error("Use positive start frame and limit and a nonnegative focus wait.")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**63 for seed in args.seeds):
        parser.error("Use distinct policy seeds in [0, 2**63).")
    refs = (args.game_over_default_reference, args.game_over_retry_reference, args.menu_crop)
    if any(refs) and (not all(refs) or len(args.menu_crop) < 2) or not 0 < args.menu_max_error < .2:
        parser.error("Supply both verified Game Over screenshots, at least two row crops, and a valid matching error.")
    torch.set_num_threads(1)
    try:
        arms = selections(args)
        if not arms:
            raise ValueError("Specify at least one frozen policy arm.")
        menu = GameOverMenu(*refs, args.menu_max_error) if all(refs) else None
        restart_hook = hook(args.restart_hook) if args.restart_hook else None
        specification = run_specification(args, arms)
        if not args.resume:
            args.out.mkdir(parents=True, exist_ok=False)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, ImportError, AttributeError) as error:
        parser.error(str(error))
    with output_lock(args.out):
        report = evaluate(args, arms, specification, args.out / "report.json", menu, restart_hook)
    print(json.dumps(dict(status=report["status"], matched_blocks=report.get("matched_blocks", 0),
                          summary=report.get("matched_summary", {}),
                          waiting_for_fresh_credit=report["waiting_for_fresh_credit"])), flush=True)


if __name__ == "__main__":
    main()

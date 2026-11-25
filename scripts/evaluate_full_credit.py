"""Evaluate frozen policies from a fresh stock TH14 credit with all native lives."""

import argparse
from collections import Counter
from contextlib import contextmanager
import ctypes
from functools import partial
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch

from scripts import audit_th14_clear_objective, evaluate_sft_native, stage_policy
from scripts.audit_th14_clear_objective import NativeState, classify
from scripts.checkpoint_storage import digest
from scripts.evaluate_sft_native import load_arms, output_lock, restore_report, summarize_arm, trial_schedule
from scripts.evaluate_stage_policies import policy_observation
from scripts.pretrain_policy import ACTION_NAMES, ACTION_WIDTHS
from scripts.train_full_policy import publish_json


RESOURCE_FIELDS = ("lives", "bombs", "power", "life_fragments", "bomb_fragments")
OPENING_RESOURCES = dict(lives=2, bombs=3, power=100, life_fragments=0, bomb_fragments=0)


def resources(env):
    values = {name: env.game.read_game_val(name) for name in RESOURCE_FIELDS}
    if any(value is None for value in values.values()):
        raise RuntimeError("The native credit inventory is unreadable.")
    return values


def continues(env):
    return env.reader.read_uint(env.reader.offsets.statics.continues, relative=True)


def phase_id(snapshot):
    boss = snapshot.boss
    return (snapshot.stage, boss.pointer, boss.max_hp, snapshot.spell_id) if boss is not None else None


def verify_opening(env, frame):
    position = float(env.snapshot.observation["context"][0]) * 192
    if ((env.current_stage(), env.current_screen(), env.game.read_game_val("game_state")) != (1, 7, 2)
            or continues(env) or env.stage_frame() != frame or env.last_frame != frame
            or not np.isfinite(position) or abs(position) > .05):
        raise RuntimeError("The trial did not start at the requested zero-Continue stock Stage 1 opening.")
    opening = resources(env)
    if opening != OPENING_RESOURCES:
        raise RuntimeError(f"Expected stock opening inventory {OPENING_RESOURCES}, got {opening}.")
    return opening


def credit_trial(env, arm, trial, args, reset_callback, record_progress):
    reset_started = perf_counter()
    env.motion_history = arm.history
    observation, reset_info = reset_callback(env, trial)
    initial_resources = verify_opening(env, args.start_frame)
    reset_seconds = perf_counter() - reset_started
    torch.manual_seed(trial["policy_seed"])
    stats = {}
    for name, width in zip(ACTION_NAMES, ACTION_WIDTHS):
        stats[name] = np.zeros(width, dtype=np.int64)
    horizontal = [0, 0]
    x = float(observation["context"][0]) * 192
    x_min = x_max = x
    phase = None
    phases = []
    stage2_entry = None
    edge_decisions = 0
    deaths = 0
    started = perf_counter()
    while True:
        before = env.snapshot
        old_x = x
        action, _ = arm.policy.predict(policy_observation(arm.policy, env, observation), deterministic=False)
        for counts, value in zip(stats.values(), action):
            counts[int(value)] += 1
        observation, _, terminated, truncated, info = env.step(action)
        stage = info["native_stage"]
        continued = continues(env)
        after = env.snapshot
        if stage == after.stage and info["native_pause_state"] == 2:
            x = float(observation["context"][0]) * 192
            x_min, x_max = min(x_min, x), max(x_max, x)
            edge_decisions += int(abs(x) >= 183.5)
        direction = int(action[0]) % 3 - 1
        if direction and before.stage == after.stage == stage and before.player_state == after.player_state == 1:
            horizontal[0] += 1
            horizontal[1] += int(direction * (x - old_x) > .05)
        observed_phase = phase_id(after) if stage == after.stage and info["native_pause_state"] == 2 else None
        milestone = observed_phase != phase
        if milestone:
            if phase is not None:
                phases[-1]["exit"] = dict(frame=info["stage_frame"], last_observed_hp=before.boss.hp,
                                          observed_native_stage=stage, resources=resources(env))
            if observed_phase is not None:
                phases.append(dict(stage=stage, spell_id=after.spell_id, max_hp=after.boss.max_hp,
                                   entry=dict(frame=info["stage_frame"], hp=after.boss.hp, resources=resources(env))))
            phase = observed_phase
        if stage2_entry is None and stage == 2 and not continued:
            stage2_entry = dict(frame=env.stage_frame(), native_pause_state=info["native_pause_state"],
                                resources=resources(env))
            milestone = True
        milestone = milestone or info["misses"] != deaths
        deaths = info["misses"]
        if continued or terminated or truncated or milestone or info["steps"] % 256 == 0:
            row = dict(info, decision_count=info["steps"], deaths=deaths,
                       actual_bombs_consumed=info["bombs_used"], native_continues=continued,
                       start_frame=args.start_frame, reset_info=reset_info, start_resources=initial_resources,
                       reset_seconds=reset_seconds, stage2_entry=stage2_entry,
                       stage2_lives=stage2_entry["resources"]["lives"] if stage2_entry else None,
                       stage2_bombs=stage2_entry["resources"]["bombs"] if stage2_entry else None,
                       boss_phase_events=phases, edge_decisions=edge_decisions, x_min=x_min, x_max=x_max,
                       horizontal_commands_while_active=horizontal[0], horizontal_responses=horizontal[1],
                       action_counts={name: values.tolist() for name, values in stats.items()},
                       seconds=perf_counter() - started)
            if continued:
                env.release()
                row.update(outcome="continued_credit_invalid", stop_on_continue=True)
                record_progress(row)
                raise RuntimeError("Native Continue counter advanced. This credit was stopped and excluded.")
            if terminated or truncated:
                env.release()
                inventory = resources(env)
                native = NativeState(stage, env.current_screen(), info["native_pause_state"], inventory["lives"],
                                     continued)
                raw_terminal_frame = env.stage_frame()
                row.update(terminal_resources=inventory, native_game_screen=native.screen,
                           native_terminal_stage_frame=raw_terminal_frame)
                record_progress(row)
                outcome = classify("stage", 1, native, training_limit_reached=truncated)
                if outcome == "ongoing":
                    raise RuntimeError("Native adapter ended an otherwise active credit.")
                failure_frame = raw_terminal_frame or info["stage_frame"]
                row.update(outcome=outcome, native_adapter_outcome=info["outcome"],
                           failure_frame=failure_frame if outcome == "game_over" else None,
                           stopped_before_continue=outcome == "game_over", game_over=outcome == "game_over",
                           reached_goal=outcome == "clear")
                return row
            record_progress(row)


def summarize_credit(episodes):
    summary = summarize_arm(episodes)
    at_stage2 = [episode for episode in episodes if episode["stage2_entry"] is not None]
    summary.update(clears=sum(row["reached_goal"] for row in episodes),
                   game_overs=sum(row["game_over"] for row in episodes),
                   time_limits=sum(row["outcome"] == "time_limit" for row in episodes),
                   stage2_entries=len(at_stage2),
                   median_stage2_lives=median(row["stage2_lives"] for row in at_stage2) if at_stage2 else None,
                   median_stage2_bombs=median(row["stage2_bombs"] for row in at_stage2) if at_stage2 else None,
                   deaths=sum(row["deaths"] for row in episodes),
                   actual_bombs_consumed=sum(row["actual_bombs_consumed"] for row in episodes),
                   edge_decisions=sum(row["edge_decisions"] for row in episodes))
    return summary


def refresh_summary(report, arms):
    blocks = len(report["completed"]) // len(arms)
    prefix = report["completed"][:blocks * len(arms)]
    report.update(matched_blocks=blocks, matched_summary={
        name: summarize_credit([row["result"] for row in prefix if row["trial"]["arm"] == name]) for name in arms
    })


@contextmanager
def native_stage(args, arms):
    from scripts.full_credit_reset import prepare_fresh_credit

    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState
    continuous = 0x80000000
    keep_awake(continuous | 0x00000001 | 0x00000002)
    try:
        prepare_fresh_credit(args.parakit)
        longest_history = max([12] + [n for arm in arms.values() for n in (arm.policy.motion_spans or [])])
        stage = stage_policy.NormalStageOne(
            args.parakit, limit=args.limit, max_misses=float("inf"), prepared=True,
            motion_history=any(arm.history for arm in arms.values()),
            history_frames=longest_history, focus_wait=args.focus_wait,
        )
        try:
            yield stage
        finally:
            stage.close()
    finally:
        keep_awake(continuous)


def run_evaluation(args, arms, path, specification, stage_session=native_stage, reset_callback=None):
    if reset_callback is None:
        from scripts.full_credit_reset import fresh_credit_reset

        reset_callback = partial(fresh_credit_reset, start_frame=args.start_frame)
    if args.resume:
        report = restore_report(path, specification)
        if report.get("in_progress") is not None:
            report["interrupted_attempts"].append(report["in_progress"])
    else:
        report = dict(specification=specification, completed=[], restarts=[], interrupted_attempts=[])
    report.update(pending=None, in_progress=None, error=None)
    refresh_summary(report, arms)
    remaining = specification["schedule"][len(report["completed"]):]
    report["status"] = "evaluating" if remaining else "complete"
    publish_json(path, report)
    if not remaining:
        return report
    try:
        with stage_session(args, arms) as stage:
            for trial in remaining:
                report["pending"] = trial
                publish_json(path, report)
                print(json.dumps(dict(status="trial", **trial)), flush=True)

                def record_progress(result):
                    report["in_progress"] = dict(trial=trial, result=result)
                    publish_json(path, report)

                result = credit_trial(stage, arms[trial["arm"]], trial, args, reset_callback, record_progress)
                report["completed"].append(dict(trial=trial, result=result))
                report.update(pending=None, in_progress=None)
                refresh_summary(report, arms)
                publish_json(path, report)
                fields = ("outcome", "first_miss_frame", "decision_count", "deaths", "actual_bombs_consumed",
                          "failure_frame", "stage2_lives", "stage2_bombs")
                print(json.dumps(dict(status="completed", **trial, **{key: result[key] for key in fields})), flush=True)
    except BaseException as error:
        report.update(status="interrupted", error=dict(type=type(error).__name__, detail=str(error)))
        publish_json(path, report)
        raise
    report["status"] = "complete"
    publish_json(path, report)
    return report


def run_specification(args, arms):
    from scripts import full_credit_reset

    sources = (audit_th14_clear_objective, evaluate_sft_native, full_credit_reset, stage_policy)
    return {
        "schema": 1,
        "task": dict(title="TH14", version="1.00b", difficulty="Normal", shot="Reimu A", first_stage=1,
                     last_stage=1, start_frame=args.start_frame, decision_limit=args.limit,
                     frames_per_action=4, all_native_lives=True, allows_continue=False,
                     expected_start_resources=OPENING_RESOURCES),
        "controller": dict(kind="frozen_sft_and_ppo", decode="sampled", geometry=False, aim_correction=False,
                           automatic_bomb=False, native_rng_seed_controlled=False,
                           bomb_accounting="inventory decreases on steps without life loss"),
        "runtime": dict(runner_sha256=digest(Path(__file__)),
                        source_sha256={module.__name__: digest(Path(module.__file__)) for module in sources},
                        parakit_offsets_sha256=digest(args.parakit / "offsets.py"),
                        torch=str(torch.__version__), stable_baselines3=evaluate_sft_native.sb3_version, device="cpu"),
        "policies": {name: arm.provenance for name, arm in arms.items()},
        "schedule": trial_schedule(arms, args.seeds),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", default=[], metavar="NAME=CHECKPOINT")
    parser.add_argument("--ppo-policy", action="append", default=[], metavar="NAME=ARCHIVE")
    parser.add_argument("--ppo-template", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1201, 1205)))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parakit", type=Path, default=Path(__file__).resolve().parents[2] / "parakit")
    parser.add_argument("--limit", type=int, default=3300, help="Decision cap for the Stage 1 credit attempt.")
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--focus-wait", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not (args.policy or args.ppo_policy) or bool(args.ppo_policy) != bool(args.ppo_template):
        parser.error("Specify at least one arm and provide --ppo-template with --ppo-policy.")
    if args.limit < 1 or args.start_frame < 1 or args.focus_wait < 0:
        parser.error("Use a positive decision limit and starting frame, and a nonnegative focus wait.")
    if len(args.seeds) != len(set(args.seeds)) or any(not 0 <= seed < 2**63 for seed in args.seeds):
        parser.error("Use distinct policy seeds in [0, 2**63).")
    if os.name != "nt":
        parser.error("The native TH14 1.00b game adapter requires Windows.")
    torch.set_num_threads(1)
    try:
        arms = load_arms(args.policy, args.ppo_policy, args.ppo_template)
        specification = run_specification(args, arms)
        if not args.resume:
            args.out.mkdir(parents=True, exist_ok=False)
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    with output_lock(args.out):
        report = run_evaluation(args, arms, args.out / "report.json", specification)
    print(json.dumps(dict(matched_blocks=report["matched_blocks"], summary=report["matched_summary"])), flush=True)


if __name__ == "__main__":
    main()

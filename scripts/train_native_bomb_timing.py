"""Collect randomized native Bomb trials, fit a contextual head, and evaluate whole credits."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from scripts import evaluate_full_credit as credit
from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import HASHES, PPO397, REPO, TEMPLATE
from scripts.evaluate_sft_native import Arm, load_arms, output_lock, restore_report
from scripts.evaluate_stage_policies import policy_observation
from scripts.full_credit_reset import fresh_credit_reset


BASE = "ppo397"
LEARNED = "native_bomb_timing"
OBSERVATION = ("raw_native_player_state", "previous_bomb_key_held", "native_bomb_stock")
SCHEMA = "th14-stock-1.00b-normal-reimu-a-stage1-bomb-intervention-v1"


def native_bomb_features(env):
    state = env.snapshot.player_state
    stock = env.game.read_game_val("bombs")
    if state is None or stock is None or stock < 0:
        raise RuntimeError("The native Bomb head state or inventory is unreadable.")
    return int(state), int("x" in env.held), int(stock)


def native_transition_state(env):
    tick = env.game.read_game_val("global_timer")
    lives = env.game.read_game_val("lives")
    stock = env.game.read_game_val("bombs")
    if tick is None or lives is None or stock is None:
        raise RuntimeError("The state during a native Bomb intervention is unreadable.")
    return {
        "global_tick": int(tick), "stage_frame": env.stage_frame(), "lives": int(lives),
        "bombs": int(stock), "player_state": int(env.snapshot.player_state),
    }


def strata(features):
    """Back off on inventory count while retaining player state and held key."""
    state, held, stock = features
    yield f"{state}:{held}:exact:{stock}"
    yield f"{state}:{held}:available:{int(stock > 0)}"


def record_label(window, outcome, deaths, consumed, frames, *, final):
    elapsed = window["transitions"][-1]["after"]["global_tick"] - window["before"]["global_tick"]
    window.update(elapsed_native_ticks=elapsed, native_life_losses=deaths - window["prior_deaths"],
                  actual_bombs_consumed=consumed - window["prior_bombs_used"], terminal_outcome=outcome)
    exact = all(transition["global_ticks"] == 4 for transition in window["transitions"])
    absorbing = outcome in ("clear", "game_over")
    window["label_valid"] = exact and (elapsed == frames or absorbing and 0 < elapsed < frames)
    if not window["label_valid"]:
        window["label_rejection"] = "clock_mismatch" if not exact else "cutoff" if final else "duration"
    return window


def collect_credit(env, arm, trial, args):
    env.motion_history = arm.history
    observation, reset = fresh_credit_reset(env, trial, start_frame=args.start_frame)
    opening = credit.verify_opening(env, args.start_frame)
    seed = trial["policy_seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(np.random.SeedSequence([args.randomization_seed, seed]))
    windows, window, rare_states = [], None, set(args.rare_state)
    deaths = consumed = 0
    rare_opportunities = 0
    while True:
        action, _ = arm.policy.predict(policy_observation(arm.policy, env, observation), deterministic=False)
        action = np.asarray(action, dtype=np.int64).copy()
        base_bomb = int(action[3])
        if window is None:
            features = native_bomb_features(env)
            player_state, _, stock = features
            rare = player_state in rare_states
            probability = args.rare_rate if rare else args.common_rate
            rare_opportunities += int(rare and stock > 0)
            if stock > 0 and rng.random() < probability:
                bomb = int(rng.integers(2))
                window = dict(features=features, rare=rare, assignment_probability=.5,
                              enrollment_probability=probability, bomb=bomb, base_bomb=base_bomb,
                              first_decision=env.steps, before=native_transition_state(env),
                              prior_deaths=deaths, prior_bombs_used=consumed, transitions=[])
        if window is not None:
            before = native_transition_state(env)
            action[3] = window["bomb"] if not window["transitions"] else 0
        observation, _, terminated, truncated, info = env.step(action)
        deaths, consumed = info["misses"], info["bombs_used"]
        continued = credit.continues(env)
        if continued:
            raise RuntimeError("Native Continue advanced during Bomb intervention collection.")
        final = terminated or truncated
        outcome = None
        if final:
            native = credit.NativeState(info["native_stage"], env.current_screen(), info["native_pause_state"],
                                        env.game.read_game_val("lives"), continued)
            outcome = credit.classify("stage", 1, native, training_limit_reached=truncated)
            if outcome == "ongoing":
                raise RuntimeError("Native adapter stopped outside a recognized stage outcome.")
        if window is not None:
            after = native_transition_state(env)
            window["transitions"].append(dict(before=before, after=after,
                                               action=action.tolist(), base_bomb=base_bomb,
                                               global_ticks=after["global_tick"] - before["global_tick"]))
            elapsed = after["global_tick"] - window["before"]["global_tick"]
            if final or elapsed >= args.label_frames or len(window["transitions"]) >= args.label_frames // 4:
                windows.append(record_label(window, outcome, deaths, consumed, args.label_frames, final=final))
                window = None
        if final:
            env.release()
            return {
                "reset": reset, "opening_resources": opening, "outcome": outcome,
                "decision_count": info["steps"], "stage_frame": info["stage_frame"],
                "first_miss_frame": info["first_miss_frame"], "deaths": deaths,
                "actual_bombs_consumed": consumed, "rare_enrollment_opportunities": rare_opportunities,
                "windows": windows,
            }


def collect(args, arm):
    arms = {BASE: arm}
    specification = credit.run_specification(args, arms)
    specification["controller"]["kind"] = "isolated_randomized_native_bomb_interventions"
    specification["runtime"]["source_sha256"][__name__] = digest(Path(__file__))
    specification["intervention"] = {
        "schema": SCHEMA, "observation": OBSERVATION,
        "rare_player_states": args.rare_state, "rare_enrollment_probability": args.rare_rate,
        "other_enrollment_probability": args.common_rate, "assignment_probability": .5,
        "assignment_seed": args.randomization_seed, "label_native_frames": args.label_frames,
        "followup_bomb_bit": 0, "requires_positive_native_bomb_stock": True,
        "bomb_counting": "Native adapter inventory decreases on steps without a life loss.",
    }
    # Normalize the specification before comparing an archived resume report.
    specification = json.loads(json.dumps(specification))
    path = args.out / "report.json"
    if not args.resume:
        args.out.mkdir(parents=True, exist_ok=False)
    with output_lock(args.out):
        report = restore_report(path, specification) if args.resume else dict(
            specification=specification, completed=[], restarts=[], pending=None,
        )
        pending = specification["schedule"][len(report["completed"]):]
        report.update(status="collecting" if pending else "complete", error=None)
        credit.publish_json(path, report)
        if not pending:
            return
        try:
            with credit.native_stage(args, arms) as stage:
                for trial in pending:
                    report["pending"] = trial
                    credit.publish_json(path, report)
                    result = collect_credit(stage, arm, trial, args)
                    report["completed"].append(dict(trial=trial, result=result))
                    report["pending"] = None
                    credit.publish_json(path, report)
                    print(json.dumps(dict(trial=trial, outcome=result["outcome"],
                                          windows=len(result["windows"]),
                                          accepted=sum(w["label_valid"] for w in result["windows"]))), flush=True)
        except BaseException as error:
            report.update(status="interrupted", error=dict(type=type(error).__name__, detail=str(error)))
            credit.publish_json(path, report)
            raise
        report["status"] = "complete"
        credit.publish_json(path, report)


def fit_head(reports, min_per_action, bomb_cost, margin):
    """Estimate action values only from matched-stratum randomized observations."""
    groups = defaultdict(lambda: [[], []])
    totals = dict(windows=0, accepted=0, rejected=0)
    seen_episodes = set()
    anchors = set()
    tasks = []
    for report in reports:
        specification = report["specification"]
        intervention = specification["intervention"]
        if (report["status"] != "complete" or intervention["schema"] != SCHEMA
                or [row["trial"] for row in report["completed"]] != specification["schedule"]):
            raise ValueError("Fit requires completed randomized native intervention reports.")
        task = {field: intervention[field] for field in ("label_native_frames", "followup_bomb_bit",
                                                         "requires_positive_native_bomb_stock")}
        tasks.append(task)
        anchors.add(specification["policies"][BASE]["weights_sha256"])
        for record in report["completed"]:
            episode = (intervention["assignment_seed"], record["trial"]["policy_seed"])
            if episode in seen_episodes:
                raise ValueError(f"Duplicate randomized credit {episode}.")
            seen_episodes.add(episode)
            for row in record["result"]["windows"]:
                totals["windows"] += 1
                if not row["label_valid"]:
                    totals["rejected"] += 1
                    continue
                if (row["assignment_probability"] != .5 or row["bomb"] not in (0, 1)
                        or len(row["features"]) != len(OBSERVATION) or row["features"][-1] <= 0):
                    raise ValueError("Unexpected intervention support or assignment probability.")
                targets = (int(row["native_life_losses"] > 0), row["actual_bombs_consumed"])
                for key in strata(row["features"]):
                    groups[key][row["bomb"]].append(targets)
                totals["accepted"] += 1
    if not tasks or any(task != tasks[0] for task in tasks) or len(anchors) != 1:
        raise ValueError("The intervention reports must share a label horizon and frozen PPO anchor.")

    table = {}
    for key, actions in sorted(groups.items()):
        evidence = []
        for samples in actions:
            if not samples:
                evidence.append(dict(count=0))
                continue
            losses, bombs = zip(*samples)
            # Jeffreys smoothing stabilizes rare binary losses. Bomb cost uses
            # measured inventory consumption, never an issued Bomb command.
            life_loss_probability = (sum(losses) + .5) / (len(samples) + 1)
            expected_bombs = mean(bombs)
            evidence.append(dict(count=len(samples), losses=sum(losses),
                                 life_loss_probability=life_loss_probability, expected_bombs=expected_bombs,
                                 value=-life_loss_probability - bomb_cost * expected_bombs))
        choice = None
        if all(arm["count"] >= min_per_action for arm in evidence):
            choice = int(evidence[1]["value"] - evidence[0]["value"] > margin)
        table[key] = dict(actions=evidence, choice=choice)

    return dict(schema=SCHEMA, observation=OBSERVATION, anchor_sha256=anchors.pop(),
                label=tasks[0], counts=totals, min_per_action=min_per_action, bomb_cost=bomb_cost,
                margin=margin, table=table)


class NativeBombPolicy:
    def __init__(self, policy, model):
        self.policy, self.model = policy, model
        self.motion_spans = policy.motion_spans
        self.retain_recent_motion = policy.retain_recent_motion
        self.stage_clock = policy.stage_clock
        self.environment = None

    def predict(self, observation, deterministic=False):
        action, _ = self.policy.predict(observation, deterministic=deterministic)
        action = np.asarray(action, dtype=np.int64).copy()
        for key in strata(native_bomb_features(self.environment)):
            record = self.model["table"].get(key)
            if record is not None and record["choice"] is not None:
                action[3] = record["choice"]
                break
        return action, None


def load_native_bomb_arm(base, model_path):
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if (model["schema"] != SCHEMA or tuple(model["observation"]) != OBSERVATION
            or model["anchor_sha256"] != base.provenance["weights_sha256"]):
        raise ValueError("Bomb head does not match the observation contract or frozen PPO anchor.")
    if not any(row["choice"] is not None for row in model["table"].values()):
        raise ValueError("No Bomb stratum has enough randomized support to evaluate a changed head.")
    if any(row["choice"] not in (None, 0, 1) for row in model["table"].values()):
        raise ValueError("Expected binary choices in the fitted Bomb table.")
    candidate = NativeBombPolicy(base.policy, model)
    return Arm(candidate, base.history, {
        "kind": "frozen_ppo_with_fitted_native_bomb_head", "anchor_sha256": model["anchor_sha256"],
        "model_sha256": digest(model_path), "observation_extension": OBSERVATION,
        "unsupported_context": "frozen_anchor_bomb_action",
    })


def evaluate(args, base):
    learned = load_native_bomb_arm(base, args.model)
    candidate = learned.policy
    arms = {BASE: base, LEARNED: learned}

    @contextmanager
    def stage_session(args, arms):
        with credit.native_stage(args, arms) as stage:
            candidate.environment = stage
            try:
                yield stage
            finally:
                candidate.environment = None

    specification = credit.run_specification(args, arms)
    specification["controller"]["kind"] = "frozen_ppo_with_fitted_native_bomb_head"
    specification["runtime"]["source_sha256"][__name__] = digest(Path(__file__))
    specification = json.loads(json.dumps(specification))
    if not args.resume:
        args.out.mkdir(parents=True, exist_ok=False)
    with output_lock(args.out):
        report = credit.run_evaluation(args, arms, args.out / "report.json", specification,
                                       stage_session=stage_session)
    print(json.dumps(report["matched_summary"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    collection = actions.add_parser("collect")
    training = actions.add_parser("fit")
    evaluation = actions.add_parser("evaluate")
    for native, seeds in ((collection, list(range(1601, 1609))), (evaluation, list(range(1701, 1705)))):
        native.add_argument("--out", required=True, type=Path)
        native.add_argument("--ppo397", type=Path, default=PPO397)
        native.add_argument("--ppo-template", type=Path, default=TEMPLATE)
        native.add_argument("--seeds", type=int, nargs="+", default=seeds)
        native.add_argument("--parakit", type=Path, default=REPO.parent / "parakit")
        native.add_argument("--limit", type=int, default=3300)
        native.add_argument("--start-frame", type=int, default=30)
        native.add_argument("--focus-wait", type=int, default=30)
        native.add_argument("--resume", action="store_true")
    collection.add_argument("--rare-state", nargs="+", type=int, default=[4])
    collection.add_argument("--rare-rate", type=float, default=1.)
    collection.add_argument("--common-rate", type=float, default=.002)
    collection.add_argument("--label-frames", type=int, default=16)
    collection.add_argument("--randomization-seed", type=int, default=404)
    training.add_argument("--data", type=Path, nargs="+", required=True)
    training.add_argument("--out", type=Path, required=True, help="Output fitted model JSON path.")
    training.add_argument("--min-per-action", type=int, default=3)
    training.add_argument("--bomb-cost", type=float, default=.05)
    training.add_argument("--margin", type=float, default=.02)
    evaluation.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "fit":
        if args.min_per_action < 1 or any(not math.isfinite(x) or x < 0 for x in (args.bomb_cost, args.margin)):
            parser.error("Expected positive support and nonnegative finite costs.")
        paths = args.data
        fingerprints = [digest(path) for path in paths]
        if len(set(fingerprints)) != len(paths):
            parser.error("Duplicate intervention inputs.")
        model = fit_head([json.loads(path.read_text(encoding="utf-8")) for path in paths],
                         args.min_per_action, args.bomb_cost, args.margin)
        model["input_reports_sha256"] = fingerprints
        model["trainer_sha256"] = digest(Path(__file__))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists():
            parser.error("Use a fresh model path for each fit.")
        credit.publish_json(args.out, model)
        print(json.dumps(dict(**model["counts"], supported=sum(row["choice"] is not None for row in
                                                                model["table"].values()),
                              model_sha256=digest(args.out))), flush=True)
        return
    if os.name != "nt":
        parser.error("The native mode requires Windows and stock TH14 1.00b.")
    if (args.limit < 1 or args.start_frame < 1 or args.focus_wait < 0
            or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**63 for seed in args.seeds)):
        parser.error("Use valid timing settings and distinct nonnegative seeds.")
    if args.command == "collect" and (args.randomization_seed < 0 or args.label_frames < 4 or args.label_frames % 4
                                      or not 0. <= args.common_rate <= args.rare_rate <= 1.):
        parser.error("Use a four-tick label horizon and valid ordered enrollment probabilities.")
    torch.set_num_threads(1)
    base = load_arms([], [f"{BASE}={args.ppo397}"], args.ppo_template)[BASE]
    if base.provenance["weights_sha256"] != HASHES[BASE]:
        parser.error("Expected the published frozen PPO-397 checkpoint.")
    if args.command == "collect":
        collect(args, base)
    else:
        evaluate(args, base)


if __name__ == "__main__":
    main()

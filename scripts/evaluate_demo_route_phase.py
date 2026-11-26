"""Evaluate an Alice route locked to native TH14 boss phases with an independent Shoot head."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts import evaluate_credit_story, fit_randomized_bomb_value, train_demo_route, train_native_bomb_timing
from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import TEMPLATE
from scripts.evaluate_sft_native import Arm, load_arm, output_lock
from scripts.evaluate_stage_policies import policy_observation
from scripts.native_story_restart import GameOverMenu
from scripts.pretrain_policy import predict_heads
from scripts.train_demo_route import DemoPolicy, SCHEMA
from scripts.train_full_policy import publish_json


def phase_knots(demo, expected_hash, actual_frames):
    if digest(demo) != expected_hash:
        raise ValueError("The phase source must be the exact cleared Alice training shard.")
    with np.load(demo, allow_pickle=False) as data:
        frames = data["stage_frame"]
        hp = np.rint(data["context"][:, 8] * 2000).astype(int)
    if not np.array_equal(actual_frames, frames):
        raise ValueError("The route artifact does not contain the hashed source timeline.")
    present = frames[hp > 0]
    meetings = np.split(present, np.flatnonzero(np.diff(present) > 64) + 1)
    meetings = [row for row in meetings if len(row) >= 16]
    if len(meetings) != 2:
        raise ValueError("Expected one visible midboss and one final boss in the training credit.")

    def first(mask):
        rows = np.flatnonzero(mask)
        if not len(rows):
            raise ValueError("The training credit lacks a visible HP transition knot.")
        return int(frames[rows[0]])

    mid, main = meetings
    previous_hp = np.r_[0, hp[:-1]]
    spell4 = first((frames > main[0]) & (hp == 1500) & (previous_hp > 1500))
    nonspell2 = first((frames > spell4) & (hp == 2000) & (previous_hp > 0) & (previous_hp < 100))
    spell8 = first((frames > nonspell2) & (hp == 1900) & (previous_hp > 1900))
    end = first((frames > spell8) & (hp == 0) & (previous_hp > 0))
    return {
        "midboss": [int(mid[0]), int(mid[-1] + 4)],
        "nonspell1": [int(main[0]), spell4],
        "spell4": [spell4, nonspell2],
        "nonspell2": [nonspell2, spell8],
        "spell8": [spell8, end],
    }


def native_phase(snapshot):
    boss = snapshot.boss
    if boss is None or snapshot.stage != 1:
        return None
    if snapshot.spell_id == 4 and boss.max_hp == 11300:
        return "spell4"
    if snapshot.spell_id == 8 and boss.max_hp == 11400:
        return "spell8"
    if snapshot.spell_id is None:
        return {9000: "midboss", 11300: "nonspell1", 11400: "nonspell2"}.get(boss.max_hp)
    return None


class PhaseRoute(DemoPolicy):
    def __init__(self, checkpoint, route, bomb, shoot, mode):
        super().__init__(checkpoint, route, bomb_model=bomb)
        self.shoot_source = shoot
        self.mode = mode
        self.knots = None
        self.motion_spans = shoot.policy.motion_spans
        self.retain_recent_motion = shoot.policy.retain_recent_motion
        self.sessions = []
        self.credit = None

    def phase_time(self, frame, stage, boss):
        env = self.environment
        if self.credit != env.reset_count:
            self.credit = env.reset_count
            self.sessions.append(dict(reset_count=self.credit, events=[], by_phase={},
                                      total_combat_decisions=0, fallback_queries=0, fallback_fire=0))
            self.phase = None
            self.offset = 0
        self.expired_phase = False
        self.last_live_phase = native_phase(env.snapshot)
        if self.mode == "shoot-only" or stage != 1:
            self.actual_reference_frame = super().phase_time(frame, stage, boss)
            return self.actual_reference_frame

        phase = self.last_live_phase
        if self.phase != phase:
            if phase is None and self.phase is not None:
                # Preserve the timing of the opening waves following the native midboss exit.
                self.offset = self.knots[self.phase][1] - frame
            self.phase = phase
            if phase is not None:
                self.phase_live_start = frame
                self.sessions[-1]["events"].append(dict(phase=phase, native_start_frame=frame,
                                                       reference_frame_interval=self.knots[phase]))
        if phase is None:
            self.actual_reference_frame = frame + self.offset
        else:
            start, stop = self.knots[phase]
            uncapped = start + (frame - self.phase_live_start)
            self.expired_phase = uncapped >= stop - 4
            self.actual_reference_frame = min(uncapped, stop - 4)
        return self.actual_reference_frame

    @torch.inference_mode()
    def predict(self, observation, deterministic=False):
        action, _ = super().predict(observation, deterministic)
        env = self.environment
        boss = env.snapshot.boss
        if boss is None or boss.max_hp <= 40 or not boss.vulnerable:
            return action, None
        session = self.sessions[-1]
        session["total_combat_decisions"] += 1
        name = self.last_live_phase or f"stage{env.snapshot.stage}_spell{env.snapshot.spell_id}_hp{boss.max_hp}"
        row = session["by_phase"].setdefault(name, dict(decisions=0, native_start=env.last_frame,
                                                        native_end=env.last_frame, fallback=0, fire=0,
                                                        fallback_fire=0, min_hp=boss.hp))
        row["decisions"] += 1
        row["native_end"] = env.last_frame
        row["min_hp"] = min(row["min_hp"], boss.hp)
        # This replaces only the Shoot bit. Both inputs and weights belong to a frozen learned policy.
        if self.expired_phase or not action[2]:
            arm = self.shoot_source
            transformed = policy_observation(arm.policy, env, observation, uses_history=arm.history)
            tensors, _ = arm.policy.obs_to_tensor(transformed)
            logits = predict_heads(arm.policy, tensors["bullets"], tensors["context"])[2]
            action[2] = int(logits.argmax(dim=-1).item())
            row["fallback"] += 1
            row["fallback_fire"] += int(action[2])
            session["fallback_queries"] += 1
            session["fallback_fire"] += int(action[2])
        row["fire"] += int(action[2])
        return action, None


def evaluate(args):
    actor_report = json.loads((args.artifact / "report.json").read_text(encoding="utf-8"))
    checkpoint, route = args.artifact / "model.pt", args.artifact / "route.npz"
    if actor_report["schema"] != SCHEMA or (digest(checkpoint), digest(route)) != (
            actor_report["model_sha256"], actor_report["route_sha256"]):
        raise ValueError("The demo artifacts do not match their archived training report.")
    bomb = json.loads(args.bomb_model.read_text(encoding="utf-8")) if args.bomb_model else None
    shoot = load_arm(args.shoot_policy)
    with np.load(route, allow_pickle=False) as archive:
        knots = phase_knots(args.demo, actor_report["train_stage_shards"]["1"], archive["frame_1"])
    arms = {}
    for mode in args.modes:
        policy = PhaseRoute(checkpoint, route, bomb, shoot, mode)
        policy.knots = knots
        provenance = dict(mode=mode, demo_replay=actor_report["train_replay"], demo_sha256=digest(args.demo),
                          route_sha256=digest(route), model_sha256=digest(checkpoint), phase_knots=knots,
                          beyond_reference_movement="Frozen neural route endpoint hold in the matching native phase.",
                          independent_shoot=shoot.provenance,
                          shoot_selection="Greedy frozen SFT Shoot only when vulnerable combat outlasts phase or route Shoot is off.",
                          learned_bomb_value_sha256=digest(args.bomb_model) if args.bomb_model else None,
                          learned_bomb_training_anchor=bomb["anchor_sha256"] if bomb else None)
        arms[mode] = Arm(policy, shoot.history, provenance)
    args.crossover = args.timing_policy = []
    args.restart_hook = None
    spec = evaluate_credit_story.run_specification(args, arms)
    spec["controller"].update(kind="demo_route_native_phase_and_frozen_shoot", allows_continue=False,
                              later_stages="encounter-only route alignment until verified phase knots exist",
                              no_runtime_geometry=True)
    spec["runtime"]["runner_sha256"] = digest(Path(__file__))
    for source in (evaluate_credit_story, train_demo_route, fit_randomized_bomb_value, train_native_bomb_timing):
        spec["runtime"]["source_sha256"][source.__name__] = digest(Path(source.__file__))
    from scripts.native_story_transitions import configure

    archive = configure(args, spec)
    refs = (args.game_over_default_reference, args.game_over_retry_reference, args.menu_crop)
    menu = GameOverMenu(*refs, args.menu_max_error) if all(refs) else None
    if not args.resume:
        args.out.mkdir(parents=True, exist_ok=False)
    telemetry = args.out / "controller-phase-telemetry.json"
    sessions = json.loads(telemetry.read_text(encoding="utf-8")) if args.resume and telemetry.is_file() else []
    with output_lock(args.out):
        try:
            report = evaluate_credit_story.evaluate(args, arms, spec, args.out / "report.json", menu, None,
                                                    transition_archive=archive)
        finally:
            sessions.append({name: arm.policy.sessions for name, arm in arms.items()})
            publish_json(telemetry, sessions)
    print(json.dumps(dict(status=report["status"], summary=report.get("matched_summary"),
                          waiting_for_fresh_credit=report.get("waiting_for_fresh_credit"))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True, help="Hashed TH14 Alice Stage 1 training NPZ.")
    parser.add_argument("--shoot-policy", type=Path, default=TEMPLATE)
    parser.add_argument("--bomb-model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parakit", type=Path, default=Path(__file__).resolve().parents[2] / "parakit")
    parser.add_argument("--modes", choices=("phase-aware", "shoot-only"), nargs="+", default=["phase-aware"])
    parser.add_argument("--target", choices=tuple(evaluate_credit_story.TARGETS), default="stage2")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--focus-wait", type=int, default=90)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2201, 2202])
    parser.add_argument("--game-over-default-reference", type=Path)
    parser.add_argument("--game-over-retry-reference", type=Path)
    parser.add_argument("--menu-crop", action="append", nargs=4, type=int)
    parser.add_argument("--menu-max-error", type=float, default=.035)
    parser.add_argument("--transition-budget-mib", type=int, default=64)
    parser.add_argument("--transition-reserve-mib", type=int, default=20480)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("The stock TH14 1.00b native adapter requires Windows.")
    if args.limit is None:
        args.limit = evaluate_credit_story.LIMITS[args.target]
    refs = (args.game_over_default_reference, args.game_over_retry_reference, args.menu_crop)
    if (min(args.start_frame, args.limit, args.transition_budget_mib) < 1 or args.focus_wait < 0
            or args.transition_reserve_mib < 0 or not 0 < args.menu_max_error < .2
            or any(refs) and (not all(refs) or len(args.menu_crop) < 2)
            or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**63 for seed in args.seeds)
            or len(set(args.modes)) != len(args.modes)):
        parser.error("Use valid episode limits, unique modes/seeds, and complete optional menu references.")
    torch.set_num_threads(1)
    evaluate(args)


if __name__ == "__main__":
    main()

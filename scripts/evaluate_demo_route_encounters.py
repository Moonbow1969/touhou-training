"""Evaluate an Alice route with stock Stage 1 phase knots and later-stage encounter clocks."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts import evaluate_credit_story, evaluate_demo_route_phase, fit_randomized_bomb_value, train_demo_route
from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import TEMPLATE
from scripts.evaluate_demo_route_phase import PhaseRoute, phase_knots
from scripts.evaluate_sft_native import Arm, load_arm, output_lock
from scripts.native_story_restart import GameOverMenu
from scripts.native_story_transitions import configure
from scripts.train_demo_route import ALICE, SCHEMA
from scripts.train_full_policy import publish_json


class EncounterClock:
    """Align only visual boss encounters. Clipped replay HP cannot identify every spell."""

    def __init__(self, spans):
        self.spans = [tuple(map(int, span)) for span in spans]
        self.next_index = 0
        self.active = None
        self.last_native = None
        self.last_reference = None
        self.offset = 0
        self.events = []

    def locate(self, native, visible, vulnerable):
        if self.last_native is not None and native < self.last_native:
            raise RuntimeError("The native stage clock moved backward within one credit stage.")
        self.last_native = native
        if self.active is not None and visible:
            self.last_seen = native
        if (self.active is not None and self.next_index < len(self.spans) and not visible
                and native - self.last_seen >= 96):
            start, stop = self.spans[self.active]
            self.offset = stop - self.last_seen
            self.events.append(dict(kind="observable_boss_exit", encounter=self.active,
                                    native_last_visible=self.last_seen, native_confirmation=native,
                                    reference_last_visible=stop, new_offset=self.offset))
            self.active = None
        if self.active is None and vulnerable and self.next_index < len(self.spans):
            index = self.next_index
            start, stop = self.spans[index]
            if abs(native + self.offset - start) <= 480:
                self.active = index
                self.next_index += 1
                self.active_from = self.last_seen = native
                self.events.append(dict(kind="vulnerable_boss_onset", encounter=index, native_frame=native,
                                        reference_visible_frame=start, previous_offset=self.offset))
        tail = False
        if self.active is None:
            reference = native + self.offset
            if self.next_index < len(self.spans):
                upcoming_start = self.spans[self.next_index][0]
                if upcoming_start <= reference <= upcoming_start + 480:
                    reference = upcoming_start - 4
        else:
            start, stop = self.spans[self.active]
            target = start + native - self.active_from
            tail = target >= stop - 4
            reference = min(stop - 4, target)
        if self.last_reference is not None:
            reference = max(self.last_reference, reference)
        self.last_reference = reference
        return reference, tail


class EncounterPhaseRoute(PhaseRoute):
    def __init__(self, checkpoint, route, bomb, shoot, align):
        super().__init__(checkpoint, route, bomb, shoot, "phase-aware")
        self.align = align
        self.encounter_scope = None
        self.later_stage_sessions = []

    def phase_time(self, frame, stage, boss):
        original = super().phase_time(frame, stage, boss)
        if stage == 1 or not self.align:
            return original
        scope = (self.environment.reset_count, stage)
        if self.encounter_scope != scope:
            self.encounter_scope = scope
            self.later_clock = EncounterClock(self.route[f"encounters_{stage}"])
            self.later_stage_sessions.append(dict(credit=scope[0], stage=stage, events=self.later_clock.events,
                                                  tail_decisions=0, active_without_alignment=0))
        native = self.environment.snapshot.boss
        visible = native is not None and native.hp > 0
        vulnerable = visible and native.vulnerable
        self.actual_reference_frame, self.expired_phase = self.later_clock.locate(frame, visible, vulnerable)
        self.later_stage_sessions[-1]["tail_decisions"] += int(self.expired_phase)
        self.later_stage_sessions[-1]["active_without_alignment"] += int(vulnerable and self.later_clock.active is None)
        return self.actual_reference_frame


def make_arms(args):
    report = json.loads((args.artifact / "report.json").read_text(encoding="utf-8"))
    checkpoint, route = args.artifact / "model.pt", args.artifact / "route.npz"
    if report["schema"] != SCHEMA or (digest(checkpoint), digest(route)) != (
            report["model_sha256"], report["route_sha256"]):
        raise ValueError("The frozen route or actor did not match its training report.")
    bomb = json.loads(args.bomb_model.read_text(encoding="utf-8")) if args.bomb_model else None
    shoot = load_arm(args.shoot_policy)
    with np.load(route, allow_pickle=False) as archive:
        knots = phase_knots(args.data / f"{ALICE}-stage1.npz", report["train_stage_shards"]["1"], archive["frame_1"])
        verified = {}
        for stage in range(2, 7):
            source = args.data / f"{ALICE}-stage{stage}.npz"
            fingerprint = digest(source)
            if fingerprint != report["train_stage_shards"][str(stage)]:
                raise ValueError(f"Stage {stage} route does not match the frozen train-only source.")
            spans = archive[f"encounters_{stage}"].tolist()
            if len(spans) != 2 or spans[1][0] - spans[0][1] <= 96:
                raise ValueError(f"Stage {stage} lacks two distinct observed boss encounters.")
            verified[stage] = dict(source_sha256=fingerprint, visible_boss_intervals=spans)
    arms = {}
    for align in ([True, False] if args.compare_reference else [True]):
        name = "macro_encounter_clock" if align else "stage1_phase_reference"
        policy = EncounterPhaseRoute(checkpoint, route, bomb, shoot, align)
        policy.knots = knots
        arms[name] = Arm(policy, shoot.history, dict(
            model_sha256=digest(checkpoint), route_sha256=digest(route), train_replay=ALICE,
            stage1_phase_knots=knots, later_stages_verified_train_sources=verified,
            method="causal vulnerable onset/confirmed visible exit; nondecreasing within encounter" if align
            else "Stage 1 knots, later encounter onset only (2201 recipe)",
            later_scope="Stage 2–6 macro encounters only; no inferred spell or HP phase IDs.",
            learned_bomb_sha256=digest(args.bomb_model) if args.bomb_model else None,
            independent_shoot=shoot.provenance,
        ))
    return arms


def evaluate(args):
    supplied = (args.game_over_default_reference, args.game_over_retry_reference, args.menu_crop)
    menu = GameOverMenu(*supplied, args.menu_max_error) if all(supplied) else None
    args.crossover = args.timing_policy = []
    args.restart_hook = None
    arms = make_arms(args)
    spec = evaluate_credit_story.run_specification(args, arms)
    spec["controller"].update(kind="neural_demo_route_encounter_clock_ablation", pure=True)
    spec["runtime"]["runner_sha256"] = digest(Path(__file__))
    for source in (evaluate_credit_story, evaluate_demo_route_phase, train_demo_route, fit_randomized_bomb_value):
        spec["runtime"]["source_sha256"][source.__name__] = digest(Path(source.__file__))
    archive = configure(args, spec)
    if not args.resume:
        args.out.mkdir(parents=True, exist_ok=False)
    with output_lock(args.out):
        try:
            report = evaluate_credit_story.evaluate(args, arms, spec, args.out / "report.json", menu, None,
                                                    transition_archive=archive)
        finally:
            path = args.out / "controller-encounter-telemetry.json"
            prior = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            prior.append({name: dict(whole_credit=arm.policy.sessions,
                                    later=arm.policy.later_stage_sessions) for name, arm in arms.items()})
            publish_json(path, prior)
    print(json.dumps(dict(status=report["status"], matched_summary=report.get("matched_summary"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--shoot-policy", type=Path, default=TEMPLATE)
    parser.add_argument("--bomb-model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parakit", type=Path, default=Path(__file__).resolve().parents[2] / "parakit")
    parser.add_argument("--compare-reference", action="store_true")
    parser.add_argument("--target", choices=tuple(evaluate_credit_story.TARGETS), default="ending")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--focus-wait", type=int, default=90)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2301])
    parser.add_argument("--game-over-default-reference", type=Path)
    parser.add_argument("--game-over-retry-reference", type=Path)
    parser.add_argument("--menu-crop", action="append", nargs=4, type=int)
    parser.add_argument("--menu-max-error", type=float, default=.035)
    parser.add_argument("--transition-budget-mib", type=int, default=64)
    parser.add_argument("--transition-reserve-mib", type=int, default=20480)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("The stock TH14 adapter requires Windows.")
    args.limit = evaluate_credit_story.LIMITS[args.target] if args.limit is None else args.limit
    menu = (args.game_over_default_reference, args.game_over_retry_reference, args.menu_crop)
    if (min(args.limit, args.start_frame, args.transition_budget_mib) < 1 or args.focus_wait < 0
            or not 0 < args.menu_max_error < .2 or args.transition_reserve_mib < 0
            or any(menu) and (not all(menu) or len(args.menu_crop) < 2)
            or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**63 for seed in args.seeds)):
        parser.error("Use valid native controls, unique seeds, and complete optional menu references.")
    torch.set_num_threads(1)
    evaluate(args)


if __name__ == "__main__":
    main()

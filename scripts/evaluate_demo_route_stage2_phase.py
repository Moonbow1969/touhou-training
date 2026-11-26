"""Evaluate the exact observed Stage 2 phase mapping, without changing the Stage 1 actor."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts import evaluate_credit_story, evaluate_demo_route_encounters, evaluate_demo_route_phase
from scripts import fit_randomized_bomb_value, train_demo_route, train_native_bomb_timing
from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import TEMPLATE
from scripts.evaluate_demo_route_encounters import EncounterPhaseRoute, make_arms
from scripts.evaluate_sft_native import load_arm, output_lock
from scripts.native_story_restart import GameOverMenu
from scripts.native_story_transitions import configure
from scripts.train_demo_route import ALICE
from scripts.train_full_policy import publish_json


NATIVE_STAGE2_IDS = {
    (12000, None): "mid_nonspell",
    (12000, 12): "spell12",
    (12400, None): "main_nonspell1",
    (12400, 16): "spell16",
    (12900, None): "main_nonspell2",
    (12900, 20): "spell20",
}


def stage2_knots(path, expected_hash, expected_frames):
    if digest(path) != expected_hash:
        raise ValueError("The phase source is not the exact Alice Stage 2 training shard.")
    with np.load(path, allow_pickle=False) as data:
        frames = data["stage_frame"]
        hp = np.rint(2000 * data["context"][:, 8]).astype(int)
    if not np.array_equal(frames, expected_frames):
        raise ValueError("The phase source timeline differs from the frozen route.")
    visible = frames[hp > 0]
    groups = [group for group in np.split(visible, np.flatnonzero(np.diff(visible) > 64) + 1) if len(group) >= 16]
    if len(groups) != 2:
        raise ValueError("Expected two clear Stage 2 boss encounters before phase extraction.")
    mid, main = groups
    previous = np.r_[0, hp[:-1]]

    def first(mask):
        rows = np.flatnonzero(mask)
        if len(rows) == 0:
            raise ValueError("Missing visibly bracketed training health knot.")
        i = rows[0]
        if frames[i] - frames[i - 1] > 6:
            raise ValueError("A health knot crosses an unresolved replay gap.")
        return int(frames[i]), [int(frames[i - 1]), int(frames[i])]

    def crossing(lower, upper, threshold):
        return first((frames > lower) & (frames < upper) & (hp > 0)
                     & (hp <= threshold) & (previous > threshold) & (previous <= 2000))

    def health_reset(lower):
        return first((frames > lower) & (frames < main[-1]) & (hp == 2000)
                     & (previous > 0) & (previous <= 100))

    spell12, spell12_bracket = crossing(mid[0], mid[-1], 1900)
    spell16, spell16_bracket = crossing(main[0], main[-1], 1400)
    nonspell2, reset1_bracket = health_reset(spell16)
    spell20, spell20_bracket = crossing(nonspell2, main[-1], 1900)
    later_unknown, reset2_bracket = health_reset(spell20)
    return {
        "mid_nonspell": [int(mid[0]), spell12],
        "spell12": [spell12, int(mid[-1]) + 4],
        "main_nonspell1": [int(main[0]), spell16],
        "spell16": [spell16, nonspell2],
        "main_nonspell2": [nonspell2, spell20],
        "spell20": [spell20, later_unknown],
    }, {
        "spell12_visible_hp_1900": spell12_bracket,
        "spell16_visible_hp_1400": spell16_bracket,
        "post_spell16_clipped_reset": reset1_bracket,
        "spell20_visible_hp_1900": spell20_bracket,
        "post_spell20_clipped_reset_unlabeled": reset2_bracket,
    }


class ExactStage2Route(EncounterPhaseRoute):
    def __init__(self, checkpoint, route, bomb, shoot, stage2_plan):
        super().__init__(checkpoint, route, bomb, shoot, True)
        self.stage2_plan = stage2_plan
        self.stage2_scope = None
        self.stage2_sessions = []

    def phase_time(self, frame, stage, boss):
        env = self.environment
        scope = (env.reset_count, stage)
        native = env.snapshot.boss
        if stage > 1 and self.encounter_scope == scope:
            clock = self.later_clock
            # The next stage ID can appear a few ticks before its opening timer resets (e.g. 28 to 2).
            if (clock.last_native is not None and frame < clock.last_native <= 64
                    and frame <= 32 and clock.next_index == 0 and native is None):
                clock.events.append(dict(kind="opening_stage_clock_reset", previous_frame=clock.last_native,
                                         native_frame=frame))
                clock.last_native = clock.last_reference = None
        macro_frame = super().phase_time(frame, stage, boss)
        if stage != 2:
            return macro_frame
        if scope != self.stage2_scope:
            self.stage2_scope = scope
            self.stage2_phase = None
            self.post_exact_from = None
            self.stage2_sessions.append(dict(credit=scope[0], events=[], phase_tail_counts={}))
        current = env.snapshot.boss
        combat = current is not None and current.max_hp > 40 and current.hp > 0
        identity = (current.max_hp, env.snapshot.spell_id) if combat else None
        label = NATIVE_STAGE2_IDS.get(identity)
        session = self.stage2_sessions[-1]
        if label is None:
            # Continue from the observed reference reset if the live credit advances to a new boss phase.
            if identity is not None and self.stage2_phase == "spell20" and self.post_exact_from is None:
                self.post_exact_from = frame
                session["events"].append(dict(phase="later_unknown", native_frame=frame,
                                              reference_frame=self.stage2_plan["spell20"][1],
                                              raw_max_hp=identity[0], raw_spell=identity[1]))
            if identity is None and self.stage2_phase != "spell20":
                self.stage2_phase = None
            if identity is not None and self.post_exact_from is not None:
                spans = self.route["encounters_2"]
                end = int(spans[-1][-1]) - 4
                elapsed = self.stage2_plan["spell20"][1] + frame - self.post_exact_from
                self.actual_reference_frame = min(end, elapsed)
                self.expired_phase = elapsed >= end
                return self.actual_reference_frame
            return macro_frame
        start, stop = self.stage2_plan[label]
        if self.stage2_phase != label:
            self.stage2_phase = label
            self.phase_native_start = frame
            self.post_exact_from = None
            session["events"].append(dict(phase=label, native_frame=frame, reference_interval=[start, stop],
                                          raw_max_hp=identity[0], raw_spell=identity[1]))
        elapsed = start + frame - self.phase_native_start
        # The tracker reads position four frames ahead. Keep both its input key and target inside this phase.
        last_input = stop - 8
        self.actual_reference_frame = min(last_input, elapsed)
        self.expired_phase = elapsed >= last_input
        if self.expired_phase:
            tail = session["phase_tail_counts"]
            tail[label] = tail.get(label, 0) + 1
        return self.actual_reference_frame


def evaluate(args):
    refs = (args.game_over_default_reference, args.game_over_retry_reference, args.menu_crop)
    menu = GameOverMenu(*refs, args.menu_max_error) if all(refs) else None
    args.crossover = args.timing_policy = []
    args.restart_hook = None
    args.compare_reference = False
    macro = make_arms(args)["macro_encounter_clock"]
    route_path = args.artifact / "route.npz"
    report = json.loads((args.artifact / "report.json").read_text(encoding="utf-8"))
    bomb = json.loads(args.bomb_model.read_text(encoding="utf-8")) if args.bomb_model else None
    source = args.data / f"{ALICE}-stage2.npz"
    with np.load(route_path, allow_pickle=False) as data:
        knots, brackets = stage2_knots(source, report["train_stage_shards"]["2"], data["frame_2"])
    shoot = load_arm(args.shoot_policy)
    policy = ExactStage2Route(args.artifact / "model.pt", route_path, bomb, shoot, knots)
    policy.knots = macro.policy.knots
    development = json.loads(args.development_audit.read_text(encoding="utf-8"))
    observed = {(row["max_hp"], None if row["spell"] < 0 else row["spell"])
                for row in development["true_native_phases"] if row["stage"] == 2}
    if (development["train_route_source_sha256"]["2"] != report["train_stage_shards"]["2"]
            or not set(NATIVE_STAGE2_IDS) <= observed):
        raise ValueError("The Stage 2 raw identifiers were not observed in the supplied development credit.")
    provenance = dict(macro.provenance, stage2_phase_intervals=knots, stage2_visible_knot_brackets=brackets,
                      stage2_raw_ids=[dict(max_hp=key[0], spell_id=key[1], phase=value)
                                      for key, value in NATIVE_STAGE2_IDS.items()],
                      development_native_credit_sha256=development["credit_sha256"],
                      development_audit_sha256=digest(args.development_audit),
                      development_tape_role="Established raw phase identities only. Not an evaluation of this variant.",
                      learned_bomb_training_anchor=bomb["anchor_sha256"] if bomb else None,
                      learned_bomb_training_stage=1 if bomb else None,
                      stage2_tail_target="Last in-phase replay position with a four-frame action lookahead.",
                      beyond_phase="After the demonstration's aligned phase would end, hold its last active neural position/Focus target and use independent frozen Shoot for real vulnerable combat.",
                      stage2_unknown_phase="Continue from the observed post-spell20 HP reset with encounter bounds. No unobserved spell ID assigned.")
    arms = {"stage2_exact_phase": replace(macro, policy=policy, provenance=provenance)}
    spec = evaluate_credit_story.run_specification(args, arms)
    spec["controller"].update(kind="frozen_demo_exact_stage1_stage2_phase",
                              stage3_plus="macro encounter alignment", learned_bomb_transfer_stage1_to_later=bool(bomb))
    spec["runtime"]["runner_sha256"] = digest(Path(__file__))
    for module in (evaluate_demo_route_encounters, evaluate_demo_route_phase, evaluate_credit_story,
                   train_demo_route, fit_randomized_bomb_value, train_native_bomb_timing):
        spec["runtime"]["source_sha256"][module.__name__] = digest(Path(module.__file__))
    if args.record_sensors:
        from scripts.native_story_sensors import configure as configure_sensors
        archive = configure_sensors(args, spec)
    else:
        archive = configure(args, spec)
    if not args.resume:
        args.out.mkdir(parents=True, exist_ok=False)
    with output_lock(args.out):
        try:
            result = evaluate_credit_story.evaluate(args, arms, spec, args.out / "report.json", menu, None,
                                                    transition_archive=archive)
        finally:
            path = args.out / "controller-stage2-phase.json"
            sessions = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
            sessions.append(dict(whole_credit=policy.sessions, stage2=policy.stage2_sessions,
                                 later_encounters=policy.later_stage_sessions))
            publish_json(path, sessions)
    print(json.dumps(dict(status=result["status"], summary=result.get("matched_summary"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--development-audit", type=Path,
                        default=Path("results/2026-11-25-th14-native-policy/demo-phase-2201-natural-stage2-audit.json"))
    parser.add_argument("--shoot-policy", type=Path, default=TEMPLATE)
    parser.add_argument("--bomb-model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parakit", type=Path, default=Path(__file__).resolve().parents[2] / "parakit")
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
    parser.add_argument("--record-sensors", action="store_true",
                        help="Record passive hazard and resource signals outside policy observations.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("The stock TH14 adapter requires Windows.")
    if args.limit is None:
        args.limit = evaluate_credit_story.LIMITS[args.target]
    refs = (args.game_over_default_reference, args.game_over_retry_reference, args.menu_crop)
    if (min(args.limit, args.start_frame, args.transition_budget_mib) < 1 or min(args.focus_wait,args.transition_reserve_mib) < 0
            or any(refs) and (not all(refs) or len(args.menu_crop) < 2)
            or not 0 < args.menu_max_error < .2 or len(set(args.seeds)) != len(args.seeds)
            or any(not 0 <= seed < 2**63 for seed in args.seeds)):
        parser.error("Use valid native limits, unique seeds and complete optional visual menu references.")
    torch.set_num_threads(1)
    evaluate(args)


if __name__ == "__main__":
    main()

"""Evaluate the frozen TH14 demo route with private two-tick movement actuation."""

import argparse
from collections import Counter
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts import diagnose_demo_route, evaluate_credit_story, native_story_restart, stage_policy, train_demo_route
from scripts.checkpoint_storage import digest
from scripts.diagnose_demo_route import recorded_word, sample_position
from scripts.evaluate_sft_native import Arm, output_lock
from scripts.native_story_restart import GameOverMenu
from scripts.train_demo_route import DemoPolicy, SCALE, SCHEMA, features


class DecisionClock:
    """Delegate native operations and replace exactly one decision wait per adapter step."""

    def __init__(self, native):
        self.native = native
        self.pending = None
        self.replaced = False

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _sleep(self, ticks=0, **kwargs):
        if self.pending is not None and not self.replaced and ticks == 4:
            ticks = self.pending
            self.replaced = True
        return self.native._sleep(ticks, **kwargs)

    @contextmanager
    def decision(self, ticks):
        if self.pending is not None:
            raise RuntimeError("Nested native decision wait.")
        self.pending, self.replaced = ticks, False
        try:
            yield
            if not self.replaced:
                raise RuntimeError("The stock adapter did not issue its expected four-tick decision wait.")
        finally:
            self.pending = None


class TwoTickStage(stage_policy.NormalStageOne):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game = DecisionClock(self.game)
        self._clear_tempo()

    def _clear_tempo(self):
        self.global_deltas, self.stage_deltas = Counter(), Counter()
        self.exact_both = self.active_steps = self.dialogue_steps = self.transition_steps = 0
        self.bomb_tempo = None

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self._clear_tempo()
        return result

    def step(self, action):
        old_clock = self.game.read_game_val("global_timer")
        old_frame, old_stage = self.stage_frame(), self.snapshot.stage
        dialogue = self.game.read_game_val("in_dialog") == -1
        with self.game.decision(4 if dialogue else 2):
            observation, reward, terminated, truncated, info = super().step(action)
        new_clock = self.game.read_game_val("global_timer")
        if dialogue:
            self.dialogue_steps += 1
        elif info["native_stage"] != old_stage or info["native_pause_state"] != 2:
            self.transition_steps += 1
        else:
            if old_clock is None or new_clock is None:
                raise RuntimeError("The active native clock is unreadable.")
            global_delta = new_clock - old_clock
            stage_delta = self.stage_frame() - old_frame
            if global_delta <= 0 or stage_delta < 0:
                raise RuntimeError("Native clock failed to advance during the two-tick decision.")
            self.global_deltas[global_delta] += 1
            self.stage_deltas[stage_delta] += 1
            self.active_steps += 1
            self.exact_both += int(global_delta == stage_delta == 2)
        info["actuator_tempo"] = dict(
            measured_active_decisions=self.active_steps, exact_joint_two_tick=self.exact_both,
            global_tick_deltas=dict(self.global_deltas), stage_tick_deltas=dict(self.stage_deltas),
            legacy_dialogue_decisions=self.dialogue_steps, transition_decisions=self.transition_steps,
        )
        info["bomb_tempo"] = self.bomb_tempo
        return observation, reward, terminated, truncated, info


class BombCadence:
    def __init__(self):
        self.credit = None
        self.tick = None
        self.stage = None

    def action(self, credit, stage, tick, predict):
        if tick is None:
            raise RuntimeError("No native global clock is available for Bomb timing.")
        if self.credit != credit:
            self.credit = credit
            self.tick = None
            self.queries = self.reused = self.clock_reanchors = 0
            self.intervals = Counter()
        elapsed = tick - self.tick if self.tick is not None else None
        if elapsed is not None and elapsed < 0:
            if stage == self.stage:
                raise RuntimeError("The Bomb decision clock moved backward within one stage.")
            self.clock_reanchors += 1
            elapsed = None
        if elapsed is None or elapsed >= 4:
            self.value = predict()
            self.tick = tick
            self.queries += 1
            if elapsed is not None:
                self.intervals[elapsed] += 1
        else:
            self.reused += 1
        self.stage = stage
        return self.value

    def report(self):
        return dict(queries=self.queries, held_decisions=self.reused,
                    native_query_tick_intervals=dict(self.intervals), stage_clock_reanchors=self.clock_reanchors)


class TwoTickDemoPolicy(DemoPolicy):
    def __init__(self, checkpoint, archive, bomb_model):
        super().__init__(checkpoint, archive, bomb_model=bomb_model)
        self.bomb_cadence = BombCadence()

    def phase_time(self, frame, stage, boss):
        scope = (self.environment.reset_count, stage)
        prior = self.time_offset if self.credit_stage == scope else 0
        time = super().phase_time(frame, stage, boss)
        if abs(time - frame - prior) < 2:
            self.time_offset = prior
            return frame + prior
        return time

    @torch.inference_mode()
    def predict(self, observation, deterministic=False):
        env = self.environment
        context = observation["context"]
        stage = env.snapshot.stage
        frame = self.phase_time(env.last_frame, stage, bool(context[8] > 0))
        ticks = 4 if env.game.read_game_val("in_dialog") == -1 else 2
        frames = self.route[f"frame_{stage}"]
        position = context[:2] * SCALE
        target = sample_position(frame + ticks, frames, self.route[f"positions_{stage}"])
        equivalent = position + (target - position) * (4 / ticks)
        word = recorded_word(frame, frames, self.route[f"keys_{stage}"])
        value = features(position, equivalent, np.asarray(bool(word & 8)), np.asarray(bool(word & 1)),
                         self.route["bounds"])
        movement, shooting = self.model(torch.from_numpy(value))
        focus, move = divmod(int(movement.argmax()), 9)
        bomb = 0
        if self.bomb_model is not None:
            from scripts.fit_randomized_bomb_value import action_values, native_bomb_features

            def predict_bomb():
                values, _ = action_values(self.bomb_model, native_bomb_features(env))
                return int(values[0].argmax())

            bomb = self.bomb_cadence.action(
                env.reset_count, stage, env.game.read_game_val("global_timer"), predict_bomb,
            )
            env.bomb_tempo = self.bomb_cadence.report()
        return np.array([move, focus, int(shooting.argmax()), bomb], dtype=np.int64), None


@contextmanager
def native_session(args, arms, menu, restart_hook):
    from environment import interface as game
    from environment.structured import StateReader, load_parakit_offsets

    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState
    continuous = 0x80000000
    keep_awake(continuous | 0x00000001 | 0x00000002)
    env = None
    try:
        reader = StateReader(game, load_parakit_offsets(args.parakit), 96)
        preparation = native_story_restart.prepare(game, reader, menu, restart_hook)
        env = TwoTickStage(
            args.parakit, limit=args.limit, max_misses=float("inf"), prepared=True,
            last_stage=evaluate_credit_story.last_stage(args.target), motion_history=False, focus_wait=args.focus_wait,
        )
        yield env, preparation
    finally:
        try:
            if env is not None:
                env.release()
                env.close()
        finally:
            keep_awake(continuous)


def make_arm(args):
    report = json.loads((args.artifact / "report.json").read_text(encoding="utf-8"))
    checkpoint, route = args.artifact / "model.pt", args.artifact / "route.npz"
    if report["schema"] != SCHEMA or (digest(checkpoint), digest(route)) != (
            report["model_sha256"], report["route_sha256"]):
        raise ValueError("The frozen Alice route and actor do not match their report.")
    bomb = json.loads(args.bomb_model.read_text(encoding="utf-8")) if args.bomb_model else None
    policy = TwoTickDemoPolicy(checkpoint, route, bomb)
    provenance = dict(schema=SCHEMA, model_sha256=report["model_sha256"], route_sha256=report["route_sha256"],
                      train_replay=report["train_replay"], replay_sha256=report["replay_sha256"],
                      phase_alignment="suppress cadence-grid reanchors smaller than two native ticks",
                      policy_decode="greedy", nominal_movement_ticks=2, dialogue_ticks=4)
    if bomb is not None:
        provenance["learned_bomb_transfer"] = dict(
            value_model_sha256=digest(args.bomb_model), anchor_sha256=bomb["anchor_sha256"],
            native_query_period_at_least_ticks=4, intervention_label_ticks=bomb["native_label_frames"],
            source="randomized Stage 1 interventions on PPO397 with forced release during 16-frame label follow-up",
            limitation="Repeated online re-evaluation on route states is transfer; the Bomb cadence is held at four ticks.",
        )
    return {"alice_route_2tick": Arm(policy, False, provenance)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--bomb-model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parakit", type=Path, default=Path(__file__).resolve().parents[2] / "parakit")
    parser.add_argument("--target", choices=tuple(evaluate_credit_story.TARGETS), default="stage2")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--focus-wait", type=int, default=90)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2101, 2102])
    parser.add_argument("--game-over-default-reference", type=Path)
    parser.add_argument("--game-over-retry-reference", type=Path)
    parser.add_argument("--menu-crop", action="append", nargs=4, type=int)
    parser.add_argument("--menu-max-error", type=float, default=.035)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--record-transitions", action="store_true")
    parser.add_argument("--transition-budget-mib", type=int, default=256)
    parser.add_argument("--transition-reserve-mib", type=int, default=2048)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("The stock TH14 1.00b native adapter requires Windows.")
    if args.limit is None:
        args.limit = 2 * evaluate_credit_story.LIMITS[args.target]
    refs = (args.game_over_default_reference, args.game_over_retry_reference, args.menu_crop)
    if (min(args.start_frame, args.limit) < 1 or args.focus_wait < 0 or not 0 < args.menu_max_error < .2
            or any(refs) and (not all(refs) or len(args.menu_crop) < 2)
            or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**63 for seed in args.seeds)):
        parser.error("Use valid decision limits, unique policy seeds and complete optional menu references.")
    torch.set_num_threads(1)
    args.crossover = args.timing_policy = []
    args.restart_hook = None
    try:
        arms = make_arm(args)
        spec = evaluate_credit_story.run_specification(args, arms)
        spec["task"].update(frames_per_action=2, dialogue_frames_per_action=4)
        spec["controller"].update(kind="frozen_demo_route_two_tick", bomb_action_value_minimum_period=4)
        spec["runtime"]["runner_sha256"] = digest(Path(__file__))
        for module in (evaluate_credit_story, train_demo_route, diagnose_demo_route):
            spec["runtime"]["source_sha256"][module.__name__] = digest(Path(module.__file__))
        if args.bomb_model:
            from scripts import fit_randomized_bomb_value, train_native_bomb_timing

            for module in (fit_randomized_bomb_value, train_native_bomb_timing):
                spec["runtime"]["source_sha256"][module.__name__] = digest(Path(module.__file__))
        menu = GameOverMenu(*refs, args.menu_max_error) if all(refs) else None
        archive = None
        if args.record_transitions:
            from scripts.native_story_transitions import configure
            archive = configure(args, spec)
        if not args.resume:
            args.out.mkdir(parents=True, exist_ok=False)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    with output_lock(args.out):
        report = evaluate_credit_story.evaluate(args, arms, spec, args.out / "report.json", menu, None,
                                                stage_factory=native_session, transition_archive=archive)
    print(json.dumps(dict(status=report["status"], summary=report.get("matched_summary"),
                          waiting_for_fresh_credit=report.get("waiting_for_fresh_credit"))), flush=True)


if __name__ == "__main__":
    main()

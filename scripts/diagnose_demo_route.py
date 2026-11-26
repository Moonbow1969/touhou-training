"""Audit frozen neural demo-route tracking without game input or collisions."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.train_demo_route import ALICE, HOLDOUTS, SCALE, DemoPolicy, encounters, features, fit_action_deltas, open_shard
from scripts.train_full_policy import publish_json


START = np.array((0., 400.), dtype=np.float32)
MODES = {
    "stock4": (4, False, True),
    "stock4_bottom_spawn_stress": (4, False, True),
    "absolute4_without_resync": (4, False, False),
    "decision_index4": (4, True, False),
    "cadence_deadband4": (4, False, True),
    "scaled_actuator2": (2, False, False),
    "scaled_actuator1": (1, False, False),
}


def sample_position(frame, frames, positions):
    return np.array([np.interp(frame, frames, coordinates) for coordinates in positions.T], dtype=np.float32)


def recorded_word(frame, frames, chunks):
    index = max(0, int(np.searchsorted(frames, frame, side="right")) - 1)
    return int(chunks[index, int(np.clip(frame - frames[index], 0, 3))])


class DiagnosticPolicy(DemoPolicy):
    """Keep the frozen network while varying the tested clock index or action duration."""

    def __init__(self, checkpoint, archive, tick, indexed, resync, deadband=False):
        super().__init__(checkpoint, archive, resync=resync)
        self.tick = tick
        self.indexed = indexed
        self.deadband = deadband
        self.reference_frame = None

    def phase_time(self, frame, stage, boss):
        if self.indexed:
            return self.reference_frame
        prior = (self.time_offset if self.credit_stage == (self.environment.reset_count, stage) else 0)
        time = super().phase_time(frame, stage, boss)
        if self.deadband and abs((time - frame) - prior) < self.tick:
            self.time_offset = prior
            return frame + prior
        return time

    @torch.inference_mode()
    def predict(self, observation, deterministic=False):
        if self.tick == 4:
            return super().predict(observation, deterministic=deterministic)
        env = self.environment
        stage = env.snapshot.stage
        context = observation["context"]
        frame = self.phase_time(env.last_frame, stage, bool(context[8] > 0))
        frames = self.route[f"frame_{stage}"]
        position = context[:2] * SCALE
        goal = sample_position(frame + self.tick, frames, self.route[f"positions_{stage}"])
        equivalent_goal = position + (goal - position) * (4 / self.tick)
        word = recorded_word(frame, frames, self.route[f"keys_{stage}"])
        sample = features(position, equivalent_goal, np.asarray(bool(word & 8)), np.asarray(bool(word & 1)),
                          self.route["bounds"])
        motion, shoot = self.model(torch.from_numpy(sample))
        focus, move = divmod(int(motion.argmax()), 9)
        return np.array([move, focus, int(shoot.argmax()), 0], dtype=np.int64), None


def summarize(items):
    if not items:
        return None
    error = np.asarray([row[0] for row in items])
    policy_focus, expert_focus, policy_shoot, expert_shoot = (
        np.asarray([row[index] for row in items]) for index in range(1, 5)
    )
    start, end = min(row[5] for row in items), max(row[5] for row in items)
    return dict(
        decisions=len(items), native_frame_interval=[start, end],
        track_error=dict(median=float(np.median(error)), p95=float(np.quantile(error, .95)), max=float(error.max())),
        peak_error_at=list(items[int(error.argmax())][6:8]),
        within_8=int((error <= 8).sum()), within_12=int((error <= 12).sum()),
        focus=dict(predicted=int(policy_focus.sum()), expert=int(expert_focus.sum()),
                   agreement=int((policy_focus == expert_focus).sum()),
                   expert_on_maintained=int((policy_focus & expert_focus).sum())),
        shoot=dict(predicted=int(policy_shoot.sum()), expert=int(expert_shoot.sum()),
                   agreement=int((policy_shoot == expert_shoot).sum()),
                   expert_on_maintained=int((policy_shoot & expert_shoot).sum())),
    )


def simulate(policy, record, dense, stage, delta, mode):
    frames, positions, keys = [record[key] for key in ("frame", "positions", "keys")]
    threshold, late = dense
    starting_frame = 30 if stage == 1 else int(frames[0])
    env = SimpleNamespace(reset_count=1, last_frame=starting_frame, snapshot=SimpleNamespace(stage=stage))
    policy.environment = env
    policy.credit_stage = None
    for field in ("frame", "positions", "keys"):
        key = f"{field}_{stage}"
        policy.route[key] = record[field]
    windows = encounters(record)
    policy.route[f"encounters_{stage}"] = np.asarray(windows, dtype=np.int32).reshape(-1, 2)
    lo, hi = policy.route["bounds"]
    stage_errors, final_errors, late_errors, dense_errors = [], [], [], []
    anchors = []
    initial_position = (START.copy() if stage == 1 or mode == "stock4_bottom_spawn_stress"
                        else positions[0].copy())
    position = initial_position.copy()
    initial_source_index = int(np.searchsorted(frames, starting_frame, side="left"))
    reference_frame = starting_frame
    tick = policy.tick
    indices = np.arange(starting_frame, int(frames[-1]) - tick + 1, tick)
    present = np.asarray([any(left <= native <= right for left, right in windows) for native in indices])
    expected_positions = np.stack([np.interp(indices + tick, frames, field) for field in positions.T], axis=1)
    density = np.interp(indices, frames, record["counts"])
    for i, frame in enumerate(indices):
        env.last_frame = int(frame)
        policy.reference_frame = int(frames[min(initial_source_index + i, len(frames) - 1)])
        obs = np.zeros(10, dtype=np.float32)
        obs[:2] = position / SCALE
        obs[8] = present[i]
        previous_offset = getattr(policy, "time_offset", 0)
        action, _ = policy.predict({"context": obs}, deterministic=True)
        move, focus, shoot = map(int, action[:3])
        position = np.clip(position + delta[focus * 9 + move] * (tick / 4), lo, hi)
        current_offset = getattr(policy, "time_offset", 0)
        if current_offset != previous_offset:
            anchors.append(dict(native_frame=int(frame), offset=int(current_offset)))
        if policy.indexed:
            reference_frame = policy.reference_frame
        else:
            reference_frame = frame + current_offset
        key = recorded_word(frame, frames, keys)
        item = (float(np.linalg.norm(position - expected_positions[i])), bool(focus), bool(key & 8),
                bool(shoot), bool(key & 1), int(frame), stage, int(frame + tick))
        stage_errors.append(item)
        if windows and windows[-1][0] <= frame <= windows[-1][-1]:
            final_errors.append(item)
            if frame >= late:
                late_errors.append(item)
            if density[i] >= threshold:
                dense_errors.append(item)
    return dict(mode=mode, start_position=initial_position.tolist(), start_frame=starting_frame,
                source_initial_position=sample_position(starting_frame, frames, positions).tolist(),
                native_to_reference_anchors=anchors,
                final_reference_minus_native_frame=int(reference_frame - indices[-1]),
                all=summarize(stage_errors), final_boss=summarize(final_errors),
                late_final_boss=summarize(late_errors), dense_final_boss=summarize(dense_errors)), (
                    stage_errors, final_errors, late_errors, dense_errors
                )


def attach_counts(data, replay, stage, record):
    with np.load(data / f"{replay}-stage{stage}.npz", allow_pickle=False) as source:
        record["counts"] = source["bullet_count"]
    spans = encounters(record)
    if not spans:
        return float("inf"), float("inf")
    start, finish = spans[-1]
    both = (record["frame"] >= start) & (record["frame"] <= finish)
    return float(np.quantile(record["counts"][both], .75)), (start + finish) / 2


def aggregate(groups):
    return {name: summarize(values) for name, values in zip(
        ("all", "final_boss", "late_final_boss", "dense_final_boss"), groups,
    )}


def diagnose(args):
    torch.set_num_threads(1)
    files = args.artifact
    training_delta = np.load(files / "route.npz", allow_pickle=False)["action_deltas"]
    report = dict(model_sha256=digest(files / "model.pt"), route_sha256=digest(files / "route.npz"),
                  method=dict(
                      movement="scale the empirical 2D displacement from constant-input four-tick native windows",
                      test_start="stock x=0 y=400 at Stage 1 frame 30; use first observed native replay position and tick for later stages. A separate bottom-spawn stress hypothesis initializes later stages at x=0 y=400.",
                      reference="position linearly interpolated in native stage-frame time; error is always relative to true clock",
                      controls="compare focus and shoot with the first recorded expert input of that decision",
                      dense_boss="top quartile of recorded projectile counts within the final boss encounter",
                      late_boss="last half of the final observed boss encounter",
                      decision_index="one recorded row drawn per four ticks; ignores missing rows",
                      alternate_actuators="use same frozen inverse map with target delta scaled to a four-tick equivalent, no resync; compare to absolute4_without_resync on identical route clock.",
                      cadence_deadband="alternate offline policy ignores boss encounter reanchors smaller than one action interval; deployed candidate remains unchanged.",
                      native_physics_limits="No collisions, hitstop, bombs, phase reaction, shot simulation or respawn; alternative ticks interpolate route and assume per-tick linear mechanics.",
                  ), datasets={})
    for replay, stages in [(ALICE, range(1, 7)), *((name, (1,)) for name in HOLDOUTS)]:
        records = {stage: open_shard(args.data, replay, stage)[0] for stage in stages}
        if replay == ALICE:
            delta = training_delta
            origin = "Alice fit"
            support = None
        else:
            delta, samples = fit_action_deltas(list(records.values()), fallback=training_delta)
            origin = "held-out native constant-input medians; Alice fallback for actions with <10 rows"
            support = samples.tolist()
        result = dict(movement_dynamics=origin, diagnostics_calibration_action_counts=support, stages={}, aggregate={})
        totals = {mode: [[], [], [], []] for mode in MODES}
        for stage, record in records.items():
            dense = attach_counts(args.data, replay, stage, record)
            row = dict(recorded_encounters=encounters(record), dense_projectile_threshold=dense[0], modes={})
            for mode, (tick, indexed, resync) in MODES.items():
                if stage == 1 and mode == "stock4_bottom_spawn_stress":
                    continue
                if stage > 1 and mode in ("absolute4_without_resync", "decision_index4",
                                           "scaled_actuator1", "scaled_actuator2"):
                    continue
                actor = DiagnosticPolicy(files / "model.pt", files / "route.npz", tick, indexed,
                                         resync, deadband=mode == "cadence_deadband4")
                summary, traces = simulate(actor, record, dense, stage, delta, mode)
                row["modes"][mode] = summary
                for bank, new in zip(totals[mode], traces):
                    bank.extend(new)
                print(json.dumps(dict(replay=replay, stage=stage, mode=mode, tracking=summary["all"]["track_error"])),
                      flush=True)
            result["stages"][stage] = row
        result["aggregate"] = {mode: aggregate(values) for mode, values in totals.items() if values[0]}
        report["datasets"][replay] = result
    publish_json(args.out, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Select a fresh output path.")
    diagnose(args)


if __name__ == "__main__":
    main()

"""Record actor inputs and native combat signals on continuous TH14 story credits."""

import json

import numpy as np

from scripts import native_credit_transitions as archive
from scripts.checkpoint_storage import digest


SCHEMA = "th14-stock-story-actor-transitions-v1"
NATIVE_FIELDS = (*archive.NATIVE_FIELDS[:-1], "screen", "boss_vulnerable", "snapshot_valid")
POSITION_FIELDS = ("player_x", "player_y", "boss_x", "boss_y")
OUTCOMES = {"ongoing": 0, "target_reached": 1, "game_over": 2, "time_limit": 3, "one_credit_clear": 4}


def read_state(env, stage=None, pause=None):
    game = env.game
    stage = env.current_stage() if stage is None else stage
    pause = game.read_game_val("game_state") if pause is None else pause
    screen = env.current_screen()
    native = dict(global_tick=game.read_game_val("global_timer"), stage_frame=env.stage_frame(),
                  lives=game.read_game_val("lives"), bombs=game.read_game_val("bombs"),
                  player_state=env.snapshot.player_state)
    if any(native[key] is None for key in ("global_tick", "stage_frame", "lives", "bombs")) or pause is None:
        raise RuntimeError("Required story transition counters are unreadable.")
    values = archive.native_fields(env, native, stage, pause if screen == 7 else -1)
    values[3] = pause
    valid = bool(values[-1])
    boss = env.snapshot.boss if valid else None
    position = [float("nan")] * 4
    if valid:
        x, y = env.snapshot.observation["context"][:2]
        position[:2] = (float(x) * 192, float(y) * 448)
    if boss is not None:
        position[2:] = (boss.x, boss.y)
    return [*values[:-1], screen, int(boss.vulnerable) if boss is not None else -1, values[-1]], position


def source_offsets(env):
    offsets = archive.source_offsets(env)
    enemy = env.reader.offsets.enemies
    offsets["boss"] = {name: getattr(enemy, name) for name in (
        "enemy_manager_pointer", "zEnemyManager_list", "zEnemy_data", "zEnemyData_flags", "zEnemyData_hp",
        "zEnemyData_pos", "zEnemyFlags_is_boss", "zEnemyFlags_no_hurtbox", "zEnemyFlags_invincible",
        "zEnemyFlags_intangible",
    )}
    return offsets


class StoryTransitions:
    def __init__(self, env):
        self.offsets = source_offsets(env)
        self.observations = []
        self.rows = []
        self.final_observation = None
        self.terminal_outcome = None

    def before_step(self, env, observation):
        self.observations.append(archive.copy_observation(observation))
        return read_state(env)

    def after_step(self, env, before, action, info, terminated, truncated, deaths, bombs):
        after = read_state(env, info["native_stage"], info["native_pause_state"])
        self.rows.append((action.tolist(), before, after, terminated, truncated, deaths, bombs))

    def finish(self, outcome, final_observation):
        self.terminal_outcome = OUTCOMES[outcome]
        if final_observation is not None and self.rows[-1][2][0][-1]:
            self.final_observation = archive.copy_observation(final_observation)

    def arrays(self):
        actions, before, after, terminated, truncated, deaths, bombs = zip(*self.rows)
        raw_before, position_before = zip(*before)
        raw_after, position_after = zip(*after)
        raw_before = np.array(raw_before, dtype=np.int64)
        raw_after = np.array(raw_after, dtype=np.int64)
        bullets, context = map(np.stack, zip(*self.observations))
        exact = ((raw_before[:, -1] == 1) & (raw_after[:, -1] == 1) & (raw_before[:, 2] == raw_after[:, 2])
                 & (raw_after[:, 0] - raw_before[:, 0] == 4) & (raw_after[:, 1] - raw_before[:, 1] == 4))
        indices = np.r_[0, np.flatnonzero(raw_before[1:, 2] != raw_before[:-1, 2]) + 1]
        stage_spans = np.stack((raw_before[indices, 2], indices, np.r_[indices[1:], len(actions)]), axis=1)
        outcomes = np.zeros(len(actions), dtype=np.uint8)
        outcomes[-1] = self.terminal_outcome
        result = dict(
            bullets=bullets, context=context, action=np.array(actions, dtype=np.uint8),
            native_before=raw_before, native_after=raw_after,
            positions_before=np.array(position_before, dtype=np.float32),
            positions_after=np.array(position_after, dtype=np.float32), stage_spans=stage_spans,
            native_life_losses=np.array(deaths, dtype=np.int16), actual_bombs_consumed=np.array(bombs, dtype=np.int16),
            terminated=np.array(terminated, dtype=np.bool_), truncated=np.array(truncated, dtype=np.bool_),
            four_tick_stage_transition=exact, terminal_outcome=outcomes,
            final_observation_valid=np.array(self.final_observation is not None, dtype=np.bool_),
        )
        if self.final_observation is not None:
            result["final_bullets"], result["final_context"] = self.final_observation
        return result


class StoryArchive:
    def __init__(self, args, specification):
        self.args = args
        self.out = args.out
        self.specification = specification

    def path(self, trial):
        return self.out / "transition-shards" / (
            f"credit-{trial['block']:03d}-arm-{trial['arm']}-seed-{trial['policy_seed']}.npz"
        )

    def metadata(self, trial, offsets, rows):
        spec = self.specification
        return dict(
            schema=SCHEMA, trial=trial, rows=rows, task=spec["task"], actor=spec["policies"][trial["arm"]],
            runtime=spec["runtime"], source_offsets=offsets, native_field_order=NATIVE_FIELDS,
            position_field_order=POSITION_FIELDS, native_key_bit_order=archive.KEY_ORDER,
            stage_span_columns=["native_stage", "first_decision_index", "end_decision_index_exclusive"],
            terminal_codes=OUTCOMES, invalid_native_field=-1,
            observation_format="Exact float32 actor input. Final actor input exists only after a valid time limit.",
            action_format="Applied policy action. The native arrays also record before/after held keys.",
            shot_alignment="Use global native clock and measured player/boss positions to align actions to later HP changes.",
        )

    def start(self, env):
        archive.storage_remaining(self.out, self.args)
        return StoryTransitions(env)

    def publish(self, trial, result, trace):
        rows = len(trace.rows)
        if rows != result["decision_count"] or len(trace.observations) != rows or trace.terminal_outcome is None:
            raise ValueError("Transition archive and finished story credit do not match.")
        return archive.publish_archive(self.out, self.args, self.path(trial),
                                       self.metadata(trial, trace.offsets, rows), result, trace.arrays())

    def recover(self, completed, remaining):
        for row in completed:
            saved = row["result"]["transition_shard"]
            if archive.descriptor(self.out, self.path(row["trial"]), saved["rows"]) != saved:
                raise ValueError("A previously committed story transition archive changed.")
        while remaining and self.path(remaining[0]).is_file():
            trial = remaining.pop(0)
            path = self.path(trial)
            metadata, result, rows = archive.read_archive(path)
            expected = self.metadata(trial, metadata["source_offsets"], rows)
            if metadata != json.loads(json.dumps(expected)) or result["decision_count"] != rows:
                raise ValueError(f"Saved story transition archive belongs to a different credit: {path.name}.")
            result["transition_shard"] = archive.descriptor(self.out, path, rows)
            completed.append(dict(trial=trial, result=result))


def configure(args, specification):
    specification["transition_recording"] = dict(
        schema=SCHEMA, budget_mib=args.transition_budget_mib, disk_reserve_mib=args.transition_reserve_mib,
    )
    sources = specification["runtime"]["source_sha256"]
    from pathlib import Path

    for name, path in ((__name__, __file__), (archive.__name__, archive.__file__)):
        sources[name] = digest(Path(path))
    return StoryArchive(args, specification)

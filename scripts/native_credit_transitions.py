"""Bounded per-credit archives of the data seen by the stock TH14 actor."""

from io import BytesIO
import json
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np

from scripts.checkpoint_storage import MIB, digest


SCHEMA = "th14-stock-actor-transitions-v1"
NATIVE_FIELDS = ("global_tick", "stage_frame", "stage", "pause", "chapter", "player_state",
                 "lives", "bombs", "previous_x", "boss_pointer", "boss_hp", "boss_max_hp",
                 "spell_id", "held_keys_mask", "snapshot_valid")
KEY_ORDER = ("left", "right", "up", "down", "shift", "z", "x", "ctrl")
TERMINAL_CODES = {"ongoing": 0, "clear": 1, "game_over": 2, "time_limit": 3}
BEHAVIOR_CODES = {"anchor": 0, "randomized_bomb": 1, "fixed_followup": 2}


def native_fields(env, native, stage, pause):
    """Keep raw counters and mark stale snapshots at non-gameplay exits."""
    snapshot = env.snapshot
    valid = pause == 2 and snapshot.stage == stage
    boss = snapshot.boss if valid else None
    spell_id = snapshot.spell_id if valid else None
    chapter = (env.reader.read_uint(env.reader.offsets.statics.stage_chapter, relative=True)
               if valid else -1)
    key_mask = sum(1 << index for index, key in enumerate(KEY_ORDER) if key in env.held)
    return [native["global_tick"], native["stage_frame"], stage, pause, chapter,
            native["player_state"] if valid else -1, native["lives"], native["bombs"],
            int("x" in env.held), boss.pointer if boss else -1, boss.hp if boss else -1,
            boss.max_hp if boss else -1, spell_id if spell_id is not None else -1, key_mask, int(valid)]


def source_offsets(env):
    offsets = env.reader.offsets
    return {
        "stage_chapter": offsets.statics.stage_chapter, "continues": offsets.statics.continues,
        "game_thread_pointer": offsets.game_thread.game_thread_pointer,
        "stage_timer": offsets.game_thread.zGameThread_stage_timer,
        "player_pointer": offsets.player.player_pointer, "player_state": offsets.player.zPlayer_state,
        "global_timer_path": env.game._OFFSETS["global_timer"],
        "lives": env.game._OFFSETS["lives"], "bombs": env.game._OFFSETS["bombs"],
    }


def copy_observation(observation):
    return tuple(np.array(observation[name], dtype=np.float32, copy=True) for name in ("bullets", "context"))


class CreditTransitions:
    def __init__(self, offsets):
        self.offsets = offsets
        self.rows = []
        self.final_observation = None

    def add(self, observed, action, before, after, base_bomb, behavior, outcome, losses, bombs):
        both_active = bool(before[-1] and after[-1] and before[2] == after[2])
        exact = both_active and after[0] - before[0] == 4 and after[1] - before[1] == 4
        self.rows.append((observed, action.tolist(), before, after, base_bomb, BEHAVIOR_CODES[behavior],
                          TERMINAL_CODES[outcome or "ongoing"], losses, bombs, exact))

    def arrays(self):
        observations, action, before, after, base_bomb, behavior, outcome, losses, bombs, exact = zip(*self.rows)
        bullets, context = map(np.stack, zip(*observations))
        arrays = dict(bullets=bullets, context=context, action=np.asarray(action, dtype=np.uint8),
                      native_before=np.asarray(before, dtype=np.int64), native_after=np.asarray(after, dtype=np.int64),
                      base_bomb=np.asarray(base_bomb, dtype=np.uint8), behavior=np.asarray(behavior, dtype=np.uint8),
                      terminal_outcome=np.asarray(outcome, dtype=np.uint8),
                      native_life_losses=np.asarray(losses, dtype=np.int16),
                      actual_bombs_consumed=np.asarray(bombs, dtype=np.int16),
                      four_tick_stage_transition=np.asarray(exact, dtype=np.bool_),
                      final_observation_valid=np.asarray(self.final_observation is not None, dtype=np.bool_))
        if self.final_observation is not None:
            arrays["final_bullets"], arrays["final_context"] = self.final_observation
        return arrays


def shard_path(out, trial):
    return out / "transition-shards" / f"credit-{trial['block']:03d}-seed-{trial['policy_seed']}.npz"


def shard_metadata(specification, trial, offsets, rows):
    runtime = specification["runtime"]
    return dict(schema=SCHEMA, trial=trial, assignment_seed=specification["intervention"]["assignment_seed"],
                rows=rows, task=specification["task"], actor=specification["policies"][trial["arm"]],
                source_sha256=runtime["source_sha256"], runner_sha256=runtime["runner_sha256"],
                parakit_offsets_sha256=runtime["parakit_offsets_sha256"], source_offsets=offsets,
                native_field_order=NATIVE_FIELDS, key_bit_order=KEY_ORDER,
                terminal_codes=TERMINAL_CODES, behavior_codes=BEHAVIOR_CODES,
                observation_format="float32 exact actor input; next input is next row unless terminal",
                action_format="Applied four-head command; held-key mask is sampled before and after native execution.",
                invalid_native_field=-1)


def storage_remaining(out, args):
    occupied = sum(path.stat().st_size for path in (out / "transition-shards").glob("credit-*.npz"))
    remaining = min(args.transition_budget_mib * MIB - occupied,
                    shutil.disk_usage(out).free - args.transition_reserve_mib * MIB)
    if remaining <= 0:
        raise RuntimeError("No transition archive budget remains above the disk reserve.")
    return remaining


def descriptor(out, path, rows):
    return dict(path=path.relative_to(out).as_posix(), sha256=digest(path), bytes=path.stat().st_size, rows=rows)


def publish_shard(out, args, specification, trial, result, transitions):
    rows = len(transitions.rows)
    if rows != result["decision_count"]:
        raise ValueError("Recorded transition rows do not match the completed native credit.")
    metadata = shard_metadata(specification, trial, transitions.offsets, rows)
    return publish_archive(out, args, shard_path(out, trial), metadata, result, transitions.arrays())


def publish_archive(out, args, path, metadata, result, arrays):
    """Write a compressed credit and its result atomically within the disk budget."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = None
    try:
        # A credit is already bounded by the native decision limit. Compress in
        # memory so the disk guard uses the actual encoded size.
        with BytesIO() as buffer:
            np.savez_compressed(buffer, metadata_json=json.dumps(metadata, sort_keys=True, allow_nan=False),
                                result_json=json.dumps(result, sort_keys=True, allow_nan=False),
                                **arrays)
            with buffer.getbuffer() as data:
                if len(data) > storage_remaining(out, args):
                    raise RuntimeError("Disk reserve dropped before publishing the native transition archive.")
                with NamedTemporaryFile(dir=path.parent, prefix=".credit-", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return descriptor(out, path, metadata["rows"])


def recover_shard(out, specification, trial):
    """Recover a published NPZ if interruption preceded the JSON report commit."""
    path = shard_path(out, trial)
    metadata, result, rows = read_archive(path)
    expected = shard_metadata(specification, trial, metadata["source_offsets"], rows)
    if metadata != json.loads(json.dumps(expected)) or result["decision_count"] != rows:
        raise ValueError(f"Saved transition shard does not match its episode specification: {path.name}.")
    result["transition_shard"] = descriptor(out, path, rows)
    return result


def read_archive(path):
    """Validate the archive and return its metadata, result and action row count."""
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"Saved transition shard has an invalid ZIP checksum: {path.name}.")
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(data["metadata_json"].item())
        result = json.loads(data["result_json"].item())
        rows = len(data["action"])
        if data["action"].shape != (rows, 4):
            raise ValueError("Saved credit actions do not use the four-headed actor format.")
    return metadata, result, rows

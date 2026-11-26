"""Extract aligned TH14 human replay inputs around observed hit and recovery states."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


SCHEMA = "th14-human-hit-recovery-v1"
BOMB_BIT = 2
HIT_STATE = 4
LEAD_FRAMES = 8
RECOVERY_FRAMES = 32
INVENTORY = ("lives", "bombs", "life_pieces", "bomb_pieces", "power")


def build_windows(arrays, words, offset, stride, control_bits):
    """Use an established replay offset without applying movement eligibility."""
    frames = np.asarray(arrays["stage_frame"], dtype=np.int64)
    ticks = np.asarray(arrays["global_tick"], dtype=np.int64)
    states = np.asarray(arrays["player_state"])
    native = np.asarray(arrays["native"], dtype=np.uint16)
    inventory = np.asarray(arrays["hit_recovery_inventory"], dtype=np.int32)
    if (stride < 1 or frames.ndim != 1 or states.shape != frames.shape or native.shape != frames.shape
            or ticks.shape != frames.shape or inventory.shape != (len(frames), len(INVENTORY))
            or (np.diff(frames) <= 0).any()):
        raise ValueError("Expected ordered native frames with clock, player state, inputs and inventory.")
    hits = frames[states == HIT_STATE]
    # A sample belongs to the union of intervals around all observed hit frames.
    # Search the earliest hit that can still place the sample in the interval.
    start = np.searchsorted(hits, frames - RECOVERY_FRAMES)
    stop = np.searchsorted(hits, frames + LEAD_FRAMES, side="right")
    selected = np.flatnonzero(start < stop)
    indices = frames[selected] + offset
    legal = (indices > 0) & (indices + stride < len(words))
    replay_inputs = np.full(len(selected), -1, dtype=np.int32)
    prior_replay_inputs = replay_inputs.copy()
    chunk = np.zeros((len(selected), stride), dtype=np.uint16)
    aligned = np.zeros(len(selected), dtype=np.bool_)
    edges = np.zeros_like(chunk, dtype=np.bool_)
    if np.any(legal):
        valid_indices = indices[legal]
        replay_inputs[legal] = words[valid_indices]
        prior_replay_inputs[legal] = words[valid_indices - 1]
        chunk[legal] = words[valid_indices[:, None] + np.arange(1, stride + 1)]
        aligned[legal] = (words[valid_indices] & control_bits) == (native[selected[legal]] & control_bits)
        held = (chunk[legal] & BOMB_BIT) != 0
        prior_held = np.column_stack(((words[valid_indices] & BOMB_BIT) != 0, held[:, :-1]))
        edges[legal] = held & ~prior_held

    previous = selected > 0
    prior_sample_inputs = np.full(len(selected), -1, dtype=np.int32)
    prior_stage_delta = np.full(len(selected), -1, dtype=np.int64)
    prior_global_delta = prior_stage_delta.copy()
    following_stage_delta = prior_stage_delta.copy()
    following_global_delta = prior_stage_delta.copy()
    sample = selected[previous]
    prior_sample_inputs[previous] = native[sample - 1]
    prior_stage_delta[previous] = frames[sample] - frames[sample - 1]
    prior_global_delta[previous] = ticks[sample] - ticks[sample - 1]
    following = selected + 1 < len(frames)
    sample = selected[following]
    following_stage_delta[following] = frames[sample + 1] - frames[sample]
    following_global_delta[following] = ticks[sample + 1] - ticks[sample]

    x_edge_at_sample = legal & ((replay_inputs & BOMB_BIT) != 0) & ((prior_replay_inputs & BOMB_BIT) == 0)
    result = {name: np.asarray(arrays[name])[selected] for name in ("bullets", "context", "controllable")}
    result.update(
        stage_frame=frames[selected], global_tick=ticks[selected], player_state=states[selected],
        hit_state=states[selected] == HIT_STATE, inventory=inventory[selected], native_input=native[selected],
        prior_sample_native_input=prior_sample_inputs, prior_sample_stage_delta=prior_stage_delta,
        prior_sample_global_delta=prior_global_delta, next_sample_stage_delta=following_stage_delta,
        next_sample_global_delta=following_global_delta, replay_input_index=indices,
        replay_input_in_range=legal, replay_input_aligned=aligned,
        replay_input_at_sample=replay_inputs, replay_input_before_sample=prior_replay_inputs,
        prior_replay_x_held=np.where(legal, (prior_replay_inputs & BOMB_BIT) != 0, -1).astype(np.int8),
        replay_x_press_edge_at_sample=x_edge_at_sample,
        native_x_held=(native[selected] & BOMB_BIT) != 0,
        expert_action_chunk=chunk, expert_x_held=(chunk & BOMB_BIT) != 0,
        expert_x_press_edge=edges,
    )
    return result


def summarize(windows):
    hit = windows["hit_state"]
    valid = windows["replay_input_aligned"]
    edges = windows["expert_x_press_edge"].any(axis=1)
    return dict(context_rows=len(hit), observed_hit_rows=int(hit.sum()),
                aligned_rows=int(valid.sum()), aligned_hit_rows=int((hit & valid).sum()),
                aligned_hit_rows_with_upcoming_x_edge=int((hit & valid & edges).sum()),
                aligned_context_rows_with_upcoming_x_edge=int((valid & edges).sum()))


def publish_sidecar(path, windows, source):
    """Keep the supplemental record separate from the movement SFT dataset."""
    metadata = dict(source, schema=SCHEMA, label_bit=BOMB_BIT, hit_state=HIT_STATE,
                    lead_frames=LEAD_FRAMES, recovery_frames=RECOVERY_FRAMES,
                    inventory_columns=INVENTORY, label_validity="Use replay_input_aligned for eligible input labels.",
                    prior_x_definition="Decoded exact preceding replay tick; prior native sample may be farther back.",
                    x_label_definition="Replay key commands. Measure any inventory change independently.",
                    **summarize(windows))
    identity_fields = ("schema", "sha256", "stage", "stride", "decoder", "offset")
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            existing = json.loads(archive["metadata_json"].item())
        if any(existing[key] != metadata[key] for key in identity_fields):
            raise ValueError(f"Existing hit/recovery sidecar has a different replay contract: {path.name}.")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".hit-recovery-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, metadata_json=json.dumps(metadata, sort_keys=True, allow_nan=False), **windows)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return metadata

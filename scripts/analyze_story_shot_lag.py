"""Measure lagged associations between held Shoot and native boss HP loss."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from scripts.native_story_transitions import NATIVE_FIELDS, POSITION_FIELDS, SCHEMA
from scripts.native_credit_transitions import KEY_ORDER
from scripts.checkpoint_storage import digest
from scripts.train_full_policy import publish_json


def valid_shot_windows(data, max_hp_step):
    columns = {name: index for index, name in enumerate(NATIVE_FIELDS)}
    before, after = data["native_before"], data["native_after"]
    phases = ["stage", "chapter", "boss_pointer", "boss_max_hp", "spell_id"]
    identifiers = [columns[name] for name in phases]
    same_phase = ((before[:, identifiers] == after[:, identifiers]).all(axis=1)
                  & (before[:, columns["boss_pointer"]] > 0))
    hp_loss = before[:, columns["boss_hp"]] - after[:, columns["boss_hp"]]
    stable = (data["four_tick_stage_transition"] & same_phase
              & (hp_loss >= 0) & (hp_loss <= max_hp_step)
              & (data["native_life_losses"] == 0) & (data["actual_bombs_consumed"] == 0))
    seam = np.zeros(len(before), dtype=bool)
    continuity = [columns[name] for name in ["global_tick", "stage_frame", *phases]]
    seam[1:] = (before[1:, continuity] != after[:-1, continuity]).any(axis=1)
    eligible = stable & (before[:, columns["boss_vulnerable"]] == 1) & (after[:, columns["boss_vulnerable"]] == 1)
    shot = (after[:, columns["held_keys_mask"]] & (1 << KEY_ORDER.index("z"))) != 0
    return (before[:, columns["stage"]], before[:, columns["global_tick"]], stable, seam, hp_loss, eligible, shot)


def summarize_shard(data, lags, max_hp_step):
    stages, native_ticks, stable, seams, hp_loss, eligible, shot = valid_shot_windows(data, max_hp_step)
    excluded = np.r_[0, np.cumsum(~stable)]
    breaks = np.r_[0, np.cumsum(seams)]
    targets = np.flatnonzero(eligible)
    positions = data["positions_before"]
    summaries = {}
    for lag in lags:
        offset = lag // 4
        dst = targets[targets >= offset]
        src = dst - offset
        exact = ((native_ticks[dst] - native_ticks[src] == lag)
                 & (excluded[dst + 1] == excluded[src]) & (breaks[dst + 1] == breaks[src + 1]))
        src, dst = src[exact], dst[exact]
        vertical = positions[src, 1] - positions[src, 3]
        lateral = np.abs(positions[src, 0] - positions[src, 2])
        for stage in np.unique(stages[dst]):
            for held in (0, 1):
                mask = (stages[dst] == stage) & (shot[src] == bool(held))
                if not np.any(mask):
                    continue
                observed = hp_loss[dst[mask]]
                summaries[(int(stage), lag, held)] = np.asarray([
                    len(observed), np.count_nonzero(observed), observed.sum(), vertical[mask].sum(), lateral[mask].sum(),
                ], dtype=np.float64)
    return summaries, int(eligible.sum())


def analyze(shards, lags, max_hp_step):
    aggregate = defaultdict(lambda: np.zeros(5, dtype=np.float64))
    sources, seen = [], set()
    for path in shards:
        fingerprint = digest(path)
        if fingerprint in seen:
            raise ValueError(f"Duplicate story transition shard: {path.name}.")
        seen.add(fingerprint)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata_json"].item())
            if (metadata["schema"] != SCHEMA or tuple(metadata["native_field_order"]) != NATIVE_FIELDS
                    or tuple(metadata["position_field_order"]) != POSITION_FIELDS
                    or tuple(metadata["native_key_bit_order"]) != KEY_ORDER):
                raise ValueError(f"Expected a native story transition archive: {path.name}.")
            arrays = {name: archive[name] for name in (
                "native_before", "native_after", "four_tick_stage_transition", "native_life_losses",
                "actual_bombs_consumed", "positions_before",
            )}
        summaries, eligible = summarize_shard(arrays, lags, max_hp_step)
        trial = metadata["trial"]
        sources.append(dict(trial=trial, sha256=fingerprint, transitions=metadata["rows"], eligible_hp_steps=eligible))
        for key, totals in summaries.items():
            aggregate[(trial["arm"], *key)] += totals

    rows = []
    for (arm, stage, lag, held), (n, positive, damage, vertical, lateral) in sorted(aggregate.items()):
        rows.append(dict(arm=arm, native_stage=stage, lag_native_ticks=lag, held_shoot=bool(held),
                         step_count=int(n), positive_hp_loss_steps=int(positive),
                         total_hp_loss=int(damage), mean_hp_loss_per_step=damage / n,
                         mean_player_minus_boss_y=vertical / n, mean_absolute_lateral_gap=lateral / n))
    return dict(
        interpretation="Observational lag scan. Shoot timing and position were chosen by the recorded policy.",
        lag_definition="Source command native tick to the start of the measured target HP interval.",
        event_filter="Exact four-tick, same-phase intervals. Window excludes life loss, Bomb consumption and clock gaps.",
        damage_filter="Both target snapshots vulnerable; positive changes bounded by max_hp_step.",
        max_hp_step=max_hp_step, lags_native_ticks=lags, sources=sources, rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--lags", nargs="+", type=int, default=list(range(0, 81, 4)))
    parser.add_argument("--max-hp-step", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.max_hp_step < 1 or any(lag < 0 or lag % 4 for lag in args.lags):
        parser.error("Use a positive HP step bound and nonnegative four-tick lags.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    publish_json(args.out, analyze(args.shards, sorted(set(args.lags)), args.max_hp_step))


if __name__ == "__main__":
    main()

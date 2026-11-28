"""Recount inventory events in the signed TH07 Normal Game Start training credits."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from scripts.audit_th07_normal_letty_reach import SOURCE_SEEDS, load
from scripts.score_th07_held_practice import event_rows
from scripts.th07_retail_fragment_intake import PARENT, game_over_boundary, require, sha, table


def credit(folder, witness, members):
    verified_path, native_path, endpoint_path = (
        folder / name for name in ("verified.json", "native-input.csv", "finished-game.json")
    )
    verified, endpoint = load(verified_path), load(endpoint_path)
    require(all(sha(path) == members[f"{folder.name}/{path.name}"]
                for path in (verified_path, native_path, endpoint_path))
            and sha(verified_path) == witness["original_verified_sha256"]
            and sha(native_path) == verified["artifacts_sha256"]["native-input.csv"],
            "Ordinary original resource sources changed")
    require(verified["seed"] == endpoint["seed"] == witness["seed"]
            and verified["role"] == endpoint["role"] == witness["role"] == "train"
            and verified["sha256"] == endpoint["sha256"] == PARENT
            and verified["passed"] and not verified["stage2"] and verified["terminal_visually_confirmed"],
            "Expected the signed ordinary Normal parent training credit")
    native = table(native_path)
    first_tick = verified["camera_start_tick"]
    end = game_over_boundary(native, endpoint, first_tick, None)
    live = native[(native["tick"] >= first_tick) & (native["ns"] <= end["ns"])
                  & ((native["flags"] & 4) != 0)]
    require(np.all(live["stage"] == 1) and np.all(live["rank"] == 1)
            and np.all(live["shot"] == 0) and not np.any(live["continues"])
            and np.all(np.diff(live["ns"]) > 0) and np.all(np.diff(live["tick"]) >= 0)
            and live[0]["tick"] == first_tick and live[-1]["tick"] == end["tick"],
            "The original playable inventory cannot be aligned to the native clock")
    deaths, bombs = (event_rows(live, count) for count in ("misses", "bombs_used"))
    require(sum(e["amount"] for e in deaths) == verified["stage1_deaths"]
            and sum(e["amount"] for e in bombs) == verified["actual_bombs"],
            "Native miss or Bomb spend disagrees with the original credit verifier")
    spare_rises = np.flatnonzero(np.diff(live["lives"]) > 0)
    bomb_rises = np.flatnonzero(np.diff(live["bombs"]) > 0)
    preceding_miss = np.searchsorted([event["tick"] for event in deaths], live["tick"][bomb_rises + 1], side="right") - 1
    arrival = witness["first_confirmed_witness"]
    arrival_stamp = None if arrival is None else arrival["native_source_ns"]
    if arrival is not None:
        matching = live[live["ns"] == arrival_stamp]
        require(len(matching) == 1 and matching[0]["tick"] == arrival["native_tick"]
                and matching[0]["lives"] == arrival["native_spares"] and matching[0]["bombs"] == arrival["native_bombs"],
                "The signed visual Letty witness failed native inventory alignment")
    miss_delta = np.diff(live["misses"])
    bomb_delta = np.diff(live["bombs_used"])
    before_witness = None
    if arrival_stamp is not None:
        before_witness = live[1:]["ns"] <= arrival_stamp
    return dict(seed=witness["seed"], start_tick=first_tick, terminal_tick=int(end["tick"]),
                native_input_sha256=sha(native_path), original_endpoint_sha256=sha(endpoint_path),
                deaths=deaths, bomb_spends=bombs, original_starts_with=[int(live[0][key]) for key in ("lives", "bombs")],
                spare_inventory_increases=[dict(tick=int(live[i + 1]["tick"]),
                                                units=int(live[i + 1]["lives"] - live[i]["lives"])) for i in spare_rises],
                bomb_inventory_increases=[dict(tick=int(live[i + 1]["tick"]),
                                               units=int(live[i + 1]["bombs"] - live[i]["bombs"]),
                                               native_ticks_since_last_miss=None if miss < 0 else
                                               int(live[i + 1]["tick"]) - deaths[miss]["tick"])
                                          for i, miss in zip(bomb_rises, preceding_miss, strict=True)],
                first_verified_letty_tick=None if arrival is None else arrival["native_tick"],
                through_visual_witness=None if arrival is None else dict(
                    deaths=int(miss_delta[before_witness].sum()),
                    bombs_spent=int(bomb_delta[before_witness].sum()),
                    spares=arrival["native_spares"], bombs=arrival["native_bombs"]))


def audit(credits, reach_source, release, destination):
    reach = load(reach_source)
    require(tuple(row["seed"] for row in reach["rows"]) == SOURCE_SEEDS,
            "Expected the exact published ordinary Normal training cohort")
    members = load(release)["member_sha256"]
    rows = [credit(credits / f"{row['seed']}-parent", row, members) for row in reach["rows"]]
    deaths = [event for row in rows for event in row["deaths"]]
    bomb_rises = [event for row in rows for event in row["bomb_inventory_increases"]]
    through_letty = [row["through_visual_witness"] for row in rows if row["through_visual_witness"] is not None]
    resource_summary = dict(
        credits=len(rows), observed_playable_native_ticks=sum(row["terminal_tick"] - row["start_tick"] for row in rows),
        deaths=sum(event["amount"] for event in deaths),
        deaths_by_prior_bomb_stock=dict(sorted(Counter(event["prior_bombs"] for event in deaths).items())),
        deaths_with_spare_and_prior_bombs=sum(event["prior_bombs"] > 0 and event["prior_spares"] > 0 for event in deaths),
        final_deaths_with_prior_bombs=sum(row["deaths"][-1]["prior_bombs"] > 0 for row in rows),
        actual_bombs_spent=sum(event["amount"] for row in rows for event in row["bomb_spends"]),
        observed_bomb_inventory_increases=len(bomb_rises),
        bomb_inventory_rises_by_ticks_since_last_miss=dict(Counter(
            str(event["native_ticks_since_last_miss"]) for event in bomb_rises)),
        observed_spare_inventory_increases=sum(len(row["spare_inventory_increases"]) for row in rows),
        visually_witnessed_arrivals=len(through_letty), deaths_through_visual_arrival=sum(row["deaths"] for row in through_letty),
        actual_bombs_spent_through_visual_arrival=sum(row["bombs_spent"] for row in through_letty),
    )
    report = dict(scope="Exploratory audit of the signed TH07 retail 1.00b Normal training Game Starts",
                  letty_witness_sha256=sha(reach_source), source_release_sha256=sha(release), policy_sha256=PARENT,
                  summary=resource_summary, rows=rows,
                  interpretation=("Bomb and miss events use increments in native counters. Prior inventory is the immediately "
                                  "previous observer sample. A counter increment does not identify item class, "
                                  "deathbomb timing, or the counterfactual survival effect of using a Bomb. "
                                  "Letty arrival counts end at the first witnessed name frame."))
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(resource_summary, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("credits", "reach_source", "release", "destination"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    audit(args.credits, args.reach_source, args.release, args.destination)

"""Verify a parent-only, original Normal full Letty survival and phase diagnostic."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.score_th07_held_practice import MARKERS, describe, load, observed_practice, save
from scripts.score_th07_nonspell2_held import arm_clear
from scripts.th07_retail_fragment_intake import (
    DATA, GAME, PARENT, intake_fragment, playable_boundary, require, sha, slot_words, table,
)


SCENARIOS = {"letty-2-3": (2, 3), "letty-0-1": (0, 1), "letty-0-0": (0, 0)}
PHASE_ORDER = ("nonspell1_sub38", "normal_spell1_id3", "nonspell2_sub39", "normal_final_spell_id7")
REQUIRED = (*MARKERS, "spell-monitor-closure.csv", "spell-phase.csv")
HELD_SEEDS = set(range(21750001, 21750029))


def phase_signatures(phase):
    passive = (phase["active"] == 0) & (phase["boss_present"] != 0)
    return (
        passive & (phase["boss_sub"] == 38) & (phase["lifetime_callback"] == 1700)
        & (phase["timer_callback_tick"] == 2400) & (phase["timer_callback_sub"] == 42),
        (phase["active"] != 0) & (phase["spell_index"] == 3),
        passive & (phase["boss_sub"] == 39) & (phase["lifetime_callback"] == 2000)
        & (phase["timer_callback_tick"] == 2400) & (phase["timer_callback_sub"] == 48),
        (phase["active"] != 0) & (phase["spell_index"] == 7),
    )


def phase_timeline(folder, measured, scenario):
    qualification_path = folder / "qualification.json"
    qualification = load(qualification_path)
    native_path = folder / "native-input.csv"
    menu = folder / "configured-practice.png"
    require(qualification["scenario"] == scenario and qualification["warp"] == "End Boss"
            and qualification["rank"] == "Normal" and qualification["team"] == "Reimu A"
            and qualification["game_sha256"] == GAME and qualification["data_sha256"] == DATA
            and qualification["stage"] == qualification["native_first_tick"] == 1
            and qualification["source_save_present"] is False and qualification["continues"] == 0
            and tuple(qualification["native_observed"]) == SCENARIOS[scenario]
            and sha(menu) == qualification["source_sha256"]["configured-practice.png"]
            and sha(native_path) == qualification["source_sha256"]["native-input.csv"],
            "Native full Letty practice reset does not match its exact original roster")
    native = table(native_path)
    live, slots = slot_words(folder, native, "practice")
    stage1 = slots[1]
    end = measured["final_native_tick"]
    require(all(tick in stage1 for tick in range(1, min(120, end) + 1)),
            "The initial playable native original is missing current/previous record slots")
    path = folder / "spell-phase.csv"
    closure_path = folder / "spell-monitor-closure.csv"
    observed = table(path)
    closure = table(closure_path)
    require(len(closure) == 1 and len(observed) > 0 and np.all(np.diff(observed["ns"]) > 0)
            and closure[0]["reason"] == "worker_finished"
            and closure[0]["last_stage_tick"] == observed[-1]["stage_tick"]
            and closure[0]["last_supervisor"] == observed[-1]["supervisor"]
            and closure[0]["monotonic_ns"] >= observed[-1]["ns"],
            "Independent passive monitor did not have a verified worker-finished closure")
    phase = observed[(observed["supervisor"] == 2) & (0 < observed["stage_tick"]) & (observed["stage_tick"] <= end)]
    phase = phase[np.array([stage1.get(int(row["stage_tick"])) == (int(row["native_keys"]) & 247)
                            for row in phase], dtype=bool)]
    stage_state = {int(row["tick"]): row for row in live if 0 < row["tick"] <= end and not row["retry"]}
    witnesses = {}
    last_phase_start = 0
    previous_phase_seen = True
    prefix = []
    for label, match in zip(PHASE_ORDER, phase_signatures(phase), strict=True):
        ticks = sorted({int(tick) for tick in phase["stage_tick"][match] if int(tick) in stage_state})
        ticks_after_prior = [tick for tick in ticks if tick > last_phase_start]
        if len(ticks) >= 2:
            state = stage_state[ticks[0]]
            witnesses[label] = dict(first_tick=ticks[0], last_tick=ticks[-1], source_matched_ticks=len(ticks),
                                    first_stock=[int(state["lives"]), int(state["bombs"])],
                                    native_misses_at_first=int(state["misses"]), enters_verified_prefix=False)
        if previous_phase_seen and len(ticks_after_prior) >= 2:
            last_phase_start = ticks_after_prior[0]
            prefix.append(label)
            witnesses[label]["enters_verified_prefix"] = True
            witnesses[label]["first_ordered_tick"] = last_phase_start
        else:
            previous_phase_seen = False
    if measured["outcome"] == "practice_stage_clear":
        visual = load(folder / "practice-result-visual.json")
        tally_tick = visual["tally_native_tick"]
        tally = phase[(phase["stage_tick"] <= tally_tick) & (phase["finished_stage"] == 2)]
        require(tuple(prefix) == PHASE_ORDER and last_phase_start < tally_tick and len(tally)
                and visual["native_phase"]["phase_sha256"] == sha(path) and tally_tick <= end,
                "Actual Stage Clear lacks the independently matched full Normal Sub38/ID3/Sub39/ID7 sequence")
    return dict(
        spell_phase_sha256=sha(path), observed_phase_native_clock_rows=len(phase),
        observer_closure_sha256=sha(closure_path), observer_last_passive_tick=int(closure[0]["last_stage_tick"]),
        samples_after_original_boundary=int(np.count_nonzero(observed["stage_tick"] > end)),
        verified_playable_phase_witnesses=witnesses, independently_confirmed_phase_prefix=prefix,
        last_confirmed_phase=prefix[-1] if prefix else None,
        observed_original_spell_counts=sorted({int(x) for x in phase["spell_count"] if x >= 0}),
        reset_menu_sha256=sha(menu), reset_qualification_sha256=sha(qualification_path),
    )


def verify(folder, actor, planned, registration_hash):
    path = folder / "finished-game.json"
    original = load(path)
    fields = ("seed", "role", "arm", "host", "lane", "scenario", "decision_cap", "initial_spares", "initial_bombs")
    require(all(original[key] == planned[key] for key in fields)
            and original["policy_sha256"] == original["sha256"] == PARENT
            and original["registration_sha256"] == registration_hash
            and original["role"] in ("held", "engineering"),
            "The full Letty original diverged from its prospective roster")
    scenario = original["scenario"]
    horizon = 4 * original["decision_cap"]
    stock = SCENARIOS[scenario]
    physical = observed_practice(folder, original, horizon=horizon, expected_stock=stock)
    # A playable cutoff also needs its actual original next camera even for physical-only review.
    if physical["outcome"] == "native_tick_cutoff":
        playable_boundary(folder, table(folder / "native-input.csv"), original)
    timeline = phase_timeline(folder, physical, scenario)
    identity = {key: planned[key] for key in fields}
    try:
        policy = intake_fragment(folder, actor, None, horizon=horizon)
    except (ValueError, OSError, KeyError, AssertionError) as exc:
        return dict(identity, **physical, **timeline, qualified=False, policy_validation_error=str(exc))
    require(policy["outcome"] == physical["outcome"] and policy["native_elapsed_ticks"] == physical["playable_native_ticks"],
            "Native physical endpoint and frozen policy receipt disagree")
    return dict(identity, **physical, **timeline, qualified=policy["qualified"], strict_source_receipt=policy)


def describe_stock(rows):
    physical = [row for row in rows if row["physical_endpoint_verified"]]
    clear = sum(row["outcome"] == "practice_stage_clear" for row in physical)
    strict = [row for row in physical if row["qualified"]]
    return dict(
        assigned=len(rows), original_outcomes=len(physical), strict_controls=len(strict),
        verified_clears_all_assigned=arm_clear(clear, len(rows)),
        verified_clears_known_originals=arm_clear(clear, len(physical)),
        verified_clears_strict_qualified=arm_clear(sum(row["outcome"] == "practice_stage_clear" for row in strict), len(strict)),
        phase_prefix_reach={phase: sum(phase in row["independently_confirmed_phase_prefix"] for row in physical)
                            for phase in PHASE_ORDER},
        last_verified_phase_counts=dict(Counter(row["last_confirmed_phase"] or "unresolved" for row in physical)),
        unresolved_physical_sources=[row["seed"] for row in rows if not row["physical_endpoint_verified"]],
        native_resources=describe(physical),
    )


def read_registration(study, filename, expected, actor):
    path = study / filename
    plan = load(path)
    rows = plan["rows"]
    require(sha(path) == expected and plan["policy_sha256"] == sha(actor) == PARENT and plan["game_sha256"] == GAME
            and plan["native_rank"] == 1 and plan["starting_power"] == 128 and plan["continues"] == 0
            and all(row["arm"] == "parent" and tuple((row["initial_spares"], row["initial_bombs"])) == SCENARIOS[row["scenario"]]
                    and 4 * row["decision_cap"] == plan["cap_native_elapsed"] for row in rows),
            "Independent full Letty registration, scope or parent bytes changed")
    return plan


def watch(study, actor, filename, expected):
    plan = read_registration(study, filename, expected, actor)
    require(len(plan["rows"]) == len(HELD_SEEDS) and {row["seed"] for row in plan["rows"]} == HELD_SEEDS
            and all(row["role"] == "held" for row in plan["rows"]) and plan["cap_native_elapsed"] == 15000,
            "Full horizon scoring requires the complete disjoint 28-original diagnostic roster")
    output = study / "independent"
    output.mkdir(exist_ok=True)
    order = {row["seed"]: row for row in plan["rows"]}
    complete = {}
    published = None
    start = time.monotonic()
    while time.monotonic() - start < 18000:
        for seed, term in order.items():
            if seed in complete:
                continue
            folder = study / "raw" / str(seed)
            target = output / f"{seed}.json"
            if target.is_file():
                row = load(target)
            else:
                failed = folder / "attempt-failed.json"
                if failed.is_file():
                    fault = load(failed)
                    require(fault["seed"] == seed and fault["role"] == "held", "Wrong preassigned diagnostic failure")
                    row = dict(term, qualified=False, physical_endpoint_verified=False, failure_sha256=sha(failed))
                elif all((folder / name).is_file() for name in REQUIRED):
                    status = load(folder / "finished-game.json")
                    if status["outcome"] == "practice_stage_clear" and not (folder / "practice-result-visual.json").is_file():
                        continue
                    try:
                        row = verify(folder, actor, term, expected)
                    except (ValueError, OSError, KeyError, AssertionError) as exc:
                        row = dict(term, qualified=False, physical_endpoint_verified=False, unresolved_reason=str(exc),
                                   occurred_source_endpoint_sha256=sha(folder / "finished-game.json"))
                else:
                    continue
                row["registration_sha256"] = expected
                save(target, row)
                print(seed, row["qualified"], row.get("outcome"), row.get("independently_confirmed_phase_prefix"), flush=True)
            require(row["seed"] == seed and row["role"] == "held" and row["registration_sha256"] == expected,
                    "An existing full Letty receipt changed role or assignment")
            complete[seed] = row
        if published != len(complete):
            missing = sorted(HELD_SEEDS - complete.keys())
            rows = [complete[seed] for seed in order if seed in complete]
            index = output / "intake-index.json"
            save(index, dict(registration_sha256=expected, role="held", dispositions=len(rows), missing=missing,
                             complete=not missing, known_originals=sum(row["physical_endpoint_verified"] for row in rows),
                             qualified=sum(row["qualified"] for row in rows),
                             rows=[dict(seed=row["seed"], receipt=f"{row['seed']}.json", sha256=sha(output / f"{row['seed']}.json"))
                                   for row in rows]))
            if not missing:
                scenarios = sorted({row["scenario"] for row in rows})
                save(study / "independent-results.json",
                     dict(role="held", scope="Original Normal full Letty parent-only diagnostic. Separate from paired variants.",
                          registration_sha256=expected, index_sha256=sha(index),
                          preliminary_stage_entry_counts_by_stock={key: describe_stock([r for r in rows if r["scenario"] == key])
                                                                  for key in scenarios},
                          phase_method=("Native slots confirm a chronologically witnessed phase prefix, tied to original key records. "
                                        "An earlier loss may close an original before a later phase. A true clear needs all phases "
                                        "and the original archived Stage Clear tally.")))
                return
            published = len(complete)
        time.sleep(8)
    raise TimeoutError("Unclosed full Letty sources remained in the independent diagnostic for five hours")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("study", type=Path)
    cli.add_argument("actor", type=Path)
    cli.add_argument("registration_sha256")
    cli.add_argument("--registration", default="registration.json")
    cli.add_argument("--engineering-seed", type=int)
    args = cli.parse_args()
    torch.set_num_threads(1)
    if args.engineering_seed is None:
        watch(args.study, args.actor, args.registration, args.registration_sha256)
    else:
        plan = read_registration(args.study, args.registration, args.registration_sha256, args.actor)
        item = next(item for item in plan["rows"] if item["seed"] == args.engineering_seed)
        require(item["role"] == "engineering", "Only original engineering may be replayed through the single audit command")
        row = verify(args.study / "raw" / str(item["seed"]), args.actor, item, args.registration_sha256)
        target = args.study / f"independent-engineering-{item['seed']}.json"
        save(target, row)
        print(json.dumps({key: row[key] for key in ("outcome", "first_native_tick", "playable_native_ticks", "qualified",
                                                    "independently_confirmed_phase_prefix")}), sha(target))

"""Seal registered original TH07 Letty practice TRAIN sources as they arrive."""

import argparse
import json
from pathlib import Path
import time
from typing import NamedTuple

import numpy as np
import torch


from scripts.th07_retail_fragment_intake import (
    DATA, GAME, PARENT, intake_fragment, require, sha, slot_words, table,
)


READY = ("qualification.json", "finished-game.json", "build.sha256", "archive.sealed", "events.bin",
         "native-input.csv", "applied.csv", "controls.csv", "policy-trace.npz", "spell-phase.csv")
FINAL = "letty-final-2-3"
PRECEDING = "letty-nonspell2-2-3"
LOW_STOCK = "letty-nonspell2-0-1"


class Scene(NamedTuple):
    warp: str
    label: str
    stock: tuple[int, int]


SCENES = {
    FINAL: Scene("Final Spell", "letty_final", (2, 3)),
    PRECEDING: Scene("Boss Nonspell 2", "letty_nonspell2", (2, 3)),
    LOW_STOCK: Scene("Boss Nonspell 2", "letty_nonspell2_low", (0, 1)),
}


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            pending.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def exact_target(source, scenario=FINAL, *, require_active_spell=False):
    reset = read(source / "qualification.json")
    scene = SCENES[scenario]
    require(reset["scenario"] == scenario and reset["stage"] == reset["native_first_tick"] == 1
            and reset["warp"] == scene.warp and reset["rank"] == "Normal" and reset["team"] == "Reimu A"
            and reset["game_sha256"] == GAME and reset["data_sha256"] == DATA
            and tuple(reset["native_observed"]) == scene.stock
            and reset["source_save_present"] is False and reset["continues"] == 0,
            "Original reset did not authenticate the preregistered practice scene and stock")
    native_path = source / "native-input.csv"
    require(sha(native_path) == reset["source_sha256"]["native-input.csv"]
            and sha(source / "configured-practice.png") == reset["source_sha256"]["configured-practice.png"],
            "Practice menu or native-clock evidence changed")
    native = table(native_path)
    require(np.all(native["stage"] == 1), "Target Stage 1 practice changed its real original stage")
    active, slots = slot_words(source, native, "practice")
    first = active[(active["tick"] >= 1) & (active["tick"] <= 120)]
    # A following row can independently witness the preceding original slot.
    # Direct observer cadence alone is not proof that a native tick was lost.
    require(all(tick in slots[1] for tick in range(1, 121))
            and np.all(first["word"] == first["record_slot"])
            and np.all(first["previous_word"] == first["previous_slot"]),
            "Practice start does not have an exact first 120 native slots")
    path = source / "spell-phase.csv"
    phase = table(path)
    selected = phase[(phase["supervisor"] == 2) & (phase["active"] != 0) & (phase["spell_index"] == 7)
                     & (phase["stage_tick"] > 0)]
    def matches(samples):
        return sorted({int(row["stage_tick"]) for row in samples
                       if slots[1].get(int(row["stage_tick"])) == (int(row["native_keys"]) & 247)})

    matched = matches(selected)
    nonspell = []
    if scenario != FINAL:
        previous = phase[(phase["supervisor"] == 2) & (phase["boss_present"] != 0)
                         & (phase["boss_sub"] == 39) & (phase["active"] == 0)
                         & (phase["lifetime_callback"] == 2000) & (phase["timer_callback_tick"] == 2400)
                         & (phase["timer_callback_sub"] == 48) & (phase["stage_tick"] > 0)]
        nonspell = matches(previous)
        require(nonspell, "No independently clocked original Boss Nonspell 2 Sub39 with native callbacks")
        matched = [tick for tick in matched if tick > nonspell[-1]]
    if scenario == FINAL or require_active_spell:
        require(matched, "No independently clocked active Normal Final Spell ID 7 in the captured original")
    return dict(spell_id=7 if matched else None, active_native_samples=len(matched),
                first_active_native_tick=matched[0] if matched else None,
                last_active_native_tick=matched[-1] if matched else None,
                original_nonspell_sub=39 if nonspell else None, preceding_native_samples=len(nonspell),
                first_preceding_tick=nonspell[0] if nonspell else None,
                last_preceding_tick=nonspell[-1] if nonspell else None,
                phase_sha256=sha(path), native_first_120_both_slots=True,
                direct_clock_rows_first_120_ticks=int(np.unique(first["tick"]).size),
                native_spell_count_observations=sorted({int(value) for value in phase["spell_count"] if value >= 0}),
                native_finished_stage_observations=sorted({int(value) for value in phase["finished_stage"] if value >= 0}),
                reset_sha256=sha(source / "qualification.json"))


def validate(source, item, actor, registration_hash, output):
    spec = read(source / "finished-game.json")
    stock = SCENES[item["scenario"]].stock
    require(all(spec[key] == item[key] for key in ("seed", "role", "host", "lane", "scenario", "decision_cap", "arm"))
            and spec["role"] in ("train", "engineering") and spec["policy_sha256"] == spec["sha256"] == PARENT
            and spec["registration_sha256"] == registration_hash and spec["native_rank"] == 1
            and (spec["initial_spares"], spec["initial_bombs"]) == stock and spec["continues"] == 0,
            "Practice source changed its authorized assignment, actor or initial inventory")
    visual_path = source / "practice-result-visual.json"
    visual = read(visual_path) if visual_path.is_file() else {}
    positive_claim = visual.get("stage_clear_verified") is True
    target = exact_target(source, item["scenario"], require_active_spell=positive_claim)
    if positive_claim:
        require(visual["native_phase"]["phase_sha256"] == target["phase_sha256"]
                and 2 in target["native_finished_stage_observations"],
                "Visual Stage Clear proof did not bind the original finished-stage progression")
    verified = intake_fragment(source, actor, output, horizon=4 * item["decision_cap"])
    require(tuple(verified["first_stock"]) == stock and verified["source_kind"] == "practice",
            "The actual first controlled practice state differs from the registered initial stock")
    return target, verified


def load_registrations(study, actor, registration_hash, keys, scenarios, horizon, allocations):
    registry = study / "registration.json"
    plan = read(registry)
    require(sha(registry) == registration_hash and sha(actor) == PARENT
            and tuple(item["seed"] for item in plan["rows"]) == keys
            and plan["role"] == "train" and plan["policy_sha256"] == PARENT
            and plan["game_sha256"] == GAME and plan["native_rank"] == 1 and plan["rank"] == "Normal"
            and plan["continues"] == 0 and plan["starting_power"] == 128 and plan["cap_native_elapsed"] == horizon
            and all((r["role"], r["scenario"], 4 * r["decision_cap"]) == ("train", scenarios[r["seed"]], horizon)
                    for r in plan["rows"]),
            "Practice registration does not bind the authorized source roster")
    verified = {registration_hash: {row["seed"]: row for row in plan["rows"]}}
    fixed_fields = ("role", "game_sha256", "policy_sha256", "native_rank", "rank", "starting_power",
                    "continues", "cap_native_elapsed", "practice_tool_sha256", "shot")
    for name, digest in allocations:
        path = study / name
        amendment = read(path)
        require(sha(path) == digest and all(plan[key] == amendment[key] for key in fixed_fields)
                and tuple(row["seed"] for row in amendment["rows"]) == keys,
                "Practice allocation changed fixed scope or did not match the authorized SHA")
        for fixed, row in zip(plan["rows"], amendment["rows"], strict=True):
            require(all(fixed[key] == row[key] for key in ("seed", "role", "scenario", "decision_cap", "arm")),
                    "Practice allocation changed a fixed row")
        verified[digest] = {row["seed"]: row for row in amendment["rows"]}
    return verified


def verify_endpoint_recovery(study, source, seed, digest):
    path = source / "endpoint-recovery.json"
    recovery = read(path)
    original_failure = source / "attempt-failed.json"
    failure = read(original_failure)
    files = recovery["source_sha256"]
    unchanged = ("controls.csv", "applied.csv", "native-input.csv", "policy-trace.npz",
                 "archive.sealed", "events.bin", "practice-result-state.csv")
    require(sha(path) == digest and recovery["seed"] == failure["seed"] == seed
            and recovery["role"] == failure["role"] == "train"
            and recovery["original_attempt_sha256"] == sha(original_failure)
            and recovery["format_amendment_sha256"] == sha(study / "endpoint-format-amendment.json")
            and all(files[name] == sha(source / name) for name in (*unchanged, "finished-game.json", "qualification.json",
                                                                   "native-practice-result.png", "postterminal-salvage.json"))
            and all(files[name] == failure["lossless_evidence_files"][name] for name in unchanged),
            "Recovered terminal package changed its authorized original or physical samples")


def watch(study, actor, output, registration_hash, first, count, scenario, horizon, low_stock_from, allocations, recoveries):
    keys = tuple(range(first, first + count))
    scenarios = {seed: LOW_STOCK if low_stock_from is not None and seed >= low_stock_from else scenario for seed in keys}
    plans = load_registrations(study, actor, registration_hash, keys, scenarios, horizon, allocations)
    output.mkdir(parents=True, exist_ok=True)
    dispositions = {}
    published_count = None
    start = time.monotonic()
    while len(dispositions) < count:
        for seed in keys:
            if seed in dispositions:
                continue
            selected = scenarios[seed]
            source = study / "raw" / str(seed)
            index = output / f"{seed}-source.json"
            normal = output / f"{seed}.npz"
            normalized = normal.with_suffix(".json")
            fault = source / "attempt-failed.json"
            fields = dict(seed=seed, scene=SCENES[selected].label, role="train", source_kind="practice", native_rank=1)
            recovered = recoveries.get(seed)
            existing = read(index) if index.is_file() else None
            if existing is not None and existing.get("endpoint_recovery_sha256") == recovered:
                result = existing
            elif fault.is_file() and recovered is None:
                result = dict(fields, qualified=False, technical_failure=read(fault), original_failure_sha256=sha(fault))
                write(index, result)
            elif all((source / marker).is_file() for marker in READY):
                spec = read(source / "finished-game.json")
                if spec["outcome"] == "practice_stage_clear" and not (source / "practice-result-visual.json").is_file():
                    continue
                try:
                    if recovered is not None:
                        verify_endpoint_recovery(study, source, seed, recovered)
                        fields.update(endpoint_recovery_sha256=recovered, original_failure_sha256=sha(fault))
                    digest = spec["registration_sha256"]
                    require(digest in plans, "Source claims a practice allocation not authorized for this roster")
                    item = plans[digest][seed]
                    if normal.is_file() and normalized.is_file():
                        receipt = read(normalized)
                        target = exact_target(source, selected, require_active_spell=receipt["outcome"] == "practice_stage_clear")
                        require(sha(normal) == receipt["sha256"] and receipt["endpoint_sha256"] == sha(source / "finished-game.json")
                                and all(receipt[key] == item[key] for key in ("seed", "role", "host", "lane", "scenario"))
                                and receipt["registration_sha256"] == digest,
                                "Existing normalized practice source changed")
                    else:
                        target, receipt = validate(source, item, actor, digest, normal)
                    result = dict(fields, **target, file=normal.name, sha256=receipt["sha256"],
                                  receipt=normalized.name, receipt_sha256=sha(normalized), qualified=receipt["qualified"],
                                  outcome=receipt["outcome"], decisions=receipt["decisions"],
                                  native_ticks=receipt["native_elapsed_ticks"],
                                  early_exhaustion=spec["outcome"] == "practice_stage_clear" and receipt["outcome"] == "game_over",
                                  verified_section_complete=receipt["outcome"] == "practice_stage_clear",
                                  original_saved_replays_sha256={
                                      path.relative_to(source).as_posix(): sha(path)
                                      for path in sorted((source / "official-replays").glob("*.rpy"))
                                      if path.is_file() and path.read_bytes()[:4] == b"T7RP"},
                                  spell_capture="unverified")
                except (OSError, ValueError, KeyError) as error:
                    result = dict(fields, qualified=False, technical_failure=str(error),
                                  native_endpoint_sha256=sha(source / "finished-game.json"))
                write(index, result)
            else:
                continue
            dispositions[seed] = result
            print(json.dumps(result), flush=True)
        rows = [dispositions[seed] for seed in keys if seed in dispositions]
        missing = [seed for seed in keys if seed not in dispositions]
        if published_count != len(rows):
            write(output / "intake-index.json", dict(
                registration_sha256=registration_hash, missing=missing, complete=not missing, rows=rows,
                qualified=sum(r["qualified"] for r in rows),
                qualified_section_completions=sum(r["qualified"] and r.get("verified_section_complete", False) for r in rows)))
            published_count = len(rows)
        if not missing:
            return
        if time.monotonic() - start > 7200:
            raise TimeoutError(f"No final original disposition for {missing} after two hours")
        time.sleep(8)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("study", type=Path)
    cli.add_argument("actor", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("registration_sha256")
    cli.add_argument("--first", type=int, default=21749601)
    cli.add_argument("--count", type=int, default=8)
    cli.add_argument("--scenario", choices=tuple(SCENES), default=FINAL)
    cli.add_argument("--horizon", type=int, default=6000)
    cli.add_argument("--low-stock-from", type=int)
    cli.add_argument("--allocation", nargs=2, action="append", default=[], metavar=("FILENAME", "SHA256"))
    cli.add_argument("--endpoint-recovery", nargs=2, action="append", default=[], metavar=("SEED", "SHA256"))
    args = cli.parse_args()
    if args.low_stock_from is not None and (args.scenario != PRECEDING
                                           or not args.first <= args.low_stock_from < args.first + args.count):
        cli.error("--low-stock-from must be a registered seed in a preceding-attack roster")
    recoveries = {int(seed): digest for seed, digest in args.endpoint_recovery}
    if not all(args.first <= seed < args.first + args.count for seed in recoveries):
        cli.error("--endpoint-recovery must only bind registered seeds")
    torch.set_num_threads(1)
    watch(args.study, args.actor, args.output, args.registration_sha256, args.first, args.count, args.scenario,
          args.horizon, args.low_stock_from, args.allocation, recoveries)

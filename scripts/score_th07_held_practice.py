"""Independently audit and score paired TH07 Normal Letty practice originals."""

import argparse
from collections import Counter
import json
from pathlib import Path
from runpy import run_path
import time

import numpy as np
import torch

from scripts.intake_th07_letty_starts import PRECEDING, exact_target
from scripts.th07_retail_fragment_intake import (
    DATA, GAME, OBSERVER, PARENT, game_over_boundary, intake_fragment, native_boundary_row,
    playable_boundary, practice_result_outcome, require, sha, slot_words, table,
)


ARMS = ("parent", "practice-ppo")
SEEDS = tuple(range(21749301, 21749321))
REGISTRATION_SHA = "e1a95efb1e6be4a3f690e01d333d1bf995dea86b4b19ad1bd01332f1b44f9416"
RELEASE_SHA = "1d980129ad840f5aa9a11422b5f618264892f0a913b668d446fe93d2a62d7615"
HORIZON = 6000
BOOTSTRAP_SEED = 21749300
BOOTSTRAP_ITERATIONS = 20000
MARKERS = ("qualification.json", "finished-game.json", "build.sha256", "archive.sealed", "events.bin",
           "native-input.csv", "applied.csv", "controls.csv", "policy-trace.npz")


def load(path):
    return json.loads(path.read_bytes())


def save(path, data):
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


def event_rows(states, field):
    delta = np.diff(states[field])
    require(np.all(delta >= 0), f"Original {field} counter went backwards")
    return [
        dict(tick=int(states[i + 1]["tick"]), prior_sample_tick=int(states[i]["tick"]), amount=int(delta[i]),
             prior_spares=int(states[i]["lives"]), prior_bombs=int(states[i]["bombs"]),
             next_spares=int(states[i + 1]["lives"]), next_bombs=int(states[i + 1]["bombs"]))
        for i in np.flatnonzero(delta)
    ]


def first_nonexact_window(controls, applied, words, end, outcome):
    ticks = controls["tick"]
    require(np.all(np.diff(ticks) > 0), "Practice action clock did not advance")
    for i, (begin, stop) in enumerate(zip(ticks, np.r_[ticks[1:], end], strict=True)):
        wanted = int(controls[i]["requested"])
        actual = [words.get(t, -1) for t in range(int(begin) + 1, int(stop) + 1)]
        valid_clock = all(applied[i][key] == begin for key in ("camera_tick", "applied_tick", "frozen_tick"))
        valid_app = applied[i]["sequence"] == controls[i]["seq"] == i and applied[i]["requested_word"] == wanted
        terminal_partial = (i == len(ticks) - 1 and outcome in ("game_over", "practice_stage_clear")
                            and 0 < len(actual) < 4 and wanted in actual and -1 not in actual)
        if not (valid_clock and valid_app and (actual == [wanted] * 4 or terminal_partial)):
            return dict(decision=i, camera_native_tick=int(begin), next_native_tick=int(stop),
                        requested_word=wanted, actual_word_slots=actual,
                        command_clock_match=bool(valid_clock), requested_echo_match=bool(valid_app))
    return None


def native_phase_duration_purity(controls, applied, words, end, phase):
    by_phase = {}
    first_impure = None
    ticks = controls["tick"]
    for i, (start, stop) in enumerate(zip(ticks, np.r_[ticks[1:], end], strict=True)):
        start, stop = int(start), int(stop)
        if start < phase["first_preceding_tick"]:
            label = "before_nonspell2"
        elif start <= phase["last_preceding_tick"]:
            label = "nonspell2"
        elif phase["first_active_native_tick"] is None or start < phase["first_active_native_tick"]:
            label = "after_nonspell2_before_observed_final_spell"
        elif start <= phase["last_active_native_tick"]:
            label = "final_spell"
        else:
            label = "after_final_spell"
        bucket = by_phase.setdefault(label, Counter())
        actual = [words.get(tick, -1) for tick in range(start + 1, stop + 1)]
        requested = int(controls[i]["requested"])
        known = bool(actual) and -1 not in actual
        control_bound = (all(applied[i][field] == start for field in ("camera_tick", "applied_tick", "frozen_tick"))
                         and applied[i]["sequence"] == controls[i]["seq"] == i and applied[i]["requested_word"] == requested)
        pure = known and control_bound and all(word == requested for word in actual)
        variable = len(actual) != 4
        bucket["windows"] += 1
        bucket["pure"] += int(pure)
        bucket["missing_slot"] += int(not known)
        bucket["command_mismatch"] += int(not control_bound or (known and not pure))
        bucket["variable_duration"] += int(variable)
        bucket["pure_variable_duration"] += int(variable and pure)
        bucket["native_ticks"] += len(actual)
        if not pure and first_impure is None:
            first_impure = dict(action_index=i, start_tick=start, next_tick=stop,
                                actual_native_words=actual, requested_native_word=requested)
    return dict(by_decision_start_phase=by_phase, first_nonpure_native_window=first_impure)


def truncated_control_followup(exposure, slots, controls, original):
    issue = original["first_nonexact_window"]
    stop = issue["camera_native_tick"]
    start = original["first_native_tick"]
    if stop <= start:
        return None
    # Stop at the camera tick that begins the first bad window.
    # Count only resource events strictly before that tick.
    deaths = [row for row in original["death_events"] if row["tick"] < stop]
    bombs = [row for row in original["native_bomb_spend_events"] if row["tick"] < stop]
    state = exposure[exposure["tick"] < stop][-1]
    observed_ticks = [word for tick, word in slots.items() if start < tick <= stop]
    return dict(
        outcome="native_control_nonexact", playable_native_ticks=stop - start, final_native_tick=stop,
        actual_deaths=sum(row["amount"] for row in deaths), death_events=deaths,
        first_loss_elapsed_ticks=deaths[0]["tick"] - start if deaths else None,
        first_loss_censor_reason=None if deaths else "native_control_nonexact",
        native_bomb_spend_events=bombs, native_bomb_units_spent=sum(row["amount"] for row in bombs),
        requested_bomb_decisions=int(np.count_nonzero(controls["requested"][:issue["decision"]] & 2)),
        original_bomb_word_ticks=sum(bool(word & 2) for word in observed_ticks),
        end_stock=[int(state[k]) for k in ("lives", "bombs")])


def observed_practice(folder, spec, *, horizon=HORIZON, expected_stock=(2, 3), verify_scene=False, phase_duration_purity=False):
    """Verify original outcomes/resources independently of actor-forward and action quality."""
    proof = run_path(str(OBSERVER / "record_proof.py"))["verify_archive"](folder)
    build = dict(row.split()[::-1] for row in (folder / "build.sha256").read_text().splitlines())
    native = table(folder / "native-input.csv")
    qualified_reset = load(folder / "qualification.json")
    require(build.get("th07.exe") == GAME and build.get("th07.dat") == DATA
            and qualified_reset["source_sha256"]["native-input.csv"] == sha(folder / "native-input.csv")
            and tuple(qualified_reset["native_observed"]) == expected_stock
            and qualified_reset["scenario"] == spec["scenario"] and qualified_reset["source_save_present"] is False,
            "Independent original practice reset or native journal changed")
    require(spec["source_kind"] == "practice" and spec["native_rank"] == 1
            and spec["initial_spares"] == expected_stock[0] and spec["initial_bombs"] == expected_stock[1]
            and spec["continues"] == 0 and np.all(native["stage"] == 1) and np.all(native["rank"] == 1)
            and np.all(native["shot"] == 0) and not np.any(native["continues"]) and np.all(np.diff(native["ns"]) > 0),
            "Original stage, registered stock, rank or Continue evidence failed")
    live, slots = slot_words(folder, native, "practice")
    controls, applied = (table(folder / name) for name in ("controls.csv", "applied.csv"))
    require(len(controls) and len(controls) == len(applied) == spec["actions"]
            and int(controls[0]["seq"]) == int(applied[0]["sequence"]) == 0,
            "No registered first original action")
    start = int(controls[0]["tick"])
    camera_stamp = controls[0]["camera_ns"]
    first = live[(live["tick"] == start) & (live["ns"] >= camera_stamp)
                 & (live["ns"] <= applied[0]["apply_ns"]) & (live["retry"] == 0)]
    require(len(first) and applied[0]["camera_tick"] == applied[0]["applied_tick"] == start
            and (first[0]["lives"], first[0]["bombs"], first[0]["misses"]) == (*expected_stock, 0),
            "Independent journal cannot identify the first controlled original state")
    outcome = spec["outcome"]
    if outcome in ("game-over", "game_over"):
        outcome = "game_over"
        endpoint = game_over_boundary(native, spec, start, start + horizon)
    elif outcome == "native_tick_cutoff":
        endpoint = playable_boundary(folder, native, spec)[0] if verify_scene else native_boundary_row(native, spec)
        require(spec["boundary_stage"] == 1 and int(endpoint["tick"]) - start == horizon,
                "A real native cutoff must reach exactly the registered playable horizon")
    elif outcome == "practice_stage_clear":
        outcome, endpoint, _ = practice_result_outcome(folder, native, spec)
    else:
        raise ValueError(f"Unregistered held original endpoint: {outcome}")
    end = int(endpoint["tick"])
    elapsed = end - start
    require(0 < elapsed <= horizon, "Original closed its physical exposure outside the registered horizon")
    exposure = live[(live["tick"] >= start) & (live["ns"] <= endpoint["ns"])]
    require(exposure[0]["tick"] == start and exposure[-1]["tick"] == end
            and np.all(np.diff(exposure["tick"]) >= 0), "Native exposure clock is discontinuous")
    deaths, bombs = event_rows(exposure, "misses"), event_rows(exposure, "bombs_used")
    require(outcome != "game_over" or deaths, "True Game Over without an observed native miss")
    first_loss = deaths[0]["tick"] - start if deaths else None
    observed_ticks = [word for tick, word in slots[1].items() if start < tick <= end]
    result = dict(
        physical_endpoint_verified=True, outcome=outcome, source_endpoint_sha256=sha(folder / "finished-game.json"),
        native_source_sha256=proof["journal_sha256"], first_native_tick=start, final_native_tick=end,
        playable_native_ticks=elapsed, observed_key_ticks=len(observed_ticks),
        sampled_actions=int(spec["actions"]), actual_deaths=sum(row["amount"] for row in deaths),
        first_loss_elapsed_ticks=first_loss, first_loss_censor_reason=outcome if first_loss is None else None,
        death_events=deaths, native_bomb_spend_events=bombs,
        native_bomb_units_spent=sum(row["amount"] for row in bombs),
        requested_bomb_decisions=int(np.count_nonzero((controls["requested"] & 2) != 0)),
        original_bomb_word_ticks=sum(bool(word & 2) for word in observed_ticks),
        end_stock=[int(endpoint[key]) for key in ("lives", "bombs")],
    )
    if verify_scene:
        require(spec["scenario"] == PRECEDING, "Held source differs from registered Normal Boss Nonspell 2")
        phase = exact_target(folder, PRECEDING, require_active_spell=outcome == "practice_stage_clear")
        if outcome == "practice_stage_clear":
            visual = load(folder / "practice-result-visual.json")
            require(visual["native_phase"]["phase_sha256"] == phase["phase_sha256"]
                    and 2 in phase["native_finished_stage_observations"],
                    "Held Stage Clear did not bind original Boss Nonspell 2 → Normal spell ID7 → Stage Clear")
        result.update(verified_phase=phase, verified_section_complete=outcome == "practice_stage_clear")
        if phase_duration_purity:
            result["native_duration_purity"] = native_phase_duration_purity(controls, applied, slots[1], end, phase)
    result["first_nonexact_window"] = first_nonexact_window(controls, applied, slots[1], end, outcome)
    if result["first_nonexact_window"] is not None:
        result["censor_at_first_nonexact"] = truncated_control_followup(exposure, slots[1], controls, result)
    return result


def survival(rows):
    ends = np.asarray([row["first_loss_elapsed_ticks"] if row["first_loss_elapsed_ticks"] is not None
                       else row["playable_native_ticks"] for row in rows], dtype=np.int64)
    has_event = np.asarray([row["first_loss_elapsed_ticks"] is not None for row in rows], dtype=bool)
    points, probability, median = [], 1.0, None
    for tick in np.unique(ends):
        risk = int((ends >= tick).sum())
        deaths = int(((ends == tick) & has_event).sum())
        censored = int(((ends == tick) & ~has_event).sum())
        probability *= 1 - deaths / risk
        if median is None and probability <= .5 + 8 * np.finfo(float).eps:
            median = int(tick)
        points.append(dict(tick=int(tick), at_risk=risk, first_losses=deaths, censored=censored, survival=probability))
    return dict(median_first_loss_ticks=median, kaplan_meier=points,
                observed_first_losses=int(has_event.sum()), right_censored=int((~has_event).sum()))


def describe(rows):
    ticks = sum(row["playable_native_ticks"] for row in rows)
    deaths = sum(row["actual_deaths"] for row in rows)
    first = [row["death_events"][0] for row in rows if row["death_events"]]
    return dict(
        physical_exposure_rows=len(rows), playable_native_ticks=ticks, actual_deaths=deaths,
        native_deaths_per_1000_playable_ticks=1000 * deaths / ticks if ticks else None,
        endpoint_counts=dict(Counter(row["outcome"] for row in rows)),
        native_bomb_units_spent=sum(row["native_bomb_units_spent"] for row in rows),
        native_bomb_spend_events=sum(len(row["native_bomb_spend_events"]) for row in rows),
        requested_bomb_decisions=sum(row["requested_bomb_decisions"] for row in rows),
        original_bomb_word_ticks=sum(row["original_bomb_word_ticks"] for row in rows),
        first_loss_prior_bombs_positive=sum(row["prior_bombs"] > 0 for row in first),
        first_loss_prior_bombs_measured=len(first),
        end_stock_counts=dict(Counter(str(row["end_stock"]) for row in rows)),
        **survival(rows),
    )


def paired_comparison(rows, *, registered_seeds=SEEDS, bootstrap_seed=BOOTSTRAP_SEED):
    lookup = {(row["seed"], row["arm"]): row for row in rows}
    seeds = [seed for seed in registered_seeds if all((seed, arm) in lookup for arm in ARMS)]
    groups = [[lookup[seed, arm] for seed in seeds] for arm in ARMS]
    result = dict(matched_pairs=len(seeds), seeds=seeds, arms={arm: describe(group) for arm, group in zip(ARMS, groups)})
    if not seeds:
        return dict(result, death_rate_ratio_candidate_parent=None, paired_bootstrap_percentile_95=None)
    deaths = np.asarray([[r["actual_deaths"] for r in group] for group in groups], dtype=np.int64)
    ticks = np.asarray([[r["playable_native_ticks"] for r in group] for group in groups], dtype=np.int64)
    original_rate = deaths.sum(axis=1) / ticks.sum(axis=1)
    draw = np.random.default_rng(bootstrap_seed).integers(0, len(seeds), (BOOTSTRAP_ITERATIONS, len(seeds)))
    # Each sampled seed contributes its two arms to the same draw.
    sample_rate = deaths[:, draw].sum(axis=2) / ticks[:, draw].sum(axis=2)
    defined = sample_rate[0] > 0
    ratios = sample_rate[1, defined] / sample_rate[0, defined]
    return dict(result, death_rate_ratio_candidate_parent=float(original_rate[1] / original_rate[0]) if original_rate[0] else None,
                paired_bootstrap_percentile_95=np.quantile(ratios, [.025, .975]).tolist() if len(ratios) else None,
                valid_bootstrap_draws=int(defined.sum()), requested_bootstrap_draws=BOOTSTRAP_ITERATIONS,
                bootstrap_seed=bootstrap_seed)


def evaluate(folder, actor, arm, enrollment, binding, arm_digest, *, horizon=HORIZON, expected_stock=(2, 3),
             verify_scene=False, phase_duration_purity=False):
    identity = dict(seed=enrollment["seed"], arm=arm, role="held", lane=enrollment["lane"])
    if verify_scene:
        identity.update(host=enrollment["host"], scenario=enrollment["scenario"])
    fault = folder / "attempt-failed.json"
    if fault.is_file():
        data = load(fault)
        expected = {key: identity[key] for key in ("seed", "role")}
        require(all(data[key] == value for key, value in expected.items()), "Failed held original has wrong provenance")
        return dict(identity, qualified=False, physical_endpoint_verified=False, technical_failure=data,
                    possible_in_game_exposure=any((folder / file).exists() for file in (
                        "native-input.csv", "applied.csv", "controls.csv", "policy-trace.npz")),
                    raw_failure_sha256=sha(fault))
    finish = folder / "finished-game.json"
    spec = load(finish)
    require(all(spec[k] == enrollment[k] for k in ("seed", "role", "host", "lane", "scenario", "decision_cap"))
            and spec["arm"] == arm and spec["registration_sha256"] == arm_digest
            and spec["policy_sha256"] == spec["sha256"] == binding["policies"][arm],
            "Held original seat, controller or policy differs from its registered arm")
    try:
        measured = observed_practice(folder, spec, horizon=horizon, expected_stock=expected_stock, verify_scene=verify_scene,
                                     phase_duration_purity=phase_duration_purity)
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        return dict(identity, qualified=False, physical_endpoint_verified=False,
                    possible_in_game_exposure=True, physical_validation_error=str(exc),
                    source_endpoint_sha256=sha(finish))
    try:
        verification = intake_fragment(folder, actor, None, horizon=horizon, expected_policy_sha=binding["policies"][arm])
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        return dict(identity, **measured, qualified=False, technical_failure=str(exc))
    require(verification["outcome"] == measured["outcome"]
            and verification["native_elapsed_ticks"] == measured["playable_native_ticks"],
            "Independent physical and causal-camera endpoints disagree")
    return dict(identity, **measured, qualified=verification["qualified"], strict_source_receipt=verification)


def bindings(study, models, *, registration_sha256=REGISTRATION_SHA, release_sha256=RELEASE_SHA,
             seeds=SEEDS, horizon=HORIZON):
    path = study / "paired-registration.json"
    registration = load(path)
    release = study / "released-binding.json"
    binding = load(release)
    require(sha(release) == release_sha256 and binding["released"] is True
            and sha(path) == binding["paired_registration_sha256"] == registration_sha256
            and registration["role"] == "held" and tuple(registration["arms"]) == ARMS
            and tuple(row["seed"] for row in registration["rows"]) == seeds
            and registration["max_native_elapsed_ticks"] == horizon
            and registration["parent_policy_sha256"] == PARENT
            and binding["policies"] == {arm: sha(models[arm]) for arm in ARMS}
            and binding["policies"]["parent"] != binding["policies"]["practice-ppo"],
            "Held cohort differs from the exact prospective paired release")
    for arm in ARMS:
        path = study / f"arm-{arm}.json"
        source = load(path)
        require(sha(path) == binding["registered_arm_sha256"][arm] and source["arm"] == arm
                and source["policy_sha256"] == binding["policies"][arm]
                and tuple(row["seed"] for row in source["rows"]) == seeds,
                "A physical held arm differs from its issued registry")
        fields = ("seed", "role", "host", "lane", "lane_pair", "scenario", "decision_cap", "order")
        require(all(all(row[k] == enrolled[k] for k in fields)
                    for row, enrolled in zip(source["rows"], registration["rows"], strict=True)),
                "A physical arm changed the prospective pair enrollment")
    return registration, binding


def final_results(study, store, rows):
    physical = [r for r in rows if r["physical_endpoint_verified"]]
    strict = [r for r in physical if r["qualified"]]
    early_censored, zero_time = [], []
    for row in physical:
        if row["first_nonexact_window"] is None:
            early_censored.append(row)
        elif row["censor_at_first_nonexact"] is None:
            zero_time.append(f"{row['seed']}-{row['arm']}")
        else:
            early_censored.append(dict(row, **row["censor_at_first_nonexact"]))
    estimates = dict(strict_matched_pairs=paired_comparison(strict),
                     all_observed_matched_pairs=paired_comparison(physical),
                     censor_at_first_nonexact_matched_pairs=paired_comparison(early_censored))
    directions = {}
    for name, estimate in estimates.items():
        rate = estimate["death_rate_ratio_candidate_parent"]
        directions[name] = (None if rate is None else "lower" if rate < 1 else "higher" if rate > 1 else "equal")
    return dict(
        intake_index_sha256=sha(store / "intake-index.json"), registration_sha256=REGISTRATION_SHA,
        release_sha256=sha(study / "released-binding.json"), registered_pairs=len(SEEDS), disposition_count=len(rows),
        **estimates, direction_by_analysis=directions,
        all_three_directions_agree=all(directions.values()) and len(set(directions.values())) == 1,
        strict_arms_descriptive={arm: describe([r for r in strict if r["arm"] == arm]) for arm in ARMS},
        all_observed_arms_descriptive={arm: describe([r for r in physical if r["arm"] == arm]) for arm in ARMS},
        censor_at_first_nonexact_arms_descriptive={arm: describe([r for r in early_censored if r["arm"] == arm])
                                                 for arm in ARMS},
        unverified=[f"{r['seed']}-{r['arm']}" for r in rows if not r["physical_endpoint_verified"]],
        pre_game_setup_failures=[f"{r['seed']}-{r['arm']}" for r in rows
                                 if r.get("possible_in_game_exposure") is False],
        technically_excluded=[f"{r['seed']}-{r['arm']}" for r in rows if not r["qualified"]],
        zero_positive_clocked_exposure_at_first_nonexact=zero_time,
        exposure="Original playable native ticks. Invulnerability time was not directly measured or subtracted.",
        event="Original native miss-counter increments, including final death. KM uses first increment.",
        censoring="Primary KM treats a real practice clear or registered live 6000-tick cap as censoring a run without a first loss.",
        clocked_sensitivity=("Follow-up ends at the camera tick starting the first independently nonexact native command. "
                             "The previous interval is the last verified exposure. Loss and Bomb events at or after that "
                             "camera tick are omitted. Clean credits keep their actual endpoint."),
        pairing="Same registered controller seed and host lane. The original game RNG was not controlled.",
        uncertainty="Percentile intervals resample whole controller-seed pairs with replacement.")


def watch(study, models):
    store = study / "intake"
    store.mkdir(parents=True, exist_ok=True)
    registration, bound = bindings(study, models)
    expected = [(pair["seed"], arm) for pair in registration["rows"] for arm in pair["order"]]
    records, receipts = {}, {}
    start = time.monotonic()
    for _ in range(1440):
        for seed, arm in expected:
            if (seed, arm) in records:
                continue
            name = f"{seed}-{arm}"
            folder = study / "raw" / name
            path = store / f"{name}.json"
            if path.exists():
                record = load(path)
            else:
                failed = (folder / "attempt-failed.json").is_file()
                if not failed and not all((folder / marker).is_file() for marker in MARKERS):
                    continue
                if not failed and load(folder / "finished-game.json")["outcome"] == "practice_stage_clear" \
                        and not (folder / "practice-result-visual.json").is_file():
                    continue
                enrollment = registration["rows"][seed - SEEDS[0]]
                record = evaluate(folder, models[arm], arm, enrollment, bound, bound["registered_arm_sha256"][arm])
                save(path, record)
                print(seed, arm, record["qualified"], record.get("outcome", "unverified"), flush=True)
            require(record["seed"] == seed and record["arm"] == arm and record["role"] == "held",
                    "Existing held receipt has the wrong role, seed or arm")
            records[seed, arm], receipts[seed, arm] = record, sha(path)
        rows = [records[key] for key in expected if key in records]
        missing = [f"{seed}-{arm}" for seed, arm in expected if (seed, arm) not in records]
        index = dict(
            complete=not missing, dispositions=len(rows), qualified=sum(r["qualified"] for r in rows),
            physical_known_endpoints=sum(r["physical_endpoint_verified"] for r in rows), missing=missing,
            registration_sha256=REGISTRATION_SHA, release_sha256=sha(study / "released-binding.json"),
            rows=[dict(seed=r["seed"], arm=r["arm"], qualified=r["qualified"],
                       physical_endpoint_verified=r["physical_endpoint_verified"],
                       receipt=f"{r['seed']}-{r['arm']}.json", sha256=receipts[r["seed"], r["arm"]]) for r in rows])
        save(store / "intake-index.json", index)
        if not missing:
            save(study / "independent-held-results.json", final_results(study, store, rows))
            print(f"Independent held complete after {(time.monotonic() - start) / 60:.2f} minutes of intake", flush=True)
            return
        if time.monotonic() - start > 14400:
            break
        time.sleep(10)
    raise TimeoutError("Held source lack or unresolved visual evidence persisted for four hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("parent", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    watch(args.study, dict(zip(ARMS, (args.parent, args.candidate))))

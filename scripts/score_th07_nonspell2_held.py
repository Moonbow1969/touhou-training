"""Independently verify and score the registered TH07 Normal Letty Nonspell 2 holdout."""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import torch

from scripts.intake_th07_letty_starts import PRECEDING
from scripts.score_th07_held_practice import (
    ARMS, BOOTSTRAP_ITERATIONS, MARKERS, bindings, describe, evaluate, load, paired_comparison, save,
)
from scripts.th07_retail_fragment_intake import GAME, require, sha


SEEDS = tuple(range(21749801, 21749821))
HORIZON = 10000
STOCK = (2, 3)
BOOTSTRAP_SEED = 21749800
REQUIRED = (*MARKERS, "spell-phase.csv")


def binomial_tail(n, probability, first, last):
    return sum(math.comb(n, k) * probability**k * (1 - probability)**(n - k)
               for k in range(first, last + 1))


def exact_binomial(success, total, alpha=.05):
    """Two-sided Clopper–Pearson interval."""
    if total == 0:
        return None

    def invert(first, last, increasing):
        lower, upper = 0., 1.
        for _ in range(56):
            p = (lower + upper) / 2
            if (binomial_tail(total, p, first, last) > alpha / 2) == increasing:
                upper = p
            else:
                lower = p
        return (lower + upper) / 2

    return [invert(success, total, True) if success else 0.,
            invert(0, success, False) if success < total else 1.]


def true_clear(row):
    return bool(row and row["physical_endpoint_verified"] and row.get("verified_section_complete") is True
                and row["outcome"] == "practice_stage_clear")


def arm_clear(success, denominator):
    return dict(clears=success, denominator=denominator, fraction=success / denominator if denominator else None,
                exact_binomial_95=exact_binomial(success, denominator))


def matched_clear(rows, *, assign_unverified_as_no_evidence=False):
    lookup = {(row["seed"], row["arm"]): row for row in rows}
    seeds = [seed for seed in SEEDS if assign_unverified_as_no_evidence or all((seed, arm) in lookup for arm in ARMS)]
    groups = np.asarray([[true_clear(lookup.get((seed, arm))) for seed in seeds] for arm in ARMS], dtype=bool)
    result = dict(matched_pairs=len(seeds), seeds=seeds,
                  arms={arm: arm_clear(int(group.sum()), len(seeds)) for arm, group in zip(ARMS, groups, strict=True)})
    if not seeds:
        return dict(result, candidate_minus_parent_rate=None, paired_bootstrap_percentile_95=None,
                    conservative_exact_difference_95=None, exact_discordant_two_sided_p=None)
    parent, candidate = groups
    favours_candidate = int((~parent & candidate).sum())
    favours_parent = int((parent & ~candidate).sum())
    discordant = favours_candidate + favours_parent
    differences = candidate.astype(np.int8) - parent.astype(np.int8)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.integers(0, len(seeds), (BOOTSTRAP_ITERATIONS, len(seeds)))
    resampled = differences[draw].mean(axis=1)
    # Simultaneous exact limits on the two discordant probabilities.
    candidate_limits = exact_binomial(favours_candidate, len(seeds), alpha=.025)
    parent_limits = exact_binomial(favours_parent, len(seeds), alpha=.025)
    conservative = [candidate_limits[0] - parent_limits[1], candidate_limits[1] - parent_limits[0]]
    exact_p = min(1., 2 * binomial_tail(discordant, .5, 0, min(favours_parent, favours_candidate)))
    return dict(
        result, candidate_minus_parent_rate=float(differences.mean()),
        candidate_only_clears=favours_candidate, parent_only_clears=favours_parent, both_clear=int((parent & candidate).sum()),
        neither_has_verified_clear=int((~parent & ~candidate).sum()),
        paired_bootstrap_percentile_95=np.quantile(resampled, [.025, .975]).tolist(),
        paired_bootstrap_draws=BOOTSTRAP_ITERATIONS, paired_bootstrap_seed=BOOTSTRAP_SEED,
        conservative_exact_difference_95=conservative, exact_discordant_two_sided_p=exact_p,
    )


def missing_bounds(rows, *, registered=len(SEEDS)):
    by_arm = {}
    for arm in ARMS:
        known = [row for row in rows if row["arm"] == arm and row["physical_endpoint_verified"]]
        verified = sum(true_clear(row) for row in known)
        unresolved = registered - len(known)
        by_arm[arm] = dict(**arm_clear(verified, registered), known_original_endpoints=len(known),
                           unresolved_original_endpoints=unresolved,
                           objective_rate_bounds=[verified / registered, (verified + unresolved) / registered])
    parent, candidate = (by_arm[arm]["objective_rate_bounds"] for arm in ARMS)
    return dict(arms=by_arm, candidate_minus_parent_bounds=[candidate[0] - parent[1], candidate[1] - parent[0]])


def timing_sensitivity(rows):
    partial_exposure = []
    fully_observed = []
    stopped_without_exposure = []
    for row in rows:
        if row["first_nonexact_window"] is None:
            partial_exposure.append(row)
            fully_observed.append(row)
        elif row["censor_at_first_nonexact"] is None:
            stopped_without_exposure.append(f"{row['seed']}-{row['arm']}")
        else:
            partial_exposure.append(dict(row, **row["censor_at_first_nonexact"]))
    return partial_exposure, fully_observed, stopped_without_exposure


def final_results(rows, registration_sha256, release_sha256, index_sha256):
    physical = [r for r in rows if r["physical_endpoint_verified"]]
    strict = [r for r in physical if r["qualified"]]
    partial, before_error, zero_time = timing_sensitivity(physical)
    strict_estimate = matched_clear(strict)
    physical_estimate = matched_clear(physical)
    bounds = missing_bounds(physical)
    candidate_floor = bounds["arms"][ARMS[1]]["objective_rate_bounds"][0]
    difference_floor = bounds["candidate_minus_parent_bounds"][0]
    reverse = [name for name, item in (("strict_pairs", strict_estimate), ("known_physical_pairs", physical_estimate))
               if item["candidate_minus_parent_rate"] is not None and item["candidate_minus_parent_rate"] < 0.]
    return dict(
        role="held", scenario=PRECEDING, original_stock=list(STOCK), max_playable_native_elapsed_ticks=HORIZON,
        registered_pairs=len(SEEDS), dispositions=len(rows), registration_sha256=registration_sha256,
        release_sha256=release_sha256, intake_index_sha256=index_sha256,
        primary_all_assigned=matched_clear(physical, assign_unverified_as_no_evidence=True),
        missing_endpoint_sensitivity=bounds,
        strict_matching_sensitivity=strict_estimate,
        known_physical_pairs_sensitivity=physical_estimate,
        known_physical_arms_descriptive={arm: arm_clear(sum(true_clear(r) for r in physical if r["arm"] == arm),
                                                       sum(r["arm"] == arm for r in physical)) for arm in ARMS},
        stage_clear_censor_at_first_nonexact=matched_clear(before_error),
        stage_clear_censor_at_first_nonexact_bounds=missing_bounds(before_error),
        advancement_gate=dict(required_candidate_rate=.50, required_rate_gain=.20, candidate_conservative_rate=candidate_floor,
                              conservative_rate_gain=difference_floor,
                              thresholds_met=candidate_floor >= .50 and difference_floor >= .20 - 1e-12,
                              companion_direction_reversals=reverse,
                              manual_review_required_for_discordance=bool(reverse)),
        native_mortality_and_resources={
            key: paired_comparison(items, registered_seeds=SEEDS, bootstrap_seed=BOOTSTRAP_SEED)
            for key, items in (("strict_matched", strict), ("known_physical_matched", physical), ("censor_first_nonexact", partial))
        },
        all_physical_arms_native_resources={
            arm: describe([r for r in physical if r["arm"] == arm]) for arm in ARMS
        },
        unknown_original_endpoints=[f"{r['seed']}-{r['arm']}" for r in rows if not r["physical_endpoint_verified"]],
        strict_exclusions=[f"{r['seed']}-{r['arm']}" for r in rows if not r["qualified"]],
        zero_positive_clock_at_first_control_error=zero_time,
        pre_game_setup_failures=[f"{r['seed']}-{r['arm']}" for r in rows if r.get("possible_in_game_exposure") is False],
        primary_endpoint="Original two-phase completion with native Boss Nonspell 2 Sub39, active Normal spell ID7 and actual Stage Clear tally.",
        missing_endpoint_method=("The registered denominator includes every assigned original. An unresolved candidate endpoint "
                                 "contributes zero verified clears to the conservative gain; an unresolved parent endpoint may clear. "
                                 "Physical/strict companion estimates may be affected by endpoint ascertainment and control quality."),
        pairing="Both arms share the prospectively registered controller seed, host and lane. Original game RNG stays uncontrolled.",
        exposure=("The native miss counter measures losses. Playable time includes original invulnerability. "
                  "For timing sensitivity, follow-up ends at the camera starting the first nonexact native window; "
                  "only earlier miss and Bomb events enter."),
        censoring="First-loss survival censors a no-loss original at a verified clear or exact live playable-time cap.",
    )


def watch(study, models, registration_sha256, release_sha256):
    registration, bound = bindings(study, models, registration_sha256=registration_sha256,
                                   release_sha256=release_sha256, seeds=SEEDS, horizon=HORIZON)
    official_sha = registration.get("original_sha256", registration.get("game_sha256"))
    require(official_sha == GAME and registration["scene"] == PRECEDING and registration["native_rank"] == 1
            and (registration["starting_spares"], registration["starting_bombs"]) == STOCK
            and registration["starting_power"] == 128 and registration["continues"] == 0,
            "Signed registry does not describe the registered Normal Nonspell 2 full-stock holdout")
    enrollment = {row["seed"]: row for row in registration["rows"]}
    require(all(row["role"] == "held" and row["scenario"] == PRECEDING and 4 * row["decision_cap"] == HORIZON
                and set(row["order"]) == set(ARMS) and len(row["order"]) == len(ARMS) for row in enrollment.values()),
            "Prospective assignment changed registered scope or paired order")
    expected = [(seed, arm) for seed in SEEDS for arm in enrollment[seed]["order"]]
    store = study / "intake"
    store.mkdir(parents=True, exist_ok=True)
    records = {}
    published_count = None
    start = time.monotonic()
    while time.monotonic() - start < 14400:
        for seed, arm in expected:
            key = seed, arm
            if key in records:
                continue
            folder = study / "raw" / f"{seed}-{arm}"
            path = store / f"{seed}-{arm}.json"
            if path.is_file():
                record = load(path)
            else:
                failed = (folder / "attempt-failed.json").is_file()
                if not failed and not all((folder / marker).is_file() for marker in REQUIRED):
                    continue
                try:
                    if not failed and load(folder / "finished-game.json")["outcome"] == "practice_stage_clear" \
                            and not (folder / "practice-result-visual.json").is_file():
                        continue
                    record = evaluate(folder, models[arm], arm, enrollment[seed], bound,
                                      bound["registered_arm_sha256"][arm], horizon=HORIZON,
                                      expected_stock=STOCK, verify_scene=True)
                except (OSError, ValueError, KeyError, AssertionError) as error:
                    record = dict(seed=seed, arm=arm, role="held", qualified=False, physical_endpoint_verified=False,
                                  possible_in_game_exposure=(folder / "native-input.csv").is_file(),
                                  source_binding_error=str(error))
                record.update(registration_sha256=registration_sha256, release_sha256=release_sha256)
                save(path, record)
                print(seed, arm, record["qualified"], record.get("outcome", "unverified"), flush=True)
            require((record["seed"], record["arm"], record["role"]) == (seed, arm, "held")
                    and record["registration_sha256"] == registration_sha256 and record["release_sha256"] == release_sha256,
                    "Existing normative held outcome has the wrong assignment hash")
            records[key] = record
        if published_count != len(records):
            rows = [records[key] for key in expected if key in records]
            missing = [f"{seed}-{arm}" for seed, arm in expected if (seed, arm) not in records]
            index = store / "intake-index.json"
            save(index, dict(complete=not missing, dispositions=len(rows), missing=missing,
                             qualified=sum(r["qualified"] for r in rows),
                             physical_known_endpoints=sum(r["physical_endpoint_verified"] for r in rows),
                             registration_sha256=registration_sha256, release_sha256=release_sha256,
                             rows=[dict(seed=r["seed"], arm=r["arm"], qualified=r["qualified"],
                                        physical_endpoint_verified=r["physical_endpoint_verified"],
                                        receipt=f"{r['seed']}-{r['arm']}.json", sha256=sha(store / f"{r['seed']}-{r['arm']}.json"))
                                   for r in rows]))
            if not missing:
                save(study / "independent-held-results.json",
                     final_results(rows, registration_sha256, release_sha256, sha(index)))
                return
            published_count = len(records)
        time.sleep(8)
    raise TimeoutError("A registered held original remained unresolved for four hours")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("study", type=Path)
    cli.add_argument("parent", type=Path)
    cli.add_argument("candidate", type=Path)
    cli.add_argument("registration_sha256")
    cli.add_argument("release_sha256")
    args = cli.parse_args()
    torch.set_num_threads(1)
    watch(args.study, dict(zip(ARMS, (args.parent, args.candidate), strict=True)),
          args.registration_sha256, args.release_sha256)

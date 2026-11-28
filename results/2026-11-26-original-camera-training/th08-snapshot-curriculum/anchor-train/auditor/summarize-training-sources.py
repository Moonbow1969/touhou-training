"""Summarize all original continuations and the subset admitted by the frozen control gates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_bytes())


def summarize(archive, destination):
    study = archive.parent
    roster = read(archive)["rows"]
    anchor = study / "anchor-train"
    attempts = read(anchor / "qualified-train.json")["attempts"]
    by_seed = {row["seed"]: row for row in attempts}
    assert set(by_seed) == {row["seed"] for row in read(anchor / "registration.json")["rows"]}
    selected = {row["seed"] for row in roster if row["source_kind"] == "anchor"}
    assert selected == {row["seed"] for row in attempts if row["qualified"]}
    receipts = {}
    for attempt in attempts:
        if "receipt" in attempt:
            path = study / attempt["receipt"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == attempt["receipt_sha256"]
            receipts[attempt["seed"]] = read(path)

    def cohort(seeds):
        signed = [receipts[seed] for seed in seeds if seed in receipts]
        if not signed:
            return {"physical_attempts": len(seeds), "verified_receipts": 0}
        controls = [row["control_evidence"] for row in signed]
        return {
            "physical_attempts": len(seeds),
            "verified_receipts": len(signed),
            "native_outcomes": dict(Counter(row["outcome"] for row in signed)),
            "observed_original_actions": sum(row["decisions"] for row in signed),
            "consumed_actions": sum(row["action_masks_with_native_consumption"] for row in signed),
            "exact_windows": sum(row["exact_windows"] for row in controls),
            "total_windows": sum(row["total_windows"] for row in controls),
            "native_words": sum(row["native_words_reconstructed"] for row in controls),
            "wholly_stale_windows": sum(row["wholly_stale_windows"] for row in signed),
            "short_exceptions": sum(row["qualified_via_short_exception"] for row in signed),
            "first_min_60_exact_windows": sum(row["first_sixty_exact_original_windows"] for row in controls),
            "first_min_60_total_windows": sum(row["first_sixty_denominator"] for row in controls),
            "median_elapsed_native_ticks": statistics.median(row["elapsed_native_ticks"] for row in signed),
            "min_elapsed_native_ticks": min(row["elapsed_native_ticks"] for row in signed),
            "max_elapsed_native_ticks": max(row["elapsed_native_ticks"] for row in signed),
            "last_observed_boundary_stock_counts": dict(Counter(str(row["end_stock"]) for row in signed)),
            "max_actor_logp_replay_error": max(row["max_actor_logp_abs"] for row in signed),
            "max_critic_replay_error": max(row["max_critic_abs"] for row in signed),
            "min_boundary_bootstrap": min(row["behavior_boundary_value"] for row in signed),
            "max_boundary_bootstrap": max(row["behavior_boundary_value"] for row in signed),
        }

    report = {
        "optimizer_manifest_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "all_restored_training_attempts": cohort(by_seed),
        "admitted_restored_training": cohort(selected),
        "excluded_restored_training": cohort(set(by_seed) - selected),
        "excluded_seeds": sorted(set(by_seed) - selected),
        "processed_source_bytes": read(anchor / "intake-progress.json")["processed_bytes"],
        "registered_limit_bytes": read(anchor / "registration.json")["max_processed_bytes"],
    }
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("manifest", type=Path)
    args.add_argument("output", type=Path)
    opts = args.parse_args()
    summarize(opts.manifest, opts.output)

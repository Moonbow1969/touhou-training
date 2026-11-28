"""Summarize sealed native rollout outcomes and measured Momentum timings."""

from collections import Counter
import json
from pathlib import Path
from statistics import median, quantiles
import sys
import tarfile


def read_json(path):
    return json.loads(path.read_bytes())


def metric(values):
    deciles = quantiles(values, n=10, method="inclusive") if len(values) > 1 else None
    return {
        "sum": sum(values),
        "min": min(values),
        "p10": deciles[0] if deciles else values[0],
        "median": median(values),
        "p90": deciles[8] if deciles else values[0],
        "max": max(values),
    }


def timing(directory, event):
    return int((directory / f"{event}-real-ns").read_text())


def guest_timing(directory, receipt):
    with tarfile.open(directory / "compact.tar.gz", "r:gz") as archive:
        source = archive.extractfile("timing.jsonl")
        timing_rows = [json.loads(line) for line in source]
    first, last = receipt["anchor_tick"], receipt["boundary_tick"]
    policy = [row for row in timing_rows if row.get("phase") == "policy"
              and first <= row["camera_native_tick"] < last]
    if not policy or policy[0]["camera_native_tick"] != first:
        raise ValueError(f"Missing exact-anchor policy timing for {directory.name}")
    ending = [row for row in timing_rows if row.get("phase") == "fragment_boundary"
              and row.get("camera_native_tick") == last]
    inference = [(row["inference_end_ns"] - row["inference_start_ns"]) / 1_000_000
                 for row in policy]
    native_active_s = None
    if ending:
        native_active_s = (ending[0]["monotonic_ns"] - policy[0]["x_applied_ns"]) / 1_000_000_000
        if native_active_s <= 0:
            raise ValueError(f"Invalid native active interval in {directory.name}")
    return native_active_s, inference


def summarize(directory):
    ledger = read_json(directory / "controller-progress.json")
    rows = []
    inference_ms = []
    sources = set()
    for item in ledger["rows"]:
        root = directory / str(item["seed"])
        receipt = read_json(root / "fragment-receipt.json")
        restore = read_json(root / "restore.json")
        ack = item["independent_ack"]
        if receipt["seed"] != item["seed"] or ack["seed"] != item["seed"]:
            raise ValueError(f"Unbound receipt or independent audit for {root.name}")
        sources.add((receipt["game_sha256"], receipt["anchor_sha256"], receipt["policy_sha256"],
                     receipt["protocol_sha256"]))
        dispatched = timing(root, "dispatch")
        released = timing(root, "released")
        finished = timing(root, "end")
        if not dispatched < released < finished:
            raise ValueError(f"Invalid host clock order for {root.name}")
        active_s, inference = guest_timing(root, receipt)
        inference_ms.extend(inference)
        rows.append({
            "seed": item["seed"],
            "accepted": item["accepted"],
            "outcome": receipt["outcome"],
            "terminal": receipt["terminal"],
            "boundary_bootstrap": receipt["boundary_bootstrap"],
            "decisions": receipt["decisions"],
            "native_ticks": receipt["elapsed_native_ticks"],
            "guest_first_applied_to_boundary_s": active_s,
            "exact_windows": ack["exact_windows"],
            "comparable_windows": ack["comparable_windows"],
            "consumed_masks": receipt["consumed_masks"],
            "stale_windows": receipt["wholly_stale_windows"],
            "restore_s": restore["operation_ms"] / 1_000,
            "dispatch_to_release_s": (released - dispatched) / 1_000_000_000,
            "release_to_compact_ready_s": (finished - released) / 1_000_000_000,
            "dispatch_to_compact_ready_s": (finished - dispatched) / 1_000_000_000,
            "dispatch_ns": dispatched,
            "finished_ns": finished,
            "momentum_case_bytes": int((root / "case-bytes.txt").read_text().split()[0]),
        })
    if not rows:
        raise ValueError("No independently audited native rollouts")
    if len(rows) != ledger["attempted"]:
        raise ValueError("Controller row count disagrees with its reported attempt count")
    if len(sources) != 1:
        raise ValueError("Timing cohort mixes sealed game, anchor, policy or protocol identities")
    game_sha, anchor_sha, policy_sha, protocol_sha = sources.pop()
    accepted = sum(row["accepted"] for row in rows)
    if accepted != ledger["qualified"] or len(rows) - accepted != ledger["excluded"]:
        raise ValueError("Independent outcomes disagree with the controller counts")
    clocks = ("restore_s", "dispatch_to_release_s", "release_to_compact_ready_s",
              "dispatch_to_compact_ready_s")
    exact = sum(row["exact_windows"] for row in rows)
    comparable = sum(row["comparable_windows"] for row in rows)
    active = sum(row["release_to_compact_ready_s"] for row in rows)
    native_ticks = sum(row["native_ticks"] for row in rows)
    timed_game = [row for row in rows if row["guest_first_applied_to_boundary_s"] is not None]
    active_game = sum(row["guest_first_applied_to_boundary_s"] for row in timed_game)
    return {
        "game_sha256": game_sha,
        "anchor_sha256": anchor_sha,
        "policy_sha256": policy_sha,
        "protocol_sha256": protocol_sha,
        "attempted": len(rows),
        "accepted": accepted,
        "excluded_seeds": [row["seed"] for row in rows if not row["accepted"]],
        "outcomes_all": dict(Counter(row["outcome"] for row in rows)),
        "terminal_outcomes_all": sum(row["terminal"] for row in rows),
        "boundary_bootstraps_all": sum(row["boundary_bootstrap"] for row in rows),
        "consumed_masks_all": sum(row["consumed_masks"] for row in rows),
        "wholly_stale_windows_all": sum(row["stale_windows"] for row in rows),
        "exact_windows_all": exact,
        "comparable_windows_all": comparable,
        "exact_fraction_all": exact / comparable,
        "native_ticks_all": native_ticks,
        "native_timing_coverage": len(timed_game),
        "native_ticks_per_guest_active_s":
            sum(row["native_ticks"] for row in timed_game) / active_game if active_game else None,
        "cpu_policy_forward_ms": metric(inference_ms),
        "native_ticks_per_release_to_compact_ready_s": native_ticks / active,
        "native_ticks_per_dispatch_to_compact_ready_s":
            native_ticks / sum(row["dispatch_to_compact_ready_s"] for row in rows),
        "native_ticks_per_continuous_study_s":
            native_ticks / ((rows[-1]["finished_ns"] - rows[0]["dispatch_ns"]) / 1_000_000_000),
        "accepted_decisions": sum(row["decisions"] for row in rows if row["accepted"]),
        "peak_recorded_momentum_case_bytes": max(row["momentum_case_bytes"] for row in rows),
        "processed_bytes": ledger["rows"][-1]["independent_ack"]["processed_bytes"],
        "elapsed_s_distribution": {clock: metric([row[clock] for row in rows]) for clock in clocks},
        "rows": rows,
    }


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))

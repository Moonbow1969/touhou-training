"""Summarize signed, completed physical Letty practice source indexes."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def physical_row(root, raw, entry):
    row = dict(entry)
    if "receipt" not in row:
        recovery_path = (raw.parent / "recovered-physical" / str(row["seed"]) / "recovered-physical-original.json"
                         if raw is not None else None)
        if recovery_path and recovery_path.is_file():
            recovery = read(recovery_path)
            if recovery["source_failure_sha256"] != row["original_failure_sha256"] or recovery["seed"] != row["seed"]:
                raise ValueError(f"Physical recovery no longer binds original failure: {row['seed']}")
            row.update({key: recovery[key] for key in ("outcome", "decisions", "native_ticks", "native_misses",
                                                       "native_bomb_units_spent", "first_stock", "end_stock")})
            row.update(verified_section_complete=True, physical_recovery_sha256=sha(recovery_path),
                       exact_native_windows=recovery["control_audit"]["exact_native_windows"],
                       original_windows=recovery["control_audit"]["original_windows"])
        return row
    receipt = root / row["receipt"]
    archive = root / row["file"]
    if sha(receipt) != row["receipt_sha256"] or sha(archive) != row["sha256"]:
        raise ValueError(f"Physical source normalization changed: {row['seed']}")
    proof = read(receipt)
    if raw is not None:
        source_root = raw / str(row["seed"])
        for replay in sorted((source_root / "official-replays").glob("*.rpy")):
            with replay.open("rb") as replay_file:
                if replay_file.read(4) == b"T7RP":
                    row["original_saved_replays_sha256"][replay.relative_to(source_root).as_posix()] = sha(replay)
    with np.load(archive, allow_pickle=False) as source:
        row["native_misses"] = int(source["next_misses"][-1] - source["misses"][0])
        row["native_bomb_units_spent"] = int(source["next_bombs_used"][-1] - source["bombs_used"][0])
        row["exact_optimizer_actions"] = int(np.count_nonzero(source["native_action_eligible"])) if row["qualified"] else 0
    row.update({key: proof[key] for key in ("first_stock", "end_stock", "control_exclusions",
                                            "exact_native_windows", "original_windows", "policy_logp_max_abs_error",
                                            "policy_critic_max_abs_error", "endpoint_sha256")})
    return row


def counts(rows):
    outcomes = Counter(row.get("outcome", "technical_failure") for row in rows)
    strict = [row for row in rows if row["qualified"]]
    return dict(
        registered_dispositions=len(rows), observed_original_outcomes=len(rows) - outcomes["technical_failure"],
        outcomes=dict(outcomes), observed_section_clears=outcomes["practice_stage_clear"],
        registered_exclusions=len(rows) - len(strict), strict_originals=len(strict),
        strict_section_clears=sum(row.get("verified_section_complete", False) for row in strict),
        physical_native_ticks=sum(row.get("native_ticks", 0) for row in rows),
        strict_native_ticks=sum(row["native_ticks"] for row in strict),
        physical_decisions=sum(row.get("decisions", 0) for row in rows),
        strict_decisions=sum(row["decisions"] for row in strict),
        exact_optimizer_actions=sum(row.get("exact_optimizer_actions", 0) for row in rows),
        exact_windows=sum(row.get("exact_native_windows", 0) for row in rows),
        compared_windows=sum(row.get("original_windows", 0) for row in rows),
        strict_exact_windows=sum(row["exact_native_windows"] for row in strict),
        strict_compared_windows=sum(row["original_windows"] for row in strict),
        physical_native_misses=sum(row.get("native_misses", 0) for row in rows),
        physical_bombs_spent=sum(row.get("native_bomb_units_spent", 0) for row in rows),
        original_saved_replays=sum(len(row.get("original_saved_replays_sha256", {})) for row in rows),
    )


def summarize(index_path, output, raw=None):
    manifest = read(index_path)
    if not manifest["complete"] or manifest["missing"]:
        raise ValueError(f"Original study is awaiting {manifest['missing']}")
    rows = [physical_row(index_path.parent, raw, entry) for entry in manifest["rows"]]
    scenes = sorted({row["scene"] for row in rows})
    report = dict(
        role="train", study=index_path.parent.name, registration_sha256=manifest["registration_sha256"],
        independent_index_sha256=sha(index_path), physical_and_strict=counts(rows),
        by_scene={scene: counts([row for row in rows if row["scene"] == scene]) for scene in scenes},
        result_scope="Physical clears require the original Stage Clear tally. Native spell capture and ordinary Stage 2 are distinct outcomes.",
        rows=rows,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pending")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report["physical_and_strict"]), sha(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw", type=Path)
    args = parser.parse_args()
    summarize(args.index, args.output, args.raw)

"""Report measured original-game throughput, reset time, and compact evidence costs."""

import json
from pathlib import Path
import runpy
import sys
import tarfile


HELPERS = runpy.run_path(str(Path(__file__).with_name("summarize-production-timing.py")))
read = HELPERS["read_json"]
metric = HELPERS["metric"]
host_clock = HELPERS["timing"]


def table(archive, name):
    with archive.extractfile(name) as source:
        fields = source.readline().decode().split()
        return [dict(zip(fields, line.decode().split())) for line in source]


def timings(directory, receipt):
    with tarfile.open(directory / "compact.tar.gz", "r:gz") as archive:
        with archive.extractfile("timing.jsonl") as source:
            trace = [json.loads(line) for line in source]
        gates = table(archive, "gates.tsv")[receipt["offset"]:]
        frozen = table(archive, "frozen.tsv")[receipt["offset"]:]
        if len(gates) != receipt["decisions"] or len(gates) != len(frozen):
            raise ValueError(f"Native camera gate trace disagrees with behavior for {directory.name}")
        late = []
        for index, (gate, stopped) in enumerate(zip(gates, frozen)):
            target, seen = int(gate["camera_target"]), int(gate["camera_observed"])
            if 0 < target < seen:
                late.append(dict(decision=index, target_tick=target, observed_tick=seen,
                                 successful_sigstop_ms=(int(stopped["stopped_ns"]) - int(stopped["stop_start_ns"])) / 1e6))
    begin, end = receipt["anchor_tick"], receipt["boundary_tick"]
    decisions = [row for row in trace if row.get("phase") == "policy"
                 and begin <= row["camera_native_tick"] < end]
    if len(decisions) != receipt["decisions"] or decisions[0]["camera_native_tick"] != begin:
        raise ValueError(f"GPU-independent timing trace disagrees with game exposure in {directory.name}")
    first = decisions[0]
    ending = trace[-1]
    if ending.get("phase") != "fragment_boundary" or ending.get("camera_native_tick") != end:
        active = None
    else:
        active = (ending["monotonic_ns"] - first["x_applied_ns"]) / 1e9
        if active <= 0:
            raise ValueError(f"Invalid in-guest active duration in {directory.name}")
    forward = [(r["inference_end_ns"] - r["inference_start_ns"]) / 1e6 for r in decisions]
    return dict(guest_first_apply_to_boundary_s=active, first_cpu_policy_forward_ms=forward[0],
                first_inference_to_x11_apply_ms=(first["x_applied_ns"] - first["inference_start_ns"]) / 1e6,
                first_hud_ready_to_inference_ms=(first["inference_start_ns"] - first["hud_ready_ns"]) / 1e6,
                late_camera_gates=late, cpu_policy_forward_ms=forward)


def collect(directory, assessed):
    receipt = read(directory / "fragment-receipt.json")
    release = read(directory / "branch-release.json")
    ack = read(directory / "intake-ack.json")
    pack = read(directory / "compact-pack.json")
    seed = assessed["seed"]
    if any(item["seed"] != seed for item in (ack, receipt, pack)) or release["controller_seed"] != seed:
        raise ValueError(f"Physical and independent held receipts disagree for {seed}")
    if release["role"] != "held" or release["arm"] != assessed["arm"] or ack["arm"] != assessed["arm"]:
        raise ValueError(f"Held timing was attributed to the wrong arm for {seed}")
    dispatched, released, ended = (host_clock(directory, event) for event in ("dispatch", "released", "end"))
    if not dispatched < released < ended:
        raise ValueError(f"Invalid host clock order for {directory.name}")
    transferred_ns = (directory / "compact-transfer.done").stat().st_mtime_ns
    acknowledged_ns = (directory / "intake-ack.json").stat().st_mtime_ns
    if acknowledged_ns < transferred_ns:
        raise ValueError(f"Invalid local custody time order for {seed}")
    compact_bytes = (directory / "compact.tar.gz").stat().st_size
    shard_bytes = (directory / "fragment.npz").stat().st_size
    if (compact_bytes != pack["bytes"] or shard_bytes != pack["fragment_bytes"]
            or pack["fragment_sha256"] != receipt["sha256"]):
        raise ValueError(f"Compactor accounting disagrees with immutable inputs for {seed}")
    guest = timings(directory, receipt)
    per_decision = guest.pop("cpu_policy_forward_ms")
    return dict(seed=seed, arm=assessed["arm"], qualified=ack["valid"], original_ticks=receipt["elapsed_native_ticks"],
                native_decisions=receipt["decisions"], dispatch_ns=dispatched, compact_ready_ns=ended,
                restore_s=read(directory / "restore.json")["operation_ms"] / 1000.,
                dispatch_to_release_s=(released - dispatched) / 1e9, dispatch_to_compact_ready_s=(ended - dispatched) / 1e9,
                action_release_to_compact_ready_s=(ended - released) / 1e9,
                local_transfer_to_independent_ack_s=(acknowledged_ns - transferred_ns) / 1e9,
                compact_tar_bytes=compact_bytes, trajectory_bytes=shard_bytes,
                peak_momentum_case_snapshot_bytes=int((directory / "case-bytes.txt").read_text().split()[0]),
                **guest), per_decision


def summarize(case_root, study):
    assessment = read(study / "original-outcomes.json")
    rows, cpu_forward = [], []
    timing_missing = []
    for attempt in assessment["attempts"]:
        seed = attempt["seed"]
        if not attempt.get("file"):
            timing_missing.append(seed)
            continue
        row, span = collect(case_root / str(seed), attempt)
        rows.append(row)
        cpu_forward.extend(span)
    if not rows:
        raise ValueError("No signed held native sources with complete timing")
    active = [row for row in rows if row["guest_first_apply_to_boundary_s"] is not None]
    wall_s = (rows[-1]["compact_ready_ns"] - rows[0]["dispatch_ns"]) / 1e9
    handoff = [(right["dispatch_ns"] - left["compact_ready_ns"]) / 1e9
               for left, right in zip(rows, rows[1:])]
    if any(gap < 0 for gap in handoff):
        raise ValueError("Measured original held launches overlapped")
    guest_time = sum(row["guest_first_apply_to_boundary_s"] for row in active)
    exposed = sum(row["original_ticks"] for row in rows)
    case_bytes = sum(row["compact_tar_bytes"] + row["trajectory_bytes"] for row in rows)
    clocks = ("restore_s", "dispatch_to_release_s", "dispatch_to_compact_ready_s",
              "action_release_to_compact_ready_s", "local_transfer_to_independent_ack_s",
              "first_cpu_policy_forward_ms", "first_inference_to_x11_apply_ms",
              "first_hud_ready_to_inference_ms")
    return {
        "registration_sha256": assessment["registration_sha256"],
        "policies_sha256": assessment["source_sha256"],
        "roster_complete": assessment["complete"],
        "physical_attempts": len(assessment["attempts"]),
        "timed_held_sources": len(rows),
        "technical_timing_unavailable_seeds": timing_missing,
        "tps_clock_definition": "Native ticks between anchor and original boundary, divided by the measured interval.",
        "continuous_elapsed_s": wall_s,
        "observed_starts_per_wall_hour": len(rows) / wall_s * 3600,
        "native_ticks": exposed,
        "native_active_timing_credits": len(active),
        "native_ticks_per_game_active_s": sum(row["original_ticks"] for row in active) / guest_time if active else None,
        "native_ticks_per_dispatch_to_compact_ready_s":
            exposed / sum(row["dispatch_to_compact_ready_s"] for row in rows),
        "native_ticks_per_continuous_study_s": exposed / wall_s,
        "compact_ready_to_next_dispatch_s": metric(handoff) if handoff else None,
        "total_compact_bytes": case_bytes,
        "compact_bytes_per_native_decision": case_bytes / sum(row["native_decisions"] for row in rows),
        "peak_momentum_case_snapshot_bytes": max(row["peak_momentum_case_snapshot_bytes"] for row in rows),
        "camera_gates_observed_after_fixed_target": sum(len(row["late_camera_gates"]) for row in rows),
        "custody": "Original per-attempt data was transferred and independently verified before redundant host copies were pruned.",
        "cpu_policy_forward_ms": metric(cpu_forward),
        "distribution": {clock: metric([row[clock] for row in rows]) for clock in clocks},
        "attempts": rows,
    }


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))

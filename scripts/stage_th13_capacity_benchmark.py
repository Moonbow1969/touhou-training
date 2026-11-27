"""Verify, plot, and package the registered native TH13 concurrency benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.benchmark_th13_accelerator_concurrency import summarize
from scripts.collect_th10_full_action import save


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(study, public):
    target = public / study.name
    target.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((study / "protocol.json").read_bytes())
    costs = []
    files = [study / name for name in ("protocol.json", "setup.json", "meter-control.json")]
    for wave_id, expected in enumerate(protocol["waves"], 1):
        wave_file = study / f"wave-{wave_id:02d}.json"
        wave = json.loads(wave_file.read_bytes())
        if any(wave[key] != value for key, value in expected.items()):
            raise ValueError(f"Unexpected assignments in {wave_file}")
        if [credit["seed"] for credit in wave["receipts"]] != expected["seeds"]:
            raise ValueError(f"Unexpected completed seeds in {wave_file}")
        files.append(wave_file)
        for row in wave["receipts"]:
            episode = study / "native13" / f"engineering-capacity{wave['concurrency']}-{row['seed']}"
            receipt = json.loads((episode / "receipt.json").read_bytes())
            raw = np.fromfile(episode / "steps.bin", "<f4").reshape(-1, 143)
            if not (digest(episode / "receipt.json") == row["receipt_sha256"]
                    and digest(episode / "steps.bin") == receipt["native_difficulty_trace_sha256"]
                    and receipt["policy_sha256"] == protocol["policy_sha256"]
                    and receipt["terminal"] and receipt["continues"] == 0
                    and len(raw) == row["steps"] and np.all(raw[:, 141] == 0)
                    and np.all(raw[:, 122] == raw[:, 123])
                    and tuple(raw[0, [125, 127, 129]]) == (1, 2, 2)):
                raise ValueError(f"Original control or stock mismatch in {episode.name}")
            interval_ms = np.diff(raw[:, 121].astype(np.float64)) * 1000
            row["decision_interval_ms"] = np.percentile(interval_ms, [50, 95, 99])
            row["outside_hold_ms"] = np.percentile(interval_ms - raw[1:, 140] * 1000, [50, 95, 99])
            files.extend(episode / name for name in ("receipt.json", "acquisition-cost.json",
                         "steps.bin", "initial.png", "ranking.png", "events.tsv", "final-stock.bin"))
        costs.append(wave)
    summarize(study)
    results = json.loads((study / "throughput.json").read_bytes())
    levels = [1, 2, 4]
    if set(results["levels"]) != {str(level) for level in levels}:
        raise ValueError("All three concurrency settings must finish before comparison")
    expanded = {}
    for level in levels:
        relevant = [w for w in costs if w["concurrency"] == level]
        rows = [r for w in relevant for r in w["receipts"]]
        by_block = {}
        for block in (1, 2):
            waves = [w for w in relevant if w["block"] == block]
            wall = sum(w["measured_wave_wall_s"] for w in waves)
            all_rows = [r for w in waves for r in w["receipts"]]
            by_block[str(block)] = dict(wall_s=wall,
                acknowledged_decisions_per_s=sum(r["native_key_matches"] for r in all_rows) / wall,
                observed_native_ticks_per_s=sum(r["valid_native_ticks"] for r in all_rows) / wall)
        expanded[str(level)] = dict(by_block=by_block,
            median_per_credit_end_to_end_decision_ms=np.median([r["decision_interval_ms"] for r in rows], axis=0).tolist(),
            median_per_credit_nonhold_ms=np.median([r["outside_hold_ms"] for r in rows], axis=0).tolist(),
            latency_quantiles=[50, 95, 99],
            latency_scope="Measured between consecutive action completions. Nonhold residual includes screen capture, inference, logging, and any native transition wait.",
            input_match_fraction=sum(r["native_key_matches"] for r in rows) / sum(r["steps"] for r in rows),
            worker_to_collector_stdout_bytes=sum(r["actual_ingress_bytes"] for r in rows),
            collector_to_worker_stdin_bytes=sum(r["actual_egress_bytes"] for r in rows),
            network_scope="Application stdout/stdin payload only. SSH framing, remote command strings, setup, and the separate host monitor are excluded.",
            largest_game_cgroup_peak_bytes=max((r["cgroup_memory_peak"] or 0) for r in rows),
            total_cgroup_read_bytes=sum(r["cgroup_io"]["rbytes"] for r in rows),
            total_cgroup_written_bytes=sum(r["cgroup_io"]["wbytes"] for r in rows),
            min_tmpfs_free_bytes_at_wave_boundaries=min(min(w["host_before"]["free_bytes"],
                w["host_after"]["free_bytes"]) for w in relevant),
            stage2_credits=sum(r["terminal_stage2"] for r in rows))
    save(study / "host-costs.json", expanded)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5), constrained_layout=True)
    charts = [
        ("native_acknowledged_decisions_per_s", "Effective policy decisions / second"),
        ("native_ticks_per_s", "Observed game ticks / second"),
        ("median_boot_to_confirmed_stock_s", "Median verified opening time (seconds)"),
        ("median_per_credit_action_p95_ms", "Median per-credit p95 action hold (ms)")]
    for panel, (ax, (key, title)) in enumerate(zip(axes.flat, charts)):
        vals = [results["levels"][str(n)][key] for n in levels]
        bars = ax.bar(range(3), vals, width=.58, color=["#6499b1", "#398279", "#234d6b"])
        ax.bar_label(bars, fmt="%.1f", padding=3)
        if panel < 2:
            for index, level in enumerate(levels):
                name = ("acknowledged_decisions_per_s", "observed_native_ticks_per_s")[panel]
                for offset, block in ((-.09, "1"), (.09, "2")):
                    ax.plot(index + offset, expanded[str(level)]["by_block"][block][name], "o",
                            markersize=4, color="#c56531")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.14)
        ax.set(xticks=range(3), xticklabels=["1 game", "2 games", "4 games"], title=title)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("TH13 Easy on Accelerator · Eight full credits per setting\n"
                 "Throughput includes startup and collection; dots show each four-seed block", fontsize=13)
    fig.savefig(study / "native-throughput.png", dpi=160)
    plt.close(fig)

    evidence = study / "native-capacity-evidence.zip"
    with ZipFile(evidence, "w", compression=ZIP_DEFLATED, compresslevel=4) as output:
        for src in files:
            output.write(src, src.relative_to(study))
    save(study / "evidence-index.json", dict(credits=24,files=len(files),
        archive_sha256=digest(evidence), archive_bytes=evidence.stat().st_size,
        local_full_frames="native13/engineering-capacity{1,2,4}-{seed}/observations.u8"))
    for name in ("protocol.json", "throughput.json", "host-costs.json", "native-throughput.png",
                 "native-capacity-evidence.zip", "evidence-index.json"):
        shutil.copy2(study / name, target / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--public", type=Path, default=Path("results/2026-11-26-original-camera-training/engineering"))
    args = parser.parse_args()
    stage(args.study, args.public)

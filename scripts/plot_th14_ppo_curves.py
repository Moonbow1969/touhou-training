"""Render archived TH14 PPO figures and their underlying tables with Matplotlib."""

import argparse
import csv
import json
from bisect import bisect_right
from io import StringIO
from itertools import accumulate
from pathlib import Path
from statistics import median

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, StrMethodFormatter


matplotlib.use("Agg")

ARCHIVE = Path(__file__).resolve().parents[1] / "results" / "2026-11-23-th14-full-policy"
FIRST_REPORT = "ppo-long-8192-sft-paired.json"
SECOND_REPORT = "ppo-long-16384-sft-fresh-paired.json"
HEAD_TO_HEAD = "ppo-budget-8192-vs-16384.json"
INK = "#152634"
MUTED = "#4A5A66"
GRID = "#E0E7EB"
WARM = "#196C61"
SFT = "#797989"
CHECKPOINT = "#2D559E"


def read_report(directory, name):
    with (directory / name).open(encoding="utf-8") as stream:
        report = json.load(stream)
    if report["status"] != "complete":
        raise ValueError(f"Incomplete report {name}")
    return report


def archive_training(reports):
    if reports[0][1]["starting_steps"] != 0:
        raise ValueError("The training reports do not form one PPO lineage.")
    for (_, parent), (_, child) in zip(reports, reports[1:]):
        if (child["starting_steps"] != parent["checkpoint_steps"]
                or child["checkpoints"]["resume_sha256"] != parent["checkpoints"]["ppo_sha256"]
                or any(parent[key] != child[key] for key in ("task", "seed", "ppo_hyperparameters", "reward"))):
            raise ValueError("The training reports do not form one PPO lineage.")

    episodes, updates = [], []
    for segment, (source, report) in enumerate(reports, 1):
        durable_steps = report["checkpoint_steps"]
        durable_updates = [row for row in report["updates"] if row["steps"] <= durable_steps]
        if not durable_updates or durable_updates[-1]["steps"] != durable_steps:
            raise ValueError(f"The saved checkpoint has no final update record in {source}.")
        raw_episodes = report["training_episodes"][:durable_updates[-1]["completed_episodes"]]
        ends = list(accumulate((ep["steps"] for ep in raw_episodes), initial=report["starting_steps"]))[1:]
        if any(bisect_right(ends, update["steps"]) != update["completed_episodes"] for update in durable_updates):
            raise ValueError(f"Cumulative episode decisions disagree with update counts in {source}.")
        for index, (end, ep) in enumerate(zip(ends, raw_episodes, strict=True), 1):
            episodes.append({
                "source_report": source, "training_seed": report["seed"], "segment": segment,
                "episode_in_segment": index, "cumulative_decisions": end, "episode_decisions": ep["steps"],
                "stage_frame": ep["stage_frame"], "first_miss_frame": ep["first_miss_frame"],
                "clears": ep["stages_cleared"], "outcome": ep["outcome"], "score": ep["score"],
                "bombs_used": ep["bombs_used"],
            })
        updates.extend({"source_report": source, "training_seed": report["seed"], "segment": segment, **row}
                       for row in durable_updates)
    return episodes, updates


def paired_archive(first, second, head_to_head):
    setups = (
        (FIRST_REPORT, first, "8,192 decisions vs frozen SFT", "sft", "ppo", 0, 8192),
        (SECOND_REPORT, second, "16,384 decisions vs frozen SFT", "sft", "ppo", 0, 16384),
        (HEAD_TO_HEAD, head_to_head, "8,192 vs 16,384 decisions", "ppo_reference", "ppo", 8192, 16384),
    )
    budget_labels = {0: "Frozen SFT", 8192: "PPO 8,192", 16384: "PPO 16,384"}
    cohorts = []
    for source, report, title, left_arm, right_arm, left_budget, right_budget in setups:
        arms = []
        for arm, budget in ((left_arm, left_budget), (right_arm, right_budget)):
            control = report["controller_settings"][arm]
            if any(control[key] for key in ("geometry", "aim_correction", "automatic_bomb")):
                raise ValueError(f"Impure evaluation controller {arm} in {source}.")
            episodes = []
            for ep in report["controllers"][arm]:
                first_seed = report.get("evaluation_first_policy_seed", report["seed"])
                seed = ep.get("policy_seed", first_seed + ep["block"] - 1)
                episodes.append({
                    "source_report": source, "cohort": title, "arm": arm, "arm_label": budget_labels[budget],
                    "training_decisions": budget, "block": ep["block"], "policy_sampling_seed": seed,
                    "stage_frame": ep["stage_frame"], "first_miss_frame": ep["first_miss_frame"],
                    "clears": ep["stages_cleared"], "outcome": ep["outcome"], "score": ep["score"],
                    "bombs_used": ep["bombs_used"],
                })
            arms.append(sorted(episodes, key=lambda ep: ep["block"]))
        left, right = arms
        pairs = list(zip(left, right, strict=True))
        if (not pairs or len({a["block"] for a, _ in pairs}) != len(pairs)
                or any((a["block"], a["policy_sampling_seed"]) != (b["block"], b["policy_sampling_seed"])
                       for a, b in pairs)):
            raise ValueError(f"Unpaired archive evaluation rows in {source}.")
        cohorts.append((title, left, right))
    return cohorts


def matched_evaluations(path, training_seed=211):
    report = read_report(path.parent, path.name)
    budgets = (0, 8192, 16384)
    arms = {f"{init}-{budget}": (init, budget) for init in ("sft", "random") for budget in budgets}
    specification = report["specification"]
    controller = specification["controller"]
    if (report["matched_blocks"] != 8 or set(specification["policies"]) != set(arms)
            or controller["decode"] != "sampled" or controller["native_rng_seed_controlled"]
            or any(controller[key] for key in ("geometry", "aim_correction", "automatic_bomb"))):
        raise ValueError("The matched figure requires eight complete blocks of six pure policy checkpoints.")

    episodes = {}
    for entry in report["completed"]:
        trial, result = entry["trial"], entry["result"]
        init, budget = arms[trial["arm"]]
        seed = trial["policy_seed"]
        key = init, budget, seed
        if key in episodes:
            raise ValueError(f"Duplicate evaluation trial {trial}.")
        episodes[key] = {
            "source_report": path.name, "ppo_training_seed": training_seed, "initialization": init,
            "training_decisions": budget, "block": trial["block"], "policy_sampling_seed": seed,
            "stage_frame": result["stage_frame"], "first_miss_frame": result["first_miss_frame"],
            "clears": result["stages_cleared"], "outcome": result["outcome"],
            "episode_decisions": result["steps"], "bombs_used": result["bombs_used"],
        }
    seeds = sorted({key[2] for key in episodes})
    expected = {(init, budget, seed) for init, budget in arms.values() for seed in seeds}
    if (len(seeds) != report["matched_blocks"] or set(episodes) != expected
            or {row["block"] for row in episodes.values()} != set(range(1, len(seeds) + 1))
            or any(len({episodes[init, budget, seed]["block"] for init, budget in arms.values()}) != 1
                   for seed in seeds)):
        raise ValueError("Every policy sampling seed must cover all six arms in one matched block.")

    gains = []
    for budget in budgets:
        for seed in seeds:
            sft, random = (episodes[init, budget, seed] for init in ("sft", "random"))
            for row in (sft, random):
                anchor = episodes[row["initialization"], 0, seed]["stage_frame"]
                row["anchor_stage_frame"] = anchor
                row["gain_from_own_anchor"] = row["stage_frame"] - anchor
            gains.append({
                "source_report": path.name, "training_decisions": budget, "block": sft["block"],
                "policy_sampling_seed": seed, "sft_stage_frame": sft["stage_frame"],
                "random_stage_frame": random["stage_frame"],
                "sft_minus_random_stage_frames": sft["stage_frame"] - random["stage_frame"],
                "sft_gain_from_zero": sft["gain_from_own_anchor"],
                "random_gain_from_zero": random["gain_from_own_anchor"],
                "sft_minus_random_gains": sft["gain_from_own_anchor"] - random["gain_from_own_anchor"],
            })
    return [episodes[key] for key in sorted(episodes)], gains, specification["task"]


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis):
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(axis="both", length=0, pad=7, labelcolor=MUTED)
    axis.grid(axis="y", color=GRID, linewidth=.7)
    axis.set_axisbelow(True)
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def heading(fig, title, subtitle):
    fig.text(.08, .95, title, fontsize=20, weight="bold", va="top", color=INK)
    fig.text(.08, .875, subtitle, fontsize=10, va="top", color=MUTED)


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=200, metadata={"Software": "Touhou Training"})
    svg = StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None})
    clean_svg = "\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n"
    (directory / f"{name}.svg").write_text(clean_svg, encoding="utf-8", newline="\n")
    plt.close(fig)


def plot_training(directory, episodes, boundaries, stem="ppo-on-policy-training"):
    fig, axis = plt.subplots(figsize=(10.3, 5.65))
    fig.subplots_adjust(left=.09, right=.97, top=.75, bottom=.22)
    seed = episodes[0]["training_seed"]
    heading(fig, "SFT-initialized PPO during training", f"TH14 1.00b · Normal Reimu A · Stage 1 · one training seed ({seed})")
    for segment in sorted({row["segment"] for row in episodes}):
        points = [row for row in episodes if row["segment"] == segment]
        x = [row["cumulative_decisions"] for row in points]
        y = [row["stage_frame"] for row in points]
        axis.scatter(x, y, s=29, color=WARM, alpha=.67, linewidth=0, zorder=3,
                     label="Completed training episode" if segment == 1 else None)
        if len(y) >= 7:
            smoothed = [median(y[i - 6:i + 1]) for i in range(6, len(y))]
            axis.plot(x[6:], smoothed, color=WARM, lw=2.2, zorder=4,
                      label="Trailing median of 7 episodes" if segment == 1 else None)
    for boundary in boundaries:
        axis.axvline(boundary, color=SFT, lw=1, ls=(0, (3, 4)), zorder=2)
    if len(boundaries) == 1:
        boundary, = boundaries
        axis.text(boundary + 230, .98, f"Resume at {boundary:,}", transform=axis.get_xaxis_transform(),
                  color=MUTED, va="top", fontsize=9)
    axis.set_ylim(bottom=0)
    axis.set_xlim(left=0)
    axis.xaxis.set_major_locator(MaxNLocator(8, integer=True))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.set_xlabel("Cumulative native training decisions", color=INK, labelpad=8)
    axis.set_ylabel("Stage 1 endpoint (native frames)", color=INK, labelpad=9)
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncols=2, frameon=False, borderaxespad=0,
                labelcolor=INK, fontsize=9)
    style_axis(axis)
    fig.text(.09, .075, "On-policy rollouts from changing weights. These are training observations rather than fixed-policy evaluations.",
             color=MUTED, fontsize=9)
    note = ("Unfinished episodes are omitted. The rolling median restarts at the checkpoint resume."
            if len(boundaries) == 1 else
            "Dashed lines mark resumes. Only episodes covered by a saved checkpoint appear.")
    fig.text(.09, .038, note, color=MUTED, fontsize=9)
    save(fig, directory, stem)


def plot_archive_evaluations(directory, cohorts):
    fig, axes = plt.subplots(1, len(cohorts), sharey=True, figsize=(11.5, 5.6))
    fig.subplots_adjust(left=.09, right=.96, top=.73, bottom=.27, wspace=.29)
    heading(fig, "Frozen policies on original TH14", "Normal Reimu A · Stage 1 · sampled actions · all compared policies cleared 0 of 8 episodes")
    for axis, (title, left, right) in zip(axes, cohorts, strict=True):
        colors = (SFT if left[0]["training_decisions"] == 0 else CHECKPOINT, WARM)
        jitter = [(i - (len(left) - 1) / 2) * .025 for i in range(len(left))]
        for shift, a, b in zip(jitter, left, right, strict=True):
            axis.plot([shift, 1 + shift], [a["stage_frame"], b["stage_frame"]], color="#ADB9C1", lw=.85, alpha=.8)
        for x, rows, color in zip((0, 1), (left, right), colors, strict=True):
            values = [row["stage_frame"] for row in rows]
            axis.scatter([x + offset for offset in jitter], values, s=27, color=color, zorder=3, linewidth=0)
            axis.hlines(median(values), x - .14, x + .14, color=color, lw=3, zorder=4)
        deltas = [b["stage_frame"] - a["stage_frame"] for a, b in zip(left, right, strict=True)]
        wins = sum(delta > 0 for delta in deltas)
        seeds = [row["policy_sampling_seed"] for row in left]
        axis.set_title(title, fontsize=10.5, weight="bold", color=INK, pad=13)
        axis.set_xticks((0, 1), [left[0]["arm_label"], right[0]["arm_label"]])
        axis.set_xlim(-.31, 1.31)
        axis.set_ylim(bottom=0)
        axis.text(.5, -.15, f"Policy sampling seeds {min(seeds)}–{max(seeds)}", ha="center",
                  transform=axis.transAxes, fontsize=8.5, color=MUTED)
        axis.text(.5, -.24, f"Right policy won {wins}/{len(deltas)} · median paired gain {median(deltas):+,.0f} frames", ha="center",
                  transform=axis.transAxes, fontsize=8.5, color=MUTED)
        style_axis(axis)
    axes[0].set_ylabel("Stage 1 endpoint (native frames)", color=INK, labelpad=8)
    fig.text(.09, .074, "Each dot is one episode. Lines pair policy sampling seeds within each panel. Thick bars show medians.",
             color=MUTED, fontsize=9)
    fig.text(.09, .036, "The panels use separate evaluation cohorts. Native game RNG was uncontrolled.", color=MUTED, fontsize=9)
    save(fig, directory, "ppo-archived-paired-evaluations")


def plot_on_policy_comparison(directory, warm, random, boundary):
    fig, axis = plt.subplots(figsize=(10.6, 5.8))
    fig.subplots_adjust(left=.09, right=.97, top=.75, bottom=.22)
    heading(fig, "PPO rollouts during training",
            f'TH14 1.00b · Normal Reimu A · Stage 1 · one PPO training seed ({warm[0]["training_seed"]})')
    for rows, color, name in ((warm, WARM, "SFT initialization"), (random, CHECKPOINT, "Random initialization")):
        for segment in sorted({row["segment"] for row in rows}):
            segment_rows = [row for row in rows if row["segment"] == segment]
            x = [row["cumulative_decisions"] for row in segment_rows]
            y = [row["stage_frame"] for row in segment_rows]
            axis.scatter(x, y, s=20, color=color, alpha=.48, linewidth=0, zorder=3)
            smoothed = [median(y[index - 6:index + 1]) for index in range(6, len(y))]
            axis.plot(x[6:], smoothed, color=color, lw=2.15, zorder=4,
                      label=name if segment == 1 else None)
    axis.axvline(boundary, color=SFT, lw=1, ls=(0, (3, 4)), zorder=2)
    axis.text(boundary + 230, .98, f"Resume at {boundary:,}", transform=axis.get_xaxis_transform(),
              color=MUTED, va="top", fontsize=9)
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Cumulative native training decisions", color=INK, labelpad=8)
    axis.set_ylabel("Stage 1 endpoint (native frames)", color=INK, labelpad=9)
    axis.xaxis.set_major_locator(MaxNLocator(8, integer=True))
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncols=2, frameon=False,
                borderaxespad=0, labelcolor=INK, fontsize=9)
    style_axis(axis)
    fig.text(.09, .075, "Dots show completed on-policy episodes. Lines show the trailing median of seven episodes.",
             color=MUTED, fontsize=9)
    fig.text(.09, .038, "Policies change during these rollouts. Review the frozen evaluations for checkpoint performance.",
             color=MUTED, fontsize=9)
    save(fig, directory, "ppo-matched-initialization-on-policy")


def plot_matched(directory, episodes, task):
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 6.25))
    fig.subplots_adjust(left=.085, right=.967, top=.745, bottom=.233, wspace=.32)
    stage = task["stage"]
    context = f'TH14 {task["version"]} · {task["difficulty"]} {task["shot"]} · Stage {stage}'
    training_seed = episodes[0]["ppo_training_seed"]
    heading(fig, "Native PPO learning curves by initialization", f"{context} · one PPO training seed ({training_seed})")
    budgets = sorted({row["training_decisions"] for row in episodes})
    for axis, field, title, label in zip(
        axes, ("stage_frame", "gain_from_own_anchor"), ("Absolute performance", "Change from each initialization"),
        ("Stage 1 endpoint (native frames)", "Gain since 0 decisions (native frames)"), strict=True,
    ):
        for budget in budgets:
            points = [row for row in episodes if row["training_decisions"] == budget]
            sft, random = ([row for row in points if row["initialization"] == init] for init in ("sft", "random"))
            for index, (a, b) in enumerate(zip(sft, random, strict=True)):
                shift = (index - (len(sft) - 1) / 2) * 85
                axis.plot([budget - 425 + shift, budget + 425 + shift], [a[field], b[field]],
                          color="#ABB8C0", linewidth=.65, alpha=.65, zorder=2)
        for init, color, marker, side, legend in (
            ("sft", WARM, "o", -1, "SFT-initialized PPO"),
            ("random", CHECKPOINT, "s", 1, "Randomly initialized PPO"),
        ):
            medians = []
            for budget in budgets:
                values = [row[field] for row in episodes if row["training_decisions"] == budget
                          and row["initialization"] == init]
                x = budget + 425 * side
                shifts = [(i - (len(values) - 1) / 2) * 85 for i in range(len(values))]
                axis.scatter([x + shift for shift in shifts], values, s=18, color=color, marker=marker,
                             alpha=.7, linewidth=0, zorder=3)
                medians.append(median(values))
            axis.plot([budget + 425 * side for budget in budgets], medians, color=color, lw=2.35,
                      marker=marker, ms=7, label=legend, zorder=4)
        axis.set_xticks(budgets, [f"{budget:,}" for budget in budgets])
        axis.set_xlim(-1_300, budgets[-1] + 1_300)
        axis.set_title(title, fontsize=11, color=INK, weight="bold", pad=10)
        axis.set_xlabel("Cumulative native training decisions", color=INK, labelpad=9)
        axis.set_ylabel(label, color=INK, labelpad=9)
        style_axis(axis)
        if field == "stage_frame":
            axis.set_ylim(bottom=0)
        else:
            axis.axhline(0, color=MUTED, lw=.75, zorder=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.08, .108), ncols=2, frameon=False,
               labelcolor=INK, fontsize=9.5)
    fig.text(.085, .076, "Solid lines connect checkpoint medians. Gray segments pair the eight policy sampling seeds at each budget.",
             color=MUTED, fontsize=9)
    fig.text(.085, .042, "Points are offset within budgets. Changes use each seed’s own zero-budget endpoint. Native game RNG was uncontrolled.",
             color=MUTED, fontsize=9)
    save(fig, directory, "ppo-matched-initialization-curve")


def plot_trial_pair(directory, path):
    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    names = sorted(report["sources"], key=lambda name: report["sources"][name]["steps"])
    if len(names) != 2:
        raise ValueError("A trial comparison requires two frozen checkpoints.")
    episodes = {name: {episode["policy_seed"]: episode for episode in report["episodes"][name]} for name in names}
    seeds = sorted(episodes[names[0]])
    if not seeds or any(set(episodes[name]) != set(seeds) or len(report["episodes"][name]) != len(seeds)
                        for name in names):
        raise ValueError("Each policy seed must appear once for each frozen checkpoint.")
    budgets = [report["sources"][name]["steps"] for name in names]
    rows = [
        {
            "policy_seed": seed, "training_decisions": budget,
            "native_stage_frame": episodes[name][seed]["native_stage_frame"],
            "outcome": episodes[name][seed]["outcome"],
        }
        for seed in seeds for name, budget in zip(names, budgets, strict=True)
    ]
    plot_frozen_checkpoints(directory, rows, report["task"]["setting"], f"trial-ppo-{budgets[0]}-vs-{budgets[1]}")


def plot_stock_pair(directory, path):
    report = read_report(path.parent, path.name)
    specification = report["specification"]
    controller = specification["controller"]
    if controller["decode"] != "sampled" or any(controller[key] for key in
                                               ("geometry", "aim_correction", "automatic_bomb")):
        raise ValueError("The stock figure requires pure sampled policies.")
    budgets = {name: policy.get("native_training_decisions", 0) for name, policy in specification["policies"].items()}
    if len(set(budgets.values())) != len(budgets):
        raise ValueError("The stock figure requires policies at distinct training budgets.")
    rows = [
        {
            "policy_seed": entry["trial"]["policy_seed"],
            "training_decisions": budgets[entry["trial"]["arm"]],
            "native_stage_frame": entry["result"]["stage_frame"],
            "outcome": entry["result"]["outcome"],
        }
        for entry in report["completed"]
    ]
    if len(rows) != report["matched_blocks"] * len(budgets):
        raise ValueError("The stock figure requires complete evaluation blocks.")
    context = "TH14 Steam 1.00b · Normal Reimu A · Stage 1"
    plot_frozen_checkpoints(directory, rows, context, f"stock-ppo-paired-through-{max(budgets.values())}")


def plot_frozen_checkpoints(directory, rows, context, stem):
    budgets = sorted({row["training_decisions"] for row in rows})
    seeds = sorted({row["policy_seed"] for row in rows})
    indexed = {(row["training_decisions"], row["policy_seed"]): row["native_stage_frame"] for row in rows}
    if not 2 <= len(budgets) <= 3 or len(indexed) != len(rows) or len(indexed) != len(seeds) * len(budgets):
        raise ValueError("The figure requires two or three budgets with the same paired seed cohort.")
    fig, axis = plt.subplots(figsize=(8.2, 5.25))
    fig.subplots_adjust(left=.13, right=.94, top=.75, bottom=.23)
    heading(fig, "Paired PPO checkpoint evaluation", f"{context} · {len(seeds)} paired policy seeds")
    offsets = [(index - (len(seeds) - 1) / 2) * .035 for index in range(len(seeds))]
    endpoints = [[indexed[budget, seed] for seed in seeds] for budget in budgets]
    for shift, *pair in zip(offsets, *endpoints, strict=True):
        axis.plot([index + shift for index in range(len(budgets))], pair, color="#ABB8C0", lw=.8, alpha=.8, zorder=2)
    for index, (values, color) in enumerate(zip(endpoints, (SFT, CHECKPOINT, WARM)[-len(budgets):], strict=True)):
        axis.scatter([index + shift for shift in offsets], values, color=color, s=30, linewidth=0, zorder=3)
        axis.hlines(median(values), index - .16, index + .16, color=color, lw=3, zorder=4)
    axis.set_xticks(range(len(budgets)), [f"{budget:,}" if budget else "Frozen SFT\n0" for budget in budgets])
    axis.set_xlim(-.35, len(budgets) - .65)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Cumulative native training decisions", color=INK, labelpad=8)
    axis.set_ylabel("Stage 1 endpoint (native frames)", color=INK, labelpad=9)
    style_axis(axis)
    fig.text(.13, .075, "Lines pair policy sampling seeds. Thick bars mark the medians.", color=MUTED, fontsize=9)
    fig.text(.13, .038, "The original game's RNG was uncontrolled.", color=MUTED, fontsize=9)
    write_csv(directory / f"{stem}.csv", rows)
    save(fig, directory, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ARCHIVE, help="Directory with the archived JSON reports.")
    parser.add_argument("--out", type=Path, help="Output directory. Defaults to figures under --results.")
    parser.add_argument("--matched", type=Path, help="Completed native report for the six matched initialization arms.")
    parser.add_argument("--random-traces", type=Path, nargs=2, metavar="REPORT",
                        help="Completed random-initialization training reports at 8,192 and 16,384 decisions.")
    parser.add_argument("--warm-continuations", type=Path, nargs="+", metavar="REPORT",
                        help="Ordered warm-start reports after 16,384, with a saved checkpoint in every report.")
    parser.add_argument("--trial-paired", type=Path, help="Paired frozen checkpoints on the original TH14 trial.")
    parser.add_argument("--stock-paired", type=Path, help="Paired frozen checkpoints on the Steam TH14 build.")
    parser.add_argument("--matched-training-seed", type=int, default=211,
                        help="PPO training seed for the matched figure. Defaults to the original paired run’s 211.")
    args = parser.parse_args()

    first = read_report(args.results, FIRST_REPORT)
    second = read_report(args.results, SECOND_REPORT)
    warm_reports = [(FIRST_REPORT, first), (SECOND_REPORT, second)]
    episodes, updates = archive_training(warm_reports)
    cohorts = paired_archive(first, second, read_report(args.results, HEAD_TO_HEAD))
    matched_run = matched_evaluations(args.matched, args.matched_training_seed) if args.matched is not None else None
    cold = None
    if args.random_traces:
        traces = [(path.name, read_report(path.parent, path.name)) for path in args.random_traces]
        if any(report.get("initialization") != "random" or report["seed"] != first["seed"]
               for _, report in traces):
            raise ValueError("Expected random-initialization training traces from the same training seed.")
        cold = archive_training(traces)
    directory = args.out or args.results / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / "ppo-on-policy-training-episodes.csv", episodes)
    write_csv(directory / "ppo-on-policy-updates.csv", updates)
    write_csv(directory / "ppo-archived-paired-evaluations.csv",
              [row for _, left, right in cohorts for row in (*left, *right)])
    with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "th14-ppo-curves"}):
        plot_training(directory, episodes, [first["trained_steps"]])
        plot_archive_evaluations(directory, cohorts)
        if args.trial_paired:
            plot_trial_pair(directory, args.trial_paired)
        if args.stock_paired:
            plot_stock_pair(directory, args.stock_paired)
        if args.warm_continuations:
            traces = list(warm_reports)
            for path in args.warm_continuations:
                with path.open(encoding="utf-8") as stream:
                    traces.append((str(Path(path.parent.name) / path.name), json.load(stream)))
            continuation_episodes, continuation_updates = archive_training(traces)
            write_csv(directory / "ppo-warm-continuation-episodes.csv", continuation_episodes)
            write_csv(directory / "ppo-warm-continuation-updates.csv", continuation_updates)
            plot_training(directory, continuation_episodes, [report["starting_steps"] for _, report in traces[1:]],
                          "ppo-warm-continuation")
        if matched_run is not None:
            matched, gains, task = matched_run
            write_csv(directory / "ppo-matched-initialization-evaluations.csv", matched)
            write_csv(directory / "ppo-matched-initialization-gains.csv", gains)
            plot_matched(directory, matched, task)
        if cold is not None:
            cold_episodes, cold_updates = cold
            for suffix, warm_rows, cold_rows in (
                ("episodes", episodes, cold_episodes), ("updates", updates, cold_updates),
            ):
                rows = [dict(initialization=init, **row) for init, source in (
                    ("sft", warm_rows), ("random", cold_rows),
                ) for row in source]
                write_csv(directory / f"ppo-matched-initialization-on-policy-{suffix}.csv", rows)
            plot_on_policy_comparison(directory, episodes, cold_episodes, first["trained_steps"])


if __name__ == "__main__":
    main()

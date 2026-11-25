"""Render a completed stock TH14 full-credit report without importing the game."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


MIDBOSS_FRAME = 2422
ARM_LABELS = {"sft": "Frozen SFT", "ppo211": "Frozen PPO · training seed 211",
              "ppo397": "Frozen PPO · training seed 397"}
COLORS = ("#0F766E", "#1D4ED8", "#B45309")
OUTCOMES = {"game_over": "Game Over", "clear": "Stage 2 arrival", "time_limit": "Decision cutoff"}
TERMINAL_MARKERS = {"game_over": "x", "clear": "D", "time_limit": ">"}


def credit_rows(report):
    specification = report["specification"]
    completed = report["completed"]
    schedule = specification["schedule"]
    if (report["status"] != "complete" or not schedule
            or [row["trial"] for row in completed] != schedule):
        raise ValueError("Expected a completed credit schedule with every scored episode.")

    rows = []
    for credit in completed:
        trial, result = credit["trial"], credit["result"]
        outcome = result["outcome"]
        stage2 = result["stage2_entry"]
        first_miss_frame = result["first_miss_frame"]
        game_over_frame = result["failure_frame"]
        if (outcome not in OUTCOMES or result["native_continues"] != 0
                or (stage2 is not None) != (outcome == "clear")
                or (game_over_frame is not None) != (outcome == "game_over")):
            raise ValueError(f"Unclassified or inconsistent credit outcome {trial}: {outcome}.")

        # The raw native clock can reset on entering Stage 2. The adapter's
        # Stage 1 frame preserves its final observation across that transition.
        end_frame = game_over_frame if outcome == "game_over" else result["stage_frame"]
        start_frame = result["start_frame"]
        if end_frame < start_frame or first_miss_frame is not None and not start_frame <= first_miss_frame <= end_frame:
            raise ValueError(f"Inconsistent native Stage 1 frame range in {trial}.")
        if first_miss_frame is None and outcome == "game_over":
            raise ValueError(f"Game Over lacks a first miss in {trial}.")

        rows.append(dict(
            arm=trial["arm"], policy_seed=trial["policy_seed"], block=trial["block"],
            order=trial["order"], start_frame=start_frame, first_miss_frame=first_miss_frame,
            first_miss_censored=first_miss_frame is None, endpoint=outcome,
            endpoint_stage1_frame=end_frame, game_over_frame=game_over_frame,
            game_over_right_censored=outcome == "time_limit",
            stage2_arrival=stage2 is not None,
            stage2_native_frame=stage2["frame"] if stage2 else None,
            stage2_lives=result["stage2_lives"], stage2_bombs=result["stage2_bombs"],
            actual_bombs_consumed=result["actual_bombs_consumed"], deaths=result["deaths"],
            native_score_gain=result["score"], decisions=result["decision_count"],
        ))
    return rows


def arm_summary(rows):
    groups = {arm: [row for row in rows if row["arm"] == arm]
              for arm in dict.fromkeys(row["arm"] for row in rows)}
    statistics = {}
    for arm, group in groups.items():
        misses = [row["first_miss_frame"] for row in group if not row["first_miss_censored"]]
        failures = [row["game_over_frame"] for row in group if row["game_over_frame"] is not None]
        statistics[arm] = dict(
            credits=len(group), stage2_arrivals=sum(row["stage2_arrival"] for row in group),
            game_overs=len(failures), game_over_right_censored=sum(row["game_over_right_censored"] for row in group),
            first_misses_observed=len(misses), first_miss_censored=len(group) - len(misses),
            median_observed_first_miss_frame=median(misses) if misses else None,
            median_observed_game_over_frame=median(failures) if failures else None,
            crossed_midboss_reference_frame=sum(row["endpoint_stage1_frame"] >= MIDBOSS_FRAME for row in group),
            deaths=sum(row["deaths"] for row in group),
            actual_bombs_consumed=sum(row["actual_bombs_consumed"] for row in group),
        )
    return groups, statistics


def draw_credits(groups, statistics, path):
    color_text = "#132337"
    count = sum(len(group) for group in groups.values())
    total_arrivals = sum(stats["stage2_arrivals"] for stats in statistics.values())
    limits = sum(stats["game_over_right_censored"] for stats in statistics.values())
    observed = {row["endpoint"] for group in groups.values() for row in group}
    observations = [row["endpoint_stage1_frame"] for group in groups.values() for row in group]
    right = max(max(observations), MIDBOSS_FRAME) * 1.055

    figure, axes = plt.subplots(len(groups), 1, sharex=True, squeeze=False,
                                figsize=(9.4, 2.05 * len(groups) + 1.7), facecolor="white")
    axes = axes[:, 0]
    figure.subplots_adjust(left=.12, right=.96, top=.815, bottom=.185, hspace=.66)
    figure.text(.12, .96, "TH14 Stage 1 · Full-credit outcomes", fontsize=22,
                color=color_text, fontweight="bold", va="top")
    figure.text(.12, .908, "Steam 1.00b  /  Normal Reimu A  /  Frozen policies  /  Zero continues",
                fontsize=10, color="#475569", va="top")
    figure.text(.12, .876, f"Stage 2 arrivals  {total_arrivals}/{count}      Decision-cutoff censors  {limits}",
                fontsize=10, color="#475569", va="top")
    figure.supylabel("Policy seed", x=.025, y=.50, fontsize=10, color=color_text)

    for index, (axis, (arm, group)) in enumerate(zip(axes, groups.items())):
        color = COLORS[index % len(COLORS)]
        group = sorted(group, key=lambda row: row["policy_seed"])
        axis.set_title(ARM_LABELS.get(arm, arm), loc="left", fontsize=12,
                       color=color, fontweight="bold", pad=7)
        axis.set_title(f"Stage 2  {statistics[arm]['stage2_arrivals']}/{len(group)}",
                       loc="right", fontsize=10, color="#475569", pad=7)
        axis.axvline(MIDBOSS_FRAME, color="#64748B", linestyle=(0, (4, 3)), linewidth=1)
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#E2E8F0", linewidth=.7)
        axis.set_ylim(len(group) - .45, -.55)
        axis.set_xlim(0, right)
        axis.set_yticks(range(len(group)), [str(row["policy_seed"]) for row in group])
        axis.tick_params(axis="both", length=0, labelsize=9, colors="#475569", pad=6)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for position, row in enumerate(group):
            endpoint = row["endpoint_stage1_frame"]
            first_miss = row["first_miss_frame"]
            axis.hlines(position, row["start_frame"], endpoint, linewidth=2.6, color=color, alpha=.35)
            symbol_lines = 1.65
            if first_miss is None:
                axis.scatter(endpoint, position - .15, marker="o", s=25, facecolors="white",
                             edgecolors=color, linewidths=1.4, zorder=5, clip_on=False)
                position += .11
            else:
                axis.scatter(first_miss, position, marker="o", s=40, color=color,
                             edgecolors="white", linewidths=.6, zorder=5)
            options = {"facecolors": "white", "edgecolors": color_text} if row["endpoint"] == "time_limit" else {"color": color_text}
            axis.scatter(endpoint, position, marker=TERMINAL_MARKERS[row["endpoint"]], s=52,
                         linewidths=symbol_lines, zorder=6, clip_on=False, **options)

    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    axes[-1].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[-1].set_xlabel("Native Stage 1 frame", fontsize=10, color=color_text, labelpad=9)
    handles = [Line2D([], [], marker="o", color="#475569", linestyle="None", label="First miss", markersize=5)]
    for outcome, label in OUTCOMES.items():
        if outcome in observed:
            handles.append(Line2D([], [], marker=TERMINAL_MARKERS[outcome], color=color_text,
                                  markerfacecolor="white" if outcome == "time_limit" else color_text,
                                  linestyle="None", label=label, markersize=6))
    if any(row["first_miss_censored"] for group in groups.values() for row in group):
        handles.append(Line2D([], [], marker="o", color="#475569", markerfacecolor="white",
                              linestyle="None", label="First miss censored", markersize=5))
    handles.append(Line2D([], [], color="#64748B", linestyle=(0, (4, 3)), label="First midboss ≈ 2,422"))
    figure.legend(handles=handles, frameon=False, loc="lower left", bbox_to_anchor=(.109, .045),
                  ncol=min(len(handles), 4), fontsize=9, labelcolor="#334155")
    figure.savefig(path, dpi=160, metadata={"Title": "TH14 Stage 1 full-credit outcomes"})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    output = args.out or args.report.parent
    rows = credit_rows(report)
    groups, statistics = arm_summary(rows)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "stage1-credits.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "stage1-summary.json").open("w", encoding="utf-8") as stream:
        json.dump(dict(
            task=report["specification"]["task"], credits=len(rows), midboss_reference_frame=MIDBOSS_FRAME,
            stage2_arrivals=sum(stats["stage2_arrivals"] for stats in statistics.values()),
            endpoint_semantics={
                "game_over": "Observed native Game Over after exhausting the credit.",
                "clear": "Observed Stage 2 arrival. The plot uses the last recorded Stage 1 frame.",
                "time_limit": "Right-censored at the last observed Stage 1 frame.",
            },
            arms=statistics,
        ), stream, indent=2, allow_nan=False)
        stream.write("\n")
    draw_credits(groups, statistics, output / "stage1-first-miss-and-endpoints.png")


if __name__ == "__main__":
    main()
